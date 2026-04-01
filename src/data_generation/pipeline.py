import os
import json
import random
import argparse
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()  # 加载根目录下的 .env 文件中的环境变量


# --- 1. 数据结构定义 (Schemas) ---
class IntentDef(BaseModel):
    brand: str
    is_common: bool
    intent_name: str
    description: str


class SynthesizedQuery(BaseModel):
    intent: IntentDef
    query: str
    is_valid: bool = True


class FinalTrainingData(BaseModel):
    messages: List[Dict[str, str]]


# --- 2. API 配置 (Configuration) ---
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")
QWEN_BASE_URL = os.environ.get(
    "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen-turbo")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- 3. 提示词定义 (Prompts) ---
SYNTHESIS_PROMPT_TMPL = """<system>你是一个用户语料生成专家。请根据提供的[意图名称]和[意图描述],生成{num_queries}条不同用户可能会说的真实查询语句。请仅输出查询语句本身, 每行一条, 不要包含任何编号或其他多余说明。</system>

<rules>
1. 语言风格需尽可能多样化: 包括口语化表述、极简短语、冗长且带情绪的抱怨、错别字或倒装句等。
2. 模拟不同人群: 例如不熟悉电子产品的老年人、急躁的年轻人等不同受众的语气。
3. 必须紧扣意图描述, 但绝对不要直接照抄“意图名称”中的词汇。
</rules>

<input>
意图名称: {intent_name}
意图描述: {intent_description}
</input>"""

JUDGE_PROMPT_TMPL = """<system>你是一个严格的数据质量审查员。请评估以下用户Query是否**明确且唯一**地指向给定的目标意图。</system>

<input>
目标意图: {intent_name}
用户Query: {query}
</input>

<instruction>
判断逻辑: 
- 如果该Query可能同时属于其他意图, 不具备唯一指向性, 请输出"REJECT"
- 如果该Query极其含糊不清, 请输出"REJECT"
- 如果指向上述意图的意图非常明确, 请输出"ACCEPT"

请仅输出 ACCEPT 或 REJECT, 不要输出其他任何多余内容。
</instruction>"""

# --- 4. 核心逻辑 ---


def load_intents(filepath: str) -> List[IntentDef]:
    """读取并展平所有的意图配置"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    intents = []
    for brand, intent_list in data.items():
        is_common = brand == "公共"
        for item in intent_list:
            intents.append(
                IntentDef(
                    brand=brand,
                    is_common=is_common,
                    intent_name=item["intent_name"],
                    description=item["description"],
                )
            )
    return intents


def generate_queries_for_intent(
    client: OpenAI, intent: IntentDef, num_queries: int = 20
) -> List[SynthesizedQuery]:
    """为单一意图生成查询语料"""
    prompt = SYNTHESIS_PROMPT_TMPL.format(
        num_queries=num_queries,
        intent_name=intent.intent_name,
        intent_description=intent.description,
    )
    try:
        logging.debug(f"[API] 请求 Qwen API 生成 Query -> {intent.intent_name}")
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            max_tokens=1024,
        )
        content = response.choices[0].message.content.strip()
        queries = [q.strip() for q in content.split("\n") if q.strip()]

        import re

        clean_queries = []
        for q in queries:
            q = re.sub(r"^[\d\.\、\-\s]+", "", q)
            if q:
                clean_queries.append(q)

        logging.debug(
            f"[API] 成功生成 {len(clean_queries)} 条 Query -> {intent.intent_name}"
        )
        return [SynthesizedQuery(intent=intent, query=q) for q in clean_queries]
    except Exception as e:
        logging.error(f"[Error] 生成 Query 失败 ({intent.intent_name}): {e}")
        return []


def node_a_synthesize(
    client: OpenAI,
    intents: List[IntentDef],
    max_workers: int = 4,
    queries_per_intent: int = 200,
    batch_size: int = 20,
) -> List[SynthesizedQuery]:
    """节点A: 并发调度所有意图生成查询"""
    logging.info(
        f">>> [节点 A] 开始批量合成 Queries (目标每个意图 {queries_per_intent} 条)..."
    )
    all_queries = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_intent = {}
        for intent in intents:
            num_batches = queries_per_intent // batch_size
            remainder = queries_per_intent % batch_size
            for _ in range(num_batches):
                future = executor.submit(
                    generate_queries_for_intent, client, intent, batch_size
                )
                future_to_intent[future] = intent
            if remainder > 0:
                future = executor.submit(
                    generate_queries_for_intent, client, intent, remainder
                )
                future_to_intent[future] = intent

        for future in tqdm(
            as_completed(future_to_intent),
            total=len(future_to_intent),
            desc="Synthesizing",
        ):
            all_queries.extend(future.result())
    logging.info(f"-> 节点 A 产出: 共计 {len(all_queries)} 条初筛 Queries.")
    return all_queries


def judge_query(client: OpenAI, item: SynthesizedQuery) -> SynthesizedQuery:
    prompt = JUDGE_PROMPT_TMPL.format(
        intent_name=item.intent.intent_name, query=item.query
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=10,
        )
        judgment = response.choices[0].message.content.strip().upper()
        if "REJECT" in judgment:
            item.is_valid = False
            logging.debug(f"[API] 拒绝 Query -> '{item.query}'")
        else:
            item.is_valid = True
            logging.debug(f"[API] 接受 Query -> '{item.query}'")
    except Exception as e:
        logging.error(f"[Error] 判断 Query 失败 '{item.query}': {e}")
        item.is_valid = False
    return item


def node_b_judge(
    client: OpenAI, queries: List[SynthesizedQuery], max_workers: int = 8
) -> List[SynthesizedQuery]:
    """节点B: 对生成的所有预料执行清洗过滤"""
    logging.info(f">>> [节点 B] 开始执行质量清洗 (LLM-as-a-Judge)...")
    valid_queries = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_query = {executor.submit(judge_query, client, q): q for q in queries}
        for future in tqdm(
            as_completed(future_to_query), total=len(queries), desc="Judging"
        ):
            res = future.result()
            if res.is_valid:
                valid_queries.append(res)
    rejected_count = len(queries) - len(valid_queries)
    logging.info(
        f"-> 节点 B 产出: 过滤后剩余有效 Queries 数量: {len(valid_queries)} (移除了 {rejected_count} 条无意义断言/二义性数据)."
    )
    return valid_queries


def node_c_assemble(
    valid_queries: List[SynthesizedQuery],
    all_intents: List[IntentDef],
    output_dir: str,
    ood_ratio: float = 0.15,
):
    """节点C: 基于向量空间挖掘难负样本并组装成最终 ShareGPT 训练格式"""
    logging.info(">>> [节点 C] 开始动态拼接与 ShareGPT 格式组装 (Dynamic Assembly)...")

    logging.info(
        "-> 正在加载 SentenceTransformers Embedding 模型 (进行 Hard Negative 挖掘)..."
    )
    # 抑制部分句向量库启动警告
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        logging.error(f"无法加载模型 {EMBEDDING_MODEL_NAME}: {e}")
        return

    # 构建品牌视角列表 (排除 公共)
    all_brands = list(set([i.brand for i in all_intents if not i.is_common]))

    # 对所有意图描述进行预向量化
    intent_descriptions = [i.description for i in all_intents]
    embeddings = model.encode(intent_descriptions)

    # 构建相似度矩阵
    sim_matrix = cosine_similarity(embeddings)

    training_records = []

    for query_item in tqdm(valid_queries, desc="Assembling"):
        true_intent = query_item.intent
        true_idx = all_intents.index(true_intent)

        # --- 视角(Tenant View) 决定逻辑 ---
        # 1. 如果意图是公共的，随机指定一个品牌视角
        if true_intent.is_common:
            target_brand = random.choice(all_brands) if all_brands else "公共"
            is_cross_brand_ood = False
        else:
            # 2. 如果是品牌专有，以 ood_ratio 的概率将其放入无关品牌视角中
            if all_brands and len(all_brands) > 1 and random.random() < ood_ratio:
                is_cross_brand_ood = True
                other_brands = [b for b in all_brands if b != true_intent.brand]
                target_brand = random.choice(other_brands)
            else:
                is_cross_brand_ood = False
                target_brand = true_intent.brand

        # 确定视角内的意图池 (View Pool)
        view_pool = [i for i in all_intents if i.is_common or i.brand == target_brand]

        # --- 采样策略分析 ---

        # 1. 挖掘 Hard Negatives: 在当前 View Pool 内寻找与真实意图最相似的干扰项
        similarities = sim_matrix[true_idx]
        sorted_indices = similarities.argsort()[::-1]

        hard_negatives = []
        for idx in sorted_indices:
            cand_intent = all_intents[idx]
            if cand_intent != true_intent and cand_intent in view_pool:
                hard_negatives.append(cand_intent)
            if len(hard_negatives) >= 2:  # 限定最多取 2 个难负样本
                break

        # 2. 随机抽取 Easy Negatives: 在 View Pool 内随机混淆项
        pool = [i for i in view_pool if i != true_intent and i not in hard_negatives]
        easy_neg_count = min(len(pool), random.randint(3, 10))
        easy_negatives = random.sample(pool, easy_neg_count) if pool else []

        # 3. 合并候选列表
        candidates = hard_negatives + easy_negatives

        # 4. 构建跨品牌 OOD (Out-Of-Distribution) 样本
        if is_cross_brand_ood:
            # 放入相反品牌环境，必然不存在正例
            label = "未知意图"
            context_log = f"-> 组装样本: [跨品牌OOD] 真实意图 [{true_intent.intent_name}] 置入 [{target_brand}] 视角。 Label: {label} | Hard Negatives: {[i.intent_name for i in hard_negatives]}"
        else:
            # 放入正样本, 并设定正确标签
            candidates.append(true_intent)
            label = true_intent.intent_name
            context_log = f"-> 组装样本: [正确] 视角 [{target_brand}] Label: {label} | Hard Negatives: {[i.intent_name for i in hard_negatives]}"

        logging.debug(context_log)

        # 5. 打乱候选意图顺序, 消除位置依赖偏置 (Shuffle)
        random.shuffle(candidates)

        # 6. 生成适用于 LLaMA-Factory 的 ShareGPT 数据组装格式
        candidate_strs = []
        for idx, cand in enumerate(candidates):
            candidate_strs.append(f"- 【{cand.intent_name}】: {cand.description}")

        candidates_block = "\n".join(candidate_strs)

        sys_prompt = "你是一个智能客服意图分类助手。请根据提供的<可选意图列表>和它们的描述, 仔细分析用户的<查询>, 并输出唯一匹配的意图名称。如果用户的查询明显不属于列表中的任何一个意图, 请输出'未知意图'。"
        user_prompt = f"<可选意图列表>\n{candidates_block}\n</可选意图列表>\n\n<查询>\n{query_item.query}\n</查询>\n\n你的判定结果是: "

        record = {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": label},
            ]
        }
        training_records.append(record)

    # 划分训练集和测试集 (85:15)
    train_records, val_records = train_test_split(
        training_records, test_size=0.15, random_state=42
    )

    train_file = os.path.join(output_dir, "train.jsonl")
    val_file = os.path.join(output_dir, "val.jsonl")

    logging.info(f"-> 正在将 {len(train_records)} 条训练数据写入: {train_file} ...")
    with open(train_file, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logging.info(f"-> 正在将 {len(val_records)} 条验证数据写入: {val_file} ...")
    with open(val_file, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logging.info(">>> 节点 C 执行完毕, 流水线构建成功！")


def main():
    parser = argparse.ArgumentParser(
        description="意图分类数据集自动生成与组装 DAG流水线"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="data/intents.json",
        help="意图字典配置路径 (如 data/intents.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="输出切分后数据集的目录 (默认 data)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="【调试日志】开启 DEBUG 级别详细日志输出"
    )
    args = parser.parse_args()

    # --- 设定日志级别 ---
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # 抑制其他库的冗余日志
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    intents = load_intents(args.config)
    logging.info(f"成功加载意图字典配置, 共识别出 {len(intents)} 条意图定义。")

    # 实例化 API Client
    if not QWEN_API_KEY:
        logging.error("[致命错误] 未检测到 QWEN_API_KEY 环境变量！")
        return

    client = OpenAI(
        api_key=QWEN_API_KEY if QWEN_API_KEY else "dummy",
        base_url=QWEN_BASE_URL,
    )

    # =============== 【 执行 DAG 核心流 】 ===============

    # [节点 A: Data Synthesis]
    queries = node_a_synthesize(client, intents)
    if not queries:
        return

    # [节点 B: Error Rate Filtering / LLM-as-a-Judge]
    valid_queries = node_b_judge(client, queries)
    if not valid_queries:
        return

    # [节点 C: Assembly and Hard Negative Mining]
    node_c_assemble(valid_queries, intents, args.output_dir)


if __name__ == "__main__":
    main()
