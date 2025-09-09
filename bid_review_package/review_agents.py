import os
import asyncio
import time
from langchain_openai import ChatOpenAI
import re
import json

os.environ["OPENAI_API_KEY"] = "sk-mxchjxfcabutrychpqivgpcfszpqdpzqyejnflmgafvmpoym"

# 创建LLM实例，使用硅基流动的端点
llm = ChatOpenAI(
    temperature=0.2,
    base_url="https://api.siliconflow.cn/v1",
    model="deepseek-ai/DeepSeek-V3",
    streaming=True  # 启用流式输出
)

def _get_text_for_review(content: str) -> str:
    """智能提取用于审查的文本，兼容两种输入格式"""
    try:
        data = json.loads(content)
        # 如果是新的JSON结构，提取 text_to_review
        if "text_to_review" in data:
            # 如果text_to_review本身也是个JSON，美化它
            if isinstance(data["text_to_review"], dict):
                 return json.dumps(data["text_to_review"], ensure_ascii=False, indent=2)
            return data["text_to_review"]
    except (json.JSONDecodeError, TypeError):
        # 如果不是JSON或者没有text_to_review，则返回原始输入
        pass
    return content

# 财政与逻辑校验Agent
async def fiscal_logic_agent(content):
    text_to_review = _get_text_for_review(content)
    review_prompt = f"""
你是财政局招投标文书审查员，请对以下招标公告JSON文稿进行财政与逻辑合规性审查：
{text_to_review}
请重点检查：
1. 预算金额是否合理、资金来源是否明确；
2. 项目工期与预算分配是否匹配；
3. 项目名称、编号、采购人信息的一致性；
4. 投标保证金金额是否符合项目规模；
5. 投标截止时间与开标时间逻辑是否合理。
请生成一个修改建议列表，格式为JSON数组，例如：
[
  {{
    "original": "修改前句子",
    "modified": "修改后句子",
    "reason": "修改原因"
  }}
]
如果没有问题，返回空数组[]。只返回JSON数组，不要包含任何额外的解释。确保返回有效的JSON格式。
"""
    # 使用异步生成器，逐块yield内容
    async for chunk in llm.astream(review_prompt):
        yield chunk.content

# 行文风格与术语审查Agent
async def style_term_agent(content):
    text_to_review = _get_text_for_review(content)
    review_prompt = f"""
你是政策处招投标文书审查员，请对以下招标公告JSON文稿进行语言润色和术语规范性审查：
{text_to_review}
请重点检查：
1. 是否使用正式、规范的政府公文语言，避免口语化表达；
2. 专业术语是否准确，如"招标人"、"投标人"、"中标人"、"投标保证金"、"投标截止时间"等；
3. 标点符号、格式是否符合公文规范；
4. JSON结构中的文本是否正式、专业，是否符合招标公告的严肃性。
请生成一个修改建议列表，格式为JSON数组，例如：
[
  {{
    "original": "修改前句子",
    "modified": "修改后句子",
    "reason": "修改原因"
  }}
]
如果没有问题，返回空数组[]。只返回JSON数组，不要包含任何额外的解释。确保返回有效的JSON格式。
"""
    async for chunk in llm.astream(review_prompt):
        yield chunk.content

# 政策合规与时效性审查Agent
async def policy_compliance_timeliness_agent(content):
    # 解析输入内容
    try:
        input_data = json.loads(content)
        text_to_review = input_data.get("text_to_review", "")
        retrieved_rules = input_data.get("retrieved_rules", [])
        
        # 如果text_to_review是字典，转换为JSON字符串
        if isinstance(text_to_review, dict):
            text_to_review = json.dumps(text_to_review, ensure_ascii=False, indent=2)
    except (json.JSONDecodeError, KeyError):
        # 如果解析失败，使用默认值
        text_to_review = content
        retrieved_rules = [
            "政府采购法第X条：招标公告应符合相关规定。",
            "招标投标法第Y条：投标截止时间应合理。",
            "最新修订：招标投标法于2023年修订，强调时效性。"
        ]

    review_prompt = f"""
你是发改局招投标文书审查员，请对以下招标公告正文进行政策合规与时效性审查：

待审查正文：
{text_to_review}

检索到的相关法规条款：
{chr(10).join(f"{i+1}. {rule}" for i, rule in enumerate(retrieved_rules))}

请重点检查：
1. 内容合规性：正文内容是否违反检索到的法规条款规定；
2. 政策时效性：检索到的条款是否为最新版本，是否存在"已废止"、"已修订"等信息，并发出预警；
3. 条款引用准确性：正文中引用的法规条款是否与检索到的条款一致；
4. 合规风险评估：是否存在潜在的政策风险或合规问题。

请生成一个修改建议列表，格式为JSON数组，例如：
[
  {{
    "original": "修改前句子",
    "modified": "修改后句子",
    "reason": "修改原因",
    "risk_level": "高/中/低"
  }}
]
如果没有问题，返回空数组[]。只返回JSON数组，不要包含任何额外的解释。确保返回有效的JSON格式。
"""
    async for chunk in llm.astream(review_prompt):
        yield chunk.content

# 文号管理Agent
async def document_number_agent(content):
    text_to_review = _get_text_for_review(content)
    # 模拟数据库查询历史文号（简化版：假设历史文号列表）
    historical_numbers = ["2023-交通-001", "2023-交通-002", "2024-交通-001"]

    # 提取文号（简化：假设从content中解析）
    doc_number_match = re.search(r'"docNumber":\s*"([^"]*)"', text_to_review)
    generated_number = doc_number_match.group(1) if doc_number_match else ""

    # 检查格式（年份-部门-序号）
    is_valid_format = bool(re.match(r'^\d{4}-[^-]+-\d{3}$', generated_number))
    is_unique = generated_number not in historical_numbers

    review_prompt = f"""
你是文号管理审查员，请对以下招标公告JSON文稿中的文号进行校验：
{text_to_review}

生成的文号：{generated_number}
历史文号列表：{', '.join(historical_numbers)}

请重点检查：
1. 文号格式：是否符合 年份-部门-序号 格式；
2. 文号唯一性：是否与历史文号重复。
请生成一个修改建议列表，格式为JSON数组，例如：
[
  {{
    "original": "修改前文号",
    "modified": "修改后文号",
    "reason": "修改原因"
  }}
]
如果没有问题，返回空数组[]。只返回JSON数组，不要包含任何额外的解释。确保返回有效的JSON格式。
"""
    async for chunk in llm.astream(review_prompt):
        yield chunk.content

# 新增：招投标公文内容完整性审查Agent
async def completeness_agent(content):
    text_to_review = _get_text_for_review(content)
    review_prompt = f"""
你是招投标文书完整性审查员，请对以下招标公告JSON文稿进行内容完整性审查：
{text_to_review}
请重点检查招标公告是否包含以下必要内容：
1. 项目概况（包括项目名称、预算金额、资金来源等）；
2. 招标条件（项目批准情况、资金落实等）；
3. 投标人资格要求（资质、信誉、财务状况等）；
4. 招标文件获取方式（时间、地点、方式等）；
5. 投标保证金（金额、提交方式等）；
6. 投标截止时间和开标时间；
7. 合同主要条款（工期、付款节点、违约责任、争议解决等）；
8. 联系方式（招标人、地址、联系人、电话等）；
9. 其他必要信息（如投标文件要求、评标办法等）。
请生成一个修改建议列表，格式为JSON数组，例如：
[
  {{
    "original": "缺失部分",
    "modified": "建议添加的内容",
    "reason": "缺失原因或补充说明"
  }}
]
如果内容完整，返回空数组[]。只返回JSON数组，不要包含任何额外的解释。确保返回有效的JSON格式。
"""
    async for chunk in llm.astream(review_prompt):
        yield chunk.content

# 新增：单个Agent调用方法（流式版本）
async def review_fiscal(content):
    """财政与逻辑校验审查（流式输出）"""
    async for chunk in fiscal_logic_agent(content):
        yield chunk

async def review_style(content):
    """行文风格与术语审查（流式输出）"""
    async for chunk in style_term_agent(content):
        yield chunk

async def review_policy(content):
    """政策合规与时效性审查（流式输出）"""
    async for chunk in policy_compliance_timeliness_agent(content):
        yield chunk

async def review_document_number(content):
    """文号管理审查（流式输出）"""
    async for chunk in document_number_agent(content):
        yield chunk

async def review_completeness(content):
    """招投标公文内容完整性审查（流式输出）"""
    async for chunk in completeness_agent(content):
        yield chunk

# 新增：并行流式审查方法（使用流式review_*方法）
async def run_parallel_streaming_reviews(content):
    """并行流式审查所有Agent，使用流式review_*方法"""
    print("\n--- 并行流式审查开始（使用流式方法）---")
    start_time = time.time()

    # 创建一个共享的异步队列
    print_queue = asyncio.Queue()

    # 生产者：运行流式review方法，解析JSON，并将结果放入队列
    async def streaming_producer(review_func, prefix):
        # 移除特殊处理，所有Agent都使用统一的content
        buffer = ""
        in_array = False

        try:
            async for chunk in review_func(content):
                buffer += chunk
                # 移除此处的实时打印，防止并行输出时内容混乱
                # print(chunk, end="", flush=True)

                # 寻找数组的开始
                if not in_array:
                    if '[' in buffer:
                        in_array = True
                        buffer = buffer[buffer.find('['):]  # 丢弃开头非JSON部分
                    else:
                        continue

                # 寻找并处理数组中的对象
                while '{' in buffer and '}' in buffer:
                    start_index = buffer.find('{')
                    end_index = buffer.find('}')

                    if start_index < end_index:
                        obj_str = buffer[start_index: end_index + 1]
                        try:
                            parsed_obj = json.loads(obj_str)
                            # 将结果放入队列
                            await print_queue.put((prefix, parsed_obj))
                            buffer = buffer[end_index + 1:]
                        except json.JSONDecodeError:
                            # 如果解析失败，说明对象不完整，跳出内循环继续接收chunk
                            break
                    else:
                        # 如果 '}' 在 '{' 前面，说明buffer有问题，丢弃错误部分
                        buffer = buffer[start_index:]
                        break

        except Exception as e:
            await print_queue.put((f"{prefix} [错误]:", str(e)))
        finally:
            # 发送结束信号
            await print_queue.put((prefix, None))

    # 消费者：从队列中取出结果并打印（整齐格式）
    async def print_consumer(num_agents):
        finished_agents = 0
        while finished_agents < num_agents:
            prefix, item = await print_queue.get()
            if item is None:
                finished_agents += 1
            else:
                print(f"\n{prefix}")
                pretty_obj = json.dumps(item, ensure_ascii=False, indent=2)
                for line in pretty_obj.splitlines():
                    print(f"  {line}")
            print_queue.task_done()

    # 定义所有流式审查任务
    review_tasks = [
        (review_fiscal, "[财政审查]:"),
        (review_style, "[风格审查]:"),
        (review_policy, "[政策审查]:"),
        (review_document_number, "[文号审查]:"),
        (review_completeness, "[完整性审查]:")
    ]

    # 启动消费者和所有生产者
    consumer_task = asyncio.create_task(print_consumer(len(review_tasks)))
    producers = [streaming_producer(func, prefix) for func, prefix in review_tasks]
    await asyncio.gather(*producers)

    # 等待队列中所有任务被处理完毕
    await print_queue.join()
    consumer_task.cancel()  # 结束消费者任务

    end_time = time.time()
    print(f"\n\n并行流式审查总运行时间: {end_time - start_time:.2f} 秒")
