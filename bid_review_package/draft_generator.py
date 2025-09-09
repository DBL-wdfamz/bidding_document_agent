import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

os.environ["OPENAI_API_KEY"] = "sk-mxchjxfcabutrychpqivgpcfszpqdpzqyejnflmgafvmpoym"

# 创建LLM实例，使用硅基流动的端点
llm = ChatOpenAI(
    temperature=0.2,
    base_url="https://api.siliconflow.cn/v1",
    model="deepseek-ai/DeepSeek-V3",
    streaming=True  # 启用流式输出
)

# 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """你是一个专业的招投标文书生成系统，专为政府采购和招标场景设计。
你的任务是根据用户提供的JSON输入，生成一份结构化的招标公告文档。

输入格式为一个JSON对象，包含三个主要部分：
1.  `user_input`: 包含项目的基本信息，如 `project_name` 和 `core_requirements`。
2.  `template_styles`: 一个对象，定义了文档中可用的样式。你需要从这个对象的键（key）中提取样式名称（如 title, bodyText, heading1 等）并应用到 `contentBlocks` 的 `style` 字段中。
3.  `retrieved_context`: 包含相关的法律法规或参考资料，你需要在生成内容时参考这些信息以确保合规性和准确性。

请严格按照以下JSON模板结构生成输出，只返回生成的JSON内容，不要包含任何额外的解释或外部字段如 `pageSetup` 或 `styles`。

输出模板结构：
{{
  "docInfo": {{
    "docType": "招标公告",
    "title": "",       // 根据 user_input 生成
    "urgency": "",     // 如果信息不足，编写
    "security": ""     // 如果信息不足，编写
  }},
  "contentBlocks": [
    {{
      "style": "...",     // 从输入的 `template_styles` 的键中选择
      "content":"..." // 根据 `user_input` 和 `retrieved_context` 生成
    }}
    // ... 更多内容块
  ]
}}

**关键要求:**
1.  **内容生成**: `content` 字段的内容必须根据 `user_input` 和 `retrieved_context` 生成。确保招标公告的完整性，涵盖项目概述、投标人资格、时间安排、保证金要求等关键部分。
2.  **`style`**: `contentBlocks` 中的 `style` 字段值必须从输入数据 `template_styles` 对象的键中动态获取。
3.  **禁止项**: 最终输出的JSON中不要包含 `pageSetup` 或 `styles` 字段。
"""
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# 创建对话链
chain = prompt | llm

# 创建消息历史
history = ChatMessageHistory()

# 使用RunnableWithMessageHistory
conversation = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="history"
)

# 生成初稿的函数（流式版本）
async def generate_draft_stream(prompt_text):
    """生成招标公告初稿（流式输出）"""
    async for chunk in conversation.astream({"input": prompt_text}, config={"configurable": {"session_id": "default"}}):
        yield chunk.content

# 生成初稿的可调用方法
async def generate_draft(prompt_text):
    """生成招标公告初稿并返回完整内容"""
    full_content = ""
    async for chunk in generate_draft_stream(prompt_text):
        full_content += chunk
    return full_content
