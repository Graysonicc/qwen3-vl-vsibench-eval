# 工具调用的逻辑
1. 在tools的文件夹下，每个工具里都有multi_deploy.py这个文件
2. 深度估计调用的是qwen3_vl_tool_use/tools/Depth-Anything-V2/multi_deploy.py L198， 
3. 目标检测调用的是qwen3_vl_tool_use/tools/LLMDet/multi_deploy.py L307

总的工具调用在qwen3_vl_tool_use/verl/workers/agent/qwen_tools.py中，看各个工具类的call方法，有的例如zoom in的工具就直接在这个文件中处理了，而depth和detection是封装成api调用的



# 推理流程
1. qwen3_vl_tool_use/eval/agent_eval.py L704 --> L663 --> L466 --> L257 看agent.chat_with_tools方法
2. --> qwen3_vl_tool_use/verl/workers/agent/api_agent.py L573, 其中generate_conversation就是多轮对话
3. --> qwen3_vl_tool_use/verl/workers/agent/api_agent.py L447 就是调用一次qwen模型， L500就是调用模型回答后要调用的工具，相关回复的prompt也在里面

# vsibench数据处理
qwen3_vl_tool_use/test.py   包括每个样本的数据形式以及prompt
