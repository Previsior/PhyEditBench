from openai import OpenAI
import os
import base64
import argparse
import pandas as pd
import json
import re
from dotenv import load_dotenv  # 导入dotenv库

# 加载环境变量
load_dotenv("/hsk/.env")

#  编码函数： 将本地文件转换为 Base64 编码的字符串
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
# 初始化OpenAI客户端
client = OpenAI(
    api_key = os.getenv("AGICTO_KEY"),
    base_url = os.getenv("AGICTO_URL"),
)

prompt_1 = """ 
Use the input image, editing prompt and the Physical Plausibility(the expected result of the editing prompt on the input image) to create the checklist used to judge the image edited on it(mandatory).  
(Note: The editing prompt is related to unreasonable scene that against the real world)
1. VIOLATED PHYSIC LAW: Which physic law is related to the editing prompt?
2. INVOLVED OBJECTS: What are the main objects interacted with the editing prompt within the image?
3. EXPECTED PHENOMENA: What is the expected editing result of the editing prompt with respect to the involved objects(detailed and precise phenomena of the involved objects)?

Guidelines for checklist creation: 
- only target things which are visually observable in the image 
- the statements in checklist needs to be assertive statements instead of questions
- only checklist no other content
"""

def generate_checklist(model, editing_prompt, input_image,expected_phenomena):
    messages=[
    {
        "role": "user",
        "content": [
            {"type":"text","text":f"From my evaluation of editing models I have generated a image using the prompt: {editing_prompt}"},
            {"type":"text","text":"here is the input image"},
            {
                "type":"image_url",
                "image_url":{"url": f"data:image/png;base64,{input_image}"},
            },
            {"type":"text","text":"here is the Physical Plausibility"},
            {"type":"text","text":expected_phenomena},
            {"type": "text", "text": prompt_1},
        ],
    },
]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        extra_body={
        'enable_thinking': True,
        "thinking_budget": 81920},
    )
    
    full_message = completion.choices[0].message
    answer_content = full_message.content or ""
    
    return answer_content

rubric_text = """
You are strict VLM-Judge objectively evaluating a generated edited image from a checklist, editing prompt and the input image. 
The checklist is provided as a reference :
1. VIOLATED PHYSIC LAW
2. INVOLVED OBJECTS
3. EXPECTED PHENOMENA

Score each rubric from 1–10: 
a) Instruction Following — Judge how well the prediction satisfies the instruction AND matches the reference target state.
b) Physical Plausibility — expected physical/chemical outcome is present according to the EXPECTED PHENOMENA(check the image in detail).
c) Consistency — whether the involved objects are the same object between the input image and the edited image beside some changes according to the EXPECTED PHENOMENA. 
d) Image Quality — The generated image should exhibit sharp details with minimal artifacts, distortions, or unnatural textures. Lighting, shadows, and reflections must remain physically coherent, while materials should demonstrate photorealistic appearance and visual consistency. Overall, the output should be free from obvious AI-generated artifacts or technical glitches.

Each rating must be supported with clear justification, drawing on specific edited image area and, when provided, the corresponding checklist items.
Important!: Give each rubic at least 1 point even it completely fail.
Scoring rubric for Consistency(1-10):
- 10: Only the intended edits occur; invariants and unrelated regions are preserved extremely well.
- 7-9: Minor unintended changes (small texture shifts, slight lighting drift), but overall consistent.
- 4-6: Noticeable unintended changes (background altered, viewpoint drift, extra objects), partially consistent.
- 1-3: Major unwanted changes; scene identity not preserved.
Scoring rubric for Instruction Following(1-10):
- 10: Prediction matches the reference target very closely and fulfills the instruction precisely.
- 7-9: Mostly correct; small differences from reference target but clearly follows instruction.
- 4-6: Partially correct; key aspects missing or wrong; noticeable mismatch vs reference.
- 1-3: Fails to perform the intended edit; does not resemble the reference target.
Scoring rubric for Image Quality(1-10):
- 10: Highly realistic, sharp where appropriate, no noticeable artifacts.
- 7-9: Minor artifacts or softness, overall high quality.
- 4-6: Clear artifacts, blur, distortions, but still recognizable.
- 1-3: Severe artifacts, unrealistic, degraded output.
"""
output_format = """
Return JSON with fields: 
{ ”scores”: { ”Instruction_Following”:1-10, ”Physical_Plausibility”:1-10, ”Consistency”:1-10, ”Image_Quality”:1-10}, 
 ”explanations”: {”summary”: string, ”issues”: [{"issue_name": string,"score_explanation":string}...]} ## issue_name's value should one of: Instruction Following, Physical Plausibility, Consistency and Image Quality
 }
"""

def vlm_judge(editing_prompt, edited_image, input_image, check_list, model_name):
    try:
        # 移除 extra_body 中的 enable_thinking，因为 gpt-4o 不需要，且可能导致代理报错
        extra_params = {}
        if "thinking" in model_name or "r1" in model_name: # 简单的判断逻辑
             extra_params = {
                'enable_thinking': True,
                "thinking_budget": 81920
            }

        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Here is the editing prompt :{editing_prompt}"},
                        {"type": "text", "text": f"here is the input image:"},
                        {
                            "type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{input_image}"},
                        },
                        {"type": "text", "text": f"here is the edited image:"},
                        {
                            "type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{edited_image}"},
                        },
                        {"type": "text", "text": f"here is the checklist:{check_list}"},
                        {"type": "text", "text": rubric_text},
                        {"type": "text", "text": f"here is the output format:{output_format}"},
                    ],
                },
            ],
            stream=False,
            extra_body=extra_params
        )

        # --- 增加安全检查 ---
        if not completion.choices or len(completion.choices) == 0:
            print(f"Warning: API returned no choices for this image. Reason: {getattr(completion, 'finish_reason', 'Unknown')}")
            # 返回一个全 1 分的默认 JSON，避免解析函数崩溃
            return json.dumps({
                "scores": {"Instruction_Following": 1, "Physical_Plausibility": 1, "Consistency": 1, "Image_Quality": 1},
                "explanations": {"summary": "API Refusal/Error", "issues": []}
            })
        
        full_message = completion.choices[0].message
        
        # 检查是否因为内容安全被拒绝
        if hasattr(full_message, 'refusal') and full_message.refusal:
            print(f"Warning: Request refused by safety system: {full_message.refusal}")
            return json.dumps({
                "scores": {"Instruction_Following": 1, "Physical_Plausibility": 1, "Consistency": 1, "Image_Quality": 1},
                "explanations": {"summary": "Content Policy Violation", "issues": []}
            })

        score_content = full_message.content or ""
        print("score_content:", score_content)
        return score_content

    except Exception as e:
        print(f"CRITICAL ERROR in vlm_judge: {e}")
        # 发生异常时返回保底数据，防止整个脚本中断
        return json.dumps({
                "scores": {"Instruction_Following": 0, "Physical_Plausibility": 0, "Consistency": 0, "Image_Quality": 0},
                "explanations": {"summary": f"Python Error: {str(e)}", "issues": []}
            })

def parse_score_content(score_content):
    """
    解析评分内容，提取四个评分和解释
    """
    try:
        # 尝试直接解析JSON
        score_data = json.loads(score_content)
        scores = score_data.get("scores", {})
        explanations = score_data.get("explanations", {})
        
        # 提取四个评分 - 使用正确的键名
        Instruction_Following = scores.get("Instruction_Following")
        Physical_Plausibility = scores.get("Physical_Plausibility")  # 修正：使用正确的键名
        Consistency = scores.get("Consistency")
        Image_Quality = scores.get("Image_Quality")
        
        # 提取解释
        summary = explanations.get("summary", "")
        issues = explanations.get("issues", [])
        
        return {
            "Instruction_Following": Instruction_Following,
            "Physical_Plausibility": Physical_Plausibility,
            "Consistency": Consistency,
            "Image_Quality": Image_Quality,
            "summary": summary,
            "issues": issues
        }
    except json.JSONDecodeError:
        print("JSON parsing failed. Trying to parse with regular expressions.")
        # 如果不是有效的JSON，尝试使用正则表达式解析
        parsed_data = {}

        # 提取评分 - 查找各个评分的值
        pc_match = re.search(r'"Instruction_Following"\s*:\s*(\d)', score_content)
        ep_match = re.search(r'"Physical_Plausibility"\s*:\s*(\d)', score_content)
        ioc_match = re.search(r'"Consistency"\s*:\s*(\d)', score_content)
        bi_match = re.search(r'"Image_Quality"\s*:\s*(\d)', score_content)

        if pc_match:
            parsed_data["Instruction_Following"] = int(pc_match.group(1))
        else:
            print("No Instruction Following score found.")
            parsed_data["Instruction_Following"] = None

        if ep_match:
            parsed_data["Physical_Plausibility"] = int(ep_match.group(1))
        else:
            print("No Physical Plausibility score found.")
            parsed_data["Physical_Plausibility"] = None

        if ioc_match:
            parsed_data["Consistency"] = int(ioc_match.group(1))
        else:
            print("No Consistency score found.")
            parsed_data["Consistency"] = None

        if bi_match:
            parsed_data["Image_Quality"] = int(bi_match.group(1))
        else:
            print("No Image Quality score found.")
            parsed_data["Image_Quality"] = None

        # 提取解释部分 - 这可能需要根据实际返回格式调整
        summary_match = re.search(r'"summary":\s*"([^"]*)"', score_content)
        if summary_match:
            parsed_data["summary"] = summary_match.group(1)
        else:
            parsed_data["summary"] = ""

        # 尝试找到issues部分
        issues_matches = re.findall(r'\{\s*"issue_name":\s*"([^"]+)",\s*"score_explanation":\s*"([^"]*)"', score_content)
        parsed_data["issues"] = []
        for issue_match in issues_matches:
            parsed_data["issues"].append({
                "issue_name": issue_match[0],
                "score_explanation": issue_match[1]
            })

        return parsed_data

def load_existing_data(file_path):
    """从JSONL文件加载已有数据"""
    data_dict = {}
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 确保行不为空
                    try:
                        data = json.loads(line)
                        data_id = data.get('data_id')
                        if data_id is not None:
                            data_dict[data_id] = data
                    except json.JSONDecodeError as e:
                        print(f"警告：无法解析JSON行: {line}, 错误: {e}")
    return data_dict

def append_to_jsonl(data, file_path):
    """向JSONL文件追加一行数据"""
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data) + '\n')

def process_dataset(gemini_gen_path, output_base_path, model_name, target_model_folder):
    # 创建输出目录（用于存放模型输出图像）
    output_model_path = os.path.join(output_base_path, target_model_folder)
    os.makedirs(output_model_path, exist_ok=True)
    
    # 加载meta.jsonl文件
    meta_file_path = os.path.join(gemini_gen_path, "meta.jsonl")
    
    # 输出文件路径：所有模型共享一个checklist，但每个模型有自己的score文件
    checklist_output_path = os.path.join(output_base_path, "checklists.jsonl")
    score_output_path = os.path.join(output_model_path, "scores.jsonl")  # score保存在各自模型目录中
    
    # 加载已经处理过的checklist和score
    processed_checklists = load_existing_data(checklist_output_path)
    processed_scores = load_existing_data(score_output_path)

    # 读取meta.jsonl
    with open(meta_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            data = json.loads(line)
            data_id = data['data_id']
            edit_prompt = data['edit_prompt']
            Physical_Plausibility = data["expected_phenomenon"]
            
            print(f"Processing data_id: {data_id}")
            
            # 检查input_image是否存在
            input_image_path = os.path.join(gemini_gen_path, f"input_data/data_{data_id}.png")
            
            if not os.path.exists(input_image_path):
                print(f"Input image does not exist for data_id {data_id}: {input_image_path}")
                continue
            
            # 生成checklist（如果没有的话）
            if data_id not in processed_checklists:
                print(f"Generating checklist for data_id {data_id}")
                
                input_image_b64 = encode_image(input_image_path)
                checklist = generate_checklist(model_name, edit_prompt, input_image_b64,Physical_Plausibility)
                
                # 保存checklist
                checklist_data = {
                    "data_id": data_id,
                    "data_type": data['data_type'],
                    "sub_id": data['sub_id'],
                    "checklist": checklist
                }
                
                append_to_jsonl(checklist_data, checklist_output_path)
                
                processed_checklists[data_id] = checklist_data
                print(f"Saved checklist for data_id {data_id}")
            else:
                print(f"Checklist already exists for data_id {data_id}")
            
            # 检查output_image是否存在
            output_image_path = os.path.join(output_model_path, f"{data_id}.png")
            
            if not os.path.exists(output_image_path):
                print(f"Output image does not exist for data_id {data_id}: {output_image_path}, skipping score generation")
                continue
            
            # 生成score（如果output图像存在且还没有score的话）
            if data_id not in processed_scores:
                print(f"Generating score for data_id {data_id}")
                
                # 获取checklist（可能刚生成或之前已存在）
                checklist = processed_checklists[data_id]['checklist']
                
                input_image_b64 = encode_image(input_image_path)
                output_image_b64 = encode_image(output_image_path)
                
                score_content = vlm_judge(edit_prompt, output_image_b64, input_image_b64, checklist, model_name)
                
                # 解析评分内容
                parsed_scores = parse_score_content(score_content)
                
                # 保存解析后的评分
                score_data = {
                    "data_id": data_id,
                    "data_type": data['data_type'], 
                    "sub_id": data['sub_id'],
                    "Instruction_Following": parsed_scores["Instruction_Following"],
                    "Physical_Plausibility": parsed_scores["Physical_Plausibility"],  # 修正字段名拼写
                    "Consistency": parsed_scores["Consistency"],
                    "Image_Quality": parsed_scores["Image_Quality"],
                    "summary": parsed_scores["summary"],
                    "issues": parsed_scores["issues"]
                }
                
                append_to_jsonl(score_data, score_output_path)
                
                processed_scores[data_id] = score_data
                print(f"Saved parsed scores for data_id {data_id}")
            else:
                print(f"Score already exists for data_id {data_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate checklist and scores for image editing evaluation.')
    parser.add_argument('--gemini-gen-path', type=str, default='/hsk/dataset_test/anti-physic/gemini_gen_antiphysic', 
                        help='Path to the gemini_gen_antiphysic folder')
    parser.add_argument('--output-base-path', type=str, default='/hsk/dataset_test/anti-physic/output_file', 
                        help='Base path to the output files folder')
    parser.add_argument('--model-name', type=str, default="gemini-3-pro-preview-thinking", 
                        help='Model used for generating checklists and scores')
    parser.add_argument('--target-model-folder', type=str, required=True, 
                        help='Target model folder name in output_base_path')
    
    args = parser.parse_args()
    
    process_dataset(args.gemini_gen_path, args.output_base_path, args.model_name, args.target_model_folder)
    
    # 统计所有数据点和各data_type内的各项评分平均值和加权平均值
    dimension_weights = {
        "Consistency": 0.2,
        "Instruction_Following": 0.3,
        "Physical_Plausibility": 0.4,
        "Image_Quality": 0.1,
    }

    scores_by_type = {}
    all_scores = {
        "Instruction_Following": [],
        "Physical_Plausibility": [],
        "Consistency": [],
        "Image_Quality": []
    }

    with open(os.path.join(args.output_base_path, args.target_model_folder, 'scores.jsonl'), "r", encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                
                # 添加到总分统计
                for score_type in ["Instruction_Following", "Physical_Plausibility", "Consistency", "Image_Quality"]:
                    score_value = data.get(score_type)
                    if score_value is not None:
                        all_scores[score_type].append(score_value)
                
                # 按data_type分类统计
                data_type = data["data_type"]
                if data_type not in scores_by_type:
                    scores_by_type[data_type] = {
                        "Instruction_Following": [],
                        "Physical_Plausibility": [],
                        "Consistency": [],
                        "Image_Quality": []
                    }
                
                for score_type in ["Instruction_Following", "Physical_Plausibility", "Consistency", "Image_Quality"]:
                    score_value = data.get(score_type)
                    if score_value is not None:
                        scores_by_type[data_type][score_type].append(score_value)

    # 计算所有数据点的平均分和加权平均分
    print("\n=== 总体评分统计 ===")
    overall_averages = {}
    for score_type in ["Instruction_Following", "Physical_Plausibility", "Consistency", "Image_Quality"]:
        if all_scores[score_type]:
            avg_score = sum(all_scores[score_type]) / len(all_scores[score_type])
            overall_averages[score_type] = avg_score
            print(f"{score_type} 平均分: {avg_score:.3f}")
        else:
            overall_averages[score_type] = 0
            print(f"{score_type} 平均分: 0")

    # 计算总体加权平均分
    weighted_average_total = 0
    for score_type, avg_score in overall_averages.items():
        weighted_average_total += avg_score * dimension_weights[score_type]
    print(f"总体加权平均分: {weighted_average_total:.3f}")

    # 计算各data_type的平均分和加权平均分
    print("\n=== 各数据类型评分统计 ===")
    for data_type, scores in scores_by_type.items():
        print(f"\n--- {data_type} ---")
        type_averages = {}
        for score_type in ["Instruction_Following", "Physical_Plausibility", "Consistency", "Image_Quality"]:
            if scores[score_type]:
                avg_score = sum(scores[score_type]) / len(scores[score_type])
                type_averages[score_type] = avg_score
                print(f"{score_type} 平均分: {avg_score:.3f}")
            else:
                type_averages[score_type] = 0
                print(f"{score_type} 平均分: 0")
        
        # 计算该类型加权平均分
        weighted_average_type = 0
        for score_type, avg_score in type_averages.items():
            weighted_average_type += avg_score * dimension_weights[score_type]
        print(f"{data_type} 加权平均分: {weighted_average_type:.3f}")
    
    # 保存统计结果到JSON文件
    stats_output_path = os.path.join(args.output_base_path, args.target_model_folder, 'statistics.json')
    statistics = {
        "dimension_weights": dimension_weights,  # 添加权重信息
        "overall_averages": overall_averages,
        "overall_weighted_average": weighted_average_total,  # 重命名使更清晰
        "by_data_type": {}
    }
    
    for data_type, scores in scores_by_type.items():
        type_averages = {}
        for score_type in ["Instruction_Following", "Physical_Plausibility", "Consistency", "Image_Quality"]:
            if scores[score_type]:
                avg_score = sum(scores[score_type]) / len(scores[score_type])
                type_averages[score_type] = avg_score
            else:
                type_averages[score_type] = 0
        
        # 计算该类型加权平均分
        weighted_average_type = 0
        for score_type, avg_score in type_averages.items():
            weighted_average_type += avg_score * dimension_weights[score_type]
        
        statistics["by_data_type"][data_type] = {
            "averages": type_averages,
            "weighted_average": weighted_average_type
        }
    
    with open(stats_output_path, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)
    
    print(f"\n统计结果已保存到: {stats_output_path}")
