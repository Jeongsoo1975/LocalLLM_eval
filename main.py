import ollama
import pandas as pd
from openai import OpenAI, AuthenticationError, APIStatusError
import os
from dotenv import load_dotenv
import re
import time
import csv
import logging  # 로깅 모듈 추가
import argparse  # 인자 파싱 모듈 추가

# --- 로깅 설정 ---
# 로그 파일명, 로그 레벨, 로그 포맷 등을 설정합니다.
# 로그 파일은 'evaluation_log.txt'에 저장되며, 실행 시마다 내용이 추가됩니다.
logging.basicConfig(
    level=logging.INFO,  # DEBUG 레벨로 설정하면 더 자세한 로그를 볼 수 있습니다.
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler("evaluation_log.txt", mode='a', encoding='utf-8'),
        logging.StreamHandler()  # 콘솔에도 로그 출력
    ]
)

# --- Configuration ---
OLLAMA_MODEL_NAME = 'gemma3:12b'
INPUT_CSV_FILE = 'korean_llm_questions_updated.csv'
OUTPUT_CSV_FILE = 'llm_evaluation_results_xai.csv'
REQUEST_DELAY_SECONDS = 2

# --- Load Environment Variables ---
load_dotenv()
xai_api_key = os.getenv("XAI_API_KEY")
xai_model_name_env = os.getenv("XAI_MODEL_NAME")

if not xai_api_key:
    logging.error("❌ XAI_API_KEY not found in .env file.")
    exit()
else:
    key_display = xai_api_key[:7] + "..." + xai_api_key[-4:] if len(xai_api_key) > 11 else xai_api_key
    logging.info(f"ℹ️ XAI_API_KEY loaded (display: {key_display})")

if not xai_model_name_env:
    logging.warning("⚠️ XAI_MODEL_NAME not found in .env file. Using default 'grok-3-mini'.")
    XAI_MODEL_NAME = 'grok-3-mini'
else:
    XAI_MODEL_NAME = xai_model_name_env
    logging.info(f"ℹ️ Using xAI model: {XAI_MODEL_NAME} (from .env file)")

# --- Initialize xAI (via OpenAI compatible client) ---
try:
    xai_client = OpenAI(
        api_key=xai_api_key,
        base_url="https://api.x.ai/v1",
    )
    logging.info("✅ xAI client initialized successfully.")
except Exception as e:
    logging.exception("❌ Error initializing xAI client.")
    exit()


def test_xai_api_key():
    logging.info(f"\n🧪 Testing xAI API key with model '{XAI_MODEL_NAME}'...")
    try:
        xai_client.chat.completions.create(
            model=XAI_MODEL_NAME,
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=10,
            temperature=0.1
        )
        logging.info("✅ xAI API key appears valid and model responded.")
        return True
    except AuthenticationError as e:
        logging.error(f"❌ xAI API Key is INVALID. Authentication failed. Details: {e}")
        return False
    except APIStatusError as e:
        logging.error(
            f"❌ Could not connect to xAI API or model issue (Status: {e.status_code}). Response: {e.response.text}")
        if e.status_code == 404:
            logging.error(f"   It seems the model '{XAI_MODEL_NAME}' was not found or you don't have access.")
        return False
    except Exception as e:
        logging.exception(f"❌ An unexpected error occurred while testing xAI API key.")
        return False


def get_ollama_response(prompt_text):
    try:
        logging.info(f"\n Ollama ({OLLAMA_MODEL_NAME}) <= 전체 질문송신 요청...")
        logging.debug(f"Ollama full prompt:\n{prompt_text}\n--------------------")
        response = ollama.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt_text}]
        )
        ollama_response_content = response['message']['content']
        logging.info(f" Ollama ({OLLAMA_MODEL_NAME}) => 전체 답변수신 완료.")
        logging.debug(f"Ollama full response:\n{ollama_response_content}\n--------------------")
        return ollama_response_content
    except Exception as e:
        logging.exception(f"❌ Error communicating with Ollama.")
        return "Error: Could not get response from Ollama."


def get_xai_evaluation(question, answer):
    evaluation_prompt = f"""
    You are an AI evaluator. Your task is to assess the appropriateness and correctness ("정합성")
    of a given answer to a specific question. Provide a percentage score from 0 to 100,
    where 0 is completely incorrect/irrelevant and 100 is perfectly correct and appropriate.
    Also, provide a brief justification for your score.

    Your response MUST strictly follow this format:
    Score: [Your numerical score between 0 and 100]
    Justification: [Your brief justification here]

    Original Question:
    {question}

    Answer to Evaluate:
    {answer}
    """
    try:
        logging.debug(
            f"\nDEBUG: xAI Grok ({XAI_MODEL_NAME}) <= 전체 평가 프롬프트 전달 예정:\n{evaluation_prompt}\n--------------------")
        logging.info(f" xAI Grok ({XAI_MODEL_NAME}) <= 평가요청...")
        chat_completion = xai_client.chat.completions.create(
            model=XAI_MODEL_NAME,
            messages=[
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0.2,
            max_tokens=2000,
        )

        logging.debug(f"DEBUG: Full xAI API Response Object:\n{chat_completion}\n--------------------")

        evaluation_text = ""
        if chat_completion.choices and len(chat_completion.choices) > 0:
            choice = chat_completion.choices[0]
            logging.debug(f"DEBUG: Finish Reason: {choice.finish_reason}")
            if choice.message:
                logging.debug(f"DEBUG: Message Object: {choice.message}")
                if choice.message.content is not None:
                    evaluation_text = choice.message.content
                    logging.debug(f"DEBUG: Message Content (evaluation_text):\n{evaluation_text}\n--------------------")
                else:
                    logging.debug("DEBUG: Message Content (evaluation_text) is None.")
            else:
                logging.debug("DEBUG: choice.message is None or empty.")
        else:
            logging.debug("DEBUG: chat_completion.choices is empty.")

        logging.info(f" xAI Grok ({XAI_MODEL_NAME}) => 평가수신 (일부): {evaluation_text[:100]}...")

        score_match = re.search(r"Score:\s*(\d+)", evaluation_text, re.IGNORECASE)
        justification_match = re.search(r"Justification:\s*(.+)", evaluation_text, re.IGNORECASE | re.DOTALL)

        score = int(score_match.group(1)) if score_match else -1
        justification = justification_match.group(
            1).strip() if justification_match else "Could not parse justification."

        if score == -1 and justification == "Could not parse justification.":
            justification = f"Failed to parse score/justification. Raw evaluation_text received: '{evaluation_text[:200]}'"

        return score, justification
    except AuthenticationError as e:
        logging.error(f"❌ xAI API Key became INVALID during evaluation. Details: {e}")
        return -1, f"xAI AuthenticationError: {e}"
    except APIStatusError as e:
        logging.error(
            f"❌ Error communicating with xAI API during evaluation (Status: {e.status_code}). Response: {e.response.text}")
        return -1, f"xAI APIStatusError (Status {e.status_code}): {e.message}"
    except Exception as e:
        logging.exception(f"❌ Unexpected error during xAI API call.")
        return -1, f"Unexpected error during xAI API call: {e}"


def main(start_from_question: int):  # 시작 번호를 인자로 받음
    logging.info(
        f"🚀 Starting LLM evaluation process (Targeting xAI API for evaluation). Starting from question number: {start_from_question}")

    if not test_xai_api_key():
        logging.error("\nExiting script due to xAI API key issue.")
        return

    logging.info("\n📖 Loading dataset...")
    try:
        try:
            questions_df = pd.read_csv(INPUT_CSV_FILE, encoding='utf-8-sig')
        except UnicodeDecodeError:
            try:
                questions_df = pd.read_csv(INPUT_CSV_FILE, encoding='utf-8')
            except UnicodeDecodeError:
                questions_df = pd.read_csv(INPUT_CSV_FILE, encoding='cp949')
        logging.info(f"✅ Successfully loaded {len(questions_df)} questions from '{INPUT_CSV_FILE}'.")

        prompt_column_name = 'Prompt_Korean'
        if prompt_column_name not in questions_df.columns:
            logging.error(f"❌ Column '{prompt_column_name}' not found in {INPUT_CSV_FILE}.")
            return
    except FileNotFoundError:
        logging.error(f"❌ Input CSV file '{INPUT_CSV_FILE}' not found.")
        return
    except UnicodeDecodeError as e:
        logging.error(f"❌ Could not decode CSV file. Tried 'utf-8-sig', 'utf-8', and 'cp949'. Final error: {e}")
        return
    except Exception as e:
        logging.exception(f"❌ Error reading CSV file.")
        return

    results = []
    total_score_sum = 0
    successfully_evaluated_count = 0

    start_index = start_from_question - 1  # 0-based index로 변환

    # CSV 파일 이어쓰기 및 헤더 처리 로직
    file_exists_and_has_content = False
    if os.path.exists(OUTPUT_CSV_FILE) and os.path.getsize(OUTPUT_CSV_FILE) > 0:
        file_exists_and_has_content = True

    # 재시작이고 파일에 내용이 있으면 이어쓰기('a+'), 아니면 새로 쓰기('w')
    open_mode = 'a+' if start_index > 0 and file_exists_and_has_content else 'w'

    logging.info(f"Output CSV '{OUTPUT_CSV_FILE}' will be opened in mode: '{open_mode}'.")

    try:
        with open(OUTPUT_CSV_FILE, open_mode, newline='', encoding='utf-8-sig') as csvfile:
            fieldnames = ['Question_ID', 'Category_Name_Korean', 'Category_Name_English',
                          'Prompt_Korean', 'Gemma3_Answer', 'XAI_Evaluation_Score', 'XAI_Evaluation_Justification']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if open_mode == 'w' or not file_exists_and_has_content:
                logging.info("Writing CSV header.")
                writer.writeheader()
            else:
                logging.info("Appending to existing CSV. Header not written.")

            logging.info(f"\n⚙️ Starting evaluation loop for {len(questions_df)} questions...")
            for index, row in questions_df.iterrows():
                current_question_number = index + 1

                if current_question_number < start_from_question:
                    logging.info(
                        f"Skipping already processed Question ID: {row.get('Question_ID', 'N/A')} (Number: {current_question_number})")
                    continue  # 지정된 시작 번호 이전의 질문은 건너뜀

                logging.info(
                    f"\n--- ❓ Processing Question ID: {row.get('Question_ID', 'N/A')} (Number: {current_question_number}/{len(questions_df)}) ---")
                original_prompt = row[prompt_column_name]

                logging.info("Requesting response from Ollama...")
                gemma_answer = get_ollama_response(original_prompt)
                if "Error: Could not get response from Ollama." in gemma_answer:
                    logging.error("Ollama failed to provide an answer. Skipping evaluation for this question.")
                    xai_score, xai_justification = -1, "Ollama failed to respond."
                else:
                    logging.info("Response received from Ollama. Requesting evaluation from xAI...")
                    time.sleep(REQUEST_DELAY_SECONDS)
                    xai_score, xai_justification = get_xai_evaluation(original_prompt, gemma_answer)

                if xai_score != -1:
                    logging.info(f"  ➡️  평가 점수 (xAI {XAI_MODEL_NAME}): {xai_score} / 100")
                    logging.info(f"  ➡️  평가 근거: {xai_justification}")
                    total_score_sum += xai_score
                    successfully_evaluated_count += 1
                else:
                    logging.warning(f"  ➡️  평가 실패 또는 오류 (xAI {XAI_MODEL_NAME})")
                    logging.warning(f"  ➡️  오류/실패 사유: {xai_justification}")

                current_result = {
                    'Question_ID': row.get('Question_ID', ''),
                    'Category_Name_Korean': row.get('Category_Name_Korean', ''),
                    'Category_Name_English': row.get('Category_Name_English', ''),
                    'Prompt_Korean': original_prompt,
                    'Gemma3_Answer': gemma_answer,
                    'XAI_Evaluation_Score': xai_score,
                    'XAI_Evaluation_Justification': xai_justification
                }
                results.append(current_result)  # results 리스트는 현재 실행에서 처리된 것만 담김
                writer.writerow(current_result)
                csvfile.flush()  # 각 행을 쓸 때마다 파일에 즉시 반영
    except Exception as e:
        logging.exception("❌ An error occurred during the main evaluation loop or CSV writing.")

    logging.info(f"\n\n🎉 Evaluation complete for this run! 🎉")
    if successfully_evaluated_count > 0:
        average_score_this_run = total_score_sum / successfully_evaluated_count
        logging.info(f"📊 이번 실행 평가 요약:")
        logging.info(f"  - 이번 실행에서 성공적으로 평가된 질문 수: {successfully_evaluated_count}")
        logging.info(f"  - 이번 실행 평균 점수 (0-100): {average_score_this_run:.2f}")
    else:
        logging.info("📊 이번 실행 평가 요약: 성공적으로 평가된 질문이 없습니다.")
    logging.info(f"💾 Results saved to/appended to '{OUTPUT_CSV_FILE}'")


if __name__ == "__main__":
    # --- 명령줄 인자 파싱 ---
    parser = argparse.ArgumentParser(description="LLM Evaluation Script with Resume Functionality.")
    parser.add_argument(
        "--start_from",
        type=int,
        default=1,
        help="Question number (1-based) to start processing from. Defaults to 1 (start from the beginning)."
    )
    args = parser.parse_args()

    main(start_from_question=args.start_from)  # main 함수에 시작 번호 전달