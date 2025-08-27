import asyncio
import pandas as pd
import time
import re
import logging
from datetime import datetime
from src.utils.utils import new_uuid
from src.graph import init_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

thread_id = new_uuid()
DEFAULT_CONFIG = {"configurable": {"thread_id": thread_id}}

async def stream_response(graph, query: str, config: dict, timeout: float = 30.0) -> str:
    agen = graph.astream(
        {"messages": [{"role": "user", "content": query}]},
        config=config,
        stream_mode="messages",
    )

    res_text = ""

    async def _consume():
        nonlocal res_text
        try:
            async for event in agen:
                try:
                    chunk = event[0].content
                except Exception:
                    chunk = str(event)
                if chunk:
                    logger.debug("Chunk: %s", chunk)
                    res_text += chunk
        finally:
            try:
                await agen.aclose()
            except Exception:
                pass

    try:
        await asyncio.wait_for(_consume(), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("stream_response timeout after %.1fs", timeout)
    except Exception as e:
        logger.exception("stream_response error: %s", e)

    return res_text

def extract_choice_from_text(text: str) -> str:
    if not text:
        return ""
    txt = text.upper()

    m = re.search(r'\b([A-D])\b', txt)
    if m:
        return m.group(1)
    m2 = re.search(r'[ABCD]', txt)
    if m2:
        return m2.group(0)
    return ""

async def evaluate_agent(csv_path: str, graph, *,
                         question_col: str = "Câu hỏi",
                         options_cols = ("A", "B", "C", "D"),
                         answer_col: str = "Đáp án",
                         delay_between_questions: float = 15,
                         per_question_timeout: float = 30.0,
                         output_path: str | None = None) -> str:
    
    df = pd.read_csv(csv_path)

    if "Agent" not in df.columns:
        df["Agent"] = ""
    if "Correct" not in df.columns:
        df["Correct"] = False
    if "AgentRaw" not in df.columns:
        df["AgentRaw"] = ""

    total = len(df)
    logger.info("Loaded %d rows from %s", total, csv_path)

    for i, row in df.iterrows():
        logger.info("Processing row %d / %d", i+1, total)

        if delay_between_questions:
            await asyncio.sleep(delay_between_questions)

        q = str(row.get(question_col, "")).strip()
        if not q:
            logger.warning("Empty question at row %d - skipping", i)
            continue

        options = {col: str(row.get(col, "")).strip() for col in options_cols}
        prompt_lines = [f"Câu hỏi: {q}"]
        for k in options_cols:
            prompt_lines.append(f"{k}. {options.get(k, '')}")
        prompt_lines.append("Hãy trả lời bằng duy nhất 1 ký tự A/B/C/D là đáp án đúng nhất của câu hỏi.")
        prompt = "\n".join(prompt_lines)

        cfg = DEFAULT_CONFIG

        try:
            text = await stream_response(graph, prompt, cfg, timeout=per_question_timeout)
        except Exception as e:
            logger.exception("Error calling stream_response for row %d: %s", i, e)
            text = ""

        df.at[i, "AgentRaw"] = text
        choice = extract_choice_from_text(text)
        df.at[i, "Agent"] = choice or text

        if answer_col in df.columns:
            true_answer = str(row.get(answer_col, "")).strip().upper()
            if true_answer:
                df.at[i, "Correct"] = (choice == true_answer)
            else:
                df.at[i, "Correct"] = False
        else:
            df.at[i, "Correct"] = False

        logger.info("Row %d -> agent: %s  correct: %s", i+1, df.at[i,"Agent"], df.at[i,"Correct"])

    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{csv_path.rsplit('.',1)[0]}_results_{ts}.csv"
    df.to_csv(output_path, index=False)
    logger.info("Saved results to %s", output_path)
    return output_path

async def main():
    graph = await init_graph()

    csv_path = r"test\70_trac_nghiem.csv"
    out = await evaluate_agent(csv_path=csv_path, graph=graph,
                               question_col="Câu hỏi",
                               options_cols=("A","B","C","D"),
                               answer_col="Answer",
                               delay_between_questions=5,
                               per_question_timeout=30.0)
    print("Finished. Results saved to:", out)

if __name__ == "__main__":
    asyncio.run(main())