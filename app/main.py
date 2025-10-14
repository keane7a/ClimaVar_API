from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
from utils import validate_input, cards_classify_claim, build_prompt, llm_answer, extract_final
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
CLIMAVAR_TOKEN = os.getenv("CLIMAVAR_TOKEN")

class Item(BaseModel):
    text: str
    token: str

app = FastAPI()
router = APIRouter()

@router.post("/check-misinformation/", status_code=200)
def check_misinformation(item: Item):
    """
        Evaluate a user-supplied climate claim and return a concise fact-check.
    """
    """
    End-to-end:
    1) Validate length
    2) CARDS classify
    3) Build prompt (instruction or one-shot)
    4) LLM generation
    5) Extract FINAL (≤300 chars)
    6) Log with columns indicating prompt family and chain-of-thought
    """
    
    # Init parameters
    prompt_version = "i0"   # choose among: i0,i1,i2,s0,s1,s2
    model = "gpt-4o-mini"
    temperature = 0.2
    
    text = item.text
    token = item.token
    
    # THIS IS BAD DESIGN DO NOT FOLLOW!! THIS IS NON-CRITICAL AND ONLY TO BLOCK BASIC ACCESS
    if token != CLIMAVAR_TOKEN:
        return HTTPException(status_code=401, detail="Invalid token")
    
    # 1) Validate
    err = validate_input(text)
    if err:
        return HTTPException(status_codxe=400, detail=err)

    # 2) CARDS classify
    cards = cards_classify_claim(text)

    # 3) Build prompt (and get metadata)
    prompt, family, uses_cot = build_prompt(text, cards, prompt_version)

    # 4) Generate
    raw = llm_answer(prompt, model=model, temperature=temperature)

    # 5) Extract

    # # 6) Log
    # log_run(
    #     user_text=user_text,
    #     cards_label=cards.label,
    #     cards_categories=cards.categories,
    #     cfg=cfg,
    #     prompt_family=family,
    #     uses_cot=uses_cot,
    #     final_text=final
    # )
    
    return {"response": extract_final(raw), "misinformation": cards.is_misinformation}

app.include_router(router)