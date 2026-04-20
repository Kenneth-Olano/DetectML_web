import os
import re
import math
import torch
import numpy as np
from collections import Counter
import joblib
from sentence_transformers import SentenceTransformer, util
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from keybert import KeyBERT

load_dotenv()

# --- Load Models Globally ---
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading SentenceTransformers...")
semantic_feat_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
kw_model = KeyBERT(model='sentence-transformers/all-MiniLM-L6-v2')
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Loading GPT2 for Perplexity...")
perp_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
perp_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

print("Loading Random Forest Classifier...")
scaler = joblib.load('models/rf_scaler.pkl')
rf_model = joblib.load('models/random_forest_classifier.pkl')

CHROMA_DIR = "chroma_rag_db"
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# The API keys should be injected into Hugging Face Spaces secrets
client = OpenAI(api_key=DEEPINFRA_API_KEY, base_url=DEEPINFRA_BASE_URL)

def get_vector_db():
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

db = get_vector_db()

# --- Intrinsic Functions ---
def get_perplexity(text, stride=512):
    if not text.strip(): return 0.0
    encodings = perp_tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.size(1)
    max_length = perp_model.config.n_positions
    nll_sum, n_tokens, prev_end_loc = 0.0, 0, 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_slice = input_ids[:, begin_loc:end_loc]
        target_ids = input_slice.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = perp_model(input_slice, labels=target_ids)
        nll_sum += outputs.loss.item() * trg_len
        n_tokens += trg_len
        prev_end_loc = end_loc
        if end_loc == seq_len: break
    if n_tokens == 0: return 0.0
    return math.exp(min(nll_sum / n_tokens, 50))

def get_burstiness(text):
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if not sentences: return 0.0
    lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
    return float(np.std(lengths))

def get_shannon_entropy(tokens):
    if not tokens: return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())

def get_word_entropy(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return round(get_shannon_entropy(words), 4)

def get_char_entropy(text):
    chars = [c.lower() for c in text if c.isalpha()]
    return round(get_shannon_entropy(chars), 4)

def get_repetition_rate(text):
    words = re.findall(r'\b\w+\b', text.lower())
    if not words: return 0.0
    counts = Counter(words)
    repeated = sum(count for count in counts.values() if count > 1)
    return round(repeated / len(words), 4)

def get_semantic_similarity(text1, text2):
    if not text1 or not text2: return 0.0
    e1 = semantic_feat_model.encode(text1, convert_to_tensor=True)
    e2 = semantic_feat_model.encode(text2, convert_to_tensor=True)
    return float(util.cos_sim(e1, e2)) * 100

def get_ngrams(text, n):
    words = re.findall(r'\b\w+\b', str(text).lower())
    return set(zip(*[words[i:] for i in range(n)]))

def get_jaccard_similarity(text1, text2, n=3):
    if not text1 or not text2: return 0.0
    ng1 = get_ngrams(text1, n)
    ng2 = get_ngrams(text2, n)
    if not ng1 or not ng2: return 0.0
    return (len(ng1 & ng2) / len(ng1 | ng2)) * 100

def extract_keybert_phrases(text):
    keywords_with_scores = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=10)
    extracted_phrases = [kw[0] for kw in keywords_with_scores]
    main_topic = extracted_phrases[0].title() if extracted_phrases else "Martial Law"
    return main_topic, ", ".join(extracted_phrases)

def generate_comparison_article(style_prompt, user_prompt, temperature):
    messages = [{"role": "system", "content": style_prompt}, {"role": "user", "content": user_prompt}]
    full_text = ""
    min_word_count = 400
    max_continuations = 3
    continuation_count = 0
    while True:
        response = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, temperature=temperature, max_tokens=1500
        )
        chunk = response.choices[0].message.content
        full_text += " " + chunk
        wc = len(full_text.split())
        if (response.choices[0].finish_reason == "length" or wc < min_word_count) and continuation_count < max_continuations:
            messages.append({"role": "assistant", "content": chunk})
            messages.append({"role": "user", "content": f"Continue expanding the essay. You are at {wc} words."})
            continuation_count += 1
            continue
        break
    return full_text.strip()

def analyze_article(text: str):
    if not text.strip():
        raise ValueError("Article text cannot be empty.")

    print("Extracting intrinsic features...")
    p_in = get_perplexity(text)
    b_in = get_burstiness(text)
    we_in = get_word_entropy(text)
    ce_in = get_char_entropy(text)
    rr_in = get_repetition_rate(text)

    print("Fetching contexts...")
    main_topic, keywords = extract_keybert_phrases(text)
    docs = db.similarity_search(f"Details about {keywords} including specific names, dates, and events.", k=5)
    context = "\n\n---\n\n".join([d.page_content.replace('</s>','').replace('<s>','') for d in docs])

    styles = {
        "Neutral": {"temp": 0.3, "prompt": "You are a neutral historian. Your goal is factual density. Write in the THIRD-PERSON. Use a clinical, journalistic tone. Avoid emotive adjectives. Focus on the 'who, what, where, when'."},
        "Passionate": {"temp": 0.7, "prompt": "You are a passionate editorial writer. Take a clear moral stance based on the provided context. Use the specific facts provided to support your emotional arguments."},
        "HighTemp": {"temp": 0.9, "prompt": "You are a Filipino blogger writing a personal essay. Use a FIRST-PERSON conversational tone. Anchor your story in the specific historical terms and keywords provided."}
    }

    from datetime import date
    user_prompt = f"""
    ### [MANDATORY REFERENCE CONTEXT]
    {context}
    
    ### [CURRENT REALITY DATA]
    - TODAY: {date.today().strftime('%B %d, %Y')}
    - ELAPSED TIME: {date.today().year - 1972} years since declaration.
    
    ### [MANDATORY VOCABULARY]
    You MUST explicitly include and discuss these specific terms: {keywords}
    
    Write a 500-word essay focused on: **{main_topic}**.
    """

    results = {}
    for style_name, config in styles.items():
        print(f"Generating {style_name} baseline...")
        gen_text = generate_comparison_article(config['prompt'], user_prompt, config['temp'])
        sem = get_semantic_similarity(text, gen_text)
        lex = get_jaccard_similarity(text, gen_text)
        results[style_name] = {
            "Semantic": sem, "Lexical": lex,
            "Perplexity_Delta": p_in - get_perplexity(gen_text),
            "Burstiness_Delta": b_in - get_burstiness(gen_text),
            "Word_Entropy_Delta": we_in - get_word_entropy(gen_text),
            "Char_Entropy_Delta": ce_in - get_char_entropy(gen_text),
            "Repetition_Delta": rr_in - get_repetition_rate(gen_text)
        }

    # Prepare feature array matched to FEATURE_COLUMNS
    features = [
        p_in, b_in, we_in, ce_in, rr_in,
        results["Neutral"]["Semantic"], results["Passionate"]["Semantic"], results["HighTemp"]["Semantic"],
        results["Neutral"]["Lexical"], results["Passionate"]["Lexical"], results["HighTemp"]["Lexical"],
        results["Neutral"]["Perplexity_Delta"], results["Passionate"]["Perplexity_Delta"], results["HighTemp"]["Perplexity_Delta"],
        results["Neutral"]["Burstiness_Delta"], results["Passionate"]["Burstiness_Delta"], results["HighTemp"]["Burstiness_Delta"],
        results["Neutral"]["Word_Entropy_Delta"], results["Passionate"]["Word_Entropy_Delta"], results["HighTemp"]["Word_Entropy_Delta"],
        results["Neutral"]["Char_Entropy_Delta"], results["Passionate"]["Char_Entropy_Delta"], results["HighTemp"]["Char_Entropy_Delta"],
        results["Neutral"]["Repetition_Delta"], results["Passionate"]["Repetition_Delta"], results["HighTemp"]["Repetition_Delta"]
    ]

    print("Running inference...")
    scaled_feats = scaler.transform([features])
    prob = float(rf_model.predict_proba(scaled_feats)[0][1]) * 100
    
    classification = "AI Generated" if prob >= 40.0 else "Human Written"

    return {
        "is_ai": bool(prob >= 40.0),
        "classification": classification,
        "ai_probability": round(prob, 2),
        "features": {
            "perplexity": round(float(p_in), 2),
            "burstiness": round(float(b_in), 2),
            "word_entropy": round(float(we_in), 2),
            "char_entropy": round(float(ce_in), 2),
            "repetition_rate": round(float(rr_in), 2)
        },
        "comparisons": results
    }
