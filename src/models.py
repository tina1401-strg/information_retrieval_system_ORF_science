import gc
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from config import (
    LLM_MODEL, 
    EMBED_MODEL, 
    KG_MODEL, 
    NER_MODEL, 
    DEVICE, 
    KG_BATCH_SIZE
)


# ── LLM ───────────────────────────────────────────────────────────────────────

class LLM:
    def __init__(self):
        print(f"  Loading LLM: {LLM_MODEL} on {DEVICE} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        self.model     = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            device_map  = {"": DEVICE},
            torch_dtype = torch.bfloat16,
        ).eval()
        print(f"  LLM loaded.")

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        messages  = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens = max_new_tokens,
                do_sample      = False,
                pad_token_id   = self.tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Embedder ──────────────────────────────────────────────────────────────────

class Embedder:
 
    def __init__(self):
        print(f"  Loading embedder: {EMBED_MODEL} on {DEVICE} ...")
        self.model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
        print(f"  Embedder loaded.")

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size           = batch_size,
            show_progress_bar    = True,
            convert_to_numpy     = True,
            normalize_embeddings = True,
        )

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode(
            f"query: {query}",
            convert_to_numpy     = True,
            normalize_embeddings = True,
        )

    def to_cpu(self):
        self.model = self.model.to("cpu")


# ── EntityExtractor ───────────────────────────────────────────────────────────

class EntityExtractor:
    """GLiNER — used for named entity extraction from queries."""
    
    LABELS = [
    "person",                     # researchers, historical figures
    "organisation",               # universities, institutes, agencies (NASA, CERN)
    "ort",                        # geographic locations
    "konzept",                    # ideas, principles
    "wissenschaftlicher Begriff", # technical terminology
    "datum",                      # dates, years, time periods
    "ereignis",                   # discoveries, missions, experiments (Mondlandung)
    "maßeinheit",                 # quantities & measurements (5 Lichtjahre, 300 K)
    "organismus",                 # species, animals, plants, microbes
    "substanz",                   # chemicals, elements, molecules, materials (CO₂, DNA)
    "krankheit",                  # diseases & medical conditions
    "körperteil",                 # anatomy / body parts
    "himmelskörper",              # planets, stars, galaxies (Mars, Andromeda)
    "technologie",                # instruments, devices, methods (Teleskop, MRT)
    "publikation",                # studies, journals, books (Nature, Studientitel)
]

    def __init__(self):
        from gliner import GLiNER
        print(f"  Loading GLiNER: {NER_MODEL} ...")
        self.model = GLiNER.from_pretrained(NER_MODEL)
        print(f"  GLiNER loaded.")

    def extract(self, text: str, threshold: float = 0.5) -> list[str]:
        """Extract entity strings from a query."""
        entities = self.model.predict_entities(text, self.LABELS, threshold=threshold, flat_ner = True)
        return [e["text"] for e in entities]


# ── KGExtractor ───────────────────────────────────────────────────────────────

class KGExtractor:

    def __init__(self):
        print(f"  Loading KG model: {EMBED_MODEL} on {DEVICE} ...")
        self._model     = AutoModelForSeq2SeqLM.from_pretrained(KG_MODEL).to(DEVICE).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(KG_MODEL, src_lang="de_DE")
        print(f"  KG model loaded.")

    def unload(self) -> None:
        if self._model is not None:
            self._model.cpu()
            del self._model
            del self._tokenizer
            self._model     = None
            self._tokenizer = None
            print("  mREBEL unloaded.")

    def extract_triples(self, chunks: list[str]) -> list[tuple]:
        if self._model is None:
            raise RuntimeError("KGExtractor not loaded — call .load() first.")
        # extraction logic lives here, moved from knowledge_graph.py
        all_triples = []
        for i in range(0, len(chunks), KG_BATCH_SIZE):
            batch  = chunks[i:i + KG_BATCH_SIZE]
            inputs = self._tokenizer(
                batch,
                return_tensors = "pt",
                max_length     = 200,
                truncation     = True,
                padding        = True,
            ).to(self._model.device)
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    forced_bos_token_id = self._tokenizer.convert_tokens_to_ids("de_DE"),
                    max_new_tokens      = 512,
                    num_beams           = 3,
                    top_p = 0,
                    top_k = 0
                )
            for out in output_ids:
                decoded = self._tokenizer.decode(out, skip_special_tokens=False)
                all_triples.extend(self._parse(decoded))
        return all_triples

    def tokenize(self, chunk: str):
        return self._tokenizer.encode(chunk, add_special_tokens=False)
    
    @staticmethod
    def _parse(text: str) -> list[tuple]:
        import re
        TAG_PATTERN = re.compile(r'<(?:per|org|loc|misc|time|num|concept|dis|date)>')
        triples     = []
        text        = re.sub(r"<s>|</s>|<pad>|\w+_\w+\s", "", text).strip()
        for chunk in [c.strip() for c in text.split("<triplet>") if c.strip()]:
            tag_positions = [(m.start(), m.group()) for m in TAG_PATTERN.finditer(chunk)]
            if len(tag_positions) < 2:
                continue
            first_idx,  first_tag  = tag_positions[0]
            second_idx, second_tag = tag_positions[1]
            subject  = chunk[:first_idx].strip()
            obj      = chunk[first_idx + len(first_tag):second_idx].strip()
            rel      = TAG_PATTERN.split(chunk[second_idx + len(second_tag):].strip())[0].strip()
            if subject and obj and rel and subject != obj:
                triples.append((subject, rel, obj))
        return triples

# ── shared cleanup ────────────────────────────────────────────────────────────

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()