import re
from typing import List, Optional, Tuple

from parallel_corpus import graph
from parallel_corpus.token import Token
from sparv import api as sparv_api  # type: ignore [import-untyped]
from transformers import (  # type: ignore [import-untyped]
    AutoTokenizer,
    T5ForConditionalGeneration,
    pipeline,
)


def bytes_length(s: str) -> int:
    return len(s.encode("utf-8"))


TOK_SEP = " "
logger = sparv_api.get_logger(__name__)
TOKENIZER_REVISION = "68377bdc18a2ffec8a0533fef03b1c513a4dd49d"
TOKENIZER_NAME = "google/byt5-small"
MODEL_REVISION = "84b138048992271be7617ccb11056bbcb9b72262"
MODEL_NAME = "viklofg/swedish-ocr-correction"

PUNCTUATION = re.compile(r"[.,:;!?]")


class OcrCorrector:
    TEXT_LIMIT: int = 127

    def __init__(self, *, tokenizer, model) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.pipeline = pipeline(
            "text2text-generation", model=model, tokenizer=tokenizer
        )

    @classmethod
    def default(cls) -> "OcrCorrector":
        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_NAME, revision=TOKENIZER_REVISION
        )
        model = T5ForConditionalGeneration.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION
        )
        return cls(model=model, tokenizer=tokenizer)

    def calculate_corrections(
        self, text: List[str]
    ) -> List[Tuple[Tuple[int, int], Optional[str]]]:
        logger.debug("Analyzing '%s'", text)

        parts: List[str] = []
        curr_part: List[str] = []
        curr_len = 0
        ocr_corrections: List[Tuple[Tuple[int, int], Optional[str]]] = []
        for word in text:
            len_word = bytes_length(word)
            if (curr_len + len_word + 1) > self.TEXT_LIMIT:
                parts.append(TOK_SEP.join(curr_part))
                curr_part, curr_len = [word], len_word
            else:
                curr_part.append(word)
                curr_len = len_word if curr_len == 0 else curr_len + len_word + 1
        if len(curr_part) > 0:
            parts.append(TOK_SEP.join(curr_part))
        curr_start = 0
        for part in parts:
            suggested_text = self.pipeline(part)[0]["generated_text"]
            suggested_text = PUNCTUATION.sub(r" \g<0>", suggested_text)
            graph_aligned = graph.init_with_source_and_target(part, suggested_text)
            span_ann, curr_start = align_and_diff(graph_aligned, curr_start=curr_start)
            ocr_corrections.extend(span_ann)

        logger.debug("Finished analyzing. ocr_corrections=%s", ocr_corrections)
        return ocr_corrections


def align_and_diff(
    g: graph.Graph, *, curr_start: int
) -> Tuple[List[Tuple[Tuple[int, int], Optional[str]]], int]:
    corrections = []

    edge_map = graph.edge_map(g)
    visited_tokens = set()
    for s_token in g.source:
        logger.debug("checking s_token=%s", s_token)
        edge = edge_map[s_token.id]

        source_ids = [id_ for id_ in edge.ids if id_.startswith("s")]
        target_ids = [id_ for id_ in edge.ids if id_.startswith("t")]
        target_ids_str = "-".join(target_ids)
        if target_ids_str in visited_tokens:
            continue
        visited_tokens.add(target_ids_str)
        logger.debug("processing s_token=%s", s_token)

        if len(source_ids) == len(target_ids):
            source_text = "".join(
                lookup_text(g.source, s_id) for s_id in source_ids
            ).strip()
            target_text = "".join(
                lookup_text(g.target, s_id) for s_id in target_ids
            ).strip()
            start = curr_start
            curr_start += 1
            corrections.append(
                (
                    (start, curr_start),
                    target_text if source_text != target_text else None,
                )
            )

        elif len(source_ids) == 1:
            target_texts = " ".join(
                lookup_text(g.target, id_).strip() for id_ in target_ids
            )
            source_text = s_token.text.strip()
            start = curr_start
            curr_start += 1

            corrections.append(
                (
                    (start, curr_start),
                    target_texts if source_text != target_texts else None,
                ),
            )
        elif len(target_ids) == 1:
            target_text = lookup_text(g.target, target_ids[0]).strip()
            start = curr_start
            curr_start += len(source_ids)
            corrections.append(((start, curr_start), target_text))
        else:
            # TODO Handle this correct (https://github.com/spraakbanken/sparv-sbx-ocr-correction/issues/50)
            raise NotImplementedError(
                f"Handle several sources, {source_ids=} {target_ids=} {g.source=} {g.target=}"  # noqa: E501
            )

    return corrections, curr_start


def lookup_text(tokens: List[Token], id_: str) -> str:
    for token in tokens:
        if token.id == id_:
            return token.text

    raise ValueError(
        f"The id={id_} isn't found in the list of tokens",
    )
