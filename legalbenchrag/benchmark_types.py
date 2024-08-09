from pydantic import BaseModel, computed_field, model_validator
from typing_extensions import Self


# max_bridge_gap_len will merge spans that are within max_bridge_gap_len characters of eachother.
def sort_and_merge_spans(
    spans: list[tuple[int, int]], *, max_bridge_gap_len: int = 0
) -> list[tuple[int, int]]:
    if len(spans) == 0:
        return []
    spans = sorted(spans, key=lambda x: x[0])
    merged_spans = [spans[0]]
    for span in spans[1:]:
        if span[0] <= merged_spans[-1][1] + max_bridge_gap_len:
            merged_spans[-1] = (merged_spans[-1][0], max(merged_spans[-1][1], span[1]))
        else:
            merged_spans.append(span)
    return merged_spans


class Snippet(BaseModel):
    file_path: str
    span: tuple[int, int]

    @computed_field  # type: ignore[misc]
    @property
    def answer(self) -> str:
        with open(f"./data/corpus/{self.file_path}") as f:
            return f.read()[self.span[0] : self.span[1]]


class QA(BaseModel):
    query: str
    snippets: list[Snippet]

    @model_validator(mode="after")
    def validate_snippet_spans(self) -> Self:
        spans = [snippet.span for snippet in self.snippets]
        for i in range(1, len(spans)):
            if spans[i - 1][1] >= spans[i][0]:
                raise ValueError("Spans are not disjoint.")
        return self


class Benchmark(BaseModel):
    tests: list[QA]
