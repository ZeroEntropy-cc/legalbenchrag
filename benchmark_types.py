from pydantic import BaseModel, computed_field


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


class Benchmark(BaseModel):
    tests: list[QA]
