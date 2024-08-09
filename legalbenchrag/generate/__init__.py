from legalbenchrag.generate.generate_contractnli import generate_contractnli
from legalbenchrag.generate.generate_cuad import generate_cuad


async def generate_all() -> None:
    await generate_contractnli()
    await generate_cuad()


__all__ = [
    "generate_all",
]
