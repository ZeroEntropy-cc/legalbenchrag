from legalbenchrag.generate.generate_contractnli import generate_contractnli
from legalbenchrag.generate.generate_cuad import generate_cuad
from legalbenchrag.generate.generate_maud import generate_maud


async def generate_all() -> None:
    await generate_contractnli()
    await generate_cuad()
    await generate_maud()


__all__ = [
    "generate_all",
]
