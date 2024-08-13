from legalbenchrag.generate.generate_contractnli import generate_contractnli
from legalbenchrag.generate.generate_cuad import generate_cuad
from legalbenchrag.generate.generate_maud import generate_maud
from legalbenchrag.generate.generate_privacy_qa import generate_privacy_qa


async def generate_all() -> None:
    await generate_contractnli()
    await generate_cuad()
    await generate_maud()
    await generate_privacy_qa()


__all__ = [
    "generate_all",
]
