from setuptools import setup, find_packages

setup(
    name="mlx_t2v_researcher",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "rich",
        "openai",
        "exa-py",
        "pydantic>=2.0.0",
        "agno",
        "psycopg2-binary",
        "sqlalchemy",
        "python-dotenv",
        "pgvector"
    ],
) 