@echo off

cd .\lawfastapi

echo "Run pinecone_module"
poetry run python .\pinecone_module\write.py

Pause