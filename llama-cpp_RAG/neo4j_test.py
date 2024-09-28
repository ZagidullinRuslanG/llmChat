import os

from langchain_community.graphs import Neo4jGraph

login = 'neo4j'
# password = 'XLR48xNsaLxKwVq'
password = 'your_password'

os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = login
os.environ["NEO4J_PASSWORD"] = password


graph = Neo4jGraph()

