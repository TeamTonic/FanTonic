from src.geo_data.utils import get_geo_data
from src.gen_ai.structured_data import agent_executor

sample_landmark = agent_executor.invoke(
        {"input": "find a landmark in Benin "},
    )

geo_data =get_geo_data(sample_landmark["landmark"])
x = 0