import pandas as pd
from pandasql import sqldf

def run_sql_query(query: str, df: pd.DataFrame):
    query = query.strip().rstrip(';')
    pysqldf = lambda q: sqldf(q, {"df": df})
    return pysqldf(query)
