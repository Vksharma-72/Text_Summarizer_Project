from textSummarizer.pipeline.data_pipeline import DataIngestionTrainingPipeline
from textSummarizer.pipeline.data_validation_pipeline import DataValidationPipeline
from textSummarizer.logging import logger 


STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<< ")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<<\n\n x==============x ")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Validation stage"
try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<< ")
    data_ingestion = DataValidationPipeline()
    data_ingestion.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<<\n\n x==============x ")
except Exception as e:
    logger.exception(e)
    raise e
