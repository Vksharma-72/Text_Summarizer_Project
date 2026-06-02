from textSummarizer.pipeline.data_pipeline import DataIngestionTrainingPipeline
from textSummarizer.pipeline.data_validation_pipeline import DataValidationPipeline
from textSummarizer.pipeline.data_transformation_pipeline import DataTransformationPipeline
from textSummarizer.pipeline.model_training_pipeline  import ModelTrainerPipeline
from textSummarizer.pipeline.model_evaluation_pipeline import ModelEvaluationPipeline
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
    data_validation = DataValidationPipeline()
    data_validation.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<<\n\n x==============x ")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Taransformation stage"
try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<< ")
    data_transformation = DataTransformationPipeline()
    data_transformation.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<<\n\n x==============x ")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Training stage"
try:
    logger.info(f">>>>> stage {STAGE_NAME} Started <<<<<<<<<")
    model_trainer = ModelTrainerPipeline()
    model_trainer.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<<<\n\n x==============X")
except Exception as e:
    logger.exception(e)
    raise e 

STAGE_NAME = "Model Evaluation stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} Started <<<<<<<<<<<<")
    model_evaluation = ModelEvaluationPipeline()
    model_evaluation.main()
    logger.info(f">>>>>>>>> stage {STAGE_NAME} Completed ")
except Exception as e:
    logger.exception(e)
    raise e
