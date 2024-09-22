import logging

class LoggerConfig:
    @staticmethod
    def setup_logger():
        logging.basicConfig(filename='eda_log.log', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    @staticmethod
    def log_message(message):
        logging.info(message)
