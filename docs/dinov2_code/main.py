import logging
import os

from dinov2.logging import setup_logging
from dinov2.run.submit import get_args_parser
from dinov2.train import get_args_parser as get_train_args_parser

logger = logging.getLogger("dinov2")


class MyTrainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        from dinov2.train import main as train_main

        self._setup_args()
        train_main(self.args)

    def checkpoint(self):
        logger.info(f"Requeuing {self.args}")

    def _setup_args(self):
        logger.info(f"Args: {self.args}")


def main():
    description = "Submitit launcher for DINOv2 training"
    train_args_parser = get_train_args_parser(add_help=False)
    parents = [train_args_parser]
    args_parser = get_args_parser(description=description, parents=parents)
    args = args_parser.parse_args()

    setup_logging()

    assert os.path.exists(args.config_file), "Configuration file does not exist!"
    trainer = MyTrainer(args)
    trainer()
    # submit_jobs(MyTrainer, args, name="dinov2:train")
    return 0


if __name__ == "__main__":
    main()
