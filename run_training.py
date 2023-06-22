import argparse
import logging
from src.training_contact_grasp_fusion import MainTrainingContactGraspFusion
from src.training_contact_grasp import MainTrainingContactGrasp

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="TRAINING NETWORK --MASTERARBEIT--NICOLAS-RODRIGUEZ"
        )
        parser.add_argument(
            "-c",
            "--config",
            help="path to a config file",
        )
        parser.add_argument(
            "-n",
            "--network",
            help="network 0- ContactGraspFusion and 1- ContactGrasp",
            default=1,
            type=int,
        )
        parser.add_argument(
            "-s",
            "--show",
            help="show dataset",
            default=True,
            type=bool,
        )
        args = parser.parse_args()
        if args.network == 0:
            main = MainTrainingContactGraspFusion(config_path=args.config)
            main.visualize_random_ds(show=args.show)
            main.run_training()
        else:
            main = MainTrainingContactGrasp(config_path=args.config)
            main.visualize_random_ds(show=args.show)
            main.run_training()

    except Exception:
        logging.exception("Exception occured")
    finally:
        logging.debug("Program was succesfull finilized")
