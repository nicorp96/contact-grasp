import argparse
import logging
import os
from src.validation_contact_grasp import ValidationContactGrasp
from src.contact_grasp import ContactGraspModel
from src.contact_grasp_fusion import ContactGraspModelFusion

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="VALIDATION NETWORK --MASTERARBEIT--NICOLAS-RODRIGUEZ"
        )
        parser.add_argument(
            "-n",
            "--network",
            help="network 0- ContactGraspFusion and 1- ContactGrasp",
            default=1,
            type=int,
        )
        parser.add_argument(
            "-f",
            "--data_file",
            help="data_contact_points_7000, data_contact_points_4000",
            default="data_contact_points_4000",
            type=str,
        )
        base = os.getcwd()
        base_dir = os.path.join(base, "evaluation_data")
        args = parser.parse_args()
        if args.network == 0:
            base_dir = os.path.join(base_dir, "contact_grasp_point_net_fusion")
            model = ContactGraspModelFusion()
            path_mode = os.path.join(base_dir,"loss_bce_2_heads")
            path = os.path.join(path_mode,"contact_grasp_fusion_weights.pth")  # "contact_grasp_point_net_fusion_epoch_700.pth"  # "contact_grasp_fusion_weights.pth"
        else:
            base_dir = os.path.join(base_dir, "contact_grasp_point_net")
            model = ContactGraspModel()
            path_mode = os.path.join(base_dir,"4000_points_session_1")
            path = os.path.join(path_mode,"contact_grasp_weights.pth")  # "evaluation_data_epoch_700.pth"  # "contact_grasp_weights.pth"

        main = ValidationContactGrasp(
            model=model,
            path_weights=path,
            object_train_pred_show=[
                "full1_handoff_cup.npz",
                "full1_handoff_apple.npz",
                "full1_handoff_banana.npz",
                "full1_handoff_elephant.npz",
                "full1_handoff_stanford_bunny.npz",
                "full1_handoff_mug.npz",
            ],
            path_data_set=args.data_file,
            device="cuda:0",
            debug=True,
        )

        # main.evaluation_train_data()
        main.evaluation_test_data()
    except Exception:
        logging.exception("Exception occured")
    finally:
        logging.debug("Program was succesfull finilized")
