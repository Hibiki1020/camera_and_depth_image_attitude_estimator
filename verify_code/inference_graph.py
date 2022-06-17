import yaml
import csv
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

class VerifyInfer:
    def __init__(self, CFG):
        self.CFG = CFG
        self.infer_log_top_directory = CFG["infer_log_top_directory"]
        self.infer_log_bottom_directory = CFG["infer_log_bottom_directory"]
        self.infer_log_top_path = self.infer_log_top_directory + self.infer_log_bottom_directory
        self.infer_log_file_name = CFG["infer_log_file_name"]

        self.data_list, self.roll_diff, self.pitch_diff = self.data_load()

    def data_load(self):
        csv_path = self.infer_log_top_path + self.infer_log_file_name
        data_list = []
        roll_diff_list = []
        pitch_diff_list = []
        
        with open(csv_path) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                data_list.append(row)
                roll_diff_list.append(float(row[6]))
                pitch_diff_list.append(float(row[7]))

        return data_list, roll_diff_list, pitch_diff_list

    def spin(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        roll = np.array(self.roll_diff)
        pitch = np.array(self.pitch_diff)

        ax.set_xlabel("MAE of Roll [deg]")
        ax.set_ylabel("MAE of Pitch [deg]")
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 60)
        plt.grid(True)
        ax.scatter(roll, pitch)
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./inference_graph.py")

    parser.add_argument(
        '--inference_graph', '-ig',
        type=str,
        required=False,
        default='inference_graph.yaml',
    )

    FLAGS, unparsed = parser.parse_known_args()
    #Load yaml file
    try:
        print("Opening yaml file %s", FLAGS.inference_graph)
        CFG = yaml.safe_load(open(FLAGS.inference_graph, 'r'))
    except Exception as e:
        print(e)
        print("Error yaml file %s", FLAGS.inference_graph)
        quit()

    verify_infer = VerifyInfer(CFG)
    verify_infer.spin()