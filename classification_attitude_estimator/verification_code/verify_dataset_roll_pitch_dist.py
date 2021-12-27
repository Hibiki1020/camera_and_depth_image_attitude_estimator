import csv
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class VerifyDataset:
    def __init__(self, dataset, csv_name):
        self.dataset = dataset
        self.csv_name = csv_name

        self.roll, self.pitch = self.load_data(dataset, csv_name)
        
        self.save_hist(self.roll, self.pitch)

    def load_data(self, dataset, csv_name):
        roll = []
        pitch = []

        for sequence in dataset:
            csv_path = sequence + csv_name
            with open(csv_path) as f:
                reader = csv.reader(f)
                for row in reader:
                    tmp_roll = float(row[4])/3.141592*180.0
                    tmp_pitch = float(row[5])/3.141592*180.0

                    roll.append(tmp_roll)
                    pitch.append(tmp_pitch)

        return roll, pitch

    def save_hist(self, roll, pitch):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        histogram = ax.hist2d(roll, pitch, bins=[np.linspace(-180, 180, 50), np.linspace(-180, 180, 50)], cmap=cm.jet)
        ax.set_xlabel('Roll')
        ax.set_ylabel('Pitch')
        ax.set_title('Distribution of Roll and Pitch in dataset')

        fig.colorbar(histogram[3], ax=ax)
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./verify_dataset_roll_pitch_dist.py")

    parser.add_argument(
        '--config', '-c',
        type=str,
        required=False,
        default='../pyyaml/verify_dataset_roll_pitch_dist.yaml',
    )

    FLAGS, unparsed = parser.parse_known_args()

    #Load Yaml file
    try:
        print("Opening config file %s", FLAGS.config)
        CFG = yaml.safe_load(open(FLAGS.config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening config file %s",FLAGS.config)
        quit()
    
    dataset = CFG["dataset"]
    csv_name = CFG["csv_name"]

    verify_dataset = VerifyDataset(
        dataset,
        csv_name
    )
