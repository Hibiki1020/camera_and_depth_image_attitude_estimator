import csv

def main():
    roll_index = 0
    pitch_index = 0

    final_index = 0

    with open('../index_dict/index_dict_360range_step2deg.csv', 'w') as f:
        writer = csv.writer(f)
        for roll in range( -180, 181):

            if (roll%2) != 0:
                continue

            for pitch in range(-180, 181):
                
                if (pitch%2) != 0:
                    continue

                writer.writerow([roll, pitch, roll_index, pitch_index, final_index])
                pitch_index += 1
                final_index += 1
            
            roll_index += 1
            pitch_index = 0


if __name__ == '__main__':
    main()