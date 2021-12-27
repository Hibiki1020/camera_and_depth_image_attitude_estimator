import csv

def main():
    deg_index = 0

    with open('../index_dict/index_dict_360x1range_step1deg.csv', 'w') as f:
        writer = csv.writer(f)
        for deg in range( -180, 181):
            writer.writerow([deg, deg_index])
            deg_index += 1

if __name__ == '__main__':
    main()