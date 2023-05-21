def create_answers_file(y_pred):
    
    
    names = "# Bernardo D'Agostino  Andrea Roncoli  Bianca Ziliotto\n"

    team_name = '# RGA\n'

    data_set_name = '# ML-CUP22 v1\n'

    date = '# 23 Jan 2023\n'

    # open txt file
    with open('RGA_ML-CUP22-TS.csv', 'w') as f:
        f.write(names)
        f.write(team_name)
        f.write(data_set_name)
        f.write(date)
        for i in range(len(y_pred)):
            f.write(str(i+1))
            f.write(',')
            f.write(str(y_pred[i, 0]))
            f.write(',')
            f.write(str(y_pred[i, 1]))
            f.write('\n')
