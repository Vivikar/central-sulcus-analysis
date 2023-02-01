FreeSurferColorLUT = '/mrhome/vladyslavz/git/central-sulcus-analysis/FreeSurferColorLUT.txt'


def fs_lut(lut_path:str=FreeSurferColorLUT):

    with open(lut_path, 'r') as f:
        r=f.readlines()
        
    fs_lut_names = {}
    fs_lut_colors = {}
    for l in r:
        line_data = [x for x in l.split(' ') if x]
        if len(line_data) == 6 and line_data[0].isdigit():
            fs_lut_names[int(line_data[0])] = line_data[1]
            fs_lut_colors[int(line_data[0])] = [int(x.strip()) for x in line_data[2:]]
    return fs_lut_names, fs_lut_colors