# concatenate txt files
folder = '../wifi_data/'
origin_txt = ['2018_04_19_08_00_10.txt', '2018_04_19_14_32_48.txt']
new_txt = '2018_04_19_08_00_09.txt'

with open(folder+new_txt, 'w') as outfile:
    for fname in origin_txt:
        print(fname)
        with open(folder+fname) as infile:
            outfile.write(infile.read())
        # outfile.write(infile.read('\n'))

