filename = r'C:\Users\Administrator\Desktop\standardPose.log'
count = 0
pointDict = {}
count = 0
with open(filename) as f:
    for line in f:
        if '----------------------------------------------' in line:
            count += 1
            continue
        spts = line.split(': (')
        xyz = spts[1][:-2].split(',')
        if pointDict.get(spts[0], None) is None:
            pointDict[spts[0]] = [0, 0, 0]
        x = pointDict[spts[0]][0] + float(xyz[0])
        y = pointDict[spts[0]][1] + float(xyz[1])
        z = pointDict[spts[0]][2] + float(xyz[2])
        pointDict[spts[0]] = [x, y, z]


pointSeq = [
    '0', '1', '3', '26', '5', '6', '7', '8',
    '12', '13', '14', '15', '18', '19', '20', '21',
    '22', '23', '24', '25', '2', '9', '10', '16', '17'
]
for s in pointSeq:
    x = pointDict[s][0] / count
    y = pointDict[s][1] / count
    z = pointDict[s][2] / count
    print(x, y, z)



print('SpineBase: ()')







