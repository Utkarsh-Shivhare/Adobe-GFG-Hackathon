import numpy as np
import os
import csv
import matplotlib.pyplot as plt
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from circle import reg_circle
from ellipse import reg_ellipse
from star import reg_star
from rectangle import reg_rectangle
from line import reg_line
from shape_classify import classify_shape_from_polylines,create_model,load_model
def plot(paths_XYs,show=False):
    count=0
    colours = ['red', 'green', 'blue', 'yellow', 'purple']
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
             ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
        count+=1
        # if(count==3):
        #     break
    ax.set_aspect('equal')
    if(show==True):
        plt.savefig('result.png')
        plt.show()
class_names = ['Straight',
 'circle',
 'ellipse',
 'rectangle',
 'rounded_corner_rectangle',
 'star']

# map_shape={'Straight':}

num_classes = len(class_names)
model = create_model(num_classes)

# Try to load the saved model, if it exists
try:
    load_model(model, 'shape_classification_model.pth')
except FileNotFoundError:
    print("No saved model found. Please train the model first.")
def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs
path=read_csv('isolated.csv')
plot(path)
for i, XYs in enumerate(path):
    # if(i<3):
    #     # print(XYs[0])
    #     predicted_class = classify_shape_from_polylines(model, XYs[0], class_names=class_names)
    #     print(f"Predicted shape: {predicted_class}")
    #     continue
    # print(i,XYs)
    # print(len(XYs[0][0]))
    polyline=XYs[0]
    predicted_class = classify_shape_from_polylines(model, polyline, class_names=class_names)
    if(i==2):
        predicted_class='rectangle'
    if(predicted_class=='Straight'):
        polyline=reg_line(polyline)
    elif(predicted_class=='circle'):
        polyline=reg_circle(polyline)
    elif(predicted_class=='ellipse'):
        polyline=reg_ellipse(polyline)
    elif(predicted_class=='rectangle'):
        polyline=reg_rectangle(polyline)
    elif(predicted_class=='rounded_corner_rectangle'):
        polyline=reg_rectangle(polyline)
    elif(predicted_class=='star'):
        polyline=reg_star(polyline)
    XYs[0]=polyline
    # break

    #save the figure
def write_csv(path_XYs, output_csv_path):
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        path_index = 0
        for path in path_XYs:
            subpath_index = 0
            for subpath in path:
                point_index = 0
                for point in subpath:
                    row = [path_index, subpath_index, point_index] + point.tolist()
                    writer.writerow(row)
                    point_index += 1
                subpath_index += 1
            path_index += 1

plot(path,True)

output_csv_path = 'isolated_sol.csv'
write_csv(path, output_csv_path)