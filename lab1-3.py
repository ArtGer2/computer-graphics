import matplotlib.pyplot as plt
import numpy as np
import math
import json

def string_to_2d_array(input_string):
    array_2d = eval(input_string)
    return array_2d

def binomial_coefficient(n, k):
        return np.math.factorial(n) / (np.math.factorial(k) * np.math.factorial(n - k))
def de_casteljau(points, t):

    if len(points) == 1:
        return points[0]
    else:
        new_points = [((1 - t) * p0 + t * p1) for p0, p1 in zip(points[:-1], points[1:])]
        return de_casteljau(new_points, t)

class Line:
    def __init__(self, start=[0 ,0 , 0], end=[0, 0, 0]):
        self.coords = []
        self.coords.append(np.append(np.array(start),1))
        self.coords.append(np.append(np.array(end),1))
    
    def plot(self,trim_matrix):
        result = np.matmul(self.coords,trim_matrix)
        plt.plot(result.T[0],result.T[1])

    def rotate(self,matrix):
        self.coords = np.matmul(self.coords,matrix)

class bezier_curve:

    def __init__(self, point_arr):
        self.coords = np.array(point_arr)
        ones_column = np.ones((len(self.coords), 1))
        self.coords = np.concatenate((self.coords, ones_column), axis=1)
        self.num_points = 100
    
    def edit(self, arr):
        self.coords = np.array(arr)
        ones_column = np.ones((len(self.coords), 1))
        self.coords = np.concatenate((self.coords, ones_column), axis=1)
    
    def plot(self,tm):
        result = np.matmul(self.coords,tm)
        result = np.delete(result,2,1)
        result = np.delete(result,2,1)
        t = np.linspace(0, 1, self.num_points)
        n = len(result) - 1
        curve = np.zeros((self.num_points, 2))

        for i in range(self.num_points):
            curve[i] = np.sum([binomial_coefficient(n, k) * ((1 - t[i])**(n - k)) * (t[i]**k) * result[k] for k in range(n + 1)], axis=0)
        plt.plot(result[:,0], result[:,1], 'o--')
        plt.plot(curve[:,0], curve[:,1],)

    def rotate(self,rm):
        self.coords = np.matmul(self.coords,rm)
    
class bezier_surface:
    
    def __init__ (self, control_points, u_samples =10, v_samples =10):

        self.coords = []

        self.control_points = control_points

        bezier_points = []

        for i in range(u_samples):
            u = i / (u_samples - 1)
            u_points = []
            for j in range(v_samples):
                v= j / (v_samples - 1)
                v_points = [de_casteljau([control_points[k][j] for k in range(len(control_points))], u)
                        for j in range(len(control_points[0]))]
                u_points.append(np.append(de_casteljau(v_points, v),1))

            bezier_points.append(u_points)
        self.coords = bezier_points


    def plot(self,tm):

        for j in self.coords:
            result = np.matmul(j,tm)
            plt.plot(result.transpose()[0],result.transpose()[1])

        for j in zip(*self.coords):
            result = np.matmul( j,tm)
            plt.plot(result.transpose()[0],result.transpose()[1])
        

    def rotate(self,rm):
        self.coords = np.matmul(self.coords,rm)

class Space:
    obj_list=[]
    camera_angle = []

    def __init__(self, angle = (0.0,0.0,0.0)):
        self.curves = []
        self.objects = []
        self.surfaces = []
        self.sys_cord()
        self.camera_angle = angle

    def add_object(self, obj):
        self.obj_list.append(obj)

    def sys_cord(self):
        self.add_object(Line(start=(0,0,0), end=(1,0,0)))
        self.add_object(Line(start=(0,0,0), end=(0,1,0)))
        self.add_object(Line(start=(0,0,0), end=(0,0,1)))

    def rotate(self,angle,axis, obj_num):
        obj_num +=2
        if (axis == 'x'):
            rotate_matrix =  [
                [1,0,0,0],
                [0,math.cos(angle),math.sin(angle),0],
                [0,-math.sin(angle),math.cos(angle),0],
                [0,0,0,1]
            ]       
        
        elif (axis == 'y'):
            rotate_matrix =  [
                [math.cos(angle),0,-math.sin(angle),0],
                [0,1,0,0],
                [math.sin(angle),0,math.cos(angle),0],
                [0,0,0,1]
            ]   

        elif (axis == 'z'):

            rotate_matrix =  [
                [math.cos(angle),math.sin(angle),0,0],
                [-math.sin(angle),math.cos(angle),0,0],
                [0,0,1,0],
                [0,0,0,1]
            ]  

        self.obj_list[obj_num].rotate(rotate_matrix)



    def plot(self):

        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        pitch, yaw,roll = self.camera_angle
        trim_projection_matrix = np.array([
            [math.cos(pitch), math.sin(pitch)*math.sin(yaw) ,0,0],
            [0,math.cos(yaw) ,0,0],
            [math.sin(pitch), -1 * math.cos(pitch)*math.sin(yaw) ,0,0],
            [0,0,0,1],
        ])

        for i in self.obj_list:
            i.plot(trim_projection_matrix)
            
class Rectangle:
    
    def __init__ (self, st, end, color = None):
        self.Lines_list = []
        self.Lines_list.append(Line((st[0],st[1],st[2]),(st[0],st[1],end[2])))
        self.Lines_list.append(Line((st[0],st[1],st[2]),(end[0],st[1],st[2]))) 
        self.Lines_list.append(Line((st[0],st[1],st[2]),(st[0],end[1],st[2])))

        self.Lines_list.append(Line((end[0],end[1],end[2]),(end[0],end[1],st[2])))
        self.Lines_list.append(Line((end[0],end[1],end[2]),(end[0],st[1],end[2])))  
        self.Lines_list.append(Line((end[0],end[1],end[2]),(st[0],end[1],end[2])))

        self.Lines_list.append(Line((end[0],end[1],st[2]),(st[0],end[1],st[2])))
        self.Lines_list.append(Line((end[0],end[1],st[2]),(end[0],st[1],st[2])))

        self.Lines_list.append(Line((end[0],st[1],end[2]),(st[0],st[1],end[2])))
        self.Lines_list.append(Line((end[0],st[1],end[2]),(end[0],st[1],st[2])))

        self.Lines_list.append(Line((st[0],end[1],end[2]),(st[0],st[1],end[2])))
        self.Lines_list.append(Line((st[0],end[1],end[2]),(st[0],end[1],st[2])))
        

    def plot(self,tm):
        for i in self.Lines_list:
            i.plot(tm)
    
    def rotate(self,tm):
         for i in self.Lines_list:
            i.rotate(tm)

if __name__ == "__main__":


    space = Space((math.radians(45), math.radians(45), 0))
    surf = bezier_surface([
    [np.array([5, 5, 0]), np.array([5, 3, 6]), np.array([5, -3, -6]), np.array([5, -5, 0])],
    [np.array([3, 5, 6]), np.array([5, 5, 5]), np.array([1, 3, 4]), np.array([3, -5, 4])],
    [np.array([-3, 5, -6]), np.array([1, 2, 3]), np.array([5, -5, -5]), np.array([-3, -5, -1])],
    [np.array([-5,5, 0]), np.array([-5, 3, -3]), np.array([-5, -4, -4]), np.array([-5, -5, 0])]
])
    space.add_object(surf)
    space.plot()


    for i in range(72): 
        plt.pause(0.001)
        plt.clf()
        space.rotate(math.radians(5),'y',1)
        space.plot()
        plt.draw()
            
            





