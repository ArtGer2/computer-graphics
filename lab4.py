import matplotlib.pyplot as plt
import numpy as np
import math
import json

BOTTOM = 1   # 00 000001
LEFT   = 2   # 00 000010
TOP    = 4   # 00 000100
RIGHT  = 8   # 00 001000
BACK   = 16  # 00 010000
FRONT  = 32  # 00 100000





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
    
    def plot(self,trim_matrix, style = None):
        result = np.matmul(self.coords,trim_matrix)
        if style != None:
            plt.plot(result.T[0],result.T[1],style)
        else:
            plt.plot(result.T[0],result.T[1])

    def rotate(self,matrix):
        self.coords = np.matmul(self.coords,matrix)


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



    def plot(self,mode = 0):

        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        pitch, yaw,roll = self.camera_angle
        trim_projection_matrix = np.array([
            [math.cos(pitch), math.sin(pitch)*math.sin(yaw) ,0,0],
            [0,math.cos(yaw) ,0,0],
            [math.sin(pitch), -1 * math.cos(pitch)*math.sin(yaw) ,0,0],
            [0,0,0,100],
        ])
        if mode == 0:
            for i in self.obj_list:
                i.plot(trim_projection_matrix)


        elif mode ==1:

            for id_line in range(4,7):
                id_rect = 3 
                

                outcode0 = 0
                outcode1 = 0
                outcodeOut = 0

                temp = [[0,0,0,1],[0,0,0,1]]
                done = False
                beg = self.obj_list[id_line].coords[0]
                end = self.obj_list[id_line].coords[1]
                anArea = self.obj_list[id_rect]
                outcode0 = GetCode(beg, anArea)
                outcode1 = GetCode(end, anArea)
            
                while not done:

                    if outcode0 & outcode1:
                    
                        done = True
                        self.obj_list[id_line].plot(trim_projection_matrix,'-')

                    elif not (outcode0 | outcode1):
                        
                        done = True
                        self.obj_list[id_line].plot(trim_projection_matrix, '-')

                    else:
                        outcodeOut = outcode0 if outcode0 else outcode1
                        
                        if outcodeOut & TOP:
                            x = beg[0] + (end[0] - beg[0]) * (anArea.st[1] - beg[1]) / (end[1] - beg[1])
                            z = beg[2] + (end[2] - beg[2]) * (anArea.st[1]- beg[1]) / (end[1] - beg[1])
                            y = anArea.st[1]
                        elif outcodeOut & BOTTOM:
                            x = beg[0] + (end[0] - beg[0]) * (anArea.end[1] - beg[1]) / (end[1] - beg[1])
                            z = beg[2] + (end[2] - beg[2]) * (anArea.end[1] - beg[1]) / (end[1] - beg[1])
                            y = anArea.end[1]
                        elif outcodeOut & RIGHT:
                            y = beg[1] + (end[1] - beg[1]) * (anArea.end[0] - beg[0]) / (end[0] - beg[0])
                            z = beg[2] + (end[2] - beg[2]) * (anArea.end[0] - beg[0]) / (end[0] - beg[0])
                            x = anArea.end[0]
                        elif outcodeOut & LEFT:
                            y = beg[1] + (end[1] - beg[1]) * (anArea.st[0] - beg[0]) / (end[0] - beg[0])
                            z = beg[2] + (end[2] - beg[2]) * (anArea.st[0] - beg[0]) / (end[0] - beg[0])
                            x = anArea.st[0]
                        elif outcodeOut & FRONT:
                            x = beg[0] + (end[0] - beg[0]) * (anArea.end[2] - beg[2]) / (end[2] - beg[2])
                            y = beg[1] + (end[1] - beg[1]) * (anArea.end[2] - beg[2]) / (end[2] - beg[2])
                            z = anArea.end[2]
                        elif outcodeOut & BACK:
                            x = beg[0] + (end[0] - beg[0]) * (anArea.st[2] - beg[2]) / (end[2] - beg[2])
                            y = beg[1] + (end[1] - beg[1]) * (anArea.st[2] - beg[2]) / (end[2] - beg[2])
                            z = anArea.st[2]

                        if outcodeOut == outcode0:
                            temp[0][0] = x
                            temp[0][1] = y
                            temp[0][2] = z
                            outcode0 = GetCode(temp[0], anArea)
                        
                        else:
                            temp[1][0] = x
                            temp[1][1] = y
                            temp[1][2] = z
                            
                            outcode1 = GetCode(temp[1], anArea)

                        
                result = np.matmul(temp, trim_projection_matrix)
                plt.plot(result.T[0],result.T[1],'o')
                plt.plot(result.T[0],result.T[1],'white')    
                anArea.plot(trim_projection_matrix,'black')
             
            

def GetCode(point, anArea):
    code = 0
    x, y, z = point[0], point[1], point[2]
    if y > anArea.st[1]:
        code |= TOP
    elif y < anArea.end[1]:
        code |= BOTTOM
    if x < anArea.end[0]:
        code |= RIGHT
    elif x > anArea.st[0]:
        code |= LEFT
    if z > anArea.st[2]:
        code |= FRONT
    elif z < anArea.end[2]:
        code |= BACK
        
    return code                

            
            
class Rectangle:
    
    def __init__ (self, st, end, color = None):
        self.st = np.append(st,1)
        self.end = np.append(end,1)
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
        

    def plot(self,tm,style = 0):
        if style:
            for i in self.Lines_list:
                i.plot(tm,style)
        else:
            for i in self.Lines_list:
                i.plot(tm)
    
    def rotate(self,tm):
         self.st = np.matmul(self.st,tm)
         self.end = np.matmul(self.end,tm)
         for i in self.Lines_list:
            i.rotate(tm)

if __name__ == "__main__":


    space = Space((math.radians(45), math.radians(45), 0))
    space.add_object(Rectangle((5,5,5),(-5,-5,-5)))
    space.add_object(Line((-9,0,0),(9,0,0)))
    space.add_object(Line((0,-9,0),(0,9,0)))
    space.add_object(Line((0,-9,0),(0,9,0)))
    space.plot(1)


    for i in range(1800): 
        plt.pause(0.01)
        plt.clf()
        space.rotate(math.radians(5),'y',2)

        space.rotate(math.radians(5),'x',3)

        space.rotate(math.radians(5),'z',4)

        space.plot(1)
        plt.draw()
            
            





