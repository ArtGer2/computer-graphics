import matplotlib.pyplot as plt
import numpy as np
import math
import json
from matplotlib.colors import LinearSegmentedColormap
import pygame
from numba import cuda, float64

# Создание пользовательской цветовой карты серого
cmap_gray = LinearSegmentedColormap.from_list('gray', [(0, 'black'), (1, 'white')])


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


class Point:
    def __init__(self, start=[0 ,0 , 0],style =None):
        self.coords = []
        self.style = style
        if len(start) == 4:
            self.coords.append(np.array(start))
        else:
            self.coords.append(np.append(np.array(start),1))

    def plot(self,trim_matrix, style = None):
        if style != None:
            self.style = style
        result = np.matmul(self.coords,trim_matrix)
    def rotate(self,matrix):
        self.coords = np.matmul(self.coords,matrix)

class Line:
    def __init__(self, start=[0 ,0 , 0], end=[0, 0, 0],style =None):
        self.coords = []
        self.style = style
        if len(start) == 4:
            self.coords.append(np.array(start))
            self.coords.append(np.array(end))
        else:
            self.coords.append(np.append(np.array(start),1))
            self.coords.append(np.append(np.array(end),1))
    
    def plot(self,trim_matrix, style = None):
        if style != None:
            self.style = style
        result = np.matmul(self.coords,trim_matrix)
        if self.style != None:
            pygame.draw.line(screen, (255, 0, 0), result[0][:2]*20 +200, result[1][:2]*20 +200, 3)
        else:
            pygame.draw.line(screen, (255, 0, 0), result[0][:2]*20 +200, result[1][:2]*20 +200, 3)

    def rotate(self,matrix):
        self.coords = np.matmul(self.coords,matrix)


class Space:
    obj_list=[]
    camera_angle = []

    def __init__(self, angle = [0.0,0.0,0.0]):
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

        pitch, yaw,roll = self.camera_angle
        trim_projection_matrix = np.array([
            [math.cos(pitch), math.sin(pitch)*math.sin(yaw) ,0,0],
            [0,math.cos(yaw) ,0,0],
            [math.sin(pitch), -1 * math.cos(pitch)*math.sin(yaw) ,0,0],
            [0,0,0,1000],
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
                anArea.plot(trim_projection_matrix,'black')
             
        elif mode ==2:
            dots_plane = create_parallel_plane_points(trim_projection_matrix)
            screenQ = np.zeros((len(dots_plane), 3))
            z_buffer =np.zeros(len(dots_plane))
            screen.fill((0,0,0))

            for i in self.obj_list[3:]:
                screenQ,z_buffer = cudaX(dots_plane,i.st,i.end,z_buffer, trim_projection_matrix,screenQ)
            for scr in screenQ:
                pygame.draw.circle(screen, (scr[2],scr[2],scr[2]), (scr[0],scr[1]), 4)




@cuda.jit
def cudaX_kernel(dots_plane, obj_st, obj_end, z_buffer, trim_projection_matrix, screen):
    idx = cuda.grid(1)
    if idx < len(dots_plane):

        start, end = point_along_vector(dots_plane[idx], (dots_plane[0],dots_plane[1]), (dots_plane[0],dots_plane[-1]), 20)

        status, intersection = line_cube_intersection(start, end, obj_st, obj_end)

        if status:

            color = (math.sqrt((intersection[0] - end[0])**2 + (intersection[1] - end[1])**2 + (intersection[2] - end[2])**2  ))**8/30000000000000
            if color > z_buffer[idx]:
                
                z_buffer[idx] = color
                
                result = cuda.local.array(4, dtype=float64)
                result[0] = intersection[0]
                result[1] = intersection[1]
                result[2] = intersection[2]
                result[3] = 1.0

                projected = cuda.local.array(4, dtype=float64)
                for i in range(4):
                    sum = 0
                    for j in range(4):
                        sum += result[j] * trim_projection_matrix[j, i]
                    projected[i] = sum

                screen[idx, 0] = projected[0] * 20 + 200  # X coordinate
                screen[idx, 1] = projected[1] * 20 + 200  # Y coordinate
                screen[idx, 2] = color #color

                

def cudaX(dots_plane, obj_st, obj_end, z_buffer, trim_projection_matrix,screen):

    threads_per_block = 256
    blocks_per_grid = (len(dots_plane) + threads_per_block - 1) // threads_per_block

    dots_plane_D = cuda.to_device(dots_plane)
    obj_st_D = cuda.to_device(obj_st)
    obj_end_D = cuda.to_device(obj_end)
    z_buffer_D = cuda.to_device(z_buffer)
    trim_projection_matrix_D = cuda.to_device(trim_projection_matrix)
    screen_D = cuda.to_device(screen)

    cudaX_kernel[blocks_per_grid, threads_per_block](dots_plane_D, obj_st_D, obj_end_D, z_buffer_D, trim_projection_matrix_D, screen_D)

    screen= screen_D.copy_to_host()

    z_buffer = z_buffer_D.copy_to_host()

    return screen,z_buffer

@cuda.jit(device=True)
def point_along_vector(start_point, segment1, segment2, distance):
    vec1_x = segment1[1][0] - segment1[0][0]
    vec1_y = segment1[1][1] - segment1[0][1]
    vec1_z = segment1[1][2] - segment1[0][2]

    vec2_x = segment2[1][0] - segment2[0][0]
    vec2_y = segment2[1][1] - segment2[0][1]
    vec2_z = segment2[1][2] - segment2[0][2]

    normal_x = vec1_y * vec2_z - vec1_z * vec2_y
    normal_y = vec1_z * vec2_x - vec1_x * vec2_z
    normal_z = vec1_x * vec2_y - vec1_y * vec2_x

    start_x = start_point[0] + normal_x * distance
    start_y = start_point[1] + normal_y * distance
    start_z = start_point[2] + normal_z * distance

    end_x = start_point[0] - normal_x * distance
    end_y = start_point[1] - normal_y * distance
    end_z = start_point[2] - normal_z * distance

    return (start_x, start_y, start_z), (end_x, end_y, end_z)

@cuda.jit(device=True)
def line_cube_intersection(start, end, min_corner, max_corner):
    # Параметры прямой
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dz = end[2] - start[2]

    # Пересечение с левой плоскостью куба
    tmin = (min_corner[0] - start[0]) / dx
    tmax = (max_corner[0] - start[0]) / dx

    # Пересечение с нижней плоскостью куба
    tymin = (min_corner[1] - start[1]) / dy
    tymax = (max_corner[1] - start[1]) / dy

    # Пересечение с передней плоскостью куба
    tzmin = (min_corner[2] - start[2]) / dz
    tzmax = (max_corner[2] - start[2]) / dz

    # Находим интервал пересечения плоскостей
    t0 = max(min(tmin, tmax), min(tymin, tymax), min(tzmin, tzmax))
    t1 = min(max(tmin, tmax), max(tymin, tymax), max(tzmin, tzmax))

    # Если интервал существует, вычисляем точку пересечения
    if t0 <= t1:
        intersection_point_x = start[0] + t0 * dx
        intersection_point_y = start[1] + t0 * dy
        intersection_point_z = start[2] + t0 * dz
        status = True 
        return status, (intersection_point_x, intersection_point_y, intersection_point_z)
    
    

def create_parallel_plane_points( trim_projection_matrix):
    points = []
    inv_mat = np.linalg.pinv(trim_projection_matrix)
    
    for i in range(-50,50):
        for j in range(-50,50):

            p3d = np.matmul((i/5,j/5,0,1), inv_mat)
            points.append(p3d[:3])
    return points

  

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
    pygame.init()
    screen = pygame.display.set_mode((400,400))
    
    space = Space([math.radians(45), math.radians(45), 0])
    space.add_object(Rectangle((4,4,4),(-4,-4,-4)))
    space.add_object(Rectangle((6,3,3),(-3,-3,-3)))
    space.add_object(Rectangle((0,0,0),(-6,-6,-6)))


    for i in range(360): 

        space.camera_angle[0] +=math.radians(5)
        space.camera_angle[1] +=math.radians(5)
        space.plot(2)
        pygame.display.flip()

    pygame.quit()




