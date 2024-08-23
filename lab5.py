import matplotlib.pyplot as plt
import numpy as np
import math
import json
from matplotlib.colors import LinearSegmentedColormap

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
        if self.style != None:
            plt.plot(result.T[0],result.T[1],self.style)
        else:
            plt.plot(result.T[0],result.T[1])
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
            plt.plot(result.T[0],result.T[1],self.style)
        else:
            plt.plot(result.T[0],result.T[1])

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

        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        pitch, yaw,roll = self.camera_angle
        trim_projection_matrix = np.array([
            [math.cos(pitch), math.sin(pitch)*math.sin(yaw) ,0,0],
            [0,math.cos(yaw) ,0,0],
            [math.sin(pitch), -1 * math.cos(pitch)*math.sin(yaw) ,0,0],
            [0,0,0,1],
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
             
        elif mode ==2:
            anti_proj_matrix = np.array([
            [math.cos(-pitch), math.sin(-pitch)*math.sin(yaw) ,0,0],
            [0,math.cos(yaw) ,0,0],
            [math.sin(-pitch), -1 * math.cos(-pitch)*math.sin(yaw) ,0,0],
            [0,0,0,1],])

            #for i in self.obj_list:
            #    i.plot(trim_projection_matrix,"black")
            dots_plane = create_parallel_plane_points(trim_projection_matrix,10,10,10)

            for i in dots_plane:
                start,end = point_along_vector(i,(dots_plane[0],dots_plane[1]),(dots_plane[0],dots_plane[50]),20)

                if line_cube_intersection(start,end,(5,5,5),(-5,-5,-5)) != None:
                    result = np.append(line_cube_intersection(start,end,(5,5,5),(-5,-5,-5)),1)

                    color = distance_between_points(end,result)/20
                    
                    result = np.matmul(result, trim_projection_matrix)

                    plt.scatter(result.T[0],result.T[1],c=(color,color,color),cmap="viridis")
                else:
                    result = np.matmul(np.append(i,1), trim_projection_matrix)
                    plt.scatter(result.T[0],result.T[1],c=(0,0,0),cmap="viridis")

            



def distance_between_points(point1, point2):
    # Извлекаем координаты точек
    x1, y1, z1 = point1
    x2, y2, z2,a = point2
    
    # Вычисляем расстояние
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    
    return distance

def ray_cube_intersection(a, b, cube_min, cube_max):
    # Пересечение луча с кубом по x
    tmin_x = (cube_min[0] - a[0]) / (b[0] - a[0])
    tmax_x = (cube_max[0] - a[0]) / (b[0] - a[0])

    if tmin_x > tmax_x:
        tmin_x, tmax_x = tmax_x, tmin_x

    # Пересечение луча с кубом по y
    tmin_y = (cube_min[1] - a[1]) / (b[1] - a[1])
    tmax_y = (cube_max[1] - a[1]) / (b[1] - a[1])

    if tmin_y > tmax_y:
        tmin_y, tmax_y = tmax_y, tmin_y

    if tmin_x > tmax_y or tmin_y > tmax_x:
        return None

    # Обновляем tmin и tmax
    tmin = max(tmin_x, tmin_y)
    tmax = min(tmax_x, tmax_y)

    # Пересечение луча с кубом по z
    tmin_z = (cube_min[2] - a[2]) / (b[2] - a[2])
    tmax_z = (cube_max[2] - a[2]) / (b[2] - a[2])

    if tmin_z > tmax_z:
        tmin_z, tmax_z = tmax_z, tmin_z

    if tmin > tmax_z or tmin_z > tmax:
        return None

    # Обновляем tmin и tmax
    tmin = max(tmin, tmin_z)
    tmax = min(tmax, tmax_z)

    # Проверяем, что пересечение находится на луче
    if tmin < 0:
        tmin = tmax
        if tmin < 0:
            return None

    # Координаты точки пересечения
    intersection_point = (a[0] + tmin * (b[0] - a[0]),
                          a[1] + tmin * (b[1] - a[1]),
                          a[2] + tmin * (b[2] - a[2]))

    return intersection_point
def spherical_to_cartesian(alpha, beta, gamma):
    # Преобразование сферических координат в декартовы
    x = np.sin(alpha) * np.cos(beta)
    y = np.sin(alpha) * np.sin(beta)
    z = np.cos(alpha)
    return np.array([x, y, z])


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
        intersection_point = (start[0] + t0*dx, start[1] + t0*dy, start[2] + t0*dz)
        return intersection_point
    else:
        return None
    




    
def rotation_matrix_from_angles(alpha, beta, gamma):
    """
    Функция для создания матрицы поворота по трем углам alpha, beta и gamma.
    """
    # Преобразуем углы в радианы
    alpha_rad = math.radians(alpha)
    beta_rad = math.radians(beta)
    gamma_rad = math.radians(gamma)

    # Создаем матрицы поворота вокруг осей X, Y и Z
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(alpha_rad), -math.sin(alpha_rad)],
                   [0, math.sin(alpha_rad), math.cos(alpha_rad)]])

    Ry = np.array([[math.cos(beta_rad), 0, math.sin(beta_rad)],
                   [0, 1, 0],
                   [-math.sin(beta_rad), 0, math.cos(beta_rad)]])

    Rz = np.array([[math.cos(gamma_rad), -math.sin(gamma_rad), 0],
                   [math.sin(gamma_rad), math.cos(gamma_rad), 0],
                   [0, 0, 1]])

    # Объединяем матрицы поворота
    rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))

    return rotation_matrix
def create_3d_line(point, angles, distance):


    # Переводим точку в массив numpy
    point = np.array(point)
    
    # Преобразуем углы в радианы
    angles_rad = np.radians(angles)
    
    # Создаем матрицы поворота вокруг каждой из осей
    rotation_x = np.array([[1, 0, 0],
                            [0, np.cos(angles_rad[0]), -np.sin(angles_rad[0])],
                            [0, np.sin(angles_rad[0]), np.cos(angles_rad[0])]])
    
    rotation_y = np.array([[np.cos(angles_rad[1]), 0, np.sin(angles_rad[1])],
                            [0, 1, 0],
                            [-np.sin(angles_rad[1]), 0, np.cos(angles_rad[1])]])
    
    rotation_z = np.array([[np.cos(angles_rad[2]), -np.sin(angles_rad[2]), 0],
                            [np.sin(angles_rad[2]), np.cos(angles_rad[2]), 0],
                            [0, 0, 1]])
    
    # Вычисляем общую матрицу поворота
    rotation_matrix = np.matmul(rotation_z, np.matmul(rotation_y, rotation_x))
    
    # Вычисляем направление вектора после поворота
    rotated_direction = np.dot(rotation_matrix, [0, 0, 1])  # Вектор (0, 0, 1) повернутый на заданные углы
    
    # Вычисляем точки в сторону и в обратную сторону от вектора
    point_forward = point + rotated_direction * distance
    point_backward = point - rotated_direction * distance
    return np.append(point_forward,1), np.append(point_backward,1)


def point_along_vector(start_point, segment1,segment2, distance):

    vec1 = segment1[1] - segment1[0]
    vec2 = segment2[1] - segment2[0]
    normal = np.cross(vec1, vec2)

    start = start_point + normal * distance
    end = start_point - normal * distance
    return start,end

def inverse_project_to_3d(point_2d, projection_matrix):




    inverse_projection_matrix = np.linalg.inv(projection_matrix)

    # Преобразуем введенную 2D-точку в гомогенные координаты
    point_homogeneous = np.array([point_2d[0], point_2d[1], 0, 1])

    # Умножаем на обратную матрицу проекции
    point_3d_homogeneous = np.dot(inverse_projection_matrix, point_homogeneous)

    # Нормализуем координаты, чтобы получить 3D-точку
    point_3d = point_3d_homogeneous[:3] / point_3d_homogeneous[3]

    return point_3d                    





def create_parallel_plane_points( trim_projection_matrix,dx,dy,dz):
    points = []
    inv_mat = np.linalg.pinv(trim_projection_matrix)
    
    for i in range(-18,20):
        for j in range(-18,20):
            p3d = np.matmul((i/2,j/2,0,1), inv_mat)
            
            points.append(p3d[:3])
    return points


"""
            z_buffer = np.zeros((800, 600)) + np.inf  # Инициализация Z-буфера
            frame_buffer = np.zeros((800, 600, 3), dtype=np.uint8)  # Инициализация кадра



            x0, y0 = int(beg[0] * 100 + 400), int(beg[1] * 100 + 300)
            x1, y1 = int(end[0] * 100 + 400), int(end[1] * 100 + 300)

            dz = end[2] - beg[2]


            if dz != 0:
                dz_dx = dz / (x1 - x0)
            else:
                dz_dx = 0
            for x in range(x0, x1 + 1):
                if y >= 0 and y < 800 and x >= 0 and x < 600 and z < z_buffer[y, x]:
                    z_buffer[y, x] = z
                    frame_buffer[y, x] = (255, 255, 255)
                z += dz_dx          """  
            

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
    
    space = Space([math.radians(45), math.radians(45), 0])

    
    space.add_object(Rectangle((5,5,5),(-5,-5,-5)))
    plt.show()

    for i in range(72): 
        plt.pause(0.011)
        plt.clf()
        space.camera_angle[0] +=math.radians(5)

        space.plot()
        plt.draw()






