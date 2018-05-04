# coding=utf-8
#Autor: Juan Pablo Cáceres Briones
#Tarea 1 CC3501

# Importación de librerías (Pablo Pizarro)

import numpy as np
import math
import matplotlib as mpl
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import matplotlib.colors as color
import sys
import os
import time

class Litoral:
    def __init__(self, ancho, largo, dh, rut, hora, tol):
        """
        Constructor
        :param ancho: Ancho
        :param largo: Largo
        :param dh: Tamaño grilla diferencial
        :param tol: Tolerancia
        :type int,float
        """
        self._ancho = ancho  # privada
        self._largo = largo
        self._dh = dh
        self._rut = rut
        self._hora = hora
        self._tol = tol

        #Numero de pixeles de alto y ancho
        self._m = int(float(ancho) / dh)
        self._n = int(float(largo) / dh)

        #Calculo de w (coeficiente de relajacion) optimo
        self._wden = 2 + math.sqrt(4 - math.pow(math.cos(math.pi / (self._n - 1))+ math.cos(math.pi / (self._m - 1)),2))
        self._wOp = 4.0 / self._wden

        #Matriz en donde se iterará la temperatura
        self._temperaturaLitoral = np.zeros((self._n, self._m))
        self._NewTemperaturaLitoral = np.zeros((self._n, self._m))

        #Matriz aux que ayuda a contruir las condiciones de borde (inspirado en la
        #minitarea uno subida)
        self._geografiaLitoral = np.zeros((self._n, self._m))


    #Definición de la Geografia del Litoral

    def createNaN(self):
            for y in range(0 , self._m):
                x = 0
                while x < self._n:
                    if self._geografiaLitoral[x,y]==1 or self._geografiaLitoral[x,y]==2 or \
                            self._geografiaLitoral[x,y]==3 or self._geografiaLitoral[x,y]==5:
                        break
                    else:
                        self._geografiaLitoral[x, y] = np.NaN
                        x = x + 1


    def geografiaLitoral(self):
        # superficie del mar en la matriz
        x1 = int(((1200 + 400 * float(self._rut) / 1000))/ self._dh)
        for i in range(0 , x1):
            self._geografiaLitoral[0 , i] = 1

        #Superficie de Chimeneas
        x2 = int(120 / self._dh) + x1
        for i in range(x1 , x2):
            self._geografiaLitoral[0 , i] = 2

        #Leve inclinación
        x3 = int(400 / self._dh) + x2
        m3 = 3
        for i in range (x2 , x3):
            j = int(i / m3 - x2 / m3)
            self._geografiaLitoral[j , i] = 3
        y3 = int(x3 / m3 - x2 / m3)

        #Inclinacion 1
        x4 = int(1200 / self._dh) + x1
        y4 = int((1500 + 200 * float(self._rut) / 1000)/ self._dh)
        m4 = float(y4 - y3)/float(x4 - x3)
        for i in range (x3, x4):
            j = int(i * m4 - x3 * m4) + y3
            self._geografiaLitoral[ j , i] = 3

        #Caida 1
        x5 = int(1500 / self._dh) + x1
        y5 = int((1300 + 200 * float(self._rut) / 1000)/ self._dh)
        m5 = float(y5 -y4) / float(x5 - x4)
        for i in range (x4, x5):
            j = int(i * m5 - x4 * m5) + y4
            self._geografiaLitoral[j , i] = 3

        #Inclinacion 2
        x6 = int(2000 / self._dh) + x1
        y6 = int((1850 + 100 * float(self._rut) / 1000)/ self._dh)
        m6 = float(y6 - y5) / float(x6 - x5)
        for i in range (x5, x6):
            j = int(i * m6 - x5 * m6) + y5
            #Nieve
            if j > 180:
                self._geografiaLitoral[j , i] = 5
            else:
                self._geografiaLitoral[j , i] = 3

        #Caida 2
        x7 = int(4000 / self._dh)
        y7 = int((1300 + 200 * float(self._rut) / 1000) / self._dh)
        m7 = float(y7 - y6) / float(x7 - x6)
        for i in range (x6, x7):
            j = int(i * m7 - x6 * m7) + y6
            #Nieve
            if j > 180:
                self._geografiaLitoral[j , i] = 5
            else:
                self._geografiaLitoral[j , i] = 3

        #Rellena con NaN
        self.createNaN()

    #Definicion de la Temperatura del Litoral

    #Se definen las condiciones de borde, ayudado por la matriz geografica Litoral
    def condicionesBorde(self):
       #Definicion de la temperaturas
       #Temperatura Mar
        if self._hora >= 0 and self._hora <= 8 :
            tMar = 4
        elif self._hora >= 8 and self._hora <= 16:
            tMar = 2 * self._hora - 12
        else:
            tMar = -2 * self._hora + 52

        tPlanta = 540 * (math.cos(math.pi * self._hora / 12) + 2)

        tSuelo = 20

        tNieve = 0

        for i in range(0, self._n):
            tAtmosfera = int(tMar - (6* self._dh)*(i)/1000.0)
            for j in range(0, self._m):
                if self._geografiaLitoral[i , j] == 1:
                    self._temperaturaLitoral[i , j] = tMar
                elif self._geografiaLitoral[i , j] == 2:
                    self._temperaturaLitoral[i , j] = tPlanta
                elif self._geografiaLitoral [i , j] == 3:
                    self._temperaturaLitoral[i , j] = tSuelo
                elif self._geografiaLitoral[i , j] == 5:
                    self._temperaturaLitoral[i , j] = 0
                elif self._geografiaLitoral[i, j] == 0:
                    self._temperaturaLitoral[i, j] = tAtmosfera
                elif np.isnan(self._geografiaLitoral[i,j]):
                    self._temperaturaLitoral[i,j] = np.NaN

    def nearNAN(self,a, i, j):
        if np.isnan(a[i - 1][j]): return True
        if np.isnan(a[i + 1][j]): return True
        if np.isnan(a[i][j - 1]): return True
        if np.isnan(a[i][j + 1]): return True
        return False

    def unicaIteracion(self, old, w, i, j):
        #Inspirado en el aux 3 y la tarea de Pablo Pizarro
        #Caso Intermedio
        if 0< i < self._n - 1 and 0 < j < self._m - 1:
            if self.nearNAN(old, i ,j):#Si es una condición de Borde Neumann
                #Condicion de Neumann igual a 0
                if np.isnan(self._temperaturaLitoral[i][j]):
                    return (old[i][j], self._tol)
                elif np.isnan(old[i][j + 1]):  # Si el lado derecho es NAN
                    if np.isnan(old[i - 1][j]):  # Si ademas abajo es NAN -> Esquina derecha
                        prom = old[i][j] + 0.25 * w * (2 * old[i + 1][j] + 2 * old[i][j - 1] - 4 * old[i][j])
                    else:  # Solo Borde derecho
                        prom = old[i][j] + 0.25 * w * (old[i + 1][j] + old[i - 1][j] + 2 * old[i][j - 1] - 4 * old[i][j])
                elif np.isnan(old[i - 1][j]):  # Si el lado de abajo es NAN
                    if np.isnan(old[i][j - 1]):  # Si el lado izquierdo es NAN -> Esquina izquierda
                        prom = old[i][j] + 0.25 * w * (2 * old[i + 1][j] + 2 * old[i][j + 1] - 4 * old[i][j])
                    else:  # Solo Borde inferior
                        prom = old[i][j] + 0.25 * w * (2 * old[i + 1][j] + old[i][j + 1] + old[i][j - 1] - 4 * old[i][j])
                elif np.isnan(old[i][j - 1]): #Si el lado izquierdo es NAN
                    prom = old[i][j] + 0.25 * w *(old[i + 1][j] + old[i + 1][j] + 2 * old[i][j + 1] - 4 * old[i][j])

            else:  # Punto general
                prom = old[i][j] + 0.25 * w * (old[i + 1][j] + old[i - 1][j] + old[i][j + 1] + old[i][j - 1] - 4 * old[i][j])
        else:
            # Esquinas
            if np.isnan(self._temperaturaLitoral[i][j]):
                return (old[i][j], self._tol)
            elif i == 0 and j == 0:  # esquina inferior izquierda
                prom = old[i][j]
            elif i == self._n-1 and j == 0:  # esquina superior izquierda
                prom = old[i][j] + 0.25 * w * (2 * old[i - 1][j] + 2 * old[i][j + 1] - 4 * old[i][j])
            elif i == self._n - 1 and j == self._m - 1:  # esquina inferior derecha
                prom = old[i][j] + 0.25 * w * (2 * old[i - 1][j] + 2 * old[i][j - 1] - 4 * old[i][j])
                # Bordes
            else:
                prom = old[i][j]  # Como son dirichlet no se hace nada
        e = abs(old[i, j] - prom)
        return (prom , e)


    def start(self, maxIteraciones):
        iteraciones = 0
        eMax = np.ones((self._n, self._m))
        t0 = time.time()  # tiempo inicial de ejecucion
        while iteraciones < maxIteraciones:
            for i in range(0, self._n):
                for j in range(0, self._m):
                    (self._temperaturaLitoral[i][j],eMax[i][j]) = self.unicaIteracion(self._temperaturaLitoral, self._wOp, i, j)
            e = np.max(eMax)
            if iteraciones == 0:
                e0=e
            print(e)
            if e <= self._tol:
                break
            iteraciones = 1 + iteraciones
        tf = int(time.time() - t0)
        return tf




    #Funciones que permiten Plotear las temperaturas
    def plotGeografia(self, **kwargs):
        #Funcion extraida de la tarea 1 de Pablo Pizarro
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.imshow(self._geografiaLitoral, interpolation='none', origin="lower")
        fig.colorbar(cax)
        cfg = pl.gcf()

        # Experimental: se modifican las etiquetas de los ejes
        xnum = len(ax.get_xticks()) - 2
        ynum = len(ax.get_yticks()) - 2
        xlabel = []
        ylabel = []
        for i in range(xnum): xlabel.append(str(int(float(2000) * i / (xnum))))
        for j in range(ynum): ylabel.append(str(int(float(4000) * j / (ynum))))
        if kwargs.get("xlabel"):
            ax.set_xticklabels([''] + xlabel)
            pl.ylabel("Ancho [m]")
        else:
            ax.set_xticklabels([''])
        if kwargs.get("ylabel"):
            ax.set_yticklabels([''] + ylabel)
            pl.ylabel("Altura [m]")
        else:
            ax.set_yticklabels([''])
        # Se establece el titulo de la ventana
        pl.title("Temperatura en t=" + str(int(self._hora)) + "\n")
        cfg.canvas.set_window_title("Temperatura en t=" + str(int(self._hora)))
        plt.show()

    def plotTemperaturaLog(self, **kwargs):
        #Funcion inspirada en la tarea 1 de Pablo Pizarro
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #Maximo valor de la matriz de temperatura
        Tmax = np.nanmax(self._temperaturaLitoral)
        bounds = np.hstack((np.array([-10, 0]), np.logspace(1, math.log(Tmax, 10), 50, endpoint=True, base=10)))
        norm = color.BoundaryNorm(boundaries=bounds, ncolors=256)
        cax = ax.imshow(self._temperaturaLitoral, interpolation='none', origin="lower", norm=norm)
        fig.colorbar(cax, norm=norm, boundaries=bounds)
        cfg = pl.gcf()

        # Si se define el color de fondo como negro
        if kwargs.get("black"):
            mpl.rcParams['axes.facecolor'] = '000000'

        # Experimental: se modifican las etiquetas de los ejes
        xnum = len(ax.get_xticks())-2
        ynum = len(ax.get_yticks())-2
        xlabel = []
        ylabel = []
        for i in range(xnum): xlabel.append(str(int(float(2000) * i / (xnum))))
        for j in range(ynum): ylabel.append(str(int(float(4000) * j / (ynum))))
        if kwargs.get("xlabel"):
            ax.set_xticklabels([''] + xlabel)
            pl.ylabel("Ancho [m]")
        else:
            ax.set_xticklabels([''])
        if kwargs.get("ylabel"):
            ax.set_yticklabels([''] + ylabel)
            pl.ylabel("Altura [m]")
        else:
            ax.set_yticklabels([''])
        # Se establece el titulo de la ventana
        pl.title("Temperatura en t=" + str(int(self._hora)) + "\n")
        cfg.canvas.set_window_title("Temperatura en t=" + str(int(self._hora)))
        plt.show()






#Instancia
r1 = Litoral(4000,2000,25,913,0,0.1)
r1.geografiaLitoral()
r1.condicionesBorde()
r1.start(1000)
r1.plotGeografia(black=True, ylabel=True)
r1.plotTemperaturaLog(black=True, ylabel=True)

r2 = Litoral(4000,2000,25,913,8,0.1)
r2.geografiaLitoral()
r2.condicionesBorde()
r2.start(1000)
r2.plotGeografia(black=True, ylabel=True)
r2.plotTemperaturaLog(black=True, ylabel=True)

r3 = Litoral(4000,2000,25,913,12,0.1)
r3.geografiaLitoral()
r3.condicionesBorde()
r3.start(1000)
r3.plotGeografia(black=True, ylabel=True)
r3.plotTemperaturaLog(black=True, ylabel=True)

r4 = Litoral(4000,2000,25,913,16,0.1)
r4.geografiaLitoral()
r4.condicionesBorde()
r4.start(1000)
r4.plotGeografia(black=True, ylabel=True)
r4.plotTemperaturaLog(black=True, ylabel=True)

r4 = Litoral(4000,2000,25,913,20,0.1)
r4.geografiaLitoral()
r4.condicionesBorde()
r4.start(1000)
r4.plotGeografia(black=True, ylabel=True)
r4.plotTemperaturaLog(black=True, ylabel=True)






























