import numpy as np
import numpy.linalg as la
import numpy.random as rnd
import matplotlib.pyplot as plt
import timeit
rnd.seed(2023)
n= 200
m= 150 

D = np.diag(1+100*rnd.rand(n))
Q = D

Qr,R = la.qr(rnd.rand(n,n))
A = (Qr.T@D@Qr)[:,:m]
b = rnd.rand(n)

def f1(x):
  return 0.5*la.norm(A@x-b)**2

def grad_f1(x):
  trans = A.T;
  return trans@((A@x)-b);

xstar_f1 = la.inv(A.T@A)@A.T@b
def rosenbrock(x, b=2):
    x1, x2 = x
    return (1 - x1)**2 + b * (x2 - x1**2)**2

def grad_rosenbrock(x, b=2):
    x1, x2 = x
    return np.array([
        2 * (x1 - 1) - 4 * b * x1 * (x2 - x1**2),
        2 * b * (x2 - x1**2)
    ])

xstar_rosenbrock = np.array([1,1])    
# Parte 2.a)
def gradient_descent(grad, x_init, xstar, alpha, tol=1e-5): 

    #Inicializo la cantidad de iteraciones en 0 y el resto de estructuras.
    it = 1
    xs = []
    es = []
    xs.append(x_init)

    x_k = x_init

    while (la.norm(grad(x_k)) >= tol):
      #Elijo la dirección d_k como d_k=-D_k.gradientef(x_k) tal que D_k es la identidad.
      d_k = (-1)*grad(x_k)
      #El paso alpha es fijo.
      #x_k+1 = x_k + alpha.d_k
      x_k = x_k + alpha*d_k
      #Adhiero a la lista de trayectorias la nueva trayectoria
      xs.append(x_k)
      #Adhiero a la lista de distancias al optimo la distancia del x_k+1 calculado
      es.append(la.norm(x_k - xstar))
      #Incremento la cantidad de iteraciones
      it = it+1
    
    x = x_k
    return x, it, np.array(xs), np.array(es)
    # Parte 2.b)
def nesterov_gradient_descent(grad, x_init, xstar, alpha, tol=1e-5):

    it = 1
    x_k = x_init
    y_k = x_k
    xs = []
    es = []
    xs.append(x_init)

      #Utilizaré el número de iteraciones como el k
    while (la.norm(grad(x_k)) >= tol):
      #x_ksig es x_k+1. No preciso y_ksig pues no utilizo el valor de y_k, simplemente actualizo la variable.
      x_ksig = y_k - alpha*grad(y_k)
      y_k = x_ksig + (it/(it+3))*(x_ksig - x_k)
      #Actualizo valor de x_k
      x_k = x_ksig
      xs.append(x_k)
      es.append(la.norm(x_k - xstar))
      it=it+1

    x = x_k
    return x, it, np.array(xs), np.array(es)
    #parte 2.c). 
#Rosenbrock con gradient descent.
x_init = np.array([1.5,2])

start = timeit.default_timer() #Inicia el timer
x_gd, it_gd, xs_gd, e_gd = gradient_descent(grad_rosenbrock, x_init, xstar_rosenbrock, alpha=0.01)
stop = timeit.default_timer() #Finaliza el timer

print('Gradient Descent')
print('x =', x_gd)
print('f(x) =', rosenbrock(x_gd))
print('||grad f(x)|| =', la.norm(grad_rosenbrock(x_gd)))
print('Iteraciones =', it_gd)
#Tiempo:
print('Tiempo Rosenbrock con gradient descent: ', stop - start) 
#Grafico:
#Siendo e_gd = ||x_k - x*|| calculado en función de cada iteración, tenemos que el gráfico de la función de error ∥x_k − x∗∥ en función de las iteraciones sería:
plt.plot(e_gd)
plt.xlabel('#Iteraciones')
plt.ylabel('||x_k - x*||')
plt.yscale('log')

#Rosenbrock con nesterov gradient descent.
x_init = np.array([1.5,2])

start = timeit.default_timer() #Inicia el timer 
x_gdN, it_gdN, xs_gdN, e_gdN = nesterov_gradient_descent(grad_rosenbrock, x_init, xstar_rosenbrock, alpha=0.01)
stop = timeit.default_timer() #Finaliza el timer

print('Nesterov Gradient Descent')
print('x =', x_gdN)
print('f(x) =', rosenbrock(x_gdN))
print('||grad f(x)|| =', la.norm(grad_rosenbrock(x_gdN)))
print('Iteraciones =', it_gdN)
#Tiempo:
print('Tiempo Rosenbrock con nesterov gradient descent: ', stop - start) 
#Grafico:
#Siendo e_gd = ||x_k - x*|| calculado en función de cada iteración, tenemos que el gráfico de la función de error ∥x_k − x∗∥ en función de las iteraciones sería:
plt.plot(e_gdN)
plt.xlabel('#Iteraciones')
plt.ylabel('||x_k - x*||')
plt.yscale('log')
#Último punto del 2.c
#Tomado del notebook de "gradient_descent_en_otras_funciones"
x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x, y)
Z = rosenbrock((X, Y))

plt.contour(X, Y, Z, levels=100, cmap='jet')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Curvas de nivel de Rosenbrock con trayectorias de ambos métodos')
#Se observa que con pasos más chicos se nota menos la diferencia de trayectorias de ambos métodos. Para este alpha casi no se diferencian.
plt.scatter(xs_gd[:,0],xs_gd[:,1], label='Gradient descent') #Ploteo los iterados generados por gradient_descent 
plt.scatter(xs_gdN[:,0],xs_gdN[:,1], label='Nesterov gradient descent') #Ploteo los iterados generados por Nesterov_gradient_descent

plt.show()
# Para la función ||Ax-b||^2, comenzar desde un punto aleatorio
x_init = rnd.rand(m)
start = timeit.default_timer() #Inicia el timer
x_gd, it_gd, xs_gd, e_gd = gradient_descent(grad_f1, x_init, xstar_f1, alpha=0.00001)
stop = timeit.default_timer() #Finaliza el timer

print('Gradient Descent')
print('x =', x_gd)
print('f(x) =', f1(x_gd))
print('||grad f(x)|| =', la.norm(grad_f1(x_gd)))
print('Iteraciones =', it_gd)
#Tiempo:
print('Tiempo f(x) con gradient descent: ', stop - start) 
#Grafico:
#Siendo e_gd = ||x_k - x*|| calculado en función de cada iteración, tenemos que el gráfico de la función de error ∥x_k − x∗∥ en función de las iteraciones sería:
plt.plot(e_gd)
plt.xlabel('#Iteraciones')
plt.ylabel('||x_k - x*||')
plt.yscale('log')
# Para la función ||Ax-b||^2, comenzar desde un punto aleatorio
x_init = rnd.rand(m)
start = timeit.default_timer() #Inicia el timer
x_gdN, it_gdN, xs_gdN, e_gdN = nesterov_gradient_descent(grad_f1, x_init, xstar_f1, alpha=0.00001)
#Se observa que para alpha del órden 0.001 se presentan errores varios, pero tomando alphas más pequeños funciona correctamente
stop = timeit.default_timer() #Finaliza el timer

print('Gradient Descent')
print('x =', x_gdN)
print('f(x) =', f1(x_gdN))
print('||grad f(x)|| =', la.norm(grad_f1(x_gdN)))
print('Iteraciones =', it_gdN)
#Tiempo:
print('Tiempo f(x) con gradient descent: ', stop - start) 
#Grafico:
#Siendo e_gd = ||x_k - x*|| calculado en función de cada iteración, tenemos que el gráfico de la función de error ∥x_k − x∗∥ en función de las iteraciones sería:
plt.plot(e_gdN)
plt.xlabel('#Iteraciones')
plt.ylabel('||x_k - x*||')
plt.yscale('log')

# Parte 2.d)
def gradient_descent_paso_decreciente(grad, x_init, xstar, tol=1e-5):
    it = 1
    xs = []
    es = []
    xs.append(x_init)

    x_k = x_init
    while (la.norm(grad(x_k)) >= tol):
      #Elijo la dirección d_k como d_k=-D_k.gradientef(x_k) tal que D_k es la identidad.
      d_k = (-1)*grad(x_k)
      #Defino el paso alpha:
      alpha = (0.1)/((it)**(1/92)) # observo que para los alpha = (0.1)/(it+1)^(1/i) con i>5 se reducen progresivamente a mayor i el número de iteraciones pero se queda por debajo del óptimo (1,1) por muy poco
      #x_k+1 = x_k + alpha.d_k
      x_k = x_k + alpha*d_k
      #Adhiero a la lista de trayectorias la nueva trayectoria
      xs.append(x_k)
      #Adhiero a la lista de distancias al optimo la distancia del x_k+1 calculado
      es.append(la.norm(x_k - xstar))
      #Incremento la cantidad de iteraciones
      it = it+1
    
    x = x_k
    return x, it, np.array(xs), np.array(es)

#Rosenbrock con gradient descent y paso decreciente.
x_init = np.array([1.5,2])

start = timeit.default_timer() #Inicia el timer
x_gd, it_gd, xs_gd, e_gd = gradient_descent_paso_decreciente(grad_rosenbrock, x_init, xstar_rosenbrock)
stop = timeit.default_timer() #Finaliza el timer

print('Gradient Descent')
print('x =', x_gd)
print('f(x) =', rosenbrock(x_gd))
print('||grad f(x)|| =', la.norm(grad_rosenbrock(x_gd)))
print('Iteraciones =', it_gd)
#Tiempo:
print('Tiempo Rosenbrock con gradient descent: ', stop - start) 
#Grafico:
#Siendo e_gd = ||x_k - x*|| calculado en función de cada iteración, tenemos que el gráfico de la función de error ∥x_k − x∗∥ en función de las iteraciones sería:
plt.plot(e_gd)
plt.xlabel('#Iteraciones')
plt.ylabel('||x_k - x*||')
plt.yscale('log')
# Parte 2.e)
def rosenbrock_diag(x, a=1, b=2):
    # La matriz diagonal D es formada con  d^2/dx^2(rosenbrock) y d^2/dy^2(rosenbrock) en D[0][0] y D[1][1] respectivamente
    xx_rosenbrock = 12*b*x[0]**2 - 4*b*x[1] + 2 #Derivada segunda de rosenbrock respecto a x
    yy_rosenbrock = 2*b #Derivada segunda de rosenbrock respecto a y 
    mat_Diag_Scal = np.diag((1/xx_rosenbrock, 1/yy_rosenbrock))
    
    return mat_Diag_Scal

def gradient_descent_diagonal_scaling(grad, diag_scaling, x_init, xstar, alpha, tol=1e-5):
  
    xs = []
    es = []
    xs.append(x_init)
    it = 1
    x_k = x_init

    while (la.norm(grad(x_k)) >= tol):
      #Elijo la dirección d_k como d_k=-D_k.gradientef(x_k) tal que D_k es definido por el diagonal scaling.
      d_k = ((-1)*grad(x_k))@diag_scaling(x_k)
      #El paso alpha es fijo.
      #x_k+1 = x_k + alpha.d_k
      x_k = x_k + alpha*d_k
      #Adhiero a la lista de trayectorias la nueva trayectoria
      xs.append(x_k)
      #Adhiero a la lista de distancias al optimo la distancia del x_k+1 calculado
      es.append(la.norm(x_k - xstar))
      #Incremento la cantidad de iteraciones
      it = it+1
    
    x = x_k
    return x, it, np.array(xs), np.array(es)

#Chequeo que los valores tengan sentido: 

x_init = np.array([1.5,2])

start = timeit.default_timer() #Inicia el timer
x_gdDS, it_gdDS, xs_gdDS, e_gdDS = gradient_descent_diagonal_scaling(grad_rosenbrock, rosenbrock_diag, x_init, xstar_rosenbrock, alpha=1)
stop = timeit.default_timer() #Finaliza el timer

print('Gradient Descent implementado con diagonal scaling')
print('x =', x_gdDS)
print('f(x) =', rosenbrock(x_gdDS))
print('||grad f(x)|| =', la.norm(grad_rosenbrock(x_gdDS)))
print('Iteraciones =', it_gdDS)
#Tiempo:
print('Tiempo Rosenbrock con gradient descent implementando diagonal scaling: ', stop - start) 
plt.plot(e_gdDS)
plt.xlabel('#Iteraciones')
plt.ylabel('||x_k - x*||')
plt.yscale('log')
