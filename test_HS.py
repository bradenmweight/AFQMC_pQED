import numpy as np

Nx = 10000
x = np.linspace(-10,10,Nx) #np.random.normal( 0.0, 1.0, size=Nx )
dx = x[1] - x[0]

Y  = 1
dt = 0.02
L = np.exp( -dt * Y**2 )
R = np.sqrt( 1/2/np.pi ) * np.sum( np.exp(-x**2 / 2) * np.exp(-1j * np.sqrt(2) * np.sqrt(dt) * x * Y) ) * dx
print( L )
print( R.real )