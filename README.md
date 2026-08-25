# Numerical simulation of Argon

This program is a numerical simulation of argon atoms, using periodic boundary conditions and the minimal image convention.

The physics of this simulation make use of the Lennard-Jones potential and the Velocity Verlet algorithm. 

To initiate the system, we use a FCC cubic lattice and a Maxwell-Boltzmann distribution.

In order to reach an equilibrated state we use a velocity relaxation technique.

See report for theory / results.

## Getting started

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

#### Dependencies

This is a python program and thus requires python in order to run.

This program makes use of the following python packages:

* numpy
* matplotlib.pyplot
* argparse

These must be installed before the program can successfully run.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

#### Executing Program

The program can ran without adding any parameters. It will then perform a simulation with the default settings (see below). 

The simulation will print the calculated value of the pressure of the system and will plot the pair correlation function and energy values of the system.



###### Default values:

* density = 0.3
* temperature = 3.0
* cells\_per\_axis = 3
* timestep = 1e-2
* simulation\_time = 1500
* velocity\_relaxation\_cutoff\_time = 500
* velocity\_relaxation\_frequency = 50
* sample\_frequency = 50
* correlation\_function\_bins = 100
* time\_slice\_start = velocity\_relaxation\_cutoff\_time
* time\_slice\_end = simulation\_time



Any of the values can be adjusted by either opening the code in an IDE and changing the values manually, which can be found at the start of the code. 

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

#### Running the program in a terminal

You can also change the settings of the simulation by giving arguments when running the program in a terminal. This will not adjust any values permanently.

A quick guide to running python scripts in a terminal:

https://realpython.com/run-python-scripts/



The passing of arguments is done using the argparse formatting: 

adding -VARIABLE\_NAME followed by the VALUE. This will look something like:



C:\\Users\\username>python PATH/cpa\_simulation.py -VARIABLE\_NAME VALUE



An example: *C:\\Users\\username>python PATH/cpa\_simulation.py -density 1.0 -simulation\_time = 2000 -no\_correlation*



You can also disable the printing of the pressure value or the plotting of the correlation function or energies by passing:

&#x20;-no\_pressure, -no\_correlation or -no\_energy\_plots respectively.



To get an overview of all possibilities and the meaning of the variables, use:



C:\\Users\\username>python PATH/cpa\_simulation.py -h



Here "-h" is the help function of argparse.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

##### Writing data to a file



Using a terminal (recommended):

You can save all system related data as a npz file by adding -save\_data.

Using an IDE:

By enabling the code under 'export data' (from line 459), args.save\_data is set to False by default.



This can then be read using the numpy.load function

A guide on this can be found here:

https://numpy.org/doc/stable/reference/generated/numpy.load.html

















