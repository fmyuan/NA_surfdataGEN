import netCDF4 as nc
from time import process_time
import matplotlib.pyplot as plt
import numpy as np

# Open the source file
src = nc.Dataset('hr_surfdata_test1.nc', 'r')

for name, variable in src.variables.items():
    start = process_time()
    print("Working on variable: "+ name + " dimensions: " + str(variable.dimensions))
    
    # Check if the last two dimensions are lsmlat and lsmlon
    if (variable.dimensions[-2:] == ('y_dim', 'x_dim')) :

        # Handle variables with two dimensions
        if (len(variable.dimensions) == 2):
            source = src[name][:]
            plt.imshow(source[::5,::5])
            plt.title('2D Imshow')
            plt.colorbar(label=name)
            plt.savefig(name + '.png')
            print("Saving variable: ", name)

            # 2D plots
            x_values = [3000, 5000, 7000]
            y_values = [100, 2100, 4100, 6100]

            # Create a figure for x-axis plots
            plt.figure()
            for x in x_values:
                plt.plot(source[:, x], label=f'x={x}')
            plt.title('Line plots over x-axis')
            plt.legend()
            plt.savefig(f'{name}_x_plots.png')

            # Create a figure for y-axis plots
            plt.figure()
            for y in y_values:
                plt.plot(source[y, :], label=f'y={y}')
            plt.title('Line plots over y-axis')
            plt.legend()
            plt.savefig(f'{name}_y_plots.png')

        # Handle variables with three dimensions
        if (len(variable.dimensions) == 3):
            for index in range(variable.shape[0]):
                source = src[name][index,:,:]
                plt.imshow(source[::5,::5])
                plt.title('2D Imshow')
                plt.colorbar(label=name)
                plt.savefig(name + '_' + str(index) + '.png')
                print("Saving variable: ", name)

                # 2D plots
                x_values = [3000, 5000, 7000]
                y_values = [100, 2100, 4100, 6100]

                # Create a figure for x-axis plots
                plt.figure()
                for x in x_values:
                    plt.plot(source[:, x], label=f'x={x}')
                plt.title('Line plots over x-axis')
                plt.legend()
                plt.savefig(f'{name}_{index}_x_plots.png')

                # Create a figure for y-axis plots
                plt.figure()
                for y in y_values:
                   plt.plot(source[y, :], label=f'y={y}')
                plt.title('Line plots over y-axis')
                plt.legend()
                plt.savefig(f'{name}_{index}_y_plots.png')


        # Handle variables with four dimensions
        if (len(variable.dimensions) == 4):
            for index1 in range(variable.shape[0]):
                for index2 in range(variable.shape[1]):
                    source = src[name][index1, index2,:, :]
                    plt.imshow(source[::5,::5])
                    plt.title('2D Imshow')
                    plt.colorbar(label=name)
                    plt.savefig(name + '_' + str(index1) + '_' + str(index2) + '.png') 
                    print("Saving variable: ", name)

                    # 2D plots
                    x_values = [3000, 5000, 7000]
                    y_values = [100, 2100, 4100, 6100]

                    # Create a figure for x-axis plots
                    plt.figure()
                    for x in x_values:
                       plt.plot(source[:, x], label=f'x={x}')
                    plt.title('Line plots over x-axis')
                    plt.legend()
                    plt.savefig(f'{name}_{index1}_{index2}_x_plots.png')

                    # Create a figure for y-axis plots
                    plt.figure()
                    for y in y_values:
                        plt.plot(source[y, :], label=f'y={y}')
                    plt.title('Line plots over y-axis')
                    plt.legend()
                    plt.savefig(f'{name}_{index1}_{index2}_y_plots.png')
                    
        end = process_time()
        print("Generating variable: " +name+ " takes  {}".format(end-start))

# Close the files
src.close()