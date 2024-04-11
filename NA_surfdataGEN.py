import sys
import netCDF4 as nc
from scipy.interpolate import griddata
import numpy as np
import pandas as pd 
from time import process_time
#from memory_profiler import profile
from pyproj import Proj
from pyproj import Transformer
from pyproj import CRS


def main():
    args = sys.argv[1:]
    input_path = args[0]
    if not input_path.endswith("/"): input_path=input_path+'/'
    output_path = args[1]
    if not output_path.endswith("/"): output_path=output_path+'/'
    user_option = args[2]
    user_option2= args[3]

    if  len(sys.argv) != 3 or sys.argv[1] == '--help':  # sys.argv includes the script name as the first argument
        print("Example use: python NA_surfdataGEN.py <option>")
        print(" <input_path>: path to the 1D source data directory")
        print(" <1>:  first part of the surfdata")
        print(" <2>:  second part of the surfdata)")
        print(" <all>:  complete list of the surfdata)")
        print(" The code generate 2D NA surfdata from 0.5x0.5 degree globla surfdata")              
        exit(0)

    UNSTRUCTURED=True   # 1D
    if (not user_option2 ==''): UNSTRUCTURED=False  # 2D

    area_km2 = 1.0   # DAYMET grid cell is 1km2 
    
    # Only wariables listed will be processed

    # nearest neighbor:"double" variables
    Variable_nearest = ['SLOPE', 'TOPO', 'PCT_GLACIER', 'PCT_LAKE', 'STD_ELEV']
    # nearest neighbor:"int" variables
    Variable_nearest += ['PFTDATA_MASK','SOIL_COLOR', 'SOIL_ORDER', 'abm']
    # nearest neighbor:"double" variables (added 11/07/22023)
    Variable_nearest += ['EF1_BTR', 'EF1_CRP', 'EF1_FDT', 'EF1_FET', 'EF1_GRS', 'EF1_SHR']

    # nearest neighbor: "double" variables (added 11/10/2023)
    #Variable_nearest += ['PCT_SAND', 'PCT_CLAY','ORGANIC' ,'PCT_NAT_PFT', 
    #        'MONTHLY_LAI', 'MONTHLY_SAI' ,'MONTHLY_HEIGHT_TOP', 'MONTHLY_HEIGHT_BOT']

    # nearest neighbor:"int" variables (gridcell)
    Variable_urban_nearest = ['URBAN_REGION_ID']

    # nearest neighbor:"int" variables (numurbl, gridcell)
    Variable_urban_nearest += ['NLEV_IMPROAD' ]

    # nearest neighbor:"double" variables (numurbl, gridcell)
    Variable_urban_nearest += ['T_BUILDING_MAX', 'T_BUILDING_MIN',
            'WIND_HGT_CANYON','WTLUNIT_ROOF','WTROAD_PERV','THICK_ROOF',
            'THICK_WALL','PCT_URBAN','HT_ROOF','EM_IMPROAD','EM_PERROAD',
            'EM_ROOF','EM_WALL','CANYON_HWR']

    # nearest neighbor:"double" variables (nlevurb, numurbl, gridcell)
    Variable_urban_nearest += ['TK_IMPROAD','TK_ROOF','TK_WALL', 
                                     'CV_IMPROAD', 'CV_ROOF', 'CV_WALL']

    # nearest neighbor:"double" variables (numrad, numurbl, gridcell)
    Variable_urban_nearest += ['ALB_IMPROAD_DIF','ALB_IMPROAD_DIR','ALB_PERROAD_DIF',
            'ALB_PERROAD_DIR','ALB_ROOF_DIF', 'ALB_ROOF_DIR',
            'ALB_WALL_DIF', 'ALB_WALL_DIR']

    Variable_nearest += Variable_urban_nearest

    # linear intrepolation of "double" variables
    Variable_linear = ['FMAX', 'Ws', 'ZWT0', 'binfl', 'gdp', 
                    'peatf', 'Ds', 'Dsmax', 'F0', 'LAKEDEPTH',
                   'LANDFRAC_PFT','P3', 'PCT_NATVEG', 'PCT_WETLAND', 
                    'SECONDARY_P', 'OCCLUDED_P', 'LABILE_P']

    # linear:"double" variables (added 11/07/22023)
    Variable_linear += ['APATITE_P', 'PCT_CROP']

    if user_option == 2:
        Variable_nearest = ['PCT_SAND', 'PCT_CLAY','ORGANIC' ,'PCT_NAT_PFT', 
        'MONTHLY_LAI', 'MONTHLY_SAI' ,'MONTHLY_HEIGHT_TOP', 'MONTHLY_HEIGHT_BOT']
        Variable_linear = []

    if user_option == 'all':
         Variable_nearest += ['PCT_SAND', 'PCT_CLAY','ORGANIC' ,'PCT_NAT_PFT', 
        'MONTHLY_LAI', 'MONTHLY_SAI' ,'MONTHLY_HEIGHT_TOP', 'MONTHLY_HEIGHT_BOT']
       
   #Proj4: +proj=lcc +lon_0=-100 +lat_1=25 +lat_2=60 +k=1 +x_0=0 +y_0=0 +R=6378137 +f=298.257223563 +units=m  +no_defs
    geoxy_proj_str = "+proj=lcc +lon_0=-100 +lat_0=42.5 +lat_1=25 +lat_2=60 +x_0=0 +y_0=0 +R=6378137 +f=298.257223563 +units=m +no_defs"
    geoxyProj = CRS.from_proj4(geoxy_proj_str)
    # EPSG: 4326
    # Proj4: +proj=longlat +datum=WGS84 +no_defs
    lonlatProj = CRS.from_epsg(4326) # in lon/lat coordinates
    Txy2lonlat = Transformer.from_proj(geoxyProj, lonlatProj, always_xy=True)
    Tlonlat2xy = Transformer.from_proj(lonlatProj, geoxyProj, always_xy=True)

    # Open the source file
    src = nc.Dataset('surfdata.nc', 'r')
    # points = [y,x] coordinates for src's grid
    src_resx = 0.5
    src_resy = 0.5
    src_lat = src.variables['LATIXY'][...]
    src_lon = src.variables['LONGXY'][...]
    src_x,src_y = Tlonlat2xy.transform(src_lon,src_lat)
    src_lon[src_lon<0.0]=360+src_lon[src_lon<0.0]
    
    
    # Create a new file
    if user_option == '1':
        output_file = "hr_surfdata_v1_part1.nc"
    if user_option == '2':
        output_file = "hr_surfdata_v1_part2.nc"
    if user_option == 'all':
        output_file = "surfdata_Daymet4.1km.2d.v1.nc"
        if UNSTRUCTURED:
            output_file = "surfdata_Daymet4.1km.1d.v1.nc"

    dst = nc.Dataset(output_file, 'w')

    # Copy dimensions
    for name, dimension in src.dimensions.items():
        dst.createDimension(
            name, (len(dimension) if not dimension.isunlimited() else None))

    # Copy global attributes
    dst.setncatts(src.__dict__)

    # get the fine resolution data and the locations (lat, lon)
    r_daymet = nc.Dataset('clmforc.Daymet4.1km.TBOT.2014-01.nc', 'r', format='NETCDF4')
    x_dim = r_daymet['x']  # 1D x-axis
    y_dim = r_daymet['y']  # 1D y-axis
    TBOT = r_daymet.variables['TBOT'][0,:,:]

    grid_ids = np.linspace(0, len(x_dim)*len(y_dim)-1, len(x_dim)*len(y_dim), dtype=int)
    grid_ids = grid_ids.reshape(TBOT.shape)
    grid_xids = np.indices(grid_ids.shape)[1]
    grid_yids = np.indices(grid_ids.shape)[0]

    # setup the bool_mask and XY mesh of the Daymet domain
    bool_mask = ~np.isnan(TBOT)
    grid_x, grid_y = np.meshgrid(x_dim,y_dim)
    lon,lat = Txy2lonlat.transform(grid_x,grid_y)

    grid_y1 = np.copy(grid_y[bool_mask])
    grid_x1 = np.copy(grid_x[bool_mask])

    gridcells= len(grid_x1)

    # masked daymet gridcell's lon/lat
    grid_lon,grid_lat = Txy2lonlat.transform(grid_x1,grid_y1)
    grid_lon[grid_lon<0.0]=360+grid_lon[grid_lon<0.0]

    print(TBOT.shape,bool_mask.shape, grid_x.shape, grid_x1.shape, gridcells)
    del grid_x, grid_y


    # prepare the data source with points_in_daymet_land
    idxy = np.nonzero((src_lat<=max(grid_lat)+src_resy) & (src_lat>=min(grid_lat)-src_resy) & \
                      (src_lon<=max(grid_lon)+src_resx) & (src_lon>=min(grid_lon)-src_resx) )
    points_in_daymet_land = {}
    points_in_daymet_land[0] = src_x[idxy]
    points_in_daymet_land[1] = src_y[idxy]
    points_in_daymet_land[2] = src_lat[idxy]
    points_in_daymet_land[3] = src_lon[idxy]
    points_in_daymet_land[4] = idxy[0]
    points_in_daymet_land[5] = idxy[1]
    land_points = len(points_in_daymet_land[0])
    points=np.zeros((land_points, 2), dtype='double')
    points[:,0] = src_y[idxy]
    points[:,1] = src_x[idxy]

    # Create new dimensions
    dst.createDimension('x_dim', x_dim.size)
    dst.createDimension('y_dim', y_dim.size)

    dst_var = dst.createVariable('x_dim', np.float64, ('x_dim'))
    dst_var.units = "m"
    dst_var.long_name = "x coordinate of projection"
    dst_var.standard_name = "projection_x_coordinate"
    dst['x_dim'][...] = np.copy(x_dim)
    dst_var = dst.createVariable('y_dim', np.float64, ('y_dim'))
    dst_var.units = "m"
    dst_var.long_name = "y coordinate of projection"
    dst_var.standard_name = "projection_y_coordinate"
    dst['y_dim'][...] = np.copy(y_dim)

    dst_var = dst.createVariable('lambert_conformal_conic', np.short)
    dst_var.grid_mapping_name = "lambert_conformal_conic"
    dst_var.longitude_of_central_meridian = -100.
    dst_var.latitude_of_projection_origin = 42.5
    dst_var.false_easting = 0.
    dst_var.false_northing = 0.
    dst_var.standard_parallel = 25., 60.
    dst_var.semi_major_axis = 6378137.
    dst_var.inverse_flattening = 298.257223563

    if UNSTRUCTURED:
        dst_var = dst.createVariable('lon', np.float64, ('y_dim','x_dim'))
        dst_var.units = "degrees_east"
        dst_var.long_name = "longitude coordinate"
        dst_var.standard_name = "longitude"
        dst['lon'][...] = np.copy(lon)
        dst_var = dst.createVariable('lat', np.float64, ('y_dim','x_dim'))
        dst_var.units = "degrees_north"
        dst_var.long_name = "latitude coordinate"
        dst_var.standard_name = "latitude"
        dst['lat'][...] = np.copy(lat)

        dst.createDimension('gridcell', gridcells)
        
        dst_var = dst.createVariable('gridID', np.int32, ('gridcell'))
        dst_var.long_name = 'gridId in the NA domain'
        dst_var.decription = "start from #0 at the upper left corner of the domain, covering all land and ocean gridcells" 
        dst['gridID'][...] = np.copy(grid_ids[bool_mask])

        dst_var = dst.createVariable('gridXID', np.int32, ('gridcell'))
        dst_var.long_name = 'gridId x in the NA domain'
        dst_var.decription = "start from #0 at the upper left corner and from west to east of the domain, with gridID=gridXID+gridYID*x_dim" 
        dst.variables['gridXID'][...] = np.copy(grid_xids[bool_mask])
    
        dst_var = dst.createVariable('gridYID', np.int32, ('gridcell'))
        dst_var.long_name = 'gridId y in the NA domain'
        dst_var.decription = "start from #0 at the upper left corner and from north to south of the domain, with gridID=gridXID+gridYID*y_dim" 
        dst.variables['gridYID'][...] = np.copy(grid_yids[bool_mask])
    else:
        dst_var = dst.createVariable('gridID', np.int32, ('y_dim','x_dim'))
        dst_var.long_name = 'gridId in the NA domain'
        dst_var.decription = "start from #0 at the upper left corner of the domain, covering all land and ocean gridcells" 
        dst.variables['gridID'][...] = np.copy(grid_ids)
    
        dst_var = dst.createVariable('gridXID', np.int32, ('y_dim','x_dim'))
        dst_var.long_name = 'gridId x in the NA domain'
        dst_var.decription = "start from #0 at the upper left corner and from west to east of the domain, with gridID=gridXID+gridYID*x_dim" 
        dst.variables['gridXID'][...] = np.copy(grid_xids)
    
        dst_var = dst.createVariable('gridYID', np.int32, ('y_dim','x_dim'))
        dst_var.long_name = 'gridId y in the NA domain'
        dst_var.decription = "start from #0 at the upper left corner and from north to south of the domain, with gridID=gridXID+gridYID*y_dim" 
        dst.variables['gridYID'][...] = np.copy(grid_yids)
        

    count = 0 # record how may 2D layers have been processed 

    # Copy variables
    for name, variable in src.variables.items():
        start = process_time()
        print("Working on varibale: "+ name + " dimensions: " + str(variable.dimensions))
        
        # Check if the last two dimensions are lsmlat and lsmlon
        if (variable.dimensions[-2:] == ('lsmlat', 'lsmlon')):
            # Determine the interpolation method
            if name in Variable_linear:
                iMethod = 'linear'
            else:
                iMethod = 'nearest'

            # create variables with the new dimensions

            if variable.datatype == np.int32:
                fill_value = -9999  # or any other value that you want to use to represent missing data
            else:
                fill_value = np.nan

            if UNSTRUCTURED:
                x = dst.createVariable(name, variable.datatype, variable.dimensions[:-2]+ ('gridcell',), fill_value = fill_value)
            else:
                x = dst.createVariable(name, variable.datatype, variable.dimensions[:-2]+ ('y_dim', 'x_dim'), fill_value = fill_value)
                
	    # Copy variable attributes
            dst[name].setncatts(src[name].__dict__)

            # prepare the array for the interpolated result
            f_data1 = np.zeros(gridcells, dtype=variable.datatype)

            # original variable data (source) that need to be interpolated
            o_data=np.zeros(land_points, dtype=variable.datatype)
             
            # Handle variables with two dimensions
            if (len(variable.dimensions) == 2):
                source = src[name][:]
                o_data = source[points_in_daymet_land[4][:],points_in_daymet_land[5][:]]
                f_data1 = griddata(points, o_data, (grid_y1, grid_x1), method=iMethod)
                if name=='AREA': f_data1[...] = area_km2
                if name=='LONGXY': f_data1[...] = grid_lon
                if name=='LATIXY': f_data1[...] = grid_lat
                
                if UNSTRUCTURED:
                    # Assign the interpolated data
                    dst[name][...] = np.copy(f_data1)
                    
                else:
      
                # put the masked data back to the data (with the daymet land mask

                    f_data = np.ma.array(np.empty((len(y_dim),len(x_dim)), dtype=variable.datatype), mask=bool_mask, fill_value=fill_value)
                    f_data =  np.where(f_data.mask, f_data, fill_value)
                    f_data[bool_mask]=f_data1 
                    
                # Assign the interpolated data
                    dst[name][...] = np.copy(f_data)
                    
                #print("o_data, f_data1, f_data, dst: max/min/sum")  
                #print(np.nanmax(o_data), np.nanmax(f_data1),np.nanmax(f_data[f_data != -9999]),np.nanmax(dst[name]))
                #print(np.nanmin(o_data), np.nanmin(f_data1),np.nanmin(f_data[f_data != -9999]),np.nanmin(dst[name]))   
                #print(np.nansum(o_data), np.nansum(f_data1),np.nansum(f_data[f_data != -9999]),np.nansum(dst[name]))  

                count = count + 1

            # Handle variables with three dimensions
            if (len(variable.dimensions) == 3):
                for index in range(variable.shape[0]):
                    # get all the source data (global)
                    source = src[name][index,:,:]
                    o_data = source[points_in_daymet_land[4][:],points_in_daymet_land[5][:]]
                    f_data1 = griddata(points, o_data, (grid_y1, grid_x1), method=iMethod)
                    
                    if UNSTRUCTURED:
                        # Assign the interpolated data
                        dst[name][index,...] = np.copy(f_data1)
                    else:

                    # create a mask array to hold the interpolated data

                        f_data = np.ma.array(np.empty((len(y_dim),len(x_dim)), dtype=variable.datatype), mask=bool_mask, fill_value=fill_value)
                        f_data =  np.where(f_data.mask, f_data, fill_value)
                        f_data[bool_mask]=f_data1 
    
                    # Assign the interpolated data to dst.variable
                        dst[name][index,...] = np.copy(f_data)
                        
                    #print("o_data, f_data1, f_data, dst: max/min/sum")  
                    #print(np.nanmax(o_data), np.nanmax(f_data1),np.nanmax(f_data[f_data != -9999]),np.nanmax(dst[name][index,:,:]))
                    #print(np.nanmin(o_data), np.nanmin(f_data1),np.nanmin(f_data[f_data != -9999]),np.nanmin(dst[name][index,:,:]))   
                    #print(np.nansum(o_data), np.nansum(f_data1),np.nansum(f_data[f_data != -9999]),np.nansum(dst[name][index,:,:]))  

                count = count + variable.shape[0]

            # Handle variables with four dimensions
            if (len(variable.dimensions) == 4):
                for index1 in range(variable.shape[0]):
                    for index2 in range(variable.shape[1]):
                        # get all the source data (global)

                        source = src[name][index1, index2,:, :]
                        o_data = source[points_in_daymet_land[4][:],points_in_daymet_land[5][:]]
                        
                        f_data1 = griddata(points, o_data, (grid_y1, grid_x1), method=iMethod)                      
                        if UNSTRUCTURED:
                            # Assign the interpolated data
                            dst[name][index1,index2,...] = np.copy(f_data1)
                        else:

                        # create a mask array to hold the interpolated data
                            f_data = np.ma.array(np.empty((len(y_dim),len(x_dim)), dtype=variable.datatype), mask=bool_mask, fill_value=fill_value)
                            f_data =  np.where(f_data.mask, f_data, fill_value)
                            f_data[bool_mask]=f_data1 
    
                        # Assign the interpolated data to dst.variable
                            dst[name][index1,index2,...] = np.copy(f_data)

                        #print("o_data, f_data1, f_data, dst: max/min/sum")  
                        #print(np.nanmax(o_data), np.nanmax(f_data1),np.nanmax(f_data[f_data != -9999]),np.nanmax(dst[name][index1,index2,:,:]))
                        #print(np.nanmin(o_data), np.nanmin(f_data1),np.nanmin(f_data[f_data != -9999]),np.nanmin(dst[name][index1,index2,:,:]))   
                        #print(np.nansum(o_data), np.nansum(f_data1),np.nansum(f_data[f_data != -9999]),np.nansum(dst[name][index1,index2,:,:]))  

                    count = count + variable.shape[1]

            end = process_time()
            print("Generating variable: " +name+ " takes  {}".format(end-start))

        else:

            # keep variables with the same dimension
            xerr = dst.createVariable(name, variable.datatype, variable.dimensions)
            # Copy variable attributes
            dst[name].setncatts(src[name].__dict__)
            # Copy the data

            dst[name][...] = src[name][...]
            
            end = process_time()
            print("Copying variable: " +name+ " takes  {}".format(end-start))
            
        if count > 50:
            dst.close()   # output the variable into file to save memory

            dst = nc.Dataset(output_file, 'a')

            count = 0
        
        print(count)

    # Close the files
    src.close()
    dst.close()

if __name__ == '__main__':
    main()
