import netCDF4 as nc

# Open the source and target netCDF files
source = nc.Dataset('NA_2Ddomain.nc', 'r')
target = nc.Dataset('hr_surfdata.nc', 'a')  # 'a' for append mode

# Copy global attributes from source to target
for name in source.ncattrs():
    target.setncattr(name, source.getncattr(name))

# Add notes as global attributes
notes = ["Surfdata for Daymet NA domain is generated from surfdata_360x720cru_simyr1850_c180216.nc", \
"Surfdata contains 2D NA domain information from Daymet4.1km.2d.domain.nc"]
for i, note in enumerate(notes, 1):
    target.setncattr(f'Note_{i}', note)

# Copy dimensions from source to target
for name, dimension in source.dimensions.items():
    if name not in target.dimensions:
        target.createDimension(name, len(dimension) if not dimension.isunlimited() else None)

# List of variables to be copied
variables_to_copy = ['gridIDs', 'mask', 'xc', 'yc', 'xc_LLC', 'yc_LLC']  # replace with your variable names

# Copy variables from source to target
for name in variables_to_copy:
    if name in source.variables and name not in target.variables:
        variable = source.variables[name]
        x = target.createVariable(name, variable.datatype, variable.dimensions)
        # Copy variable attributes
        target[name].setncatts(source[name].__dict__)
        target[name][:] = source[name][:]

Target.title = "2D surface properties dataset for the Daymet NA region"

# Close the netCDF files
source.close()
target.close()