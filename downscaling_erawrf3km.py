import pandas as pd
import numpy as np
import os
from netCDF4 import Dataset
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

"""
    1. nc file data structure inspection
"""
temp_file_era = "data/windera/era_wind_2018120612.nc"
temp_file_wrf = "data/WRF3Km/2018120612.nc"

ds_era = xr.open_dataset(temp_file_era)
ds_wrf = xr.open_dataset(temp_file_wrf)

print(f'Data structure of ERA5 file:\n')
print(ds_era)
for var in ds_era.variables.keys():
    print(f'Variable {var} structure:\n {ds_era.variables[var]}')

#Visualize ERA5 file (06/12/2018 - 12:00)
def visualize_nc_file(ds, title1, title2):
    u_era_10, v_era_10 = ds['u10m'], ds['v10m']
    u_era_100, v_era_100 = ds['u100m'], ds['v100m']
    lon_era, lat_era = ds['lon'].values, ds['lat'].values

    global_min = 0
    global_max = 20

    #https://help.marine.copernicus.eu/en/articles/5487266-how-to-average-winds
    magnitude_era_10 = np.sqrt(u_era_10**2 + v_era_10**2)
    magnitude_era_100 = np.sqrt(u_era_100**2 + v_era_100**2)


    fig, axs = plt.subplots(1, 2, figsize=(14, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    extent = [90, 132.5, -2.5, 34]

    axs[0].add_feature(cfeature.COASTLINE, linewidth=1)
    axs[0].add_feature(cfeature.BORDERS, linestyle=':')
    axs[0].set_extent(extent, crs = ccrs.PlateCarree())
    q1_plot = axs[0].quiver(lon_era, lat_era, u_era_10, v_era_10, magnitude_era_10, scale = 100, cmap = "viridis")
    axs[0].set_title(title1)

    axs[1].add_feature(cfeature.COASTLINE, linewidth=1)
    axs[1].add_feature(cfeature.BORDERS, linestyle=':')
    axs[1].set_extent(extent, crs = ccrs.PlateCarree())
    q2_plot = axs[1].quiver(lon_era, lat_era, u_era_100, v_era_100, magnitude_era_100, scale = 100, cmap = "viridis")
    axs[1].set_title(title2)

    plt.show()

visualize_nc_file(ds_era, "ERA5 10M", "ERA5 100M")

print(f'\nData structure of WRF3Km file:\n')
print(ds_wrf)
for var in ds_wrf.variables.keys():
    print(f'Variable {var} structure:\n {ds_wrf.variables[var]}')

visualize_nc_file(ds_wrf, "WRF3Km 10M", "WRF3Km 100M")


"""
    2. Time-series data
"""
def process_time_from_era_file(file):
  name_list = file.split("_")
  raw_date = name_list[2].replace(".nc","")
  year = raw_date[:4]
  month = raw_date[4:6]
  day = raw_date[6:8]
  hour = raw_date[8:10]
  return pd.to_datetime(f"{year}-{month}-{day} {hour}:00:00")

def process_time_from_wrf_file(file):
  raw_date = file.replace(".nc","")
  year = raw_date[:4]
  month = raw_date[4:6]
  day = raw_date[6:8]
  hour = raw_date[8:10]
  return pd.to_datetime(f"{year}-{month}-{day} {hour}:00:00")

def split_ds_based_on_height(file,time):
  ds = xr.open_dataset(file)
  ds_10m = ds[["u10m","v10m"]].assign_coords(time=("time", [time]))
  ds_100m = ds[["u100m","v100m"]].assign_coords(time=("time", [time]))

  return ds_10m, ds_100m

def process_all_files(folder_path, process_time_from_file):
    all_ds_10m = []
    all_ds_100m = []
    total_files = len([f for f in os.listdir(folder_path) if f.endswith(".nc")])

    for idx, file in enumerate(sorted(os.listdir(folder_path)), start=1):
        if file.endswith(".nc"):
            print(f"Processing file {idx}/{total_files}: {file}")
            file_path = os.path.join(folder_path, file)
            time = process_time_from_file(file)
            print(f"Current time: {time}")
            ds_10m, ds_100m = split_ds_based_on_height(file_path, time)

            all_ds_10m.append(ds_10m)
            all_ds_100m.append(ds_100m)

    print(f"Current process finished")
    combined_10m = xr.concat(all_ds_10m, dim="time")
    combined_100m = xr.concat(all_ds_100m, dim="time")

    return combined_10m, combined_100m

def xr_to_parquet(ds, output_file):
    df = ds.to_dataframe().reset_index()
    df.to_parquet(output_file, index=False)
    print(f"Saved DataFrame to Parquet: {output_file}")


era_folder = "data/windera"
wrf_folder = "data/WRF3Km"

print("Processing ERA5 files...")
era_10m, era_100m = process_all_files(era_folder, process_time_from_era_file)

print("Processing WRF3km files...")
wrf_10m, wrf_100m = process_all_files(wrf_folder, process_time_from_wrf_file)

print("Saving ERA5 CSVs...")
xr_to_parquet(era_10m, "era5_10m_combined.parquet")
xr_to_parquet(era_100m, "era5_100m_combined.parquet")

print("Saving WRF3km CSVs...")
xr_to_parquet(wrf_10m, "wrf3km_10m_combined.parquet")
xr_to_parquet(wrf_100m, "wrf3km_100m_combined.parquet")

print("Processing completed.")

"""
    3. Simple interpolation (Nearest Neighbour)
"""
def interpolate_all_era_files(input_folder, output_folder, wrf_file):
    # Load the WRF3km dataset to get the target grid
    ds_wrf = xr.open_dataset(wrf_file)
    target_lat = ds_wrf['lat']
    target_lon = ds_wrf['lon']

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the input folder
    for file in sorted(os.listdir(input_folder)):
        if file.endswith(".nc"):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file)
            ds_era = xr.open_dataset(input_path)
            ds_era_downscaled = ds_era.interp(lat=target_lat, lon=target_lon, method='nearest')
            ds_era_downscaled.to_netcdf(output_path)
            print(f"Saved downscaled file: {output_path}")


input_folder = "data/windera"
output_folder = "data/EraCut"
wrf_file = "data/WRF3Km/2019123118.nc"
# interpolate_all_era_files(input_folder, output_folder, wrf_file)


def visualize_era_and_wrf_files(ds_era, ds_wrf, version, title1, title2):
    u_era, v_era = ds_era[version], ds_era[version]
    u_wrf, v_wrf = ds_wrf[version], ds_wrf[version]

    lon, lat = ds_era['lon'].values, ds_era['lat'].values

    global_min = 0
    global_max = 20

    magnitude_era = np.sqrt(u_era ** 2 + v_era ** 2)
    magnitude_wrf = np.sqrt(u_wrf ** 2 + v_wrf ** 2)

    fig, axs = plt.subplots(1, 2, figsize=(14, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    extent = [90, 132.5, -2.5, 34]

    global_min = min(magnitude_era.min().item(), magnitude_wrf.min().item())
    global_max = max(magnitude_era.max().item(), magnitude_wrf.max().item())

    axs[0].add_feature(cfeature.COASTLINE, linewidth=1)
    axs[0].add_feature(cfeature.BORDERS, linestyle=':')
    axs[0].set_extent(extent, crs=ccrs.PlateCarree())
    q1_plot = axs[0].quiver(lon, lat, u_era, v_era, magnitude_era, scale=100, cmap="viridis")
    axs[0].set_title(title1)

    axs[1].add_feature(cfeature.COASTLINE, linewidth=1)
    axs[1].add_feature(cfeature.BORDERS, linestyle=':')
    axs[1].set_extent(extent, crs=ccrs.PlateCarree())
    q2_plot = axs[1].quiver(lon, lat, u_wrf, v_wrf, magnitude_wrf, scale=100, cmap="viridis")
    axs[1].set_title(title2)

    cbar = fig.colorbar(q2_plot, ax=axs, orientation='vertical', fraction=0.03, pad=0.04)
    cbar.set_label("Wind Magnitude (m/s)")

    plt.show()

"""
    Compare ERA5 and WRF3Km files' grids
"""
ds_era = xr.open_dataset('data/windera/era_wind_2019030606.nc')
ds_wrf = xr.open_dataset('data/WRF3Km/2019030606.nc')

ds_era_3km = ds_era.interp(lat=ds_wrf['lat'].values, lon=ds_wrf['lon'].values, method='nearest')

lat_era, lon_era = np.meshgrid(ds_era_3km['lat'].values, ds_era_3km['lon'].values, indexing='ij')
lat_wrf, lon_wrf = np.meshgrid(ds_wrf['lat'].values, ds_wrf['lon'].values, indexing='ij')

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.set_extent([100, 130, 0, 30], crs=ccrs.PlateCarree())

ax.scatter(lon_era, lat_era, color='blue', s=10, label='Era25km Grid', transform=ccrs.PlateCarree())

ax.scatter(lon_wrf, lat_wrf, color='red', s=1, label='WRF3km Grid', transform=ccrs.PlateCarree())

ax.legend(loc='upper right', fontsize=10)
ax.set_title("Grid Alignment: Era25km vs WRF3km", fontsize=14)

plt.show()

"""
    Compare ERA5 and WRF3Km post-interpolation
"""
visualize_era_and_wrf_files(ds_era_3km, ds_wrf, 'u10m', "ERA5 10M downscaled", "WRF3Km 10M")
