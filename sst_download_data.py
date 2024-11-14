import cdsapi
import os

def download_sst_data():
    client = cdsapi.Client()

    area = [60, 100, 0, 180]
    years = [str(y) for y in range(1979, 2022)]
    months = [f"{m:02d}" for m in range(1, 13)]
    days = [f"{d:02d}" for d in range(1, 32)]
    times = ['00:00', '06:00', '12:00', '18:00']

    request_params = {
        'product_type': 'reanalysis',
        'data_format': 'netcdf',
        'variable': 'sea_surface_temperature',
        'year': years,
        'month': months,
        'day': days,
        'time': times,
        'area': area,
    }

    output_file = 'sst_data_1979_2021_full.nc'

    print(f"Starting download of SST data to {output_file}")

    client.retrieve(
        'reanalysis-era5-single-levels',
        request_params,
        output_file
    )

    print(f"Download completed. File saved as {output_file}")

if __name__ == "__main__":
    download_sst_data()