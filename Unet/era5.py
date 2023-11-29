import os
import numpy as np
import onnx
import onnxruntime as ort
import xarray as xr
import dask
import numpy as np
import gcsfs
from typing import OrderedDict
import yaml
import time


model_24 = onnx.load('pangu_weather_24.onnx')

options = ort.SessionOptions()
options.enable_cpu_mem_arena=False
options.enable_mem_pattern = False
options.enable_mem_reuse = False
options.intra_op_num_threads = 1

cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',}

ort_session_24 = ort.InferenceSession('pangu_weather_24.onnx', sess_options=options, providers=['CPUExecutionProvider'])


def load_vars(var_config_file):
    with open(var_config_file, "r") as f:
        var_config = yaml.load(f, Loader=yaml.Loader)

    return var_config

def main():
    ar_full_37_1h = xr.open_zarr(
    'gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2/'
    )
    print("Model surface dataset size {} TiB".format(ar_full_37_1h.nbytes/(1024**4)))

    variables = OrderedDict(load_vars('./train_40yr_config.yaml'))
    surface_data = ar_full_37_1h[variables['input']['surface']]
    high_data = ar_full_37_1h[variables['input']['high']].sel(level=variables['input']['levels'])
    sur  = surface_data.sel(time=slice('2020-01-01', '2020-1-31'))
    hig  = high_data.sel(time=slice('2020-01-01', '2020-1-31'))
    sur_dat = xr.concat([sur['2m_temperature'],sur['mean_sea_level_pressure'],sur['10m_u_component_of_wind'],sur['10m_v_component_of_wind']],"var")
    hig_dat = xr.concat([hig['geopotential'],hig['temperature'],hig['specific_humidity'],hig['u_component_of_wind'],hig['v_component_of_wind']],'var')

    res_sur  = sur_dat.transpose('time','var','latitude','longitude')
    res_high = hig_dat.transpose('time','var','level','latitude','longitude')


    for i in range(720):
        now = str(res_sur[i]['time'].to_numpy())
        now = now.split(":")[0]
        print(now)

        sur = res_sur[i].to_numpy()
        high = res_high[i].to_numpy()

        input = high.astype(np.float32)
        surface = sur.astype(np.float32)
        output, output_surface = ort_session_24.run(None, {'input':input, 'input_surface':surface})
        np.save('./output/upper_24_%s.npy' % now, output[:, :, 30*4:90*4+1, 70*4:140*4+1])
        np.save('./output/surface_24_%s.npy' % now, output_surface[:, 30*4:90*4+1, 70*4:140*4+1])
        print('./output/upper_%s.npy' % now)


if __name__ == '__main__':
    main()

