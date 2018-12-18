#
# Compute CESM-LENS / POP2 Tracer Budgets
#

#*****************************************************************************#
from tracer_budget_tools import *
from dask.diagnostics import ProgressBar
import os
#get_ipython().run_line_magic("matplotlib", "inline")

#*****************************************************************************#
# I/O
TRACER = "TEMP"
COMPSET = "BRCP85C5CNBDRD"
COMPSET = "B20TRC5CNBDRD"
dir_lens = "/chuva/db2/CESM-LENS/fully_coupled/mon/pop/"
ens_member = 4
ens_str = "{:0>3d}".format(ens_member)
file_tracer = glob(dir_lens + TRACER + "/" + COMPSET + "/b.e11."+ COMPSET +".f09_g16."+ens_str+".pop.h."+TRACER+"*.nc")[0]
fileout = "./nc/b.e11."+ COMPSET +".f09_g16."+ens_str+".pop.h.budget.nc"

klo = 0
khi = 25
tlo = 491   #912
thi = 611   #1032

#*****************************************************************************#
# read tracer files
ds_tracer = xr.open_dataset(file_tracer,decode_times=False,mask_and_scale=True,chunks={"time": 36})
# select 120 months
temp = ds_tracer[TRACER].isel(time=slice(tlo,thi))

# grid vars
dz = ds_tracer["dz"];
tarea = ds_tracer["TAREA"]
kmt = ds_tracer["KMT"]

#*****************************************************************************#
mask3d = tracer_budget_mask3d(temp[0])
vol3d = tracer_budget_vol3d(tarea,dz,kmt)
mask2d = tracer_budget_mask2d(ds_tracer["REGION_MASK"])
area2d = tarea*mask2d
# select 120 months
time = ds_tracer["time"].isel(time=slice(tlo,thi))
time_bound = ds_tracer["time_bound"].isel(time=slice(tlo,thi))

#*****************************************************************************#
# temp_zint
temp_zint_map = tracer_budget_var3d_zint_map (temp, vol3d, klo, khi)

# temp_tend
print("computing total temp tend")
temp_tend = tracer_budget_tend_appr(TRACER, time_bound,temp_zint_map)
temp_tend_anom = rmMonAnnCyc(pop_decode_time(temp_tend))
temp_tend_anom = temp_tend_anom.assign_coords(time=time)
#*****************************************************************************#
# Creating Dataset
ds_out = temp_tend.to_dataset()
ds_out["temp_tend_anom"] = temp_tend_anom
ds_out["temp_zint_map"] = temp_zint_map
ds_out["vol3d"] = vol3d
ds_out["time_bound"] = time_bound
print(ds_out)

#*****************************************************************************#
#
print("computing temp_lat_adv_res")
temp_lat_adv_res = tracer_budget_lat_adv_resolved(TRACER, vol3d, COMPSET, ens_member, klo, khi, tlo, thi)
temp_lat_adv_res_anom = rmMonAnnCyc(pop_decode_time(temp_lat_adv_res))
temp_lat_adv_res_anom = temp_lat_adv_res_anom.assign_coords(time=time)
ds_out["temp_lat_adv_res"] = temp_lat_adv_res
ds_out["temp_lat_adv_res_anom"] = temp_lat_adv_res_anom
#
print("computing temp_vert_adv_res")
temp_vert_adv_res = tracer_budget_vert_adv_resolved (TRACER, vol3d, COMPSET, ens_member, klo, khi, tlo, thi)
temp_vert_adv_res_anom = rmMonAnnCyc(pop_decode_time(temp_vert_adv_res))
temp_vert_adv_res_anom = temp_vert_adv_res_anom.assign_coords(time=time)
ds_out["temp_vert_adv_res"] = temp_vert_adv_res
ds_out["temp_vert_adv_res_anom"] = temp_vert_adv_res_anom
#
print("computing temp_lat_mix")
temp_lat_mix = tracer_budget_hmix (TRACER, vol3d, COMPSET, ens_member, klo, khi, tlo, thi)

temp_lat_mix_anom = rmMonAnnCyc(pop_decode_time(temp_lat_mix))
temp_lat_mix_anom = temp_lat_mix_anom.assign_coords(time=time)
ds_out["temp_lat_mix"] = temp_lat_mix
ds_out["temp_lat_mix_anom"] = temp_lat_mix_anom
#
print("computing temp_vmix")
temp_dia_vmix = tracer_budget_dia_vmix (TRACER, tarea, kmt, klo, khi, COMPSET, ens_member, tlo, thi)
temp_adi_vmix = tracer_budget_adi_vmix(TRACER, vol3d, COMPSET, ens_member, klo, khi, tlo, thi)
temp_vmix = temp_dia_vmix + temp_adi_vmix
temp_vmix.attrs = {"long_name" : "vertical (diabatic+adiabatic) mixing flux"}
temp_vmix.name = "temp_vmix"
temp_vmix_anom = rmMonAnnCyc(pop_decode_time(temp_vmix))
temp_vmix_anom = temp_vmix_anom.assign_coords(time=time)
ds_out["temp_vmix"] = temp_vmix
ds_out["temp_vmix_anom"] = temp_vmix_anom
ds_out["temp_dia_vmix"] = temp_dia_vmix
ds_out["temp_adi_vmix"] = temp_adi_vmix

#*****************************************************************************#
# fluxes terms
temp_qflux = tracer_budget_sflux(TRACER, "QFLUX", area2d, COMPSET, ens_member, tlo, thi)
temp_senh_f = tracer_budget_sflux(TRACER, "SENH_F", area2d, COMPSET, ens_member, tlo, thi)
temp_lwdn_f = tracer_budget_sflux(TRACER, "LWDN_F", area2d, COMPSET, ens_member, tlo, thi)
temp_lwup_f = tracer_budget_sflux(TRACER, "LWUP_F", area2d, COMPSET, ens_member, tlo, thi)
temp_melth_f = tracer_budget_sflux(TRACER, "MELTH_F", area2d, COMPSET, ens_member, tlo, thi)
temp_shf_qsw = tracer_budget_sflux(TRACER, "SHF_QSW", area2d, COMPSET, ens_member, tlo, thi)
temp_evap_f = tracer_budget_sflux(TRACER, "EVAP_F", area2d, COMPSET, ens_member, tlo, thi)
temp_snow_f = tracer_budget_sflux(TRACER, "SNOW_F", area2d, COMPSET, ens_member, tlo, thi)
temp_ioff_f = tracer_budget_sflux(TRACER, "IOFF_F", area2d, COMPSET, ens_member, tlo, thi)
temp_shf = tracer_budget_sflux(TRACER, "SHF", area2d, COMPSET, ens_member, tlo, thi)

#*****************************************************************************#
# compute KPP_temp flux (OBS. klo=1 else ERROR)
temp_kpp_src = tracer_budget_kpp_src (TRACER, vol3d,COMPSET,ens_member,klo,khi,tlo,thi)
temp_kpp_src.name = "temp_kpp_src"
print(temp_kpp_src)
#*****************************************************************************#
# total surface flux: tend
print("computing net shf tendency")
temp_shf_tend = temp_qflux + temp_senh_f + temp_lwdn_f + temp_lwup_f + temp_melth_f + \
                temp_shf_qsw + temp_evap_f + temp_snow_f + temp_ioff_f + temp_kpp_src
temp_shf_tend.name = "temp_shf_tend"
temp_shf_tend.attrs = temp_shf.attrs
print(temp_shf_tend)
temp_shf_tend = temp_shf_tend.assign_coords(time=time)
temp_shf_tend_anom = rmMonAnnCyc(pop_decode_time(temp_shf_tend))
temp_shf_tend_anom = temp_shf_tend_anom.assign_coords(time=time)
ds_out["temp_shf_tend"] = temp_shf_tend
ds_out["temp_shf_tend_anom"] = temp_shf_tend_anom

#*****************************************************************************#
# Advection & Diffusion Tendencies 
print("computing net adv tendency")
temp_adv_tend = temp_lat_adv_res_anom +  temp_vert_adv_res_anom
temp_adv_tend.name = "temp_adv_tend"
temp_adv_tend.attrs = temp_lat_adv_res.attrs
ds_out["temp_adv_tend"] = temp_adv_tend

print("computing net diff tendency")
temp_diff_tend = temp_lat_mix_anom - temp_vmix_anom
temp_diff_tend.name = "temp_diff_tend"
temp_diff_tend.attrs = temp_lat_mix.attrs
ds_out["temp_diff_tend"] = temp_diff_tend

#*****************************************************************************#
# Residual tendency
temp_advdiff_tend = temp_adv_tend +  temp_diff_tend
print("computing net rhs tendency")
temp_rhs_tend = temp_shf + temp_qflux + temp_adv_tend + temp_diff_tend + temp_kpp_src

print("computing residual")
temp_resid_tend = temp_rhs_tend - temp_tend
temp_resid_tend.name = "temp_resid_tend"
ds_out["temp_resid_tend"] = temp_resid_tend

#*****************************************************************************#
# write data set to netCDF file
if os.path.isfile(fileout):
    os.remove(fileout)
print("saving file: " + fileout)
with ProgressBar():
    ds_out.to_netcdf(fileout,mode="w",format="NETCDF4")

#*****************************************************************************#
