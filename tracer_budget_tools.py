
# coding: utf-8

# # POP Tracer Budget

import os 
import xarray as xr
import numpy as np
from dask.diagnostics import ProgressBar
from glob import glob
get_ipython().run_line_magic('matplotlib', 'inline')


TRACER = "TEMP"
COMPSET = "BRCP85C5CNBDRD"
COMPSET = "B20TRC5CNBDRD"
dir_lens = "/chuva/db2/CESM-LENS/fully_coupled/mon/pop/"
ens_member = 4
ens_str = "{:0>3d}".format(ens_member)
file_tracer = glob(dir_lens + TRACER + "/" + COMPSET + "/b.e11."+ COMPSET +".f09_g16."+ens_str+".pop.h."+TRACER+"*.nc")[0]
ds_tracer = xr.open_dataset(file_tracer,decode_times=False,mask_and_scale=True,chunks={'time': 84})
dz = ds_tracer['dz'];
tarea = ds_tracer['TAREA']
kmt = ds_tracer['KMT']
temp = ds_tracer[TRACER]


def tracer_budget_vol3d (tarea, dz, kmt):
    """
    Arguments: cell area, cell height, max vertical indx
    Returns global 3D volume DataArray: vold3(nz,ny,nx) dtype=float64
    NOTE: does not include SSH variations
    """
    vol3d = (dz*tarea.astype('float64')).load()
    for i in range(dz.shape[0]):
        vol3d[i,:,:] = vol3d[i].where(kmt > i)
    vol3d.attrs = {'units' : 'cm3', 'long_name' : 'Tcell volume'}
    vol3d = vol3d.drop(('ULAT','ULONG'))
    return vol3d

def tracer_budget_mask2d (region_mask, sel_area = 0):
    """
    Return surface mask: if ocean than 1 else nan
    """
    mask = region_mask
    mask = mask.where(region_mask != sel_area,np.nan)
    return (mask/mask)

def tracer_budget_mask3d (var3d):
    """
    Return volume mask: if ocean than 1 else nan
    """
    mask3d = var3d/var3d
    mask3d.attrs = {'units' : '1 / np.nan', 'long_name' : 'mask3d'}
    return mask3d.where(mask3d != 0.,np.nan)

mask3d = tracer_budget_mask3d(temp[0])
vol3d = tracer_budget_vol3d(tarea,dz,kmt)
mask2d = tracer_budget_mask2d(ds_tracer['REGION_MASK'])
area2d = tarea*mask2d


def tracer_budget_var3d_zint_map (tracer, vol3d, klo=0, khi=59):
    """
    Arguments: var4d tracer(t,z,y,x), vol3d cell volume, klo : lowest k index, khi : highest k index
    Returns a 2d tracer map vertical integrated
    !checked!
    """
    units = tracer.units + ' cm^3'
    description = 'Int_V {' + tracer.name + '} dV'
    long_name = tracer.name + ' vertical average'
    attr = {'long_name' : long_name, 'units' : units, 'description': description,            "k_range" : str(klo)+" - "+str(khi)}
    var = tracer[:,klo:khi] * vol3d[klo:khi]
    var_zint_map = var.sum(dim='z_t')
    var_zint_map = var_zint_map.where(var_zint_map != 0.)
    var_zint_map.attrs = attr
    var_zint_map.name = tracer.name + "_zint" 
    return var_zint_map

temp_zint_map = tracer_budget_var3d_zint_map (temp,vol3d)



# check anomaly: ok
#temp_zint_mean = temp_zint_map[0:119].mean(dim='time').load()
#anom = temp_zint_map[996] - temp_zint_mean
#(anom*1e-17).plot(vmin=-30,vmax=30,cmap='RdBu_r')

def tracer_budget_tend_appr (time, time_bnd, var_zint):
    """
    Computes approximate TRACER budget tendency given vertically-integrated POP
    TRACER based on differencing successive monthly means
    NOTE: Assumes monthly POP output with timestamp at end-of-month
          rather than mid-month; assumes time has dimension "days".
          
    TODO: check function; add attributes; check secperday
    """
    secperday = 60.*60*24
    nt = time.shape[0]
    dt = (time_bnd[:,1] - time_bnd[:,0])*secperday
    vfill_value = np.ones(var_zint[0].shape)*np.nan
    
    units = var_zint.units + '/s'
    long_name = var_zint.long_name + ' tendency'
    attr = {'long_name' : long_name, 'units' : units}
    
    with ProgressBar():
        var1 = var_zint.load()
    var2 = var1
    
    #TODO: .roll()
    aux = (var1[0:nt-2].values + var1[1:nt-1].values)*0.5
    var1[0:nt-2].values = aux
    var1[nt-1] = vfill_value
    
    var2[1:nt-2].values = (var1[1:nt-2].values - var1[0:nt-3].values)
    var2[0] = vfill_value 
    var2[nt-1,:,:] = vfill_value
    
    #units per seconds
    var_zint_tend = var2/dt
    var_zint_tend.attrs = attr
    return var_zint_tend
#temp_tend = tracer_budget_tend_appr(ds_tracer['time'],ds_tracer['time_bound'],temp_zint_map)

def tracer_budget_lat_adv_resolved (TRACER, vol3d, COMPSET="B20TRC5CNBDRD", ens_member=4):
    """
    tracer lateral advection 
    based on tracer_budget_adv.ncl
    """
    ens_str = "{:0>3d}".format(ens_member)
    dir_budget = "/chuva/db2/CESM-LENS/download/budget/"
    
    if TRACER == "TEMP":
        var_name1 = "UET"
        var_name2 = "VNT"
        units = "degC cm^3/s"
    else:
        var_name1 = "UES"
        var_name2 = "VNS"
        units = "PSU cm^3/s"
        
    f1 = glob(dir_budget+var_name1+"/b.e11."+COMPSET+".f09_g16.004.pop.h."+var_name1+"*.nc")[0]
    f2 = glob(dir_budget+var_name2+"/b.e11."+COMPSET+".f09_g16.004.pop.h."+var_name2+"*.nc")[0]
    
    long_name = "lateral advective flux (resolved)"
    description = "Int_z{-Div[<"+var_name1+">, <"+var_name2+">]}"
    attr = {'long_name' : long_name, 'units' : units, 'description' : description}
    
    # read tracer associate variable
    ds1 = xr.open_dataset(f1,decode_times=False,mask_and_scale=True,chunks={'time': 84})
    ds2 = xr.open_dataset(f2,decode_times=False,mask_and_scale=True,chunks={'time': 84})
    u_e = ds1[var_name1]
    v_n = ds2[var_name2]
    # shift vol3d
    vol_c = vol3d
    vol_w = vol3d.shift(nlat=-1)
    vol_s = vol3d.shift(nlon=-1)
    # shift
    u_w = u_e.shift(nlat=-1)
    v_s = v_n.shift(nlon=-1)
    # e.g.: degC cm^3/s
    var1 = u_e*vol_c
    var2 = u_w*vol_w
    var3 = v_n*vol_c
    var4 = v_s*vol_s
    # Div []
    var5 = (var2-var1) + (var4-var3)
    # vertical integration
    var_lat_adv_res_map = var5.sum(dim='z_t')
    var_lat_adv_res_map.attrs = attr
    var_lat_adv_res_map.name = TRACER.lower() + "_lat_adv_res"
    var_lat_adv_res_map = var_lat_adv_res_map.drop(("ULONG","ULAT"))
    return var_lat_adv_res_map.where(var_lat_adv_res_map != 0.)

temp_lat_adv_res = tracer_budget_lat_adv_resolved(TRACER,vol3d)


def tracer_budget_vert_adv_resolved (TRACER, vol3d, COMPSET="B20TRC5CNBDRD", ens_member=4):
    """
    tracer vertical advection
    """
    ens_str = "{:0>3d}".format(ens_member)
    dir_budget = "/chuva/db2/CESM-LENS/download/budget/"
    
    if TRACER == "TEMP":
        var_name = "WTT"
        units = "degC cm^3/s"
    else:
        var_name = "WTS"
        units = "PSU cm^3/s"
    
    long_name = "vertical advective flux (resolved)"
    description = "Int_z{-d[<"+var_name+">]/dz}"
    attr = {'long_name' : long_name, 'units' : units, 'description' : description}
    # read tracer associate variable
    f1 = glob(dir_budget+var_name+"/b.e11."+COMPSET+".f09_g16.004.pop.h."+var_name+"*.nc")[0]
    ds1 = xr.open_dataset(f1,decode_times=False,mask_and_scale=True,chunks={'time': 84})
    FIELD = ds1[var_name]
    vol3d.rename({'z_t' : 'z_w_top'})
    volc = vol3d[0].values
    # e.g. degC cm^3/s
    var1 = FIELD[:,1]*volc
    var_vert_adv_res_map = -1*var1
    var_vert_adv_res_map.attrs = attr
    return var_vert_adv_res_map

temp_vert_adv_res = tracer_budget_vert_adv_resolved(TRACER,vol3d)


def tracer_budget_hmix (TRACER, vol3d, COMPSET="B20TRC5CNBDRD", ens_member=4):
    """
    tracer horizontal mixing
    compute tracer hmix integrals
    vertical fluxes are positive up
    """
    ens_str = "{:0>3d}".format(ens_member)
    dir_budget = "/chuva/db2/CESM-LENS/download/budget/"
    
    if TRACER == "TEMP":
        units = "degC cm^3/s"
    else:
        units = "PSU cm^3/s"
    #diffusive flux variable names
    var_name1 = "HDIFE_"+TRACER
    var_name2 = "HDIFN_"+TRACER
    
    long_name = "lateral diffusive flux (resolved)"
    description = "Int_z{-Div[<"+var_name1+">, <"+var_name2+">]}"
    attr = {'long_name' : long_name, 'units' : units, 'description' : description}
    
    # read tracer associate variable
    f1 = glob(dir_budget+var_name1+"/b.e11."+COMPSET+".f09_g16.004.pop.h."+var_name1+"*.nc")[0]
    f2 = glob(dir_budget+var_name2+"/b.e11."+COMPSET+".f09_g16.004.pop.h."+var_name2+"*.nc")[0]
    
    ds1 = xr.open_dataset(f1,decode_times=False,mask_and_scale=True,chunks={'time': 84})
    ds2 = xr.open_dataset(f2,decode_times=False,mask_and_scale=True,chunks={'time': 84})
    u_e = ds1[var_name1]
    v_n = ds2[var_name2]
    
    # shift vol3d
    vol_c = vol3d
    vol_w = vol3d.shift(nlat=-1)
    vol_s = vol3d.shift(nlon=-1)
    
    # shift
    u_w = u_e.shift(nlat=-1)
    v_s = v_n.shift(nlon=-1)
    
    # e.g.: degC cm^3/s
    var1 = u_e*vol_c
    var2 = u_w*vol_w
    var3 = v_n*vol_c
    var4 = v_s*vol_s
    # Div []
    #var5 = (var2-var1) + (var4-var3)
    var5 = (var1 - var2) + (var3-var4)
    # vertical integration
    var_lat_mix_res_map = var5.sum(dim='z_t')
    var_lat_mix_res_map.attrs = attr
    return var_lat_mix_res_map.where(var_lat_mix_res_map != 0.)

temp_lat_mix = tracer_budget_hmix(TRACER, vol3d)


def tracer_budget_dia_vmix (TRACER, tarea, kmt, klo=0, khi=59, COMPSET="B20TRC5CNBDRD", ens_member=4):
    """
    Computes vertical integral of diabatic vertical mixing (DIA_IMPVF_), ie. KPP
    """
    ens_str = "{:0>3d}".format(ens_member)
    dir_budget = "/chuva/db2/CESM-LENS/download/budget/"
    
    if TRACER == "TEMP":
        units = "degC cm^3/s"
    else:
        units = "PSU cm^3/s"
    #variable name
    var_name = "DIA_IMPVF_"+TRACER
    
    long_name = "vertical (diabatic) mixing flux (resolved)"
    description = "Int_z{-d[<"+var_name+">]/dz}" 
    attr = {'long_name' : long_name, 'units' : units, 'description' : description}
    
    # read tracer associate variable
    f1 = glob(dir_budget+var_name+"/b.e11."+COMPSET+".f09_g16.004.pop.h."+var_name+"*.nc")[0]
    ds1 = xr.open_dataset(f1,decode_times=False,mask_and_scale=True,chunks={'time': 84})
    FIELD = ds1[var_name]
    # zero diffusive flux across sea surface -> 0 
    FIELD_TOP = FIELD[:,klo]
    FIELD_BOT = FIELD[:,khi]
    tarea_bot = tarea.where(kmt > khi,0)
    tarea_top = tarea.where(kmt > klo,0)
    #
    FIELD_BOT = FIELD_BOT*tarea_bot
    FIELD_TOP = FIELD_TOP*tarea_top
    var_vert_mix_map = -(FIELD_BOT.fillna(0.) - FIELD_TOP)
    return var_vert_mix_map
temp_dia_vmix = tracer_budget_dia_vmix(TRACER, tarea, kmt,0,59)


def tracer_budget_adi_vmix (TRACER, vol3d, klo, khi):
    """
    Computes vertical integral of adiabatic vertical mixing (HDIFB_), ie. GM+Submeso
    """
    if TRACER == "TEMP":
        units = "degC cm^3/s"
    else:
        units = "PSU cm^3/s"
    #variable name
    var_name = "HDIFB_"+TRACER
    
    long_name = "vertical (adiabatic) mixing flux (resolved)"
    description = "Int_z{-d[<"+var_name+">]/dz}" 
    attr = {'long_name' : long_name, 'units' : units, 'description' : description}
    # read tracer associate variable
    f1 = "/chuva/db2/CESM-LENS/download/budget/"+var_name+"/b.e11.B20TRC5CNBDRD.f09_g16.004.pop.h."+var_name+".192001-200512.nc"
    ds1 = xr.open_dataset(f1,decode_times=False,mask_and_scale=True,chunks={'time': 84})
    FIELD = ds1[var_name]
    # zero diffusive flux across sea surface -> 0 
    FIELD_TOP = FIELD[:,klo]
    FIELD_BOT = FIELD[:,khi]
    #
    FIELD_BOT = FIELD_BOT*vol3d[khi]
    FIELD_TOP = FIELD_TOP*vol3d[klo]
    var_vert_mix_map = -(FIELD_BOT.fillna(0.) - FIELD_TOP)
    return var_vert_mix_map
temp_adi_vmix = tracer_budget_adi_vmix(TRACER, vol3d, 0, 59)


def tracer_budget_sflux (TRACER, var_name, area2d, COMPSET="B20TRC5CNBDRD", ens_member=4):
    """
    compute domain-specific maps of tracer surface fluxes
    
    Note: fluxes positive are down!
 
    based on tracer_budget_srf_flux.ncl
    """
    ens_str = "{:0>3d}".format(ens_member)
    dir_budget = "/chuva/db2/CESM-LENS/download/budget/"
    
    f1 = glob(dir_budget+var_name+"/b.e11."+COMPSET+".f09_g16.004.pop.h."+var_name+"*.nc")[0]
    # read tracer associate variable
    ds1 = xr.open_dataset(f1,decode_times=False,mask_and_scale=True,chunks={'time': 84})

    rho_sw = ds1["rho_sw"]              # density of saltwater (g/cm^3)
    rho_sw = rho_sw * 1.e-3             # (kg/cm^3)
    
    cp_sw = ds1["cp_sw"]                # spec. heat of saltwater (erg/g/K)
    cp_sw = cp_sw * 1.e-7 * 1.e3        # (J/kg/K)
    rho_cp = rho_sw * cp_sw             # (J/cm^3/K)
    latvap = ds1["latent_heat_vapor"]   # lat heat of vaporiz. (J/kg)
    latfus = ds1["latent_heat_fusion"]  # lat heat of fusion (erg/g)
    latfus = latfus * 1.e-7 * 1.e3      # (J/kg)

    if var_name in ['SHF', 'QFLUX', 'SENH_F', 'LWDN_F', 'LWUP_F', 'SHF_QSW', 'MELTH_F']:
        scale_factor = 1.e-4 * (1./rho_cp)          #W/m^2 -> degC cm/s
    elif var_name in ["SNOW_F","IOFF_F"]:
        scale_factor = -latfus*1.e-4 * (1./rho_cp)  #kg/m^2/s -> degC cm/s
    elif var_name is "EVAP_F":
        scale_factor = latvap*1.e-4 * (1./rho_cp)   #kg/m^2/s -> degC cm/s
    else :
        scale_factor = 1.
    
    if TRACER == "TEMP":
        units = "degC cm^3/s"
    else:
        units = "PSU cm^3/s"
        
    FIELD = ds1[var_name]
    var1 = FIELD * scale_factor
    var_sflux_map = var1*area2d
    long_name = "vertical flux across sea surface"
    #description = "Int_z{-Div[<"+var_name1+">, <"+var_name2+">]}"
    attr = {'long_name' : long_name, 'units' : units}#, 'description' : description}
    var_sflux_map = var_sflux_map.drop(("ULAT","ULONG" ))
    var_sflux_map.attrs = attr
    var_sflux_map.name = TRACER.lower() + "_" + var_name 
    return var_sflux_map


# fluxes terms
temp_qflux = tracer_budget_sflux(TRACER, "QFLUX", area2d)
temp_senh_f = tracer_budget_sflux(TRACER, "SENH_F", area2d)
temp_lwdn_f = tracer_budget_sflux(TRACER, "LWDN_F", area2d)
temp_lwup_f = tracer_budget_sflux(TRACER, "LWUP_F", area2d)
temp_melth_f = tracer_budget_sflux(TRACER, "MELTH_F", area2d)
temp_shf_qsw = tracer_budget_sflux(TRACER, "SHF_QSW", area2d)
temp_evap_f = tracer_budget_sflux(TRACER, "EVAP_F", area2d)
temp_snow_f = tracer_budget_sflux(TRACER, "SNOW_F", area2d)
temp_ioff_f = tracer_budget_sflux(TRACER, "IOFF_F", area2d)
temp_shf = tracer_budget_sflux(TRACER, "SHF", area2d)


# KPP_SRC_TEMP
var_name = "KPP_SRC_TEMP"
ens_str = "{:0>3d}".format(ens_member)
dir_budget = "/chuva/db2/CESM-LENS/download/budget/"    
f1 = glob(dir_budget+var_name+"/b.e11."+COMPSET+".f09_g16.004.pop.h."+var_name+"*.nc")[0]
# read tracer associate variable
ds1 = xr.open_dataset(f1,decode_times=False,mask_and_scale=True,chunks={'time': 84})
KPP_SRC_TEMP = ds1[var_name]
KPP_SRC_TEMP = KPP_SRC_TEMP.where(KPP_SRC_TEMP != 0.)
# compute temp flux
temp_kpp_src = tracer_budget_var3d_zint_map(KPP_SRC_TEMP,vol3d,0,44)

# total surface flux
temp_shf_tend = temp_qflux +  temp_senh_f + temp_lwdn_f + temp_lwup_f +     temp_melth_f + temp_shf_qsw + temp_evap_f + temp_snow_f + temp_ioff_f + temp_kpp_src
