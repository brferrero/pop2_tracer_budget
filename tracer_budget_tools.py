# coding: utf-8

# POP Tracer Budget: compute tracer budget terms

import os 
import xarray as xr
import numpy as np
from glob import glob
from dask.diagnostics import ProgressBar

#*****************************************************************************#
# use to sort by suffix (date)
def get_suffix(x):
    return x[-17:]

# devolve lista ordenada pelo label de tempo do arquivo
def get_filelist (basedir, scenario, freq, realm, varname):
    filelist = glob(basedir + "/" + scenario + "/" + realm + "/" +\
                    freq + "/" + varname + "/b.e11." + scenario + "*.nc")
    return (sorted(filelist,key=get_suffix))

# devolve um arquivo/lista de arquivos de uma variavel e ens_member
def get_filemember (basedir, scenario, freq, realm, ens_member, varname):
    return sorted(glob(basedir + "/" + scenario + "/" + realm + "/" + freq + "/" + \
                varname + "/b.e11." + scenario + ".f09_g16." + ens_member + ".pop.h.*.nc"), key=get_suffix)

# recebe um arquivo/lista de arquivos e devolve um dataset
def read_cesm_pop (file, chunk_sz):
    return xr.open_mfdataset(file,decode_times=False,mask_and_scale=True,\
                             concat_dim="time",data_vars="minimal",chunks={'time': chunk_sz})
#*****************************************************************************#
# decode time axis (cesm/pop)
def pop_decode_time (var): 
    varname = var.name
    time = var.time
    time.values = time.values - 16
    #var = var.assign_coords(time=time)
    ds = xr.decode_cf(var.to_dataset(),decode_times=True)
    time.values = time.values + 16
    return ds[varname]

# remove a media climatologica de var
def rmMonAnnCyc (var):
    climatology = var.groupby('time.month').mean('time')
    anomalies = (var.groupby("time.month") - climatology).squeeze()
    return anomalies.drop("month")

# devolve o indice referente ao ano inicial e ao ano final
def get_time_index (time, initial_year, final_year):
    aux = time
    aux.name = "t"
    t = pop_decode_time(aux)
    yr_i = (t.isel(time=0).values).all().year
    yr_f = (t.isel(time=-1).values).all().year
    #print("Initial year: " + str(yr_i))
    #print("Final year:" + str(yr_f))
    idx_i = (initial_year - yr_i)*12
    #idx_f = t.size - (yr_f - final_year)*12
    idx_f = idx_i + (final_year - initial_year + 1)*12
    del(t)
    return idx_i,idx_f

#*****************************************************************************#
def tracer_budget_vol3d (tarea, dz, kmt):
    """
    Arguments: cell area, cell height, max vertical indx
    Returns global 3D volume DataArray: vold3(nz,ny,nx) dtype=float64
    NOTE: does not include SSH variations
    """
    vol3d = (dz*tarea.astype('float64')).load()
    for i in range(dz.shape[0]):
        vol3d[i,:,:] = vol3d[i].where(kmt > i, 0.)
    vol3d.attrs = {'units' : 'cm3', 'long_name' : 'Tcell volume'}
    vol3d = vol3d.drop(('ULAT','ULONG'))
    vol3d.name = "vol3d"
    return vol3d

#*****************************************************************************#
def tracer_budget_mask2d (region_mask, sel_area = 0):
    """
    Return surface mask: if ocean than 1 else nan
    """
    mask = region_mask
    mask = mask.where(region_mask != sel_area,np.nan)
    return (mask/mask)

#*****************************************************************************#
def tracer_budget_mask3d (var3d):
    """
    Return volume mask: if ocean than 1 else nan
    """
    mask3d = var3d/var3d
    mask3d.attrs = {'units' : '1 / np.nan', 'long_name' : 'mask3d'}
    return mask3d.where(mask3d != 0.,np.nan)

#*****************************************************************************#
def tracer_budget_var3d_zint_map (tracer, vol3d, klo=0, khi=25):
    """
    Arguments: var4d tracer(t,z,y,x), vol3d cell volume, 
               klo : lowest k index, khi : highest k index
    Returns a 2d tracer map vertical integrated
    """
    units = tracer.units + " cm^3"
    description = "Int_V {" + tracer.name + "} dV"
    long_name = tracer.name + " vertical average"
    attr = {"long_name" : long_name, "units" : units, "description": description, \
            "k_range" : str(klo)+" - "+str(khi)}
    var = tracer.isel(z_t=slice(klo,khi)) * vol3d.isel(z_t=slice(klo,khi))
    var_zint_map = var.sum(dim="z_t")
    var_zint_map.attrs = attr
    var_zint_map.name = tracer.name + "_zint" 
    var_zint_map = var_zint_map.drop(("ULONG","ULAT"))
    return var_zint_map.where(var_zint_map != 0.)

#*****************************************************************************#
def tracer_budget_tend_appr (TRACER, time_bnd, var_zint):
    """
    Computes approximate TRACER budget tendency given vertically-integrated POP
    TRACER based on differencing successive monthly means
    NOTE: Assumes monthly POP output with timestamp at end-of-month
          rather than mid-month; assumes time has dimension "days".
    Obs: from tracer_budget_util.ncl
    """
    secperday = 60.*60*24
    # days in each month * sec/day 
    dt = (time_bnd.isel(d2=1) - time_bnd.isel(d2=0))*secperday
    units = var_zint.units + "/s"
    
    long_name = var_zint.long_name + " tendency"
    attr = {"long_name" : long_name, "units" : units}
    
    # apprx to end of month 
    # X = [X_t + X_(t+1)]/2
    X = (var_zint + var_zint.shift(time=-1))*0.5
    
    # X = X_t - X_(t-1)
    dX = X - X.shift(time=1)
    
    #units per seconds
    var_zint_tend = dX/dt
    var_zint_tend.attrs = attr
    var_zint_tend.name = TRACER.lower() + "_tend"
    return var_zint_tend

#*****************************************************************************#
def tracer_budget_lat_adv_resolved (f_ue, f_vn, TRACER, vol3d, \
                                    klo=0, khi=25, tlo=490, thi=610, z_int=True):
    """
    compute tracer lateral advection integral 
    based on tracer_budget_adv.ncl
    """
    if TRACER == "TEMP":
        var_name1 = "UET"
        var_name2 = "VNT"
        units = "degC cm^3/s"
    else:
        var_name1 = "UES"
        var_name2 = "VNS"
        units = "PSU cm^3/s"
        
    long_name = "lateral advective flux (resolved)"
    description = "Int_z{-Div[<"+var_name1+">, <"+var_name2+">]}"
    
    # read tracer associate variable
    ds1 = read_cesm_pop (f_ue, 60)
    ds2 = read_cesm_pop (f_vn, 60)
    ue = (ds1[var_name1]).isel(z_t=slice(klo,khi),time=slice(tlo,thi))
    vn = (ds2[var_name2]).isel(z_t=slice(klo,khi),time=slice(tlo,thi))
    zlo = (ds1["z_w"]).isel(z_w=klo).values
    zhi = (ds1["z_w"]).isel(z_w=(khi)).values
    attr = {"long_name" : long_name, "units" : units, "description" : description,\
            "k_range" : str(klo)+" - "+str(khi),\
            "depth_range" :  "{0:3.2f} - {1:3.2f} m".format((zlo/100),(zhi/100))}
    # vol3d
    vol = vol3d.isel(z_t=slice(klo,khi))
    # e.g.: degC cm^3/s
    ue = ue*vol  # Tcell_(i,j)
    vn = vn*vol  # Tcell_(i,j)
    # shift grid:
    uw = ue.roll(nlon=1,roll_coords=False) # Tcell_(i-1,j)
    vs = vn.roll(nlat=1,roll_coords=False) # Tcell_(i,j-1)
    # Div [du/dx + dv/dy]
    hdiv = (uw-ue) + (vs-vn)
    # vertical integration
    if z_int:
        var_lat_adv_res_map = hdiv.sum(dim="z_t",keep_attrs=True)
    else:
        var_lat_adv_res_map = hdiv
    var_lat_adv_res_map.attrs = attr
    var_lat_adv_res_map.name = TRACER.lower() + "_lat_adv_res"
    var_lat_adv_res_map = var_lat_adv_res_map.drop(("ULONG","ULAT"))
    return var_lat_adv_res_map.where(var_lat_adv_res_map != 0.)

#*****************************************************************************#
def tracer_budget_vert_adv_resolved (f_wt, TRACER, vol3d,\
                                     klo=0, khi=25, tlo=912, thi=1032, z_int=True):
    """
    tracer vertical advection integral
    Obs. klo==0 -> wtt=0. => klo=1; khi => khi+1 (max:=59)
    """
    if TRACER == "TEMP":
        var_name = "WTT"
        units = "degC cm^3/s"
    else:
        var_name = "WTS"
        units = "PSU cm^3/s"
    
    long_name = "vertical advective flux (resolved)"
    description = "Int_z{-d[<"+var_name+">]/dz}"
    attr = {"long_name" : long_name, "units" : units, "description" : description,\
            "k_range" : str(klo)+" - "+str(khi)}
    # read tracer associate variable
    ds1 = read_cesm_pop (f_wt, 60)
    wt = ds1[var_name].isel(time=slice(tlo,thi))
    wt = wt.rename({"z_w_top" : "z_t"})
    wt["z_t"] = vol3d.z_t
    # e.g. degC cm^3/s
    wt = wt*vol3d
    if z_int:
        var_top = wt.isel(z_t=klo)
        var_bottom = wt.isel(z_t=khi+1)
        # since it has NaN (NOTE: carefull with zeros)
        var_bottom = var_bottom.where(~np.isnan(var_bottom),0.)
        # vertical convergence
        var_vert_adv_res_map = (var_bottom - var_top)
    else:
        wt_z  = wt.shift(z_t=-1).fillna(0.)
        var_vert_adv_res_map = wt_z.isel(z_t=slice(klo,khi)) - wt.isel(z_t=slice(klo,khi))

    var_vert_adv_res_map.attrs = attr
    var_vert_adv_res_map.name = TRACER.lower()+"_vert_adv_res"
    var_vert_adv_res_map = var_vert_adv_res_map.drop(("ULONG","ULAT"))
    return var_vert_adv_res_map

#*****************************************************************************#
def tracer_budget_hmix (f_e, f_n, TRACER, vol3d,\
                        klo=0, khi=25, tlo=912, thi=1032, z_int=True):
    """
    tracer horizontal mixing
    compute tracer hmix integrals from Horiz Diffusive Fluxes
    vertical fluxes are positive up
    """
    if TRACER == "TEMP":
        units = "degC cm^3/s"
    else:
        units = "PSU cm^3/s"
    #diffusive flux variable names
    var_name1 = "HDIFE_"+TRACER
    var_name2 = "HDIFN_"+TRACER
    
    long_name = "lateral diffusive flux (resolved)"
    description = "Int_z{-Div[<"+var_name1+">, <"+var_name2+">]}"
    # read tracer associate variable
    ds1 = read_cesm_pop (f_e, 60) 
    ds2 = read_cesm_pop (f_n, 60) 
    
    ue = (ds1[var_name1]).isel(z_t=slice(klo,khi),time=slice(tlo,thi))
    vn = (ds2[var_name2]).isel(z_t=slice(klo,khi),time=slice(tlo,thi))
    zlo = (ds1["z_w"]).isel(z_w=klo).values
    zhi = (ds1["z_w"]).isel(z_w=(khi)).values
    
    attr = {"long_name" : long_name, "units" : units, "description" : description,\
            "k_range" : str(klo)+" - "+str(khi),\
            "depth_range" :  "{0:3.2f} - {1:3.2f} m".format((zlo/100),(zhi/100))}
    # vol3d
    vol = vol3d.isel(z_t=slice(klo,khi))
    # e.g.: degC cm^3/s
    ue = ue*vol  # Tcell_(i,j)
    vn = vn*vol  # Tcell_(i,j)
    
    # shift
    uw = ue.roll(nlon=1,roll_coords=False)  # Tcell_(i-1,j)
    vs = vn.roll(nlat=1,roll_coords=False)  # Tcell_(i,j-1)
    
    # Divergence
    hdiv = (uw-ue) + (vs-vn)
    #hdiv = (ue-uw) + (vn-vs)
    # copy coordinates
    #hdiv = hdiv.assign_coords(TLAT=ue.coords.get("TLAT"))
    # vertical integration
    if z_int:
        var_lat_mix_res_map = hdiv.sum(dim="z_t")
    else:
        var_lat_mix_res_map = hdiv
    var_lat_mix_res_map.attrs = attr
    var_lat_mix_res_map.name = TRACER.lower() + "_lat_mix_res"
    var_lat_mix_res_map = var_lat_mix_res_map.drop(("ULONG","ULAT"))
    return var_lat_mix_res_map.where(var_lat_mix_res_map != 0.)

#*****************************************************************************#
def tracer_budget_dia_vmix (f_dia, TRACER, tarea, kmt, \
                            klo=0, khi=25, tlo=912, thi=1032, z_int=True):
    """
    Computes vertical integral of diabatic vertical mixing (DIA_IMPVF_), ie. KPP
    """
    if TRACER == "TEMP":
        units = "degC cm^3/s"
    else:
        units = "PSU cm^3/s"
    #variable name
    var_name = "DIA_IMPVF_"+TRACER
    
    long_name = "vertical (diabatic) mixing flux (resolved)"
    description = "Int_z{-d[<"+var_name+">]/dz}" 
    attr = {"long_name" : long_name, "units" : units, "description" : description,\
            "k_range" : str(klo)+" - "+str(khi)}
    
    # read tracer associate variable
    ds = read_cesm_pop (f_dia, 60)
    FIELD = ds[var_name].isel(time=slice(tlo,thi)) # degC cm/s
    FIELD = FIELD.rename({"z_w_bot" : "z_t"})
    FIELD["z_t"] = ds["z_t"]
    # degC cm^3/s
    FIELD = FIELD*tarea
    if z_int:
        # zero diffusive flux across sea surface -> 0 
        FIELD_TOP = FIELD.isel(z_w_bot=klo)
        FIELD_BOT = FIELD.isel(z_w_bot=khi)
        #tarea_bot = tarea.where(kmt > khi,0.)
        #tarea_top = tarea.where(kmt > klo,0.)
        #
        FIELD_BOT = FIELD_BOT.fillna(0.)
        var_vert_mix_map = -(FIELD_BOT - FIELD_TOP)
    else:
        diadiff_z = FIELD.shift(z_t=1).fillna(0.)
        var_vert_mix_map = diadiff_z.isel(z_t=slice(klo,khi)) - FIELD.isel(z_t=slice(klo,khi))

    var_vert_mix_map.name = TRACER.lower() + "_dia_vmix"
    var_vert_mix_map.attrs = attr
    var_vert_mix_map = var_vert_mix_map.drop(("ULONG","ULAT"))
    return var_vert_mix_map.where(var_vert_mix_map != 0.)

#*****************************************************************************#
def tracer_budget_adi_vmix (f_adi, TRACER, vol3d, \
                            klo=0, khi=25, tlo=912, thi=1032, z_int=True):
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
    attr = {"long_name" : long_name, "units" : units, "description" : description,\
            "k_range" : str(klo)+" - "+str(khi)}
    # read tracer associate variable
    ds = read_cesm_pop (f_adi, 60)
    FIELD = ds[var_name].isel(time=slice(tlo,thi)) # degC/s
    FIELD = FIELD.rename({"z_w_bot" : "z_t"})
    FIELD["z_t"] = vol3d.z_t
    FIELD = FIELD*vol3d
    # zero diffusive flux across sea surface -> 0 
    FIELD_TOP = FIELD.isel(z_t=klo)
    FIELD_BOT = FIELD.isel(z_t=khi)
    #
    if z_int:
        var_vert_mix_map = -(FIELD_BOT.fillna(0.) - FIELD_TOP)
    else:
        hdifb_z = FIELD.shift(z_t=1).fillna(0.)
        var_vert_mix_map = hdifb_z.isel(z_t=slice(klo,khi)) - FIELD.isel(z_t=slice(klo,khi)).fillna(0.)

    var_vert_mix_map.attrs = attr
    var_vert_mix_map.name = TRACER.lower() + "_adi_vmix"
    var_vert_mix_map = var_vert_mix_map.drop(("ULONG","ULAT"))
    return var_vert_mix_map.where(var_vert_mix_map != 0.)

#*****************************************************************************#
def tracer_budget_sflux (f_flx, TRACER, var_name, area2d, tlo=912, thi=1032):
    """
    compute domain-specific maps of tracer surface fluxes
    
    Note: fluxes positive are down!
 
    based on tracer_budget_srf_flux.ncl
    """
    # read tracer associate variable
    ds = read_cesm_pop (f_flx, 60)
    rho_sw = ds["rho_sw"]              # density of saltwater (g/cm^3)
    rho_sw = rho_sw * 1.e-3             # (kg/cm^3)
    cp_sw = ds["cp_sw"]                # spec. heat of saltwater (erg/g/K)
    cp_sw = cp_sw * 1.e-7 * 1.e3        # (J/kg/K)
    rho_cp = rho_sw * cp_sw             # (J/cm^3/K)
    latvap = ds["latent_heat_vapor"]   # lat heat of vaporiz. (J/kg)
    latfus = ds["latent_heat_fusion"]  # lat heat of fusion (erg/g)
    latfus = latfus * 1.e-7 * 1.e3      # (J/kg)
    # scale factors
    if var_name in ["SHF", "QFLUX", "SENH_F", "LWDN_F", "LWUP_F", "SHF_QSW", "MELTH_F"]:
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
        
    FIELD = ds[var_name].isel(time=slice(tlo,thi))
    var1 = FIELD * scale_factor
    var_sflux_map = var1*area2d
    long_name = "vertical flux across sea surface"
    attr = {"long_name" : long_name, "units" : units}
    var_sflux_map.attrs = attr
    var_sflux_map.name = TRACER.lower() + "_" + var_name 
    var_sflux_map = var_sflux_map.drop(("ULONG","ULAT"))
    return var_sflux_map

#*****************************************************************************#
def tracer_budget_kpp_src (f_kpp, TRACER, vol3d,\
                           klo=1, khi=25, tlo=912, thi=1032):
    """
    compute tendency from KPP non local mixing term
    """
    var_name = "KPP_SRC_"+TRACER
    # read tracer associate variable
    ds = read_cesm_pop (f_kpp, 60)
    KPP_SRC = ds[var_name].isel(time=slice(tlo,thi))
    #KPP_SRC = KPP_SRC.where(KPP_SRC != 0.)
    # compute temp flux
    temp_kpp_src = tracer_budget_var3d_zint_map(KPP_SRC,vol3d,klo,khi)
    return temp_kpp_src

#*****************************************************************************#
