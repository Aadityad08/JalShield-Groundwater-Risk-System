#!/usr/bin/env python3
"""Fuse pump sensor data with approximate geospatial and water-infrastructure context.

Reference datasets expected in Datasets(1).zip (the user's uploaded archive):
- Jal Dharohar KML water-body locations (used as nearby infrastructure proxies,
  NOT asserted to be exact pump locations)
- Tamil Nadu groundwater-level station CSVs (latest available telemetry reading,
  with quarterly manual data as fallback)
- FHTCDataTable JJM district coverage figures transcribed from the uploaded report.

If sensor.csv is absent, a synthetic sensor dataset is generated so the pipeline
can run end-to-end. Synthetic pump sensor rows/status labels remain synthetic.

Usage:
  python data_fusion.py
  python data_fusion.py /path/to/sensor.csv --reference-zip /path/to/Datasets(1).zip
  python data_fusion.py --synthetic-rows 1000
"""
from __future__ import annotations
import argparse, math, re, zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

DISTRICTS = [
    "Ariyalur", "Chengalpattu", "Chennai", "Coimbatore", "Cuddalore",
    "Dharmapuri", "Dindigul", "Erode", "Kallakurichi", "Kancheepuram",
    "Kanniyakumari", "Karur", "Krishnagiri", "Madurai", "Mayiladuthurai",
    "Nagapattinam", "Namakkal", "Nilgiris", "Perambalur", "Pudukkottai",
    "Ramanathapuram", "Ranipet", "Salem", "Sivaganga", "Tenkasi",
    "Thanjavur", "Theni", "Thoothukudi", "Tiruchirappalli", "Tirunelveli",
    "Tirupathur", "Tiruppur", "Tiruvallur", "Tiruvannamalai", "Tiruvarur",
    "Vellore", "Viluppuram", "Virudhunagar",
]
STATUS_MAP = {"NORMAL": 0, "RECOVERING": 1, "BROKEN": 2}

# Household counts transcribed from the uploaded FHTCDataTable.pdf (printed 10 Oct 2026).
# Chennai is not listed in that PDF; it is deliberately left missing rather than invented.
JJM = {
 "Ariyalur":(207503,207503,100.00), "Coimbatore":(372578,372578,100.00),
 "Cuddalore":(522440,522440,100.00), "Kancheepuram":(216311,216311,100.00),
 "Kanniyakumari":(216738,216738,100.00), "Mayiladuthurai":(202551,202551,100.00),
 "Namakkal":(362228,362228,100.00), "Ranipet":(189334,189334,100.00),
 "Nilgiris":(96916,96916,100.00), "Theni":(185013,185013,100.00),
 "Tiruvallur":(475915,475915,100.00), "Tirunelveli":(280241,280241,100.00),
 "Tirupathur":(212012,212012,100.00), "Tiruvannamalai":(530392,530392,100.00),
 "Vellore":(212528,212528,100.00), "Viluppuram":(437317,437317,100.00),
 "Chengalpattu":(415800,415800,100.00), "Thanjavur":(421955,421955,100.00),
 "Madurai":(448939,442054,98.47), "Tiruchirappalli":(472403,456267,96.58),
 "Tiruppur":(456206,433944,95.12), "Dindigul":(452054,427395,94.55),
 "Erode":(418973,393680,93.96), "Virudhunagar":(468095,439753,93.95),
 "Perambalur":(146735,136787,93.22), "Karur":(204464,189167,92.52),
 "Salem":(645738,592571,91.77), "Thoothukudi":(365524,322464,88.22),
 "Nagapattinam":(157116,129467,82.40), "Tiruvarur":(304730,248958,81.70),
 "Kallakurichi":(301647,239784,79.49), "Krishnagiri":(409360,324686,79.32),
 "Pudukkottai":(367886,287800,78.23), "Sivaganga":(327234,254004,77.62),
 "Tenkasi":(342341,259185,75.71), "Dharmapuri":(342782,209368,61.08),
 "Ramanathapuram":(332191,158752,47.79),
}
# Name harmonization between district datasets and present district names.
def norm_district(x: object) -> str:
    s = re.sub(r"\s+", " ", str(x or "").strip()).casefold()
    aliases = {
      "the nilgiris":"Nilgiris", "nilgiri":"Nilgiris", "kanyakumari":"Kanniyakumari",
      "kanniyakumari":"Kanniyakumari", "villuppuram":"Viluppuram", "viluppuram":"Viluppuram",
      "thoothukkudi":"Thoothukudi", "thoothukudi":"Thoothukudi", "thiruvallur":"Tiruvallur",
      "thiruvannamalai":"Tiruvannamalai", "thirunelveli":"Tirunelveli",
      "thiruchirappalli":"Tiruchirappalli", "tiruchirapalli":"Tiruchirappalli",
      "thiruvarur":"Tiruvarur", "kanchipuram":"Kancheepuram", "ramanathapuram":"Ramanathapuram",
      "sivagangai":"Sivaganga", "tirupathur":"Tirupathur", "tiruppur":"Tiruppur",
      "madras":"Chennai", "nilgiris":"Nilgiris"
    }
    return aliases.get(s, s.title())


def create_synthetic_sensor_data(rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    if rows < 1: raise ValueError("synthetic_rows must be at least 1")
    rng = np.random.default_rng(seed)
    data = {f"sensor_{i:02d}": rng.normal(50+i, 10+i*0.1, rows) for i in range(52) if i != 15}
    data["sensor_04"] = rng.lognormal(2.0, 0.45, rows)
    data["sensor_10"] = rng.normal(55, 8, rows)
    data["machine_status"] = rng.choice(["NORMAL","RECOVERING","BROKEN"], rows, p=[.88,.09,.03])
    data["timestamp"] = pd.date_range("2025-01-01", periods=rows, freq="h").astype(str)
    data["sensor_15"] = rng.normal(size=rows); data["Unnamed: 0"] = np.arange(rows)
    df = pd.DataFrame(data)
    for c in df.select_dtypes(include=[np.number]).columns:
        df.loc[rng.random(rows)<.01,c] = np.nan
    return df


def load_sensor_data(path: Path, rows: int, seed: int):
    if path.exists():
        df = pd.read_csv(path)
        if df.empty: raise ValueError(f"Input dataset contains no rows: {path}")
        return df, False
    df = create_synthetic_sensor_data(rows, seed)
    out = path if path.name.lower() != "sensor.csv" else path.with_name("sensor_synthetic_generated.csv")
    out.parent.mkdir(parents=True, exist_ok=True); df.to_csv(out,index=False)
    print(f"NOTE: {path} not found; generated synthetic sensor source: {out}")
    return df, True


def find_reference_zip(explicit: str | None, script_dir: Path) -> Path | None:
    candidates = [Path(explicit)] if explicit else []
    candidates += [script_dir/"Datasets(1).zip", script_dir/"Datasets.zip", Path.cwd()/"Datasets(1).zip", Path.cwd()/"Datasets.zip"]
    return next((p for p in candidates if p.exists()), None)


def read_kml_locations(zf: zipfile.ZipFile) -> pd.DataFrame:
    rows=[]
    names=[n for n in zf.namelist() if n.lower().endswith('.kml')]
    for name in names:
        try:
            with zf.open(name) as fh:
                for _, el in ET.iterparse(fh, events=("end",)):
                    if el.tag.endswith("Placemark"):
                        props={}
                        for node in el.iter():
                            if node.tag.endswith("SimpleData") or node.tag.endswith("Data"):
                                key=node.attrib.get("name","")
                                val=node.text
                                if not val:
                                    valnode=next((c for c in node if c.tag.endswith("value")),None)
                                    val=valnode.text if valnode is not None else None
                                if key and val is not None: props[key]=val
                        coord=None
                        for node in el.iter():
                            if node.tag.endswith("coordinates") and node.text:
                                coord=node.text.strip().split()[0].split(','); break
                        district=norm_district(props.get('district_nwic') or props.get('district') or props.get('district_iw'))
                        if coord and len(coord)>=2 and district in DISTRICTS:
                            try: lon,lat=float(coord[0]),float(coord[1])
                            except ValueError: lon=lat=np.nan
                            if 76.0 <= lon <= 81.0 and 7.5 <= lat <= 14.0:
                                rows.append({"district":district,"latitude":lat,"longitude":lon,
                                  "waterbody_name":props.get('village_nwic') or props.get('village') or '',
                                  "waterbody_type":props.get('water_body_type',''),
                                  "waterbody_id":props.get('unique_id') or props.get('objectid',''),
                                  "location_source":"Jal Dharohar water-body coordinate (proxy; not a pump coordinate)",
                                  "location_dataset":Path(name).name})
                        el.clear()
        except ET.ParseError as e: print(f"WARNING: could not parse {name}: {e}")
    return pd.DataFrame(rows).drop_duplicates(subset=['district','latitude','longitude']) if rows else pd.DataFrame()


def read_groundwater_stations(zf: zipfile.ZipFile) -> pd.DataFrame:
    # Prefer the 2021-2025 six-hourly telemetry dataset; process in chunks to limit RAM.
    telemetry=next((n for n in zf.namelist() if 'gwl_tel_6_hourly_tamil_nadu' in n.lower() and n.lower().endswith('.csv')),None)
    manual=next((n for n in zf.namelist() if 'gwl_manual_quarterly_tamil-nadu' in n.lower() and n.lower().endswith('.csv')),None)
    result=pd.DataFrame()
    if telemetry:
        cols=['Station','District','Latitude','Longitude','Data Acquisition Time','Groundwater Level Telemetry 6 Hourly (meter)']
        latest={}
        try:
            with zf.open(telemetry) as fh:
                for chunk in pd.read_csv(fh,usecols=lambda c:c in cols,chunksize=250000,low_memory=False):
                    chunk['obs_time']=pd.to_datetime(chunk['Data Acquisition Time'],dayfirst=True,errors='coerce')
                    chunk['gw_level_m']=pd.to_numeric(chunk['Groundwater Level Telemetry 6 Hourly (meter)'],errors='coerce')
                    chunk['latitude']=pd.to_numeric(chunk['Latitude'],errors='coerce'); chunk['longitude']=pd.to_numeric(chunk['Longitude'],errors='coerce')
                    chunk['district']=chunk['District'].map(norm_district)
                    chunk=chunk.dropna(subset=['Station','obs_time','gw_level_m','latitude','longitude'])
                    chunk=chunk[(chunk['district'].isin(DISTRICTS)) & chunk['latitude'].between(7.5,14.0) & chunk['longitude'].between(76,81)]
                    if chunk.empty: continue
                    chunk=chunk.sort_values('obs_time').drop_duplicates(subset=['Station','latitude','longitude'],keep='last')
                    for r in chunk[['Station','district','latitude','longitude','obs_time','gw_level_m']].to_dict('records'):
                        key=(str(r['Station']).strip(),round(float(r['latitude']),5),round(float(r['longitude']),5))
                        old=latest.get(key)
                        if old is None or r['obs_time']>old['obs_time']: latest[key]=r
            result=pd.DataFrame(latest.values())
        except Exception as e: print(f"WARNING: telemetry groundwater CSV could not be read ({e}); trying manual data.")
    if result.empty and manual:
        cols=['Station','District','Latitude','Longitude','Data Acquisition Time','Groundwater Level Quarterly Manual (meter)']
        try:
            with zf.open(manual) as fh:
                df=pd.read_csv(fh,usecols=lambda c:c in cols,low_memory=False)
            df['obs_time']=pd.to_datetime(df['Data Acquisition Time'],dayfirst=True,errors='coerce')
            df['gw_level_m']=pd.to_numeric(df['Groundwater Level Quarterly Manual (meter)'],errors='coerce')
            df['latitude']=pd.to_numeric(df['Latitude'],errors='coerce'); df['longitude']=pd.to_numeric(df['Longitude'],errors='coerce')
            df['district']=df['District'].map(norm_district)
            df=df.dropna(subset=['Station','obs_time','gw_level_m','latitude','longitude'])
            df=df[(df['district'].isin(DISTRICTS)) & df['latitude'].between(7.5,14) & df['longitude'].between(76,81)]
            result=df.sort_values('obs_time').drop_duplicates(['Station','latitude','longitude'],keep='last')[['Station','district','latitude','longitude','obs_time','gw_level_m']]
        except Exception as e: print(f"WARNING: manual groundwater CSV could not be read: {e}")
    if not result.empty:
        result=result.rename(columns={'Station':'groundwater_station','district':'gw_station_district','latitude':'gw_station_latitude','longitude':'gw_station_longitude','obs_time':'groundwater_observation_time','gw_level_m':'groundwater_level_m'})
    return result.reset_index(drop=True)


def haversine_km(lat1, lon1, lat2, lon2):
    rad=np.pi/180.0
    dlat=(np.asarray(lat2)-lat1)*rad; dlon=(np.asarray(lon2)-lon1)*rad
    a=np.sin(dlat/2)**2+np.cos(lat1*rad)*np.cos(np.asarray(lat2)*rad)*np.sin(dlon/2)**2
    return 6371.0088*2*np.arcsin(np.sqrt(np.clip(a,0,1)))


def attach_locations(df: pd.DataFrame, zf: zipfile.ZipFile | None, seed: int) -> pd.DataFrame:
    n=len(df); rng=np.random.default_rng(seed+10)
    locs=read_kml_locations(zf) if zf else pd.DataFrame()
    stations=read_groundwater_stations(zf) if zf else pd.DataFrame()
    # If real mapped water-body coordinates exist, sample one within the assigned district.
    df['latitude']=np.nan; df['longitude']=np.nan; df['waterbody_name']=''; df['waterbody_type']=''; df['waterbody_id']=''
    df['location_source']='unavailable; no reference KML found'
    if not locs.empty:
        bydist={d:g.reset_index(drop=True) for d,g in locs.groupby('district')}
        for district, idx in df.groupby('district').groups.items():
            candidates=bydist.get(district)
            if candidates is None or candidates.empty:
                continue
            chosen=rng.integers(0,len(candidates),size=len(idx))
            df.loc[idx,'latitude']=candidates.iloc[chosen]['latitude'].to_numpy()
            df.loc[idx,'longitude']=candidates.iloc[chosen]['longitude'].to_numpy()
            for col in ['waterbody_name','waterbody_type','waterbody_id','location_source']:
                df.loc[idx,col]=candidates.iloc[chosen][col].fillna('').to_numpy() if col!='location_source' else candidates.iloc[chosen][col].to_numpy()
        missing=df['latitude'].isna()
        # Fallback for districts without KML point coverage: use groundwater station coordinate if available.
        if missing.any() and not stations.empty:
            for idx in df.index[missing]:
                d=df.at[idx,'district']; cand=stations[stations['gw_station_district']==d]
                if cand.empty: cand=stations
                if cand.empty: continue
                # For no known water-body location, use a groundwater station itself and label it explicitly.
                row=cand.iloc[0]
                df.at[idx,'latitude']=row['gw_station_latitude']; df.at[idx,'longitude']=row['gw_station_longitude']
                df.at[idx,'location_source']='groundwater monitoring station coordinate (proxy; not a pump coordinate)'
    elif not stations.empty:
        # No KML found: assign actual station coordinates, explicitly labelled as station proxies.
        for district, idx in df.groupby('district').groups.items():
            cand=stations[stations['gw_station_district']==district]
            if cand.empty: cand=stations
            for j in idx:
                row=cand.iloc[(j+seed)%len(cand)]
                df.at[j,'latitude']=row['gw_station_latitude']; df.at[j,'longitude']=row['gw_station_longitude']
                df.at[j,'location_source']='groundwater monitoring station coordinate (proxy; not a pump coordinate)'
    # District-level JJM household coverage metrics.
    df['jjm_total_households']=df['district'].map({d:v[0] for d,v in JJM.items()}).astype('Float64')
    df['jjm_households_with_tap']=df['district'].map({d:v[1] for d,v in JJM.items()}).astype('Float64')
    df['jjm_tap_coverage_pct']=df['district'].map({d:v[2] for d,v in JJM.items()}).astype('Float64')
    df['jjm_households_without_tap']=df['jjm_total_households']-df['jjm_households_with_tap']
    df['jjm_data_source']='FHTCDataTable.pdf uploaded by user; printed 2026-10-10; district-level'
    df.loc[df['jjm_total_households'].isna(),'jjm_data_source']='not reported in uploaded FHTCDataTable.pdf'
    # Attach nearest groundwater station; prioritize same district, then fall back to nearest statewide station.
    for c in ['groundwater_station','gw_station_district','groundwater_observation_time']:
        df[c]=pd.Series([None]*len(df), index=df.index, dtype='object')
    for c in ['groundwater_level_m','gw_station_latitude','gw_station_longitude','groundwater_station_distance_km']:
        df[c]=np.nan
    if not stations.empty:
        for district, idx in df.groupby('district').groups.items():
            same=stations[stations['gw_station_district']==district]
            for j in idx:
                lat,lon=df.at[j,'latitude'],df.at[j,'longitude']
                if pd.isna(lat) or pd.isna(lon): continue
                pool=same if not same.empty else stations
                distances=haversine_km(float(lat),float(lon),pool['gw_station_latitude'].to_numpy(float),pool['gw_station_longitude'].to_numpy(float))
                k=int(np.argmin(distances)); r=pool.iloc[k]
                df.at[j,'groundwater_station']=str(r['groundwater_station'])
                df.at[j,'gw_station_district']=str(r['gw_station_district'])
                df.at[j,'groundwater_observation_time']=str(pd.Timestamp(r['groundwater_observation_time']))
                df.at[j,'groundwater_level_m']=float(r['groundwater_level_m'])
                df.at[j,'gw_station_latitude']=float(r['gw_station_latitude']); df.at[j,'gw_station_longitude']=float(r['gw_station_longitude'])
                df.at[j,'groundwater_station_distance_km']=round(float(distances[k]),3)
        df['groundwater_level_source']='uploaded Tamil Nadu groundwater-level station data; latest available station reading'
    else:
        df['groundwater_level_source']='not available: groundwater reference CSV not read'
    return df, len(locs), len(stations)


def fuse_data(input_path: Path, output_path: Path, seed: int=42, synthetic_rows: int=1000,
              reference_zip: Path | None=None) -> pd.DataFrame:
    df, is_synthetic=load_sensor_data(input_path,synthetic_rows,seed)
    df=df.drop(columns=['sensor_15','Unnamed: 0'],errors='ignore')
    num=df.select_dtypes(include=[np.number]).columns.tolist()
    if num: df[num]=SimpleImputer(strategy='median').fit_transform(df[num])
    if 'sensor_04' not in df:
        candidates=[c for c in df if c.startswith('sensor_')]
        df['sensor_04']=pd.to_numeric(df[candidates[0]],errors='coerce') if candidates else np.nan
    if 'sensor_10_motor_temp' not in df:
        df['sensor_10_motor_temp']=pd.to_numeric(df['sensor_10'],errors='coerce') if 'sensor_10' in df else np.random.default_rng(seed+1).normal(55,8,len(df))
    if 'machine_status' not in df:
        df['machine_status']=np.random.default_rng(seed+2).choice(['NORMAL','RECOVERING','BROKEN'],len(df),p=[.88,.09,.03])
        print('NOTE: machine_status absent; synthetic labels generated.')
    num=df.select_dtypes(include=[np.number]).columns.tolist()
    if num: df[num]=SimpleImputer(strategy='median').fit_transform(df[num])
    rng=np.random.default_rng(seed); n=len(df)
    # Distribute rows evenly over districts so the prototype can map all 38 districts.
    df['district']=[DISTRICTS[i % len(DISTRICTS)] for i in range(n)]
    df['population_served']=rng.integers(150,5001,size=n)
    # These three chemistry fields remain simulated; groundwater-level observations below are sourced from uploaded data.
    df['Fluoride_Level']=rng.uniform(.2,3.5,n).round(3)
    df['Nitrate_Level']=rng.uniform(5,60,n).round(3)
    df['TDS']=rng.uniform(150,2500,n).round(2)
    chem=df[['Fluoride_Level','Nitrate_Level','TDS']].to_numpy()
    scaled=MinMaxScaler().fit_transform(chem)
    pollution=.40*scaled[:,0]+.35*scaled[:,1]+.25*scaled[:,2]
    wqi=100*(1-pollution)-20*(df['Fluoride_Level'].to_numpy()>1.5)-20*(df['Nitrate_Level'].to_numpy()>45)
    df['WQI']=np.clip(wqi,0,100).round(2)
    temp=pd.to_numeric(df['sensor_10_motor_temp'],errors='coerce').fillna(55)
    df['CCI']=((df['TDS']/1000)*(temp/40)).round(4)
    flow=pd.to_numeric(df['sensor_04'],errors='coerce').fillna(0)
    fmin,fmax=float(flow.min()),float(flow.max())
    df['lpcd_draw']=55.0 if np.isclose(fmin,fmax) else (30+(flow-fmin)/(fmax-fmin)*50).round(2)
    df['residual_chlorine']=rng.uniform(.05,.8,n).round(3)
    statuses=df['machine_status'].astype('string').str.strip().str.upper()
    unknown=sorted(set(statuses.dropna().unique())-set(STATUS_MAP))
    if unknown: raise ValueError('Unexpected machine_status values: '+', '.join(unknown))
    if statuses.isna().any(): raise ValueError('machine_status contains missing values.')
    df['machine_status']=statuses; df['status_group']=statuses.map(STATUS_MAP).astype('int64')
    df['sensor_data_source']='synthetic' if is_synthetic else 'uploaded sensor.csv'
    zf=None
    if reference_zip:
        try: zf=zipfile.ZipFile(reference_zip)
        except Exception as e: print(f'WARNING: cannot open reference ZIP {reference_zip}: {e}')
    try:
        df, nloc, ngw=attach_locations(df,zf,seed)
    finally:
        if zf: zf.close()
    df['water_chemistry_source']='synthetic; replace with lab/CGWB water-quality observations before interpreting WQI'
    output_path.parent.mkdir(parents=True,exist_ok=True); df.to_csv(output_path,index=False)
    saved=pd.read_csv(output_path)
    assert len(saved)==n
    assert saved['district'].isin(DISTRICTS).all()
    assert saved['population_served'].between(150,5000).all()
    assert saved['Fluoride_Level'].between(.2,3.5).all() and saved['Nitrate_Level'].between(5,60).all()
    assert saved['TDS'].between(150,2500).all() and saved['WQI'].between(0,100).all()
    assert saved['lpcd_draw'].between(30,80).all() and saved['residual_chlorine'].between(.05,.8).all()
    assert saved['status_group'].isin([0,1,2]).all()
    print(f'Reference data: {reference_zip if reference_zip else "not found"}')
    print(f'KML water-body reference points loaded: {nloc:,}; groundwater stations loaded: {ngw:,}')
    print(f'Rows with coordinates: {saved["latitude"].notna().sum():,}/{len(saved):,}')
    print(f'Rows with groundwater level: {saved["groundwater_level_m"].notna().sum():,}/{len(saved):,}')
    return saved


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',nargs='?',default='sensor.csv',help='Sensor CSV path (synthetic fallback if absent)')
    parser.add_argument('--output',default='fused_tn_pumps.csv',help='Output CSV path')
    parser.add_argument('--reference-zip',default=None,help='Uploaded reference archive containing KML, groundwater CSVs, and FHTC PDF')
    parser.add_argument('--synthetic-rows',type=int,default=1000,help='Rows to generate if sensor CSV is missing')
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    script_dir=Path(__file__).resolve().parent
    ref=find_reference_zip(args.reference_zip,script_dir)
    result=fuse_data(Path(args.input),Path(args.output),args.seed,args.synthetic_rows,ref)
    print(f'Fusion successful: {len(result):,} rows, {len(result.columns)} columns')
    print(f'Saved to: {Path(args.output).resolve()}')
    print(f'Districts represented: {result.district.nunique()} / {len(DISTRICTS)}')
    print('District rows:'); print(result.district.value_counts().reindex(DISTRICTS,fill_value=0).to_string())
    print('JJM coverage unavailable for districts:', ', '.join(sorted(result.loc[result.jjm_total_households.isna(),'district'].unique())))
    print('Validation: PASSED')

if __name__=='__main__': main()
