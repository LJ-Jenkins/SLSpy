"""
-sls-
Functions for NNRCMP wave and water level data.

Functions
---------
- `load_wave` -- loads NNRCMP wave data from all files in a directory.
- `load_wl` -- loads NNRCMP water level data from all files in a directory.
- `site_info` -- returns information about a site from the NNRCMP database.

Luke Jenkins Feb 2024
L.Jenkins@soton.ac.uk
"""

import os
import pandas as pd
import difflib
import re
import requests
from . import tools


def load_wave(sitecode, directory, flags=None, dispfile="n"):
    """
    Description
    ----------
    Loads NNRCMP qc wave data from files that contain the given sitecode in a directory.

    Parameters
    ----------
    sitecode : str
        NNRCMP three letter sitecode e.g., "PBy" (case insensitive).
    directory : str
        Directory of the files.
    flags : list, containing int
        Flags for which data should be removed, if 'df' is given then defaults are used (see below).
    dispfile : str ('y' or 'n')
        Choose to show the file being loaded at the command line.

    Returns
    -------
    t :
        Dataframe of time, water level (cd), water level (od), and flags.

    Additional Information
    -------
    Parameters

    Significant wave height: Hs (m)
    Spectrally-derived zero-crossing wave period: Tz (s)
    Peak wave period: Tp (s)
    Wave direction: Dir (°)
    (Directional) Spread: Spd (°)
    Maximum observed wave height: Hmax (m)
    Sea surface temperature: SST (C°)

    CCO Wave QC Data Quality Control Flags (as of January 2022)

    -Note flag 7 (bouy off location) is set to remove all data parameters (except lat/lon) if specified,
    not none like the table below suggests. Defaults are: 1, 2, 3, 4, 8, 9.

    Flag	Suspect Parameters          Not Suspect Parameters      Description

    0	                                All data pass

    1	    Hs, Hmax Tz, Tp,            SST                         Either Hs or Tz fail,
            Direction, Spread		                                so all data fail, except SST

    2	    Tp, Direction, Spread	    Hs, Hmax, Tz, SST	        Tp fail + derivatives

    3	    Direction, Spread	        Tp, Hs, Hmax, Tz, SST	    Direction fail + derivatives

    4	    Spread	                    Direction, Tp, Hs, Hmax,    Spread fail
                                        Tz, SST

    7		                            Hs, Hmax, Tz, Tp,           Buoy off location
                                        Direction, Spread, SST

    8	    SST	                        Hs, Hmax, Tz, Tp,           SST fail
                                        Direction, Spread

    9	    Hs, Hmax, Tz, Tp,                                       Missing data
            Direction, Spread, SST
    """

    files = tools.fnames(
        directory, ends_with=".txt", pattern=sitecode, case_sensitive=False
    )

    t_list = [0] * len(files)  # List to store DataFrames

    for i, file in enumerate(files):
        if dispfile == "y":
            print(file)
        filein = os.path.join(directory, file)
        T = pd.read_csv(filein, skiprows=1, header=None, delimiter="\t")
        t_list[i] = T

    t = pd.concat(t_list, ignore_index=True)
    t.columns = [
        "time",
        "latitude",
        "longitude",
        "flags",
        "Hs (m)",
        "Hmax (m)",
        "Tp (s)",
        "Tz (s)",
        "Dir (degrees)",
        "Spd (degrees)",
        "SST (degrees C)",
    ]

    t = tools.sortdftime(t)

    if flags is not None:
        # Define flags and their corresponding columns to set as NaN
        flag_columns_mapping = {
            1: [
                "Hs (m)",
                "Hmax (m)",
                "Tp (s)",
                "Tz (s)",
                "Dir (degrees)",
                "Spd (degrees)",
            ],
            2: ["Tp (s)", "Dir (degrees)", "Spd (degrees)"],
            3: ["Dir (degrees)", "Spd (degrees)"],
            4: ["Spd (degrees)"],
            7: [
                "Hs (m)",
                "Hmax (m)",
                "Tp (s)",
                "Tz (s)",
                "Dir (degrees)",
                "Spd (degrees)",
                "SST (degrees C)",
            ],
            8: ["SST (degrees C)"],
            9: [
                "Hs (m)",
                "Tp (s)",
                "Tz (s)",
                "Dir (degrees)",
                "Spd (degrees)",
                "SST (degrees C)",
            ],
        }
        if flags == "df":
            flags = [1, 2, 3, 4, 8, 9]
        for flag in flags:
            if flag in flag_columns_mapping:
                cols = flag_columns_mapping[flag]
                t.loc[t["flags"] == flag, cols] = pd.NA

    return t


def load_wl(sitecode, directory, flags=None, dispfile="n"):
    """
    Description
    ----------
    Loads NNRCMP qc water level data from files that contain the given sitecode in a directory.

    Parameters
    ----------
    sitecode : str
        NNRCMP three letter sitecode e.g., "PBy" (case insensitive).
    directory : str
        Directory of the files.
    flags : list, containing int
        Flags for which data should be removed, if 'df' is given then defaults are used (see below).
    dispfile : str, 'y' or 'n'
        Choose to show the file being loaded at the command line.

    Returns
    -------
    t :
        Dataframe of time, water level (cd), water level (od), and flags.

    Additional Information
    -------
    Data quality is marked by one column of flags.
    Defaults are: 3, 4, 7, 8, 9

    Flag = 1: Correct value
    Flag = 2: Interpolated value
    Flag = 3: Doubtful value
    Flag = 4: Isolated spike or wrong value
    Flag = 5: Correct but extreme value
    Flag = 6: Reference change detected
    Flag = 7: Constant value
    Flag = 8: Out of range
    Flag = 9: Missing data
    """

    files = tools.fnames(
        directory, ends_with=".txt", pattern=sitecode, case_sensitive=False
    )

    t_list = [0] * len(files)  # List to store DataFrames

    for i, file in enumerate(files):
        if dispfile == "y":
            print(file)
        filein = os.path.join(directory, file)
        T = pd.read_csv(filein, skiprows=1, header=None, delimiter="\t")
        t_list[i] = T

    t = pd.concat(t_list, ignore_index=True)
    t.columns = ["time", "wl (OD) (m)", "wl (CD) (m)", "residual (m)", "flags"]

    t = tools.sortdftime(t)

    if flags is not None:
        if flags == "df":
            flags = [3, 4, 7, 8, 9]
        cols = ["wl (od) (m)", "wl (cd) (m)", "residual (m)"]
        t.loc[t["flags"].isin(flags), cols] = pd.NA  # remove flagged data
        for col in cols:  # remove any remaining 99 or -99 data
            t.loc[(t[col] > 90) | (t[col] < -90), col] = pd.NA

    return t


def get_site_info(var):
    """
    Description
    ----------
    Returns information about all sites in the NNRCMP database, scraped from the website.

    Parameters
    ----------
    var : str
        Variable to return information about. Options are 'waves', 'tides', or 'met'.

    Returns
    -------
    df : DataFrame
        DataFrame containing information about all sites.

    Additional Information
    -------
    The information returned (where applicable) is as follows:

    - Site: Site name.
    - Location: Latitude and longitude.
    - WMO code: WMO code.
    - Water Depth (m): Approximate water depth.
    - Water depth datum: Water depth datum.
    - Spring tidal range (m): Approximate spring tidal range.
    - Storm alert threshold (m): Storm alert threshold.
    """

    url = "https://coastalmonitoring.org/realtimedata/"
    options = {"Timeout": 30}
    data = requests.get(url, options).text

    locnames = {
        "waves": [
            "Wave buoy location",
            "Step gauge location",
            "Wave radar location",
            "Pressure array location",
            "Step gauge and met station location",
            "Rex location",
        ],
        "tides": [
            "Tide gauge location",
            "Platform location",
            "Wave radar location",
            "Step gauge location",
            "Pressure array location",
            "Rex location",
            "Step gauge and met station location",
        ],
        "met": [
            "Met station location",
            "Meteorological station location",
            "Platform location",
            "Step gauge and met station location",
            "Met station location",
            "M<strong>et station location",
        ],
    }

    pattern = {
        "waves": "chart=\d+&tab=waves",
        "tides": "chart=\d+&tab=tides",
        "met": "chart=\d+&tab=met",
    }

    v = ["waves", "tides", "met", "GPS"]
    x = [index for index, item in enumerate(v) if item == var][0]

    i = data.find('<div class="table-responsive" id="{}">'.format(v[x]))
    j = data.find('<div class="table-responsive" id="{}">'.format(v[x + 1]))
    d = data[i:j]

    matches = list(set(re.findall(pattern[var], d)))
    scodes = [re.findall("\d+", match)[0] for match in matches]

    def py_nnrcmp_locs(string):
        rs = re.findall(
            r"(?<!#)\b\d+\.?\d*\b",
            re.sub(r"[^\d.\s]", "", re.sub(r"&#[^;]+;", " ", string)),
        )
        if len(rs) == 6:
            rs = [
                rs[0],
                rs[1] + "." + rs[2] if "." not in rs[2] else rs[1] + rs[2],
                rs[3],
                rs[4] + "." + rs[5] if "." not in rs[5] else rs[4] + rs[5],
            ]
        return rs

    funcs = [
        lambda str: re.search(re.escape("Monitoring - ") + r"(.*)", str)
        .group(1)
        .strip(),
        py_nnrcmp_locs,
        lambda str: int(re.search(r"(\d+(\.\d+)?)", str)[0]),
        lambda str: (
            (
                float(re.search(r"(\d+(\.\d+)?)", str).group(1))
                if re.search(r"(\d+(\.\d+)?)", str)
                else None
            ),
            (
                re.search(r"\b(CD|OD)\b", str).group(1)
                if re.search(r"\b(CD|OD)\b", str)
                else "no datum specified"
            ),
        ),
        lambda str: float(re.findall("(\d+.*?) m", str, re.DOTALL)[0]),
        lambda str: float(re.findall("(\d+.*?) m", str, re.DOTALL)[0]),
    ]

    site_info = []
    for scode in scodes:
        sitedata = requests.get(
            url + "?chart=" + scode + "&tab=info&disp_option="
        ).text
        site_out = []

        for idx, info in enumerate(
            [
                ["<title>National Coastal Monitoring"],
                locnames[var],
                ["WMO code"],
                ["Approximate water depth"],
                ["Approximate spring tidal range"],
                ["Storm alert threshold", "Storm threshold"],
            ]
        ):
            if len(info) == 1:
                ii = sitedata.find(info[0])
            else:
                ii = None
                for index, term in enumerate(info):
                    ii = sitedata.find(term)
                    if ii != -1:
                        break

            if ii == -1:
                if idx == 3:
                    site_out.append((pd.NA, pd.NA))
                else:
                    site_out.append(pd.NA)
            else:
                if idx == 0:
                    k = "</title>"
                else:
                    k = "</tr>"
                nxtline = sitedata[ii:].find(k)
                sinfo = sitedata[ii : ii + nxtline]
                if idx == 1 and len(sinfo) > 500:
                    f1 = sitedata[ii:].find("[endif]-->")
                    f2 = sitedata[ii + f1 :].find("<!")
                    sinfo = sitedata[ii + f1 : ii + f1 + f2]
                site_out.append(funcs[idx](sinfo))

        site_info.append(site_out)

    df = pd.DataFrame(
        site_info,
        columns=[
            "Site",
            "Location",
            "WMO code",
            "to drop",
            "Spring tidal range (m)",
            "Storm alert threshold (m)",
        ],
    )
    df[["Water Depth (m)", "Water depth datum"]] = pd.DataFrame(
        df["to drop"].tolist(), index=df.index
    )
    df = df.drop(columns=["to drop"])
    df = df.sort_values(by="Site")
    df.loc[df["Site"].str.contains("Gwynt"), "Site"] = "Gwynt Y Mor"
    df = df.reset_index(drop=True)

    return df


def site_info(site, var):
    """
    Description
    ----------
    Returns information about a site from the NNRCMP database.

    Parameters
    ----------
    site : str
        Site name.
    var : str
        Variable to return information about. Options are 'waves', 'tides', or 'met'.

    Returns
    -------
    info : dict
        Dictionary containing information about the site.

    Additional Information
    -------
    The information returned is as follows:

    - Site: Site name.
    - L1: Latitude (degrees).
    - L2: Latitude (minutes).
    - L3: Longitude (degrees).
    - L4: Longitude (minutes).
    - WMO code: WMO code.
    - Water depth (m): Water depth in metres.
    - Water depth datum: Water depth datum.
    - Spring tidal range (m): Spring tidal range in metres.
    - Storm alert threshold (m): Storm alert threshold in metres.
    """

    if var.lower() == "waves":
        data = [
            [
                "Bideford Bay",
                "51",
                "03.48",
                "004",
                "16.62",
                6201024,
                11,
                "CD",
                8.4,
                5.04,
            ],
            [
                "Blakeney Overfalls",
                "53",
                "03",
                "001",
                "06",
                62042,
                23,
                "Not specified",
                None,
                3.31,
            ],
            [
                "Boscombe",
                "50",
                "42.69",
                "001",
                "50.39",
                6201008,
                10.4,
                "CD",
                1.5,
                2.61,
            ],
            [
                "Bracklesham Bay",
                "50",
                "43.43",
                "000",
                "50.30",
                6201012,
                10.4,
                "CD",
                4.6,
                3.19,
            ],
            [
                "Chapel Point",
                "53",
                "14",
                "000",
                "26",
                6201050,
                13,
                "Not specified",
                6,
                2.64,
            ],
            [
                "Chesil",
                "50",
                "36.13",
                "002",
                "31.37",
                6201006,
                12,
                "CD",
                3.1,
                4.18,
            ],
            [
                "Cleveleys",
                "53",
                "53.70",
                "003",
                "11.78",
                6201028,
                10,
                "CD",
                8.2,
                3.74,
            ],
            [
                "Dawlish",
                "50",
                "34.80",
                "003",
                "25.04",
                6201027,
                11,
                "CD",
                3.9,
                2.63,
            ],
            [
                "Deal Pier",
                "51",
                "13.428",
                "001",
                "24.556",
                None,
                None,
                "NA",
                None,
                1.47,
            ],
            [
                "Felixstowe",
                "51",
                "56",
                "001",
                "23",
                6201052,
                8,
                "Not specified",
                3.4,
                1.94,
            ],
            [
                "Folkestone",
                "51",
                "03.76",
                "001",
                "07.67",
                6201017,
                12.7,
                "CD",
                6.5,
                2.48,
            ],
            [
                "Goodwin Sands",
                "51",
                "14.99",
                "001",
                "29.00",
                6201018,
                10,
                "CD",
                1.5,
                2.49,
            ],
            [
                "Gwynt Y Mor",
                "53",
                "28.62",
                "03",
                "30.20",
                None,
                10,
                "CD",
                7.2,
                3.55,
            ],
            [
                "Happisburgh",
                "52",
                "49",
                "001",
                "32",
                6201051,
                10,
                "Not specified",
                2.6,
                2.66,
            ],
            [
                "Hastings Pier",
                "50",
                "51.053",
                "000",
                "34.372",
                None,
                None,
                "NA",
                None,
                2.16,
            ],
            [
                "Hayling Island",
                "50",
                "43.90",
                "000",
                "57.54",
                6201011,
                10,
                "CD",
                4.6,
                2.78,
            ],
            [
                "Herne Bay",
                "51",
                "22.919",
                "001",
                "06.934",
                None,
                0.5,
                "CD",
                None,
                0.71,
            ],
            [
                "Hornsea",
                "53",
                "55.02",
                "000",
                "03.95",
                6201019,
                12,
                "CD",
                5,
                3.04,
            ],
            [
                "Looe Bay",
                "50",
                "20.33",
                "004",
                "24.65",
                6201025,
                10,
                "CD",
                4.8,
                3.7,
            ],
            [
                "Lowestoft",
                "52",
                "28",
                "001",
                "49",
                6201059,
                20,
                "Not specified",
                1.9,
                3.06,
            ],
            [
                "Lymington",
                "50",
                "44.4198",
                "001",
                "30.4268",
                None,
                2,
                "CD",
                None,
                0.79,
            ],
            [
                "Milford",
                "50",
                "42.75",
                "001",
                "36.91",
                6201009,
                10,
                "CD",
                2,
                2.74,
            ],
            [
                "Minehead",
                "51",
                "13.68",
                "003",
                "28.15",
                6201004,
                10,
                "CD",
                9.6,
                2.2,
            ],
            [
                "Morecambe Bay",
                "53",
                "59.38",
                "003",
                "03.96",
                6201029,
                10,
                "CD",
                8.2,
                3.21,
            ],
            [
                "New Brighton",
                "53",
                "26.57",
                "003",
                "02.06",
                None,
                5,
                "CD",
                8.4,
                1.64,
            ],
            [
                "Newbiggin",
                "55",
                "11.11",
                "001",
                "28.69",
                6201047,
                18,
                "CD",
                4.2,
                3.42,
            ],
            [
                "North Well",
                "53",
                "03",
                "000",
                "28",
                62041,
                31,
                "Not specified",
                6.25,
                2.2,
            ],
            [
                "Penarth",
                "51",
                "26.080",
                "003",
                "09.887",
                None,
                None,
                "NA",
                None,
                None,
            ],
            [
                "Penzance",
                "50",
                "06.86",
                "005",
                "30.18",
                6201000,
                10,
                "CD",
                4.8,
                3.06,
            ],
            [
                "Perranporth",
                "50",
                "21.19",
                "005",
                "10.47",
                6201001,
                14,
                "CD",
                6.1,
                5.4,
            ],
            [
                "Pevensey Bay",
                "50",
                "46.91",
                "000",
                "25.10",
                6201015,
                9.8,
                "CD",
                6.1,
                3.2,
            ],
            [
                "Port Isaac",
                "50",
                "35.651",
                "004",
                "50.065",
                None,
                None,
                "NA",
                None,
                3,
            ],
            [
                "Porthleven",
                "50",
                "03.76",
                "005",
                "18.44",
                6201044,
                15,
                "CD",
                4.7,
                4.93,
            ],
            [
                "Rhyl Flats",
                "53",
                "22.92",
                "003",
                "36.21",
                None,
                10,
                "CD",
                7.2,
                2.89,
            ],
            [
                "Rustington",
                "50",
                "44.06",
                "000",
                "29.64",
                6201013,
                9.9,
                "CD",
                6.1,
                3.37,
            ],
            [
                "Rye Bay",
                "50",
                "51.083",
                "000",
                "47.433",
                None,
                10,
                "CD",
                6.1,
                3.52,
            ],
            [
                "Sandown Bay",
                "50",
                "39.03",
                "001",
                "07.68",
                6201010,
                10.7,
                "CD",
                3.3,
                2.48,
            ],
            [
                "Sandown Pier",
                "50",
                "39.067",
                "001",
                "09.189",
                None,
                None,
                "NA",
                None,
                1.41,
            ],
            [
                "Scarborough",
                "54",
                "17.60",
                "000",
                "19.06",
                6201045,
                19,
                "CD",
                4.8,
                4.22,
            ],
            [
                "Seaford",
                "50",
                "45.99",
                "000",
                "04.53",
                6201014,
                11,
                "CD",
                6.1,
                3.78,
            ],
            [
                "Second Severn Crossing",
                "51",
                "34.456",
                "002",
                "41.999",
                None,
                None,
                "NA",
                None,
                0.74,
            ],
            [
                "St Mary's Sound",
                "49",
                "53.53",
                "06",
                "18.76",
                6201053,
                53,
                "CD",
                5,
                4.43,
            ],
            [
                "Start Bay",
                "50",
                "17.53",
                "003",
                "36.99",
                6201002,
                10,
                "CD",
                4.4,
                2.98,
            ],
            [
                "Swanage Pier",
                "50",
                "36.562",
                "001",
                "56.954",
                None,
                None,
                "NA",
                None,
                1.14,
            ],
            [
                "Teignmouth Pier",
                "50",
                "32.632",
                "003",
                "29.529",
                None,
                None,
                "NA",
                None,
                1.75,
            ],
            [
                "Tor Bay",
                "50",
                "26.02",
                "003",
                "31.08",
                6201003,
                11,
                "CD",
                4,
                2.2,
            ],
            [
                "Wave Hub",
                "50",
                "20.84",
                "005",
                "36.84",
                None,
                50,
                "CD",
                6.1,
                6.81,
            ],
            [
                "West Anglesey (SEACAMS)",
                "53",
                "13.0017",
                "004",
                "43.4438",
                None,
                45,
                "CD",
                None,
                None,
            ],
            [
                "West Bay",
                "50",
                "41.63",
                "002",
                "45.06",
                6201005,
                10,
                "CD",
                3.5,
                4.08,
            ],
            [
                "Weston Bay",
                "51",
                "21.13",
                "003",
                "01.23",
                6201026,
                13,
                "CD",
                11.2,
                1.94,
            ],
            [
                "Weymouth",
                "50",
                "37.36",
                "002",
                "24.85",
                6201007,
                None,
                "NA",
                2,
                2.11,
            ],
            [
                "Whitby",
                "54",
                "30.27",
                "000",
                "36.41",
                6201046,
                None,
                "NA",
                4.6,
                4.16,
            ],
        ]
    elif var.lower() == "tides":
        data = [
            [
                "Arun Platform",
                "50",
                "46.200",
                "000",
                "29.500",
                None,
                None,
                "NA",
                None,
                None,
            ],
            [
                "Brighton",
                "50",
                "48.707",
                "000",
                "06.071",
                None,
                None,
                "NA",
                None,
                None,
            ],
            [
                "Deal Pier",
                "51",
                "13.428",
                "001",
                "24.556",
                None,
                None,
                "NA",
                None,
                1.47,
            ],
            [
                "Exmouth",
                "50",
                "37.043",
                "03",
                "25.415",
                None,
                None,
                "NA",
                None,
                None,
            ],
            [
                "Hastings Pier",
                "50",
                "51.053",
                "000",
                "34.372",
                None,
                None,
                "NA",
                None,
                2.16,
            ],
            [
                "Herne Bay",
                "51",
                "22.919",
                "001",
                "06.934",
                None,
                0.5,
                "CD",
                None,
                0.71,
            ],
            [
                "Lymington",
                "50",
                "44.4198",
                "001",
                "30.4268",
                None,
                2,
                "CD",
                None,
                0.79,
            ],
            [
                "Penarth",
                "51",
                "26.080",
                "003",
                "09.887",
                None,
                None,
                "NA",
                None,
                None,
            ],
            [
                "Port Isaac",
                "50",
                "35.651",
                "004",
                "50.065",
                None,
                None,
                "NA",
                None,
                3,
            ],
            [
                "Sandown Pier",
                "50",
                "39.067",
                "001",
                "09.189",
                None,
                None,
                "NA",
                None,
                1.41,
            ],
            [
                "Scarborough",
                "54",
                "16.948",
                "00",
                "23.408",
                None,
                None,
                "NA",
                None,
                None,
            ],
            [
                "Second Severn Crossing",
                "51",
                "34.456",
                "002",
                "41.999",
                None,
                None,
                "NA",
                None,
                0.74,
            ],
            [
                "Swanage Pier",
                "50",
                "36.562",
                "001",
                "56.954",
                None,
                None,
                "NA",
                None,
                1.14,
            ],
            [
                "Teignmouth Pier",
                "50",
                "32.632",
                "003",
                "29.529",
                None,
                None,
                "NA",
                None,
                1.75,
            ],
            [
                "West Bay Harbour",
                "50",
                "42.532",
                "002",
                "45.847",
                None,
                None,
                "NA",
                None,
                None,
            ],
            [
                "Whitby Harbour",
                "54",
                "29.318",
                "00",
                "36.878",
                None,
                None,
                "NA",
                None,
                None,
            ],
        ]
    elif var.lower() == "met":
        data = [
            [
                "Arun Platform",
                "50",
                "46.200",
                "000",
                "29.500",
                None,
                None,
                "NA",
                None,
                None,
            ],
            [
                "Brighton",
                "50",
                "48.844",
                "000",
                "06.046",
                None,
                None,
                "NA",
                None,
                None,
            ],
            [
                "Bude",
                "50",
                "49.847",
                "004",
                "33.003",
                None,
                None,
                "NA",
                None,
                None,
            ],
            [
                "Chapel Point",
                "53",
                "13",
                "000",
                "20",
                "6201050",
                "13",
                "Not specified",
                "6",
                "2.64",
            ],
            [
                "Deal Pier",
                "51",
                "13.435",
                "001",
                "24.540",
                None,
                None,
                "NA",
                None,
                1.47,
            ],
            [
                "Exmouth",
                "50",
                "36.657",
                "03",
                "23.945",
                None,
                None,
                "NA",
                None,
                None,
            ],
            [
                "Felixstowe",
                "51",
                "56",
                "001",
                "19",
                "6201052",
                "8",
                "Not specified",
                "3.4",
                "1.94",
            ],
            [
                "Folkestone",
                "51",
                "04.77",
                "001",
                "10.19",
                "6201017",
                "12.7",
                "CD",
                "6.5",
                "2.48",
            ],
            [
                "Happisburgh",
                "52",
                "48",
                "001",
                "33",
                "6201051",
                "10",
                "Not specified",
                "2.6",
                "2.66",
            ],
            [
                "Herne Bay",
                "51",
                "22.370",
                "001",
                "07.460",
                None,
                "0.5",
                "CD",
                None,
                "0.71",
            ],
            [
                "Looe Bay",
                "50",
                "20.70",
                "004",
                "27.17",
                "6201025",
                "10",
                "CD",
                "4.8",
                "3.7",
            ],
            [
                "Lymington",
                "50",
                "44.4198",
                "001",
                "30.4268",
                None,
                "2",
                "CD",
                None,
                "0.79",
            ],
            [
                "Minehead",
                "51",
                "12.427",
                "003",
                "27.734",
                "6201004",
                "10",
                "CD",
                "9.6",
                "2.2",
            ],
            [
                "Penarth",
                "51",
                "26.089",
                "003",
                "09.889",
                None,
                None,
                "NA",
                None,
                None,
            ],
            [
                "Penzance",
                "50",
                "07.04",
                "005",
                "31.79",
                "6201000",
                "10",
                "CD",
                "4.8",
                "3.06",
            ],
            [
                "Perranporth",
                "50",
                "20.77",
                "005",
                "09.71",
                "6201001",
                "14",
                "CD",
                "6.1",
                "5.4",
            ],
            [
                "Port Isaac",
                "50",
                "35.408",
                "004",
                "49.426",
                None,
                None,
                "NA",
                None,
                "3",
            ],
            [
                "Sandown Pier",
                "50",
                "39.070",
                "001",
                "09.190",
                None,
                None,
                "NA",
                None,
                "1.41",
            ],
            [
                "Southwold",
                "52",
                "18",
                "001",
                "40",
                None,
                None,
                "NA",
                None,
                None,
            ],
            [
                "Swanage Pier",
                "50",
                "36.562",
                "001",
                "56.954",
                None,
                None,
                "NA",
                None,
                "1.14",
            ],
            [
                "Teignmouth Pier",
                "50",
                "32.633",
                "003",
                "29.527",
                None,
                None,
                "NA",
                None,
                "1.75",
            ],
            [
                "West Bay Harbour",
                "50",
                "42.640",
                "002",
                "45.837",
                None,
                None,
                "NA",
                None,
                None,
            ],
            [
                "Weston Bay",
                "51",
                "20.65",
                "002",
                "58.90",
                "6201026",
                "13",
                "CD",
                "11.2",
                "1.94",
            ],
            [
                "Weymouth",
                "50",
                "34.21",
                "002",
                "27.31",
                "6201007",
                None,
                "NA",
                "2",
                "2.11",
            ],
            [
                "Worthing Pier",
                "50",
                "48.422",
                "000",
                "22.128",
                None,
                None,
                "NA",
                None,
                None,
            ],
        ]
    else:
        raise ValueError(
            "Invalid 'var' argument. Expected 'waves', 'tides', or 'met'."
        )

    columns = [
        "Site",
        "L1",
        "L2",
        "L3",
        "L4",
        "WMO code",
        "Water depth (m)",
        "Water depth datum",
        "Spring tidal range (m)",
        "Storm alert threshold (m)",
    ]
    df = pd.DataFrame(data, columns=columns)
    df["Location"] = df[["L1", "L2", "L3", "L4"]].apply(
        lambda x: " ".join(x), axis=1
    )
    df = df.drop(columns=["L1", "L2", "L3", "L4"])

    mask = df["Site"].str.contains(site, case=False)
    if mask.sum() != 1:
        closest_match = difflib.get_close_matches(
            site.lower(), df["Site"].str.lower(), 1
        )[0]
        mask = df["Site"].str.lower().str.contains(closest_match)
    info = df[mask].iloc[0].to_dict()

    return info
