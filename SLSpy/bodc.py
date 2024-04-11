"""
-sls-
Functions for BODC UKNTGN sea level data.

Functions
---------
- `tide_gauge_info` -- gives basic information for specified tide gauge.
- `load` -- loads BODC sea level data from all files in a directory.
- `tidy_downloads` -- sorts BODC downloaded files into folders by sitecode.

Luke Jenkins Feb 2024
L.Jenkins@soton.ac.uk
"""

import os
import pandas as pd
import shutil
import re
from . import tools

def tide_gauge_info(sid):
    """
    Description
    ----------
    Gives the basic information on a tide gauge: name, sitecode, ordanance datum conversion value, and latitude/longitude.

    Parameters
    ----------
    sid : str
        'site identifier' either BODC three letter sitecode or full site name (case insensitive) (e.g., "abe" or "aberdeen").

    Returns
    -------
    latitude_longitude : 
        Latitude and longitude.
    full_site_name :
        Full site name e.g., "Aberdeen".
    site_code :
        3 letter site code e.g., "ABE".
    datum_conversion :
        Datum conversion value for chart datum to ordnance datum conversion (to be added to water levels).
    """
    
    # Define site information
    sites = {
        "dov": ["Dover", -3.67, [51.114389, 1.322528], "DOV"],
        "har": ["Harwich", -2.02, [51.948000, 1.292056], "HAR"],
        "hey": ["Heysham", -4.90, [54.031833, -2.920250], "HEY"],
        "hin": ["Hinkley Point", -5.90, [51.215250, -3.134472], "HIN"],
        "low": ["Lowestoft", -1.50, [52.473083, 1.750250], "LOW"],
        "nha": ["Newhaven", -3.52, [50.781778, 0.057028], "NHA"],
        "abe": ["Aberdeen", -2.25, [57.144028, -2.080222], "ABE"],
        "ptm": ["Portsmouth", -2.73, [50.802194, -1.111250], "PTM"],
        "avo": ["Avonmouth", -6.50, [51.507750, -2.712750], "AVO"],
        "bou": ["Bournemouth", -1.40, [50.714333, -1.874861], "BOU"],
        "ban": ["Bangor", -2.01, [54.664750, -5.669472], "BAN"],
        "bar": ["Barmouth", -2.44, [52.719333, -4.045028], "BAR"],
        "cro": ["Cromer", -2.75, [52.934194, 1.301639], "CRO"],
        "dev": ["Devonport", -3.22, [50.368389, -4.185250], "DEV"],
        "mum": ["Mumbles", -5.00, [51.570000, -3.975472], "MUM"],
        "fis": ["Fishguard", -2.44, [52.013222, -4.983750], "FIS"],
        "ilf": ["Ilfracombe", -4.80, [51.211139, -4.112389], "ILF"],
        "hol": ["Holyhead", -3.05, [53.313944, -4.620417], "HOL"],
        "imm": ["Immingham", -3.90, [53.630417, -0.187528], "IMM"],
        "kin": ["Kinlochbervie", -2.50, [58.456694, -5.050222], "KIN"],
        "lei": ["Leith", -2.90, [55.989833, -3.181694], "LEI"],
        "ler": ["Lerwick", -1.22, [60.154028, -1.140306], "LER"],
        "liv": ["Liverpool", -4.93, [53.449694, -3.018139], "LIV"],
        "lla": ["Llandudno", -3.85, [53.331667, -3.825222], "LLA"],
        "mha": ["Milford Haven", -3.71, [51.707389, -5.051778], "MHA"],
        "new": ["Newlyn", -3.05, [50.103000, -5.542750], "NEW"],
        "npo": ["Newport", -5.81, [51.550000, -2.987444], "NPO"],
        "nsh": ["North Shields", -2.60, [55.007444, -1.439778], "NSH"],
        "she": ["Sheerness", -2.90, [51.445639, 0.743361], "SHE"],
        "tob": ["Tobermory", -2.39, [56.623111, -6.064222], "TOB"],
        "ull": ["Ullapool", -2.75, [57.895250, -5.158056], "ULL"],
        "wey": ["Weymouth", -0.93, [50.608500, -2.447944], "WEY"],
        "whi": ["Whitby", -3.00, [54.490000, -0.614694], "WHI"],
        "wic": ["Wick", -1.71, [58.440972, -3.086389], "WIC"],
        "wor": ["Workington", -4.20, [54.650722, -3.567167], "WOR"],
        "sto": ["Stornoway", -2.71, [58.207722, -6.388889], "STO"],
        "stm": ["St. Mary's", -2.91, [49.917833, -6.317139], "STM"],
        "jer": ["St. Helier", -5.88, [49.183333, -2.116667], "JER"],
        "isl": ["Port Ellen", -0.19, [55.627583, -6.189917], "ISL"],
        "iom": ["Port Erin", -2.75, [54.085222, -4.768056], "IOM"],
        "por": ["Portpatrick", -1.80, [54.842556, -5.120028], "POR"],
        "pru": ["Portrush", -1.24, [55.206778, -6.656833], "PRU"],
        "fel": ["Felixstowe", -1.95, [51.956750, 1.348389], "FEL"],
        "mor": ["Moray Firth", -2.22, [57.599167, -4.002222], "MOR"],
        "ptb": ["Portbury", -6.50, [51.500000, -2.728472], "PTB"],
        "mil": ["Millport", -1.62, [55.749806, -4.906333], "MIL"],
        "belfast": ["Belfast", float('nan'), [float('nan'), float('nan')], float('nan')],
        "exmouth": ["Exmouth", float('nan'), [float('nan'), float('nan')], float('nan')],
        "padstow": ["Padstow", float('nan'), [float('nan'), float('nan')], float('nan')],
    }
    gauge_info = [value for key, value in sites.items() if sid.lower() == key or sid.lower() == value[0].lower()]
    full_site_name = gauge_info[0][0]
    datum_conversion = gauge_info[0][1]
    latitude_longitude = gauge_info[0][2]
    site_code = gauge_info[0][3]
    if gauge_info:
        return latitude_longitude, full_site_name, site_code, datum_conversion
    else:
        # Return False for all outputs if sid is not found
        return [False] * 4

def load(sitecode, directory, datum='od', flags=None, dispfile='n'):
    """
    Description
    ----------
    Loads BODC data from files that contain the given sitecode from a directory.

    Parameters
    ----------
    sitecode : str
        BODC three letter sitecode e.g., "ABE" (case insensitive).
    directory : str
        Directory of the files.
    datum : str, 'cd' or 'od'
        Specify whether to convert values from chart datum to ordnance datum.
    flags : list, containing str
        Flags for which data should be removed, if 'df' is given then defaults are used (see below).
    dispfile : str, 'y' or 'n'
        Choose to show the file being loaded at the command line. 

    Returns
    -------
    t : 
        Dataframe of time, water level (cd or od), water level flags, residuals, and residual flags.

    Additional Information
    -------
     BODC Quality Control Flags (as of March 2023)
     Defaults are: <, >, A, B, C, D, E, G, I, K, L, M, N, O, P, Q, U, W, X
     (some not relevant for sea level but included anyway)
     
     FLAG	DESCRIPTION
     Blank	Unqualified
     <	    Below detection limit
     >	    In excess of quoted value
     A	    Taxonomic flag for affinis (aff.)
     B	    Beginning of CTD Down/Up Cast
     C	    Taxonomic flag for confer (cf.)
     D	    Thermometric depth
     E	    End of CTD Down/Up Cast
     G	    Non-taxonomic biological characteristic uncertainty
     H	    Extrapolated value
     I	    Taxonomic flag for single species (sp.)
     K	    Improbable value - unknown quality control source
     L	    Improbable value - originator's quality control
     M	    Improbable value - BODC quality control
     N	    Null value
     O	    Improbable value - user quality control
     P	    Trace/calm
     Q	    Indeterminate
     R	    Replacement value
     S	    Estimated value
     T	    Interpolated value
     U	    Uncalibrated
     W	    Control value
     X	    Excessive difference
    """

    #files = [file for file in os.listdir(directory) if file.endswith('.txt') and sitecode.lower() in file.lower()]
    files = tools.fnames(directory, ends_with='.txt', pattern=sitecode, case_sensitive=False)

    t_list = [0] * len(files) # List to store DataFrames

    for i, file_name in enumerate(files):
        if dispfile == 'y':
            print(file_name)

        file_path = os.path.join(directory, file_name)

        with open(file_path, 'r') as f:
            for _ in range(10): # Ignore the first 10 lines
                f.readline()  

            first_line = ' '+f.readline().strip()

        indices = []
        for index, char in enumerate(first_line):
            # Check if the character is not a space and the previous character is a space or it's the first character
            if char != ' ' and (index == 0 or first_line[index - 1] == ' '):
                indices.append(index)

        widths=[(indices[1],indices[6]+3), # time
                (indices[6]+3,indices[7]), # wl
                (indices[7],indices[7]+1), # flags wl
                (indices[7]+1,indices[8]), # residual
                (indices[8],indices[8]+1)] # flags res

        df = pd.read_fwf(file_path, colspecs=widths, skiprows=11,
                    names=['time', 'water level (cd) (m)', 'wl flags', 'residual (m)', 'rs flags'],
                    dtype={'time': str, 'wl flags': str, 'rs flags': str})

        t_list[i] = df  # add the DataFrame to the list

    t = pd.concat(t_list, ignore_index=True) 

    t = tools.sortdftime(t)

    if datum == 'od':
        _, _, _, od = tide_gauge_info(sitecode)
        t.loc[t['water level (cd) (m)'] != -99, 'water level (cd) (m)'] += od
        t.rename(columns={'water level (cd) (m)': 'water level (od) (m)'}, inplace=True)

    if flags is not None:
        if flags == 'df':
            flags = ['<','>','A','B','C','D','E','G','I','K','L','M','N','O','P','Q', 'U','W','X']
        colcon = [
            (2, 1, flags), # wl flags
            (4, 3, flags), # rs flags
            (2, 1, (t.iloc[:, 2] > 90) | (t.iloc[:, 2] < -90)), # any remaining 99 or -9 data
            (4, 3, (t.iloc[:, 4] > 90) | (t.iloc[:, 4] < -90))
        ]
        for condition_col, target_column, condition in colcon:
            t.loc[t.iloc[:, condition_col].isin(condition), t.columns[target_column]] = pd.NA

    return t

def tidy_downloads(files_directory, out_directory, sem='m', html='m', pdf='m'):
    """
    Description
    ----------
    Tidies a list of BODC UKNTGN files into folders of sites, each containing relevant files. 
    Works for data from primary channel, values + residuals, surges and extremes. 
    Existing files are overwritten.

    Parameters
    ----------
    files_directory : str
        Where the downloaded files are kept.
    out_directory : str
        Where the tidied folders and files should go.
    sem : str, 'm' or 'd'
        Move or delete the 'surges.txt', 'extremes.txt', and 'means.txt' files.
    html : str, 'm' or 'd'
        Move or delete the .html files.
    pdf : str, 'm' or 'd'
        Move or delete the .pdf files (user agreement and format). 
    """

    # Get the file names
    #file_list = os.listdir(files_directory)

    # Extract PDFs from the file list
    #pdfs = [file for file in file_list if file.endswith('.pdf')]
    pdfs = tools.fnames(files_directory, ends_with='.pdf')

    # Filter out files that aren't .txt
    #file_list = [file for file in file_list if file.endswith('.txt')]
    file_list = tools.fnames(files_directory, ends_with='.txt')

    sem = ['surges', 'extremes', 'means']
    for s in sem:
        if s+'.txt' in file_list:
            file_list.remove(s+'.txt')
            if sem == 'm':
                shutil.move(os.path.join(files_directory, s+'.txt'), out_directory)
            elif sem == 'd':
                os.remove(os.path.join(files_directory, s+'.txt'))

    # Create a list of site codes
    site_code_list = [''.join(re.findall(r'[a-zA-Z]+',file.split('.')[0])) for file in file_list]
    u_sites = list(dict.fromkeys(site_code_list)) # Get unique sitecodes (preserves order), list(set(site_code_list)) does not preserve order 

    for site in u_sites:
        site_dir = os.path.join(out_directory, site)
        if not os.path.exists(site_dir):
            os.mkdir(site_dir)
        files2move = [file for file in file_list if site in file]
        for mfile in files2move:
            shutil.move(os.path.join(files_directory, mfile), site_dir)

        # Handle .html files
        html_file = os.path.join(files_directory, site + '.html')
        if os.path.exists(html_file):
            if html == 'm':
                shutil.move(html_file, site_dir)
            elif html == 'd':
                os.remove(html_file)

    # Move or delete PDFs
    if len(pdfs) != 0: 
        for mfile in pdfs:
            if pdf == 'm':
                shutil.move(os.path.join(files_directory, mfile), out_directory)
            elif pdf == 'd':
                os.remove(os.path.join(files_directory, mfile))


                