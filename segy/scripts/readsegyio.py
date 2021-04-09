#=============================================================================
# Start imports
# Get segyio from https://github.com/equinor/segyio
#=============================================================================
import time
import segyio
import re

import numpy             as np
import pandas            as pd

# My own modules
#from SEGYtools_segyio    import segyio_tools
#=============================================================================
# End imports
#=============================================================================
class ReadSegyio():
    """
    Process a segyio object

    Create an object containing
    data = pandas DataFrame with 2D gathers specified by "gather_id"
           gather_id has to be a segyio header name
    headers = SEGY trace headers

    Parameters:
    segyfile                 : SEGY filename
    keep_hdrs=[]             : SEGY header to keep
    drop_hdrs=[]             : SEGY headers to remove
    gather_id="FieldRecord"  : SEGY header name of gathers to read
    verbose=False            : 0,1  print information

    Returns:
    int: Description of return value

    """

    def _parse_trace_headers(self, segyfile, n_traces):
        """
        Extract trace headers from SEGY file

        Returns:
        df: Dataframe with trace headers
        """
             # Get all header keys
        headers = segyio.tracefield.keys
        # Initialize dataframe with trace id as index and headers as columns
        df = pd.DataFrame(index=range(1, n_traces + 1),columns=headers.keys())
        # Fill dataframe with all header values
        for k, v in headers.items():
            df[k] = segyfile.attributes(v)[:]
        return df

    def _parse_text_header(self, segyfile):
        """
        Format segy EBCDIC header into a readable, clean dictionary

        """
        raw_header = segyio.tools.wrap(segyfile.text[0])
        # Cut on C*int pattern
        cut_header = re.split(r'C ', raw_header)[1::]
        # Remove end of line return
        text_header = [x.replace('\n', ' ') for x in cut_header]
        text_header[-1] = text_header[-1][:-2]
        # Format in dict
        clean_header = {}
        i = 1
        for item in text_header:
            key = "C" + str(i).rjust(2, '0')
            i += 1
            clean_header[key] = item
        return clean_header

    def clean_headers(self, keep=[], verbose=0):
        """
        Remove all zero/constant headers except those specified in "keep"
        """
        # Drop columns with all zeros or all values equal
        cols      = self.hdrs.columns
        keep_hdrs =  cols.intersection(keep)
        tmp       = self.hdrs[keep_hdrs]
        n0        = self.hdrs.shape[1]
        cols0     = set(self.hdrs.columns)

        # Remove colums that contain only 0
        self.hdrs = self.hdrs.loc[:,self.hdrs.sum() != 0]
        # Remove colums with std= 0, i.e. constant values
        self.hdrs = self.hdrs.loc[:,self.hdrs.std() != 0]
        # Ensure 'keep' columns that are actually present in the headers
        # are not removed
        for hdr in tmp.columns:
            if (hdr not in self.hdrs.columns):
                self.hdrs[hdr] = tmp[hdr]
        n1 = self.hdrs.shape[1]
        cols1 = set(self.hdrs.columns)
        if verbose:
            if len(cols0-cols1) != 0:
                print("The following %d columns were removed"%(n0-n1+1))
                print(cols0-cols1)
                print("Shape of trace header array = ",self.hdrs.shape)
            else:
                print("No constant or zero value headers were removed")

    # Drop the duplicates and columns with constant values
    def drop_headers(self,drop_list,verbose=0):
        """
        Remove headers specified in "drop"
        """
        hdrs_removed=list()
        for hdr in drop_list:
            if (hdr in self.hdrs.columns):
                self.hdrs.drop(hdr,axis=1,inplace=True)
                hdrs_removed.append(hdr)
        if verbose:
            if len(hdrs_removed) != 0:
                print("Removed the following headers: ",hdrs_removed)
                print("Shape of trace header array = ",self.hdrs.shape)
            else:
                print("No headers were removed")


    def info(self,ncol=5):
        """
        Print extended header information
        """
        pd.set_option('display.max_columns',15)
        pd.set_option('display.width',1000)
        pd.options.display.float_format = '{:,.2f}'.format

        n = len(self.hdrs.columns)
        print("There are %3d trace headers"%n)
        for i in range(int(np.ceil(n/ncol))):
            cols=self.hdrs.columns[i*ncol:(i+1)*ncol]
            print(self.hdrs[cols].describe())
            #print(self.hdrs[cols].describe().apply(lambda x: format(x, 'f')))

    def info2(self,ind,header_list):
        """
        Print min//max/range/number-of-unique-value/increment for specified headers

        Find the following information for a trace header
        maximum, minimum, range, # of unique values and increment
        Run on a single shot/receiver combination (i.e. a 2D line)
        """
        for hdr in header_list:
           if (hdr in self.hdrs.columns):
               min_hdr    = self.hdrs.loc[ind,[hdr]].min(axis=0)[0]
               max_hdr    = self.hdrs.loc[ind,[hdr]].max(axis=0)[0]
               range_hdr  = max_hdr - min_hdr
               n_uniq_hdr = len(self.hdrs.loc[ind,hdr].unique())
               incr_hdr   = int(np.round(range_hdr / n_uniq_hdr))
               print("Header %s: min/max/range/number/increment = %5s,%5s,%5s,%5s,%5s"%
                     (hdr,min_hdr,max_hdr,range_hdr,n_uniq_hdr,incr_hdr))
           else:
               print("Header {} not present".format(hdr))

    def __init__(self, segyfile, keep_hdrs=[], drop_hdrs=[],
                 gather_id="FieldRecord", verbose=False):
        """
        Initialize SEGY data loader

        Input_data = segy file
        """
        if keep_hdrs == None:
            keep_hdrs=[]
        keep_hdrs.append(gather_id)
        self.segyfile     = segyfile
        self.verbose      = verbose
        self.keep_hdrs    = keep_hdrs
        self.drop_hdrs    = drop_hdrs
        self.gather_id    = gather_id

        #==============================================================================
        # Read seismic data
        #==============================================================================
        with segyio.open(segyfile, ignore_geometry=True) as f:
            mapped = f.mmap()
            if self.verbose:
                if mapped: print( "File is memory mapped!" )
            self.n_traces    = f.tracecount
            self.sample_rate = segyio.tools.dt(f) / 1000
            self.n_samples   = f.samples.size
            twt         = f.samples
            #data        = f.trace.raw[:]  # Get all data into memory (could cause on big files)
            # Load headers
            self.bin_hdr = f.bin
            self.hdrs    = self._parse_trace_headers(f, self.n_traces)
            self.txt_hdr = self._parse_text_header(f)

        #self.headers      = segyio_tools(self.hdrs)

        if self.verbose:
            print("SEGY file               = ",segyfile)
            print("Sampling interval       = ",self.sample_rate)
            print("Number of traces        = ",self.n_traces)
            print("Number of samples       = ",self.n_samples)
            print("Initial shape of trace header array = ",self.hdrs.shape)

        # Drop the duplicates and columns with constant values, except the keep_hdrs
        if self.verbose:
            print("Drop the duplicates and columns with constant values, except the keep_hdrs")
        self.clean_headers(self.keep_hdrs,verbose=verbose)

        # Drop remaining headers specified by user
        if self.verbose:
            print("Drop remaining headers specified by user")
        self.drop_headers(self.drop_hdrs,verbose=self.verbose)

        # Print info on remaining headers
        if self.verbose:
            self.info()
            print("")
            print("Shape of trace header array after editing = ",self.hdrs.shape)
            print("")

        # Create DataFrame with shotline, shotpoint and receiver line header
        # Sort, and keep only unique values
        # reset_index renumbers the row index to that have have a number/label per
        # shot, starting at 1 with increment 1
        shots = pd.DataFrame({self.gather_id:self.hdrs[self.gather_id]})
        #shots = pd.DataFrame({'SHOT':headers.hdrs['SHOT']})
        shots.sort_values(by=[self.gather_id])
        # Keep only unqiue values of SHOT
        shots.drop_duplicates(inplace=True)
        # Reset the DataFrame index, starting at 0 with increment 1.
        # Do not insert theoriginal index as a column (drop=True)
        shots.reset_index(inplace=True,drop=True)
        # Define an empty DataFrame holding per shot the seismic data and the mask.
        # Also included shot identification labels ("SHOT")

        self.data = pd.DataFrame(index=shots.index,
                                 columns=["gather",self.gather_id])

        self.data.index.name = "gather"

        # Print dataframe index
        if self.verbose:
            print("")
            print("Number of gathers = {}".format(shots.shape[0]))
            print("data columns      = ",self.data.columns)
            print("data shape        = ",self.data.shape)

        if self.verbose == True:
            time.sleep(.2) # wait untill everything is printed

        for sht_num in shots.index:

            # Find indices of current shot gather and store shot in dataframe
            # Store shot gather number as 16bit integer
            self.data.loc[sht_num][self.gather_id] = np.int32(shots.loc[sht_num][self.gather_id])
            with segyio.open(self.segyfile, ignore_geometry=True) as f:
                self.data.loc[sht_num]["gather"]  = f.trace.raw[:]

        return
