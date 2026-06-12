from abc import ABC, abstractmethod
from ont_fast5_api.fast5_interface import get_fast5_file
import pod5
import numpy as np


class SignalReaderBase(ABC):
    """
    Abstract base class for signal readers.

    Provides a unified API for reading signals from different nanopore file formats.
    """

    def __init__(self, filename):
        """
        Initialize with the path to a FAST5 or POD5 file.
        :param filename: Path to the file.
        """
        self.filename = filename

    @abstractmethod
    def __enter__(self):
        """
        Open the file and return self.
        Used with context manager (with ... as ...).
        """
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Close the file when exiting the context manager.
        """
        pass

    @abstractmethod
    def get_read_ids(self):
        """
        Return a list of all read IDs in the file.
        """
        pass

    @abstractmethod
    def get_signal(self, read_id):
        """
        Retrieve the raw signal (converted to pA) for a given read ID.
        :param read_id: The read identifier.
        :return: Numpy array of signal values in picoamperes.
        """
        pass

    @abstractmethod
    def get_channel_info(self, read_id):
        """
        Retrieve the channel information for a given read ID.
        :param read_id: The read identifier.
        :return: A dictionary containing channel metadata (e.g., channel_number).
        """
        pass

    def get_total_reads(self):
        """
        Return the total number of reads (works for both FAST5 and POD5).
        """
        ids = self.get_read_ids()
        try:
            return len(ids)
        except TypeError:
            return sum(1 for _ in ids)


class Fast5Reader(SignalReaderBase):
    """
    Signal reader implementation for FAST5 files.
    """

    def __enter__(self):
        self.infile = get_fast5_file(self.filename, mode="r")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.infile.close()

    def get_read_ids(self):
        return self.infile.get_read_ids()

    def get_channel_info(self, read_id):
        """
        Returns a dictionary containing:
        - channel_number
        - digitisation
        - offset
        - range
        - sampling_rate
        """
        f5_read = self.infile.get_read(read_id)
        return f5_read.get_channel_info()

    def get_signal(self, read_id):
        f5_read = self.infile.get_read(read_id)
        # Retrieve raw signal and convert to pA
        raw_signal = f5_read.get_raw_data(scale=True)
        return np.array(raw_signal, dtype=np.float32)


class Pod5Reader(SignalReaderBase):
    """
    Signal reader implementation for POD5 files.
    """

    def __enter__(self):
        self.infile = pod5.Reader(self.filename)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.infile.close()

    def get_read_ids(self):
        return self.infile.read_ids

    def get_channel_info(self, read_id):
        """
        Returns a dictionary containing channel location info.
        Note: Pod5 separates calibration (offset/scale) from channel info.
        """
        r = next(self.infile.reads(selection=[read_id]))
        return {
            'channel_number': r.channel.channel,
            'well': r.channel.well
        }

    def get_signal(self, read_id):
        """
        Retrieve calibrated signal directly using signal_pa.
        """
        r = next(self.infile.reads(selection=[read_id]))
        return np.array(r.signal_pa, dtype=np.float32)


def get_reader(filename):
    """
    Factory function to get the appropriate reader based on file extension.
    :param filename: Path to the FAST5 or POD5 file.
    :return: Instance of Fast5Reader or Pod5Reader.
    :raises ValueError: If the file format is not supported.
    """
    if filename.endswith(".pod5"):
        return Pod5Reader(filename)
    elif filename.endswith(".fast5"):
        return Fast5Reader(filename)
    else:
        raise ValueError(f"Unsupported file format: {filename}")
