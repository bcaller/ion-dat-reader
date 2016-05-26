# Copyright Ben Caller 2016

import struct
from collections import namedtuple
from math import ceil, floor

import argparse
import numpy as np

# Struct definitions
# https://github.com/iontorrent/TS/blob/4a916c73c28801dc0949d996438c794c199b1e6c/Analysis/datahdr.h
FileHeader = namedtuple("FileHeader", "sig struct_version header_size data_size")
SIZEOF_FILE_HEADER = 4 * 4
ExperimentHeader = namedtuple("ExperimentHeader",
                              "wall_time rows columns x_region_size y_region_size "
                              "frame_count uncomp_frame_count sample_rate "
                              "channel_offset_1 channel_offset_2 channel_offset_3 channel_offset_4 "
                              "hw_interlace_type interlaceType")
SIZEOF_EXPT_HEADER = 2 * 6 + 4 + 2 * 6 + 4
CompressedFrameHeader = namedtuple("CompressedFrameHeader", "timestamp size Transitions total sentinel")
SIZEOF_COMPRESSED_HEADER_DATA = 16
FILE_SIGNATURE = 0xdeadbeef
PLACEKEY_SENTINEL = 0xdeadbeef
SKIP = 4294967295
SIGNAL_MASK = np.uint16(0x3fff)
KEY_16_1 = 187  # 0xBB
KEY_STATE_CHANGE = 127  # 0x7F


def check_file_header(header):
    if not header.struct_version == 4:
        raise Exception("This program only works with version 4 of the dat file")
    if not header.sig == FILE_SIGNATURE:
        raise Exception("The signature was wrong for an Ion dat file")
    assert header.header_size == SIZEOF_EXPT_HEADER


def main(filename):
    with open(filename, "rb") as dat:
        header = FileHeader(*struct.unpack('>4I', dat.read(SIZEOF_FILE_HEADER)))

        check_file_header(header)

        experiment = ExperimentHeader(*struct.unpack('>I6HI6H', dat.read(SIZEOF_EXPT_HEADER)))

        if experiment.uncomp_frame_count < experiment.frame_count or experiment.uncomp_frame_count >= experiment.frame_count * 4:
            raise Exception("Unknown compression")

        regions_x = ceil(experiment.columns / experiment.x_region_size)
        regions_y = ceil(experiment.rows / experiment.y_region_size)

        well_number = lambda row, col: row * experiment.columns + col
        row_number = lambda w: floor(w / experiment.columns)
        col_number = lambda w: w % experiment.columns

        print(header)
        print(experiment)
        voltage_signals = np.empty((experiment.frame_count, experiment.rows, experiment.columns), np.dtype('H'))

        for frame_num in range(experiment.frame_count):  # experiment.frame_count
            read_frame(dat, experiment, frame_num, regions_x, regions_y, voltage_signals)

        print("Cell 0, 0 has values")
        print(", ".join(str(t) for t in voltage_signals[:, 0, 0]))
        print("Cell 722, 720 has values")
        print(", ".join(str(t) for t in voltage_signals[:, 722, 720]))
        print("Cell 616, 616 has values")
        print(", ".join(str(t) for t in voltage_signals[:, 616, 616]))


def read_frame(dat, experiment, frame_num, regions_x, regions_y, voltage_signals):
    timestamp, compressed = struct.unpack('>2I', dat.read(2 * 4))
    print(timestamp, compressed)
    if compressed > 1:
        raise Exception("Not implemented")
    elif compressed == 1:
        frame_header = CompressedFrameHeader(timestamp,
                                             *struct.unpack('>4I', dat.read(SIZEOF_COMPRESSED_HEADER_DATA)))
        print(frame_num, frame_header)

        if not frame_header.sentinel == PLACEKEY_SENTINEL:
            raise Exception("Bad sentinel")
            # https://github.com/iontorrent/TS/blob/4a916c73c28801dc0949d996438c794c199b1e6c/Analysis/Image/deInterlace.cpp#L376
            # dat.read(frame_header.size - SIZEOF_COMPRESSED_HEADER_DATA)

        region_offsets = read_region_offsets(dat, frame_header, regions_x, regions_y)
        data_size = frame_header.size - SIZEOF_COMPRESSED_HEADER_DATA - regions_x * regions_y * 4
        print("Data size:", data_size, "Bytes per well:", (frame_header.size - SIZEOF_COMPRESSED_HEADER_DATA) / (experiment.rows * experiment.columns))
        if data_size < 4:
            print("No data for this frame")
            if data_size > 0:
                dat.read(data_size)
        else:
            raw_data = np.fromfile(dat, dtype=np.dtype('B'), count=data_size)
            voltage_signals[frame_num] = voltage_signals[frame_num - 1].copy()
            transitions_seen = 0
            total = np.zeros(1, dtype=np.uint32)
            for yr in range(regions_y):
                for xr in range(regions_x):
                    region_id = yr * regions_x + xr
                    offset = region_offsets[region_id]
                    if not offset == SKIP:
                        # print("REGION", yr, xr, region_id, offset, raw_data[offset])
                        state = 0
                        x_size, y_size = region_size(experiment, xr, yr)
                        x_start, y_start = experiment.x_region_size * xr, experiment.y_region_size * yr
                        x_end, y_end = x_start + x_size, y_start + y_size
                        old_signal_for_region = voltage_signals[frame_num - 1, y_start:y_end, x_start:x_end]
                        wells_in_region = old_signal_for_region.reshape(-1)  # Makes copy
                        # Each block of eight adjacent wells is interlaced
                        for start_of_eight_wells in range(0, ceil(x_size * y_size / 8) * 8, 8):

                            # Check for state change
                            if raw_data[offset] == KEY_STATE_CHANGE:
                                if raw_data[offset + 1] & 0x0f == KEY_16_1:
                                    state = 16
                                else:
                                    state = raw_data[offset + 1] & 0xf
                                    if state > 8:
                                        state = 16
                                offset += 2
                                transitions_seen += 1
                                # print("STATE CHANGE", state, transitions_seen, frame_header.Transitions)
                            deinterlace_eight_values(
                                wells_in_region[start_of_eight_wells:(start_of_eight_wells + 8)],
                                raw_data[offset:(offset + state)],
                                state)
                            # State means the number of bytes encoding the eight wells
                            offset += state
                            # if frame_num > 8:
                            #     print(state, total[0], frame_header.total, transitions_seen, frame_header.Transitions)
                        voltage_signals[frame_num, y_start:y_end, x_start:x_end] = \
                            wells_in_region.reshape(y_size, x_size)
                        total += np.sum(wells_in_region)
            if not transitions_seen == frame_header.Transitions:
                raise Exception(
                    "Expected {} transitions: {} seen".format(frame_header.Transitions, transitions_seen))
            if not np.asscalar(total) == frame_header.total:
                raise Exception("Expected total of {}: {} seen".format(frame_header.total, total))
                # print(voltage_signals[frame_num])
                # np.set_printoptions(threshold=np.nan)
                # print(signal)
                # break
    else:  # First Frame
        stride = experiment.rows * experiment.columns * 2
        # dat.read(stride)
        # dat.read((15 * experiment.columns + 15) * 2)
        # for well in range(100):
        voltage_signals[frame_num] = np.bitwise_and(
            np.fromfile(dat, dtype=np.dtype('>H'), count=experiment.columns * experiment.rows),
            SIGNAL_MASK).reshape(experiment.rows, experiment.columns)

        # signal = np.bitwise_and(struct.unpack('>H', dat.read(2))[0], SIGNAL_MASK)
        print(voltage_signals[frame_num])
        total = np.sum(voltage_signals[frame_num])
        print(total)
        # break


        # LoadCompressedRegionImage
        # https://github.com/iontorrent/TS/blob/4a916c73c28801dc0949d996438c794c199b1e6c/Analysis/Image/deInterlace.cpp#L708


def deinterlace_eight_values(cells_in_region, signal, state):
    # https://github.com/iontorrent/TS/blob/master/Analysis/Image/deInterlace.cpp#L1049
    eight_values = np.zeros(8, dtype=np.dtype('i2'))
    if state == 3:
        eight_values[0] = (signal[0] >> 5) & 0x7
        eight_values[1] = (signal[0] >> 2) & 0x7
        eight_values[2] = ((signal[0] << 1) & 0x6) | ((signal[1] >> 7) & 1)
        eight_values[3] = ((signal[1] >> 4) & 0x7)
        eight_values[4] = ((signal[1] >> 1) & 0x7)
        eight_values[5] = ((signal[1] << 2) & 0x4) | ((signal[2] >> 6) & 3)
        eight_values[6] = ((signal[2] >> 3) & 0x7)
        eight_values[7] = ((signal[2]) & 0x7)
    elif state == 4:
        eight_values[0] = (signal[0] >> 4) & 0xf
        eight_values[1] = (signal[0]) & 0xf
        eight_values[2] = (signal[1] >> 4) & 0xf
        eight_values[3] = (signal[1]) & 0xf
        eight_values[4] = (signal[2] >> 4) & 0xf
        eight_values[5] = (signal[2]) & 0xf
        eight_values[6] = (signal[3] >> 4) & 0xf
        eight_values[7] = (signal[3]) & 0xf
    elif state == 5:
        eight_values[0] = (signal[0] >> 3) & 0x1f
        eight_values[1] = ((signal[0] << 2) & 0x1c) | ((signal[1] >> 6) & 0x3)
        eight_values[2] = (signal[1] >> 1) & 0x1f
        eight_values[3] = ((signal[1] << 4) & 0x10) | ((signal[2] >> 4) & 0xf)
        eight_values[4] = ((signal[2] << 1) & 0x1e) | ((signal[3] >> 7) & 0x1)
        eight_values[5] = (signal[3] >> 2) & 0x1f
        eight_values[6] = ((signal[3] << 3) & 0x18) | ((signal[4] >> 5) & 0x7)
        eight_values[7] = (signal[4]) & 0x1f
    elif state == 6:
        eight_values[0] = (signal[0] >> 2) & 0x3f
        eight_values[1] = ((signal[0] << 4) & 0x30) | ((signal[1] >> 4) & 0xf)
        eight_values[2] = ((signal[1] << 2) & 0x3c) | ((signal[2] >> 6) & 0x3)
        eight_values[3] = (signal[2] & 0x3f)
        eight_values[4] = (signal[3] >> 2) & 0x3f
        eight_values[5] = ((signal[3] << 4) & 0x30) | ((signal[4] >> 4) & 0xf)
        eight_values[6] = ((signal[4] << 2) & 0x3c) | ((signal[5] >> 6) & 0x3)
        eight_values[7] = (signal[5] & 0x3f)
    elif state == 7:
        eight_values[0] = (signal[0] >> 1) & 0x7f
        eight_values[1] = ((signal[0] << 6) & 0x40) | ((signal[1] >> 2) & 0x3f)
        eight_values[2] = ((signal[1] << 5) & 0x60) | ((signal[2] >> 3) & 0x1f)
        eight_values[3] = ((signal[2] << 4) & 0x70) | ((signal[3] >> 4) & 0x0f)
        eight_values[4] = ((signal[3] << 3) & 0x78) | ((signal[4] >> 5) & 0x07)
        eight_values[5] = ((signal[4] << 2) & 0x7c) | ((signal[5] >> 6) & 0x3)
        eight_values[6] = ((signal[5] << 1) & 0x7e) | ((signal[6] >> 7) & 0x1)
        eight_values[7] = (signal[6] & 0x7f)
    elif state == 8:
        np.copyto(eight_values, signal)  # Signal has 8 values
    elif state == 16:
        eight_values[0] = (signal[0] << 8) | signal[1]
        eight_values[1] = (signal[2] << 8) | signal[3]
        eight_values[2] = (signal[4] << 8) | signal[5]
        eight_values[3] = (signal[6] << 8) | signal[7]
        eight_values[4] = (signal[8] << 8) | signal[9]
        eight_values[5] = (signal[10] << 8) | signal[11]
        eight_values[6] = (signal[12] << 8) | signal[13]
        eight_values[7] = (signal[14] << 8) | signal[15]
    else:
        raise Exception("Bad state: {}".format(state))
    if not state == 16:
        eight_values -= 1 << (state - 1)

    cells_in_region += eight_values[:len(cells_in_region)]


def read_region_offsets(dat, frame_header, regions_x, regions_y):
    # Could not find this in the C++ repo so used the Java one
    # https://github.com/chenopodium/TorrentScout/blob/c02a644480c386f767974e06b41682de0be38b45/TorrentScoutWellsAccess/src/com/iontorrent/rawdataaccess/pgmacquisition/PGMRegionFrameReader.java#L568
    max_offset = frame_header.size - SIZEOF_COMPRESSED_HEADER_DATA - 4 - regions_x * regions_y * 4
    region_offsets = np.fromfile(dat, dtype=np.dtype('>I'), count=regions_x * regions_y)
    delta = 0
    for yr in range(regions_y):
        for xr in range(regions_x):
            region_id = yr * regions_x + xr

            if delta == 0 and not region_offsets[region_id] == SKIP:
                delta = region_offsets[region_id]

            if region_offsets[region_id] == SKIP:
                pass
            elif region_offsets[region_id] - delta > max_offset:
                raise Exception("Region diff beyond end of frame. Should mean skip!")
            else:
                region_offsets[region_id] -= delta
                if region_offsets[region_id] < 0:
                    raise Exception("negative region offset < 0 AFTER SUBTRACTING DELTA")
    return region_offsets


def region_generator(experiment, xr, yr, x_size, y_size):
    for y in range(y_size * yr, y_size * (yr + 1)):
        for x in range(x_size * xr, x_size * (xr + 1)):
            # yield (x, y)
            yield y * experiment.columns + x


def region_size(experiment, xr, yr):
    x_size = experiment.x_region_size if (xr + 1) * experiment.x_region_size <= experiment.columns \
        else experiment.columns % experiment.x_region_size
    y_size = experiment.y_region_size if (yr + 1) * experiment.y_region_size <= experiment.rows \
        else experiment.rows % experiment.y_region_size
    return x_size, y_size


if __name__ == "__main__":
    print("IonDatReader by Ben Caller")
    print("Reads each frame of a dat file and prints out the values for certain wells")
    parser = argparse.ArgumentParser(description="Read an Ion Torrent dat file",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dat", help="Filename of dat file to read")
    args = parser.parse_args()
    main(args.dat)
