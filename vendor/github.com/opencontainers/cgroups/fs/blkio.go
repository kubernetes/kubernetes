package fs

import (
	"bufio"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/opencontainers/cgroups"
)

type BlkioGroup struct {
	weightFilename       string
	weightDeviceFilename string
}

func (s *BlkioGroup) Name() string {
	return "blkio"
}

func (s *BlkioGroup) Apply(path string, _ *cgroups.Resources, pid int) error {
	return apply(path, pid)
}

func (s *BlkioGroup) Set(path string, r *cgroups.Resources) error {
	s.detectWeightFilenames(path)
	if r.BlkioWeight != 0 {
		if err := cgroups.WriteFile(path, s.weightFilename, strconv.FormatUint(uint64(r.BlkioWeight), 10)); err != nil {
			return err
		}
	}

	if r.BlkioLeafWeight != 0 {
		if err := cgroups.WriteFile(path, "blkio.leaf_weight", strconv.FormatUint(uint64(r.BlkioLeafWeight), 10)); err != nil {
			return err
		}
	}
	for _, wd := range r.BlkioWeightDevice {
		if wd.Weight != 0 {
			if err := cgroups.WriteFile(path, s.weightDeviceFilename, wd.WeightString()); err != nil {
				return err
			}
		}
		if wd.LeafWeight != 0 {
			if err := cgroups.WriteFile(path, "blkio.leaf_weight_device", wd.LeafWeightString()); err != nil {
				return err
			}
		}
	}
	for _, td := range r.BlkioThrottleReadBpsDevice {
		if err := cgroups.WriteFile(path, "blkio.throttle.read_bps_device", td.String()); err != nil {
			return err
		}
	}
	for _, td := range r.BlkioThrottleWriteBpsDevice {
		if err := cgroups.WriteFile(path, "blkio.throttle.write_bps_device", td.String()); err != nil {
			return err
		}
	}
	for _, td := range r.BlkioThrottleReadIOPSDevice {
		if err := cgroups.WriteFile(path, "blkio.throttle.read_iops_device", td.String()); err != nil {
			return err
		}
	}
	for _, td := range r.BlkioThrottleWriteIOPSDevice {
		if err := cgroups.WriteFile(path, "blkio.throttle.write_iops_device", td.String()); err != nil {
			return err
		}
	}

	return nil
}

/*
examples:

    blkio.sectors
    8:0 6792

    blkio.io_service_bytes
    8:0 Read 1282048
    8:0 Write 2195456
    8:0 Sync 2195456
    8:0 Async 1282048
    8:0 Total 3477504
    Total 3477504

    blkio.io_serviced
    8:0 Read 124
    8:0 Write 104
    8:0 Sync 104
    8:0 Async 124
    8:0 Total 228
    Total 228

    blkio.io_queued
    8:0 Read 0
    8:0 Write 0
    8:0 Sync 0
    8:0 Async 0
    8:0 Total 0
    Total 0
*/

func splitBlkioStatLine(r rune) bool {
	return r == ' ' || r == ':'
}

func getBlkioStat(dir, file string) ([]cgroups.BlkioStatEntry, error) {
	var blkioStats []cgroups.BlkioStatEntry
	f, err := cgroups.OpenFile(dir, file, os.O_RDONLY)
	if err != nil {
		if os.IsNotExist(err) {
			return blkioStats, nil
		}
		return nil, err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	for sc.Scan() {
		// format: dev type amount
		fields := strings.FieldsFunc(sc.Text(), splitBlkioStatLine)
		if len(fields) < 3 {
			if len(fields) == 2 && fields[0] == "Total" {
				// skip total line
				continue
			} else {
				return nil, malformedLine(dir, file, sc.Text())
			}
		}

		v, err := strconv.ParseUint(fields[0], 10, 64)
		if err != nil {
			return nil, &parseError{Path: dir, File: file, Err: err}
		}
		major := v

		v, err = strconv.ParseUint(fields[1], 10, 64)
		if err != nil {
			return nil, &parseError{Path: dir, File: file, Err: err}
		}
		minor := v

		op := ""
		valueField := 2
		if len(fields) == 4 {
			op = fields[2]
			valueField = 3
		}
		v, err = strconv.ParseUint(fields[valueField], 10, 64)
		if err != nil {
			return nil, &parseError{Path: dir, File: file, Err: err}
		}
		blkioStats = append(blkioStats, cgroups.BlkioStatEntry{Major: major, Minor: minor, Op: op, Value: v})
	}
	if err := sc.Err(); err != nil {
		return nil, &parseError{Path: dir, File: file, Err: err}
	}

	return blkioStats, nil
}

func (s *BlkioGroup) GetStats(path string, stats *cgroups.Stats) error {
	type blkioStatInfo struct {
		filename            string
		blkioStatEntriesPtr *[]cgroups.BlkioStatEntry
	}
	bfqDebugStats := []blkioStatInfo{
		{
			filename:            "blkio.bfq.sectors_recursive",
			blkioStatEntriesPtr: &stats.BlkioStats.SectorsRecursive,
		},
		{
			filename:            "blkio.bfq.io_service_time_recursive",
			blkioStatEntriesPtr: &stats.BlkioStats.IoServiceTimeRecursive,
		},
		{
			filename:            "blkio.bfq.io_wait_time_recursive",
			blkioStatEntriesPtr: &stats.BlkioStats.IoWaitTimeRecursive,
		},
		{
			filename:            "blkio.bfq.io_merged_recursive",
			blkioStatEntriesPtr: &stats.BlkioStats.IoMergedRecursive,
		},
		{
			filename:            "blkio.bfq.io_queued_recursive",
			blkioStatEntriesPtr: &stats.BlkioStats.IoQueuedRecursive,
		},
		{
			filename:            "blkio.bfq.time_recursive",
			blkioStatEntriesPtr: &stats.BlkioStats.IoTimeRecursive,
		},
		{
			filename:            "blkio.bfq.io_serviced_recursive",
			blkioStatEntriesPtr: &stats.BlkioStats.IoServicedRecursive,
		},
		{
			filename:            "blkio.bfq.io_service_bytes_recursive",
			blkioStatEntriesPtr: &stats.BlkioStats.IoServiceBytesRecursive,
		},
	}
	bfqStats := []blkioStatInfo{
		{
			filename:            "blkio.bfq.io_serviced_recursive",
			blkioStatEntriesPtr: &stats.BlkioStats.IoServicedRecursive,
		},
		{
			filename:            "blkio.bfq.io_service_bytes_recursive",
			blkioStatEntriesPtr: &stats.BlkioStats.IoServiceBytesRecursive,
		},
	}
	cfqStats := []blkioStatInfo{
		{
			filename:            "blkio.sectors_recursive",
			blkioStatEntriesPtr: &stats.BlkioStats.SectorsRecursive,
		},
		{
			filename:            "blkio.io_service_time_recursive",
			blkioStatEntriesPtr: &stats.BlkioStats.IoServiceTimeRecursive,
		},
		{
			filename:            "blkio.io_wait_time_recursive",
			blkioStatEntriesPtr: &stats.BlkioStats.IoWaitTimeRecursive,
		},
		{
			filename:            "blkio.io_merged_recursive",
			blkioStatEntriesPtr: &stats.BlkioStats.IoMergedRecursive,
		},
		{
			filename:            "blkio.io_queued_recursive",
			blkioStatEntriesPtr: &stats.BlkioStats.IoQueuedRecursive,
		},
		{
			filename:            "blkio.time_recursive",
			blkioStatEntriesPtr: &stats.BlkioStats.IoTimeRecursive,
		},
		{
			filename:            "blkio.io_serviced_recursive",
			blkioStatEntriesPtr: &stats.BlkioStats.IoServicedRecursive,
		},
		{
			filename:            "blkio.io_service_bytes_recursive",
			blkioStatEntriesPtr: &stats.BlkioStats.IoServiceBytesRecursive,
		},
	}
	throttleRecursiveStats := []blkioStatInfo{
		{
			filename:            "blkio.throttle.io_serviced_recursive",
			blkioStatEntriesPtr: &stats.BlkioStats.IoServicedRecursive,
		},
		{
			filename:            "blkio.throttle.io_service_bytes_recursive",
			blkioStatEntriesPtr: &stats.BlkioStats.IoServiceBytesRecursive,
		},
	}
	baseStats := []blkioStatInfo{
		{
			filename:            "blkio.throttle.io_serviced",
			blkioStatEntriesPtr: &stats.BlkioStats.IoServicedRecursive,
		},
		{
			filename:            "blkio.throttle.io_service_bytes",
			blkioStatEntriesPtr: &stats.BlkioStats.IoServiceBytesRecursive,
		},
	}
	orderedStats := [][]blkioStatInfo{
		bfqDebugStats,
		bfqStats,
		cfqStats,
		throttleRecursiveStats,
		baseStats,
	}

	var blkioStats []cgroups.BlkioStatEntry
	var err error

	for _, statGroup := range orderedStats {
		for i, statInfo := range statGroup {
			if blkioStats, err = getBlkioStat(path, statInfo.filename); err != nil || blkioStats == nil {
				// if error occurs on first file, move to next group
				if i == 0 {
					break
				}
				return err
			}
			*statInfo.blkioStatEntriesPtr = blkioStats
			// finish if all stats are gathered
			if i == len(statGroup)-1 {
				return nil
			}
		}
	}
	return nil
}

func (s *BlkioGroup) detectWeightFilenames(path string) {
	if s.weightFilename != "" {
		// Already detected.
		return
	}
	if cgroups.PathExists(filepath.Join(path, "blkio.weight")) {
		s.weightFilename = "blkio.weight"
		s.weightDeviceFilename = "blkio.weight_device"
	} else {
		s.weightFilename = "blkio.bfq.weight"
		s.weightDeviceFilename = "blkio.bfq.weight_device"
	}
}
