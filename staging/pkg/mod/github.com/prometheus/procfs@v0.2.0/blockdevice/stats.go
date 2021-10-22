// Copyright 2018 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package blockdevice

import (
	"bufio"
	"fmt"
	"github.com/prometheus/procfs/internal/util"
	"io"
	"io/ioutil"
	"os"
	"strings"

	"github.com/prometheus/procfs/internal/fs"
)

// Info contains identifying information for a block device such as a disk drive
type Info struct {
	MajorNumber uint32
	MinorNumber uint32
	DeviceName  string
}

// IOStats models the iostats data described in the kernel documentation
// https://www.kernel.org/doc/Documentation/iostats.txt,
// https://www.kernel.org/doc/Documentation/block/stat.txt,
// and https://www.kernel.org/doc/Documentation/ABI/testing/procfs-diskstats
type IOStats struct {
	// ReadIOs is the number of reads completed successfully.
	ReadIOs uint64
	// ReadMerges is the number of reads merged.  Reads and writes
	// which are adjacent to each other may be merged for efficiency.
	ReadMerges uint64
	// ReadSectors is the total number of sectors read successfully.
	ReadSectors uint64
	// ReadTicks is the total number of milliseconds spent by all reads.
	ReadTicks uint64
	// WriteIOs is the total number of writes completed successfully.
	WriteIOs uint64
	// WriteMerges is the number of reads merged.
	WriteMerges uint64
	// WriteSectors is the total number of sectors written successfully.
	WriteSectors uint64
	// WriteTicks is the total number of milliseconds spent by all writes.
	WriteTicks uint64
	// IOsInProgress is number of I/Os currently in progress.
	IOsInProgress uint64
	// IOsTotalTicks is the number of milliseconds spent doing I/Os.
	// This field increases so long as IosInProgress is nonzero.
	IOsTotalTicks uint64
	// WeightedIOTicks is the weighted number of milliseconds spent doing I/Os.
	// This can also be used to estimate average queue wait time for requests.
	WeightedIOTicks uint64
	// DiscardIOs is the total number of discards completed successfully.
	DiscardIOs uint64
	// DiscardMerges is the number of discards merged.
	DiscardMerges uint64
	// DiscardSectors is the total number of sectors discarded successfully.
	DiscardSectors uint64
	// DiscardTicks is the total number of milliseconds spent by all discards.
	DiscardTicks uint64
	// FlushRequestsCompleted is the total number of flush request completed successfully.
	FlushRequestsCompleted uint64
	// TimeSpentFlushing is the total number of milliseconds spent flushing.
	TimeSpentFlushing uint64
}

// Diskstats combines the device Info and IOStats
type Diskstats struct {
	Info
	IOStats
	// IoStatsCount contains the number of io stats read. For kernel versions 5.5+,
	// there should be 20 fields read. For kernel versions 4.18+,
	// there should be 18 fields read. For earlier kernel versions this
	// will be 14 because the discard values are not available.
	IoStatsCount int
}

// BlockQueueStats models the queue files that are located in the sysfs tree for each block device
// and described in the kernel documentation:
// https://www.kernel.org/doc/Documentation/block/queue-sysfs.txt
// https://www.kernel.org/doc/html/latest/block/queue-sysfs.html
type BlockQueueStats struct {
	// AddRandom is the status of a disk entropy (1 is on, 0 is off).
	AddRandom uint64
	// Dax indicates whether the device supports Direct Access (DAX) (1 is on, 0 is off).
	DAX uint64
	// DiscardGranularity is the size of internal allocation of the device in bytes, 0 means device
	// does not support the discard functionality.
	DiscardGranularity uint64
	// DiscardMaxHWBytes is the hardware maximum number of bytes that can be discarded in a single operation,
	// 0 means device does not support the discard functionality.
	DiscardMaxHWBytes uint64
	// DiscardMaxBytes is the software maximum number of bytes that can be discarded in a single operation.
	DiscardMaxBytes uint64
	// HWSectorSize is the sector size of the device, in bytes.
	HWSectorSize uint64
	// IOPoll indicates if polling is enabled (1 is on, 0 is off).
	IOPoll uint64
	// IOPollDelay indicates how polling will be performed, -1 for classic polling, 0 for hybrid polling,
	// with greater than 0 the kernel will put process issuing IO to sleep for this amount of time in
	// microseconds before entering classic polling.
	IOPollDelay int64
	// IOTimeout is the request timeout in milliseconds.
	IOTimeout uint64
	// IOStats indicates if iostats accounting is used for the disk (1 is on, 0 is off).
	IOStats uint64
	// LogicalBlockSize is the logical block size of the device, in bytes.
	LogicalBlockSize uint64
	// MaxHWSectorsKB is the maximum number of kilobytes supported in a single data transfer.
	MaxHWSectorsKB uint64
	// MaxIntegritySegments is the max limit of integrity segments as set by block layer which a hardware controller
	// can handle.
	MaxIntegritySegments uint64
	// MaxSectorsKB is the maximum number of kilobytes that the block layer will allow for a filesystem request.
	MaxSectorsKB uint64
	// MaxSegments is the number of segments on the device.
	MaxSegments uint64
	// MaxSegmentsSize is the maximum segment size of the device.
	MaxSegmentSize uint64
	// MinimumIOSize is the smallest preferred IO size reported by the device.
	MinimumIOSize uint64
	// NoMerges shows the lookup logic involved with IO merging requests in the block layer. 0 all merges are
	// enabled, 1 only simple one hit merges are tried, 2 no merge algorithms will be tried.
	NoMerges uint64
	// NRRequests is the number of how many requests may be allocated in the block layer for read or write requests.
	NRRequests uint64
	// OptimalIOSize is the optimal IO size reported by the device.
	OptimalIOSize uint64
	// PhysicalBlockSize is the physical block size of device, in bytes.
	PhysicalBlockSize uint64
	// ReadAHeadKB is the maximum number of kilobytes to read-ahead for filesystems on this block device.
	ReadAHeadKB uint64
	// Rotational indicates if the device is of rotational type or non-rotational type.
	Rotational uint64
	// RQAffinity indicates affinity policy of device, if 1 the block layer will migrate request completions to the
	// cpu “group” that originally submitted the request, if 2 forces the completion to run on the requesting cpu.
	RQAffinity uint64
	// SchedulerList contains list of available schedulers for this block device.
	SchedulerList []string
	// SchedulerCurrent is the current scheduler for this block device.
	SchedulerCurrent string
	// WriteCache shows the type of cache for block device, "write back" or "write through".
	WriteCache string
	// WriteSameMaxBytes is the number of bytes the device can write in a single write-same command.
	// A value of ‘0’ means write-same is not supported by this device.
	WriteSameMaxBytes uint64
	// WBTLatUSec is the target minimum read latency, 0 means feature is disables.
	WBTLatUSec int64
	// ThrottleSampleTime is the time window that blk-throttle samples data, in millisecond. Optional
	// exists only if CONFIG_BLK_DEV_THROTTLING_LOW is enabled.
	ThrottleSampleTime *uint64
	// Zoned indicates if the device is a zoned block device and the zone model of the device if it is indeed zoned.
	// Possible values are: none, host-aware, host-managed for zoned block devices.
	Zoned string
	// NRZones indicates the total number of zones of the device, always zero for regular block devices.
	NRZones uint64
	// ChunksSectors for RAID is the size in 512B sectors of the RAID volume stripe segment,
	// for zoned host device is the size in 512B sectors.
	ChunkSectors uint64
	// FUA indicates whether the device supports Force Unit Access for write requests.
	FUA uint64
	// MaxDiscardSegments is the maximum number of DMA entries in a discard request.
	MaxDiscardSegments uint64
	// WriteZeroesMaxBytes the maximum number of bytes that can be zeroed at once.
	// The value 0 means that REQ_OP_WRITE_ZEROES is not supported.
	WriteZeroesMaxBytes uint64
}

const (
	procDiskstatsPath   = "diskstats"
	procDiskstatsFormat = "%d %d %s %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d"
	sysBlockPath        = "block"
	sysBlockStatFormat  = "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d"
	sysBlockQueue       = "queue"
)

// FS represents the pseudo-filesystems proc and sys, which provides an
// interface to kernel data structures.
type FS struct {
	proc *fs.FS
	sys  *fs.FS
}

// NewDefaultFS returns a new blockdevice fs using the default mountPoints for proc and sys.
// It will error if either of these mount points can't be read.
func NewDefaultFS() (FS, error) {
	return NewFS(fs.DefaultProcMountPoint, fs.DefaultSysMountPoint)
}

// NewFS returns a new blockdevice fs using the given mountPoints for proc and sys.
// It will error if either of these mount points can't be read.
func NewFS(procMountPoint string, sysMountPoint string) (FS, error) {
	if strings.TrimSpace(procMountPoint) == "" {
		procMountPoint = fs.DefaultProcMountPoint
	}
	procfs, err := fs.NewFS(procMountPoint)
	if err != nil {
		return FS{}, err
	}
	if strings.TrimSpace(sysMountPoint) == "" {
		sysMountPoint = fs.DefaultSysMountPoint
	}
	sysfs, err := fs.NewFS(sysMountPoint)
	if err != nil {
		return FS{}, err
	}
	return FS{&procfs, &sysfs}, nil
}

// ProcDiskstats reads the diskstats file and returns
// an array of Diskstats (one per line/device)
func (fs FS) ProcDiskstats() ([]Diskstats, error) {
	file, err := os.Open(fs.proc.Path(procDiskstatsPath))
	if err != nil {
		return nil, err
	}
	defer file.Close()

	diskstats := []Diskstats{}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		d := &Diskstats{}
		d.IoStatsCount, err = fmt.Sscanf(scanner.Text(), procDiskstatsFormat,
			&d.MajorNumber,
			&d.MinorNumber,
			&d.DeviceName,
			&d.ReadIOs,
			&d.ReadMerges,
			&d.ReadSectors,
			&d.ReadTicks,
			&d.WriteIOs,
			&d.WriteMerges,
			&d.WriteSectors,
			&d.WriteTicks,
			&d.IOsInProgress,
			&d.IOsTotalTicks,
			&d.WeightedIOTicks,
			&d.DiscardIOs,
			&d.DiscardMerges,
			&d.DiscardSectors,
			&d.DiscardTicks,
			&d.FlushRequestsCompleted,
			&d.TimeSpentFlushing,
		)
		// The io.EOF error can be safely ignored because it just means we read fewer than
		// the full 20 fields.
		if err != nil && err != io.EOF {
			return diskstats, err
		}
		if d.IoStatsCount >= 14 {
			diskstats = append(diskstats, *d)
		}
	}
	return diskstats, scanner.Err()
}

// SysBlockDevices lists the device names from /sys/block/<dev>
func (fs FS) SysBlockDevices() ([]string, error) {
	deviceDirs, err := ioutil.ReadDir(fs.sys.Path(sysBlockPath))
	if err != nil {
		return nil, err
	}
	devices := []string{}
	for _, deviceDir := range deviceDirs {
		if deviceDir.IsDir() {
			devices = append(devices, deviceDir.Name())
		}
	}
	return devices, nil
}

// SysBlockDeviceStat returns stats for the block device read from /sys/block/<device>/stat.
// The number of stats read will be 15 if the discard stats are available (kernel 4.18+)
// and 11 if they are not available.
func (fs FS) SysBlockDeviceStat(device string) (IOStats, int, error) {
	stat := IOStats{}
	bytes, err := ioutil.ReadFile(fs.sys.Path(sysBlockPath, device, "stat"))
	if err != nil {
		return stat, 0, err
	}
	count, err := fmt.Sscanf(strings.TrimSpace(string(bytes)), sysBlockStatFormat,
		&stat.ReadIOs,
		&stat.ReadMerges,
		&stat.ReadSectors,
		&stat.ReadTicks,
		&stat.WriteIOs,
		&stat.WriteMerges,
		&stat.WriteSectors,
		&stat.WriteTicks,
		&stat.IOsInProgress,
		&stat.IOsTotalTicks,
		&stat.WeightedIOTicks,
		&stat.DiscardIOs,
		&stat.DiscardMerges,
		&stat.DiscardSectors,
		&stat.DiscardTicks,
	)
	// An io.EOF error is ignored because it just means we read fewer than the full 15 fields.
	if err == io.EOF {
		return stat, count, nil
	}
	return stat, count, err
}

// SysBlockDeviceQueueStats returns stats for /sys/block/xxx/queue where xxx is a device name.
func (fs FS) SysBlockDeviceQueueStats(device string) (BlockQueueStats, error) {
	stat := BlockQueueStats{}
	// files with uint64 fields
	for file, p := range map[string]*uint64{
		"add_random":             &stat.AddRandom,
		"dax":                    &stat.DAX,
		"discard_granularity":    &stat.DiscardGranularity,
		"discard_max_hw_bytes":   &stat.DiscardMaxHWBytes,
		"discard_max_bytes":      &stat.DiscardMaxBytes,
		"hw_sector_size":         &stat.HWSectorSize,
		"io_poll":                &stat.IOPoll,
		"io_timeout":             &stat.IOTimeout,
		"iostats":                &stat.IOStats,
		"logical_block_size":     &stat.LogicalBlockSize,
		"max_hw_sectors_kb":      &stat.MaxHWSectorsKB,
		"max_integrity_segments": &stat.MaxIntegritySegments,
		"max_sectors_kb":         &stat.MaxSectorsKB,
		"max_segments":           &stat.MaxSegments,
		"max_segment_size":       &stat.MaxSegmentSize,
		"minimum_io_size":        &stat.MinimumIOSize,
		"nomerges":               &stat.NoMerges,
		"nr_requests":            &stat.NRRequests,
		"optimal_io_size":        &stat.OptimalIOSize,
		"physical_block_size":    &stat.PhysicalBlockSize,
		"read_ahead_kb":          &stat.ReadAHeadKB,
		"rotational":             &stat.Rotational,
		"rq_affinity":            &stat.RQAffinity,
		"write_same_max_bytes":   &stat.WriteSameMaxBytes,
		"nr_zones":               &stat.NRZones,
		"chunk_sectors":          &stat.ChunkSectors,
		"fua":                    &stat.FUA,
		"max_discard_segments":   &stat.MaxDiscardSegments,
		"write_zeroes_max_bytes": &stat.WriteZeroesMaxBytes,
	} {
		val, err := util.ReadUintFromFile(fs.sys.Path(sysBlockPath, device, sysBlockQueue, file))
		if err != nil {
			return BlockQueueStats{}, err
		}
		*p = val
	}
	// files with int64 fields
	for file, p := range map[string]*int64{
		"io_poll_delay": &stat.IOPollDelay,
		"wbt_lat_usec":  &stat.WBTLatUSec,
	} {
		val, err := util.ReadIntFromFile(fs.sys.Path(sysBlockPath, device, sysBlockQueue, file))
		if err != nil {
			return BlockQueueStats{}, err
		}
		*p = val
	}
	// files with string fields
	for file, p := range map[string]*string{
		"write_cache": &stat.WriteCache,
		"zoned":       &stat.Zoned,
	} {
		val, err := util.SysReadFile(fs.sys.Path(sysBlockPath, device, sysBlockQueue, file))
		if err != nil {
			return BlockQueueStats{}, err
		}
		*p = val
	}
	scheduler, err := util.SysReadFile(fs.sys.Path(sysBlockPath, device, sysBlockQueue, "scheduler"))
	if err != nil {
		return BlockQueueStats{}, err
	}
	var schedulers []string
	xs := strings.Split(scheduler, " ")
	for _, s := range xs {
		if strings.HasPrefix(s, "[") && strings.HasSuffix(s, "]") {
			s = s[1 : len(s)-1]
			stat.SchedulerCurrent = s
		}
		schedulers = append(schedulers, s)
	}
	stat.SchedulerList = schedulers
	// optional
	throttleSampleTime, err := util.ReadUintFromFile(fs.sys.Path(sysBlockPath, device, sysBlockQueue, "throttle_sample_time"))
	if err == nil {
		stat.ThrottleSampleTime = &throttleSampleTime
	}
	return stat, nil
}
