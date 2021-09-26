// Copyright 2019 The Prometheus Authors
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

package btrfs

import (
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/prometheus/procfs/internal/fs"
	"github.com/prometheus/procfs/internal/util"
)

// SectorSize contains the Linux sector size.
// > Linux always considers sectors to be 512 bytes long independently
// > of the devices real block size.
const SectorSize = 512

// FS represents the pseudo-filesystem sys, which provides an interface to
// kernel data structures.
type FS struct {
	sys *fs.FS
}

// NewDefaultFS returns a new Bcache using the default sys fs mount point. It will error
// if the mount point can't be read.
func NewDefaultFS() (FS, error) {
	return NewFS(fs.DefaultSysMountPoint)
}

// NewFS returns a new Btrfs filesystem using the given sys fs mount point. It will error
// if the mount point can't be read.
func NewFS(mountPoint string) (FS, error) {
	if strings.TrimSpace(mountPoint) == "" {
		mountPoint = fs.DefaultSysMountPoint
	}
	sys, err := fs.NewFS(mountPoint)
	if err != nil {
		return FS{}, err
	}
	return FS{&sys}, nil
}

// Stats retrieves Btrfs filesystem runtime statistics for each mounted Btrfs filesystem.
func (fs FS) Stats() ([]*Stats, error) {
	matches, err := filepath.Glob(fs.sys.Path("fs/btrfs/*-*"))
	if err != nil {
		return nil, err
	}

	stats := make([]*Stats, 0, len(matches))
	for _, uuidPath := range matches {
		s, err := GetStats(uuidPath)
		if err != nil {
			return nil, err
		}

		// Set the UUID from the path when it could not be retrieved from the filesystem.
		if s.UUID == "" {
			s.UUID = filepath.Base(uuidPath)
		}

		stats = append(stats, s)
	}

	return stats, nil
}

// GetStats collects all Btrfs statistics from sysfs
func GetStats(uuidPath string) (*Stats, error) {
	r := &reader{path: uuidPath}
	s := r.readFilesystemStats()
	if r.err != nil {
		return nil, r.err
	}

	return s, nil
}

type reader struct {
	path     string
	err      error
	devCount int
}

// readFile reads a file relative to the path of the reader.
// Non-existing files are ignored.
func (r *reader) readFile(n string) string {
	b, err := util.SysReadFile(path.Join(r.path, n))
	if err != nil && !os.IsNotExist(err) {
		r.err = err
	}
	return strings.TrimSpace(string(b))
}

// readValues reads a number of numerical values into an uint64 slice.
func (r *reader) readValue(n string) (v uint64) {
	// Read value from file
	s := r.readFile(n)
	if r.err != nil {
		return
	}

	// Convert number
	v, _ = strconv.ParseUint(s, 10, 64)
	return
}

// listFiles returns a list of files for a directory of the reader.
func (r *reader) listFiles(p string) []string {
	files, err := ioutil.ReadDir(path.Join(r.path, p))
	if err != nil {
		r.err = err
		return nil
	}

	names := make([]string, len(files))
	for i, f := range files {
		names[i] = f.Name()
	}
	return names
}

// readAllocationStats reads Btrfs allocation data for the current path.
func (r *reader) readAllocationStats(d string) (a *AllocationStats) {
	// Create a reader for this subdirectory
	sr := &reader{path: path.Join(r.path, d), devCount: r.devCount}

	// Get the stats
	a = &AllocationStats{
		// Read basic allocation stats
		MayUseBytes:      sr.readValue("bytes_may_use"),
		PinnedBytes:      sr.readValue("bytes_pinned"),
		ReadOnlyBytes:    sr.readValue("bytes_readonly"),
		ReservedBytes:    sr.readValue("bytes_reserved"),
		UsedBytes:        sr.readValue("bytes_used"),
		DiskUsedBytes:    sr.readValue("disk_used"),
		DiskTotalBytes:   sr.readValue("disk_total"),
		Flags:            sr.readValue("flags"),
		TotalBytes:       sr.readValue("total_bytes"),
		TotalPinnedBytes: sr.readValue("total_bytes_pinned"),
		Layouts:          sr.readLayouts(),
	}

	// Pass any error back
	r.err = sr.err

	return
}

// readLayouts reads all Btrfs layout statistics for the current path
func (r *reader) readLayouts() map[string]*LayoutUsage {
	files, err := ioutil.ReadDir(r.path)
	if err != nil {
		r.err = err
		return nil
	}

	m := make(map[string]*LayoutUsage)
	for _, f := range files {
		if f.IsDir() {
			m[f.Name()] = r.readLayout(f.Name())
		}
	}

	return m
}

// readLayout reads the Btrfs layout statistics for an allocation layout.
func (r *reader) readLayout(p string) (l *LayoutUsage) {
	l = new(LayoutUsage)
	l.TotalBytes = r.readValue(path.Join(p, "total_bytes"))
	l.UsedBytes = r.readValue(path.Join(p, "used_bytes"))
	l.Ratio = r.calcRatio(p)

	return
}

// calcRatio returns the calculated ratio for a layout mode.
func (r *reader) calcRatio(p string) float64 {
	switch p {
	case "single", "raid0":
		return 1
	case "dup", "raid1", "raid10":
		return 2
	case "raid5":
		return float64(r.devCount) / (float64(r.devCount) - 1)
	case "raid6":
		return float64(r.devCount) / (float64(r.devCount) - 2)
	default:
		return 0
	}
}

// readDeviceInfo returns the information for all devices associated with this filesystem.
func (r *reader) readDeviceInfo(d string) map[string]*Device {
	devs := r.listFiles("devices")
	info := make(map[string]*Device, len(devs))
	for _, n := range devs {
		info[n] = &Device{
			Size: SectorSize * r.readValue("devices/"+n+"/size"),
		}
	}

	return info
}

// readFilesystemStats reads Btrfs statistics for a filesystem.
func (r *reader) readFilesystemStats() (s *Stats) {
	// First get disk info, and add it to reader
	devices := r.readDeviceInfo("devices")
	r.devCount = len(devices)

	s = &Stats{
		// Read basic filesystem information
		Label:          r.readFile("label"),
		UUID:           r.readFile("metadata_uuid"),
		Features:       r.listFiles("features"),
		CloneAlignment: r.readValue("clone_alignment"),
		NodeSize:       r.readValue("nodesize"),
		QuotaOverride:  r.readValue("quota_override"),
		SectorSize:     r.readValue("sectorsize"),

		// Device info
		Devices: devices,

		// Read allocation data
		Allocation: Allocation{
			GlobalRsvReserved: r.readValue("allocation/global_rsv_reserved"),
			GlobalRsvSize:     r.readValue("allocation/global_rsv_size"),
			Data:              r.readAllocationStats("allocation/data"),
			Metadata:          r.readAllocationStats("allocation/metadata"),
			System:            r.readAllocationStats("allocation/system"),
		},
	}
	return
}
