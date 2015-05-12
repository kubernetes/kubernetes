// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build linux

// Provides Filesystem Stats
package fs

/*
 extern int getBytesFree(const char *path, unsigned long long *bytes);
 extern int getBytesTotal(const char *path, unsigned long long *bytes);
 extern int getBytesAvail(const char *path, unsigned long long *bytes);
*/
import "C"

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"syscall"
	"unsafe"

	"github.com/docker/docker/pkg/mount"
	"github.com/golang/glog"
)

var partitionRegex = regexp.MustCompile("^(:?(:?s|xv)d[a-z]+\\d*|dm-\\d+)$")

const (
	LabelSystemRoot   = "root"
	LabelDockerImages = "docker-images"
)

type partition struct {
	mountpoint string
	major      uint
	minor      uint
}

type RealFsInfo struct {
	// Map from block device path to partition information.
	partitions map[string]partition
	// Map from label to block device path.
	// Labels are intent-specific tags that are auto-detected.
	labels map[string]string
}

type Context struct {
	// docker root directory.
	DockerRoot string
}

func NewFsInfo(context Context) (FsInfo, error) {
	mounts, err := mount.GetMounts()
	if err != nil {
		return nil, err
	}
	partitions := make(map[string]partition, 0)
	fsInfo := &RealFsInfo{}
	fsInfo.labels = make(map[string]string, 0)
	for _, mount := range mounts {
		if !strings.HasPrefix(mount.Fstype, "ext") && mount.Fstype != "btrfs" {
			continue
		}
		// Avoid bind mounts.
		if _, ok := partitions[mount.Source]; ok {
			continue
		}
		partitions[mount.Source] = partition{mount.Mountpoint, uint(mount.Major), uint(mount.Minor)}
	}
	glog.Infof("Filesystem partitions: %+v", partitions)
	fsInfo.partitions = partitions
	fsInfo.addLabels(context)
	return fsInfo, nil
}

func (self *RealFsInfo) addLabels(context Context) {
	dockerPaths := getDockerImagePaths(context)
	for src, p := range self.partitions {
		if p.mountpoint == "/" {
			if _, ok := self.labels[LabelSystemRoot]; !ok {
				self.labels[LabelSystemRoot] = src
			}
		}
		self.updateDockerImagesPath(src, p.mountpoint, dockerPaths)
		// TODO(rjnagal): Add label for docker devicemapper pool.
	}
}

// Generate a list of possible mount points for docker image management from the docker root directory.
// Right now, we look for each type of supported graph driver directories, but we can do better by parsing
// some of the context from `docker info`.
func getDockerImagePaths(context Context) []string {
	// TODO(rjnagal): Detect docker root and graphdriver directories from docker info.
	dockerRoot := context.DockerRoot
	dockerImagePaths := []string{}
	for _, dir := range []string{"devicemapper", "btrfs", "aufs"} {
		dockerImagePaths = append(dockerImagePaths, path.Join(dockerRoot, dir))
	}
	for dockerRoot != "/" && dockerRoot != "." {
		dockerImagePaths = append(dockerImagePaths, dockerRoot)
		dockerRoot = filepath.Dir(dockerRoot)
	}
	dockerImagePaths = append(dockerImagePaths, "/")
	return dockerImagePaths
}

// This method compares the mountpoint with possible docker image mount points. If a match is found,
// docker images label is added to the partition.
func (self *RealFsInfo) updateDockerImagesPath(source string, mountpoint string, dockerImagePaths []string) {
	for _, v := range dockerImagePaths {
		if v == mountpoint {
			if i, ok := self.labels[LabelDockerImages]; ok {
				// pick the innermost mountpoint.
				mnt := self.partitions[i].mountpoint
				if len(mnt) < len(mountpoint) {
					self.labels[LabelDockerImages] = source
				}
			} else {
				self.labels[LabelDockerImages] = source
			}
		}
	}
}

func (self *RealFsInfo) GetDeviceForLabel(label string) (string, error) {
	dev, ok := self.labels[label]
	if !ok {
		return "", fmt.Errorf("non-existent label %q", label)
	}
	return dev, nil
}

func (self *RealFsInfo) GetLabelsForDevice(device string) ([]string, error) {
	labels := []string{}
	for label, dev := range self.labels {
		if dev == device {
			labels = append(labels, label)
		}
	}
	return labels, nil
}

func (self *RealFsInfo) GetMountpointForDevice(dev string) (string, error) {
	p, ok := self.partitions[dev]
	if !ok {
		return "", fmt.Errorf("no partition info for device %q", dev)
	}
	return p.mountpoint, nil
}

func (self *RealFsInfo) GetFsInfoForPath(mountSet map[string]struct{}) ([]Fs, error) {
	filesystems := make([]Fs, 0)
	deviceSet := make(map[string]struct{})
	diskStatsMap, err := getDiskStatsMap("/proc/diskstats")
	if err != nil {
		return nil, err
	}
	for device, partition := range self.partitions {
		_, hasMount := mountSet[partition.mountpoint]
		_, hasDevice := deviceSet[device]
		if mountSet == nil || (hasMount && !hasDevice) {
			total, free, avail, err := getVfsStats(partition.mountpoint)
			if err != nil {
				glog.Errorf("Statvfs failed. Error: %v", err)
			} else {
				deviceSet[device] = struct{}{}
				deviceInfo := DeviceInfo{
					Device: device,
					Major:  uint(partition.major),
					Minor:  uint(partition.minor),
				}
				fs := Fs{deviceInfo, total, free, avail, diskStatsMap[device]}
				filesystems = append(filesystems, fs)
			}
		}
	}
	return filesystems, nil
}

func getDiskStatsMap(diskStatsFile string) (map[string]DiskStats, error) {
	diskStatsMap := make(map[string]DiskStats)
	file, err := os.Open(diskStatsFile)
	if err != nil {
		if os.IsNotExist(err) {
			glog.Infof("not collecting filesystem statistics because file %q was not available", diskStatsFile)
			return diskStatsMap, nil
		}
		return nil, err
	}

	defer file.Close()
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := scanner.Text()
		words := strings.Fields(line)
		if !partitionRegex.MatchString(words[2]) {
			continue
		}
		// 8      50 sdd2 40 0 280 223 7 0 22 108 0 330 330
		deviceName := path.Join("/dev", words[2])
		wordLength := len(words)
		offset := 3
		var stats = make([]uint64, wordLength-offset)
		if len(stats) < 11 {
			return nil, fmt.Errorf("could not parse all 11 columns of /proc/diskstats")
		}
		var error error
		for i := offset; i < wordLength; i++ {
			stats[i-offset], error = strconv.ParseUint(words[i], 10, 64)
			if error != nil {
				return nil, error
			}
		}
		diskStats := DiskStats{
			ReadsCompleted:  stats[0],
			ReadsMerged:     stats[1],
			SectorsRead:     stats[2],
			ReadTime:        stats[3],
			WritesCompleted: stats[4],
			WritesMerged:    stats[5],
			SectorsWritten:  stats[6],
			WriteTime:       stats[7],
			IoInProgress:    stats[8],
			IoTime:          stats[9],
			WeightedIoTime:  stats[10],
		}
		diskStatsMap[deviceName] = diskStats
	}
	return diskStatsMap, nil
}

func (self *RealFsInfo) GetGlobalFsInfo() ([]Fs, error) {
	return self.GetFsInfoForPath(nil)
}

func major(devNumber uint64) uint {
	return uint((devNumber >> 8) & 0xfff)
}

func minor(devNumber uint64) uint {
	return uint((devNumber & 0xff) | ((devNumber >> 12) & 0xfff00))
}

func (self *RealFsInfo) GetDirFsDevice(dir string) (*DeviceInfo, error) {
	var buf syscall.Stat_t
	err := syscall.Stat(dir, &buf)
	if err != nil {
		return nil, fmt.Errorf("stat failed on %s with error: %s", dir, err)
	}
	major := major(buf.Dev)
	minor := minor(buf.Dev)
	for device, partition := range self.partitions {
		if partition.major == major && partition.minor == minor {
			return &DeviceInfo{device, major, minor}, nil
		}
	}
	return nil, fmt.Errorf("could not find device with major: %d, minor: %d in cached partitions map", major, minor)
}

func (self *RealFsInfo) GetDirUsage(dir string) (uint64, error) {
	out, err := exec.Command("du", "-s", dir).CombinedOutput()
	if err != nil {
		return 0, fmt.Errorf("du command failed on %s with output %s - %s", dir, out, err)
	}
	usageInKb, err := strconv.ParseUint(strings.Fields(string(out))[0], 10, 64)
	if err != nil {
		return 0, fmt.Errorf("cannot parse 'du' output %s - %s", out, err)
	}
	return usageInKb * 1024, nil
}

func getVfsStats(path string) (total uint64, free uint64, avail uint64, err error) {
	_p0, err := syscall.BytePtrFromString(path)
	if err != nil {
		return 0, 0, 0, err
	}
	res, err := C.getBytesFree((*C.char)(unsafe.Pointer(_p0)), (*_Ctype_ulonglong)(unsafe.Pointer(&free)))
	if res != 0 {
		return 0, 0, 0, err
	}
	res, err = C.getBytesTotal((*C.char)(unsafe.Pointer(_p0)), (*_Ctype_ulonglong)(unsafe.Pointer(&total)))
	if res != 0 {
		return 0, 0, 0, err
	}
	res, err = C.getBytesAvail((*C.char)(unsafe.Pointer(_p0)), (*_Ctype_ulonglong)(unsafe.Pointer(&avail)))
	if res != 0 {
		return 0, 0, 0, err
	}
	return total, free, avail, nil
}
