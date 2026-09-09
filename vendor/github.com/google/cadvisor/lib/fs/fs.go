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

//go:build linux

// Provides Filesystem Stats
package fs

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"syscall"

	mount "github.com/moby/sys/mountinfo"

	"k8s.io/klog/v2"
)

const (
	LabelSystemRoot     = "root"
	LabelDockerImages   = "docker-images"
	LabelCrioImages     = "crio-images"
	LabelCrioContainers = "crio-containers"
)

const (
	// The block size in bytes.
	statBlockSize uint64 = 512
	// The maximum number of `disk usage` tasks that can be running at once.
	maxConcurrentOps = 20
)

// A pool for restricting the number of consecutive `du` and `find` tasks running.
var pool = make(chan struct{}, maxConcurrentOps)

func init() {
	for i := 0; i < maxConcurrentOps; i++ {
		releaseToken()
	}
}

func claimToken() {
	<-pool
}

func releaseToken() {
	pool <- struct{}{}
}

type partition struct {
	mountpoint string
	major      uint
	minor      uint
	fsType     string
	blockSize  uint
}

type RealFsInfo struct {
	// Map from block device path to partition information.
	partitions map[string]partition
	// Map from label to block device path.
	// Labels are intent-specific tags that are auto-detected.
	labels map[string]string
	// Map from mountpoint to mount information.
	mounts map[string]mount.Info
	// fsUUIDToDeviceName is a map from the filesystem UUID to its device name.
	fsUUIDToDeviceName map[string]string
}

func NewFsInfo(context Context) (FsInfo, error) {
	fileReader, err := os.Open("/proc/self/mountinfo")
	if err != nil {
		return nil, err
	}
	mounts, err := mount.GetMountsFromReader(fileReader, nil)
	if err != nil {
		return nil, err
	}

	fsUUIDToDeviceName, err := getFsUUIDToDeviceNameMap()
	if err != nil {
		// UUID is not always available across different OS distributions.
		// Do not fail if there is an error.
		klog.Warningf("Failed to get disk UUID mapping, getting disk info by uuid will not work: %v", err)
	}

	fsInfo := &RealFsInfo{
		partitions:         processMounts(mounts, nil),
		labels:             make(map[string]string),
		mounts:             make(map[string]mount.Info),
		fsUUIDToDeviceName: fsUUIDToDeviceName,
	}

	for _, mnt := range mounts {
		fsInfo.mounts[mnt.Mountpoint] = *mnt
	}

	fsInfo.addDockerImagesLabel(context, mounts)
	fsInfo.addCrioImagesLabel(context, mounts)

	klog.V(1).Infof("Filesystem UUIDs: %+v", fsInfo.fsUUIDToDeviceName)
	klog.V(1).Infof("Filesystem partitions: %+v", fsInfo.partitions)
	fsInfo.addSystemRootLabel(mounts)
	return fsInfo, nil
}

// getFsUUIDToDeviceNameMap creates the filesystem uuid to device name map
// using the information in /dev/disk/by-uuid. If the directory does not exist,
// this function will return an empty map.
func getFsUUIDToDeviceNameMap() (map[string]string, error) {
	const dir = "/dev/disk/by-uuid"

	if _, err := os.Stat(dir); os.IsNotExist(err) {
		return make(map[string]string), nil
	}

	files, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	fsUUIDToDeviceName := make(map[string]string)
	for _, file := range files {
		fpath := filepath.Join(dir, file.Name())
		target, err := os.Readlink(fpath)
		if err != nil {
			klog.Warningf("Failed to resolve symlink for %q", fpath)
			continue
		}
		device, err := filepath.Abs(filepath.Join(dir, target))
		if err != nil {
			return nil, fmt.Errorf("failed to resolve the absolute path of %q", filepath.Join(dir, target))
		}
		fsUUIDToDeviceName[file.Name()] = device
	}
	return fsUUIDToDeviceName, nil
}

func processMounts(mounts []*mount.Info, excludedMountpointPrefixes []string) map[string]partition {
	partitions := make(map[string]partition)

	for _, mnt := range mounts {
		// Use plugin system to determine if filesystem is supported
		plugin := GetPluginForFsType(mnt.FSType)
		if plugin == nil {
			continue
		}

		// Avoid bind mounts, but allow tmpfs duplicates (handled by plugin's ProcessMount)
		if _, ok := partitions[mnt.Source]; ok {
			if mnt.FSType != "tmpfs" {
				continue
			}
		}

		// Check for excluded mountpoint prefixes
		hasPrefix := false
		for _, prefix := range excludedMountpointPrefixes {
			if strings.HasPrefix(mnt.Mountpoint, prefix) {
				hasPrefix = true
				break
			}
		}
		if hasPrefix {
			continue
		}

		// Let plugin process the mount (handles filesystem-specific modifications)
		include, processedMnt, err := plugin.ProcessMount(mnt)
		if err != nil {
			klog.Warningf("error processing mount for %s: %v", mnt.FSType, err)
			continue
		}
		if !include {
			continue
		}

		partitions[processedMnt.Source] = partition{
			fsType:     processedMnt.FSType,
			mountpoint: processedMnt.Mountpoint,
			major:      uint(processedMnt.Major),
			minor:      uint(processedMnt.Minor),
		}
	}

	return partitions
}

// addSystemRootLabel attempts to determine which device contains the mount for /.
func (i *RealFsInfo) addSystemRootLabel(mounts []*mount.Info) {
	for _, m := range mounts {
		if m.Mountpoint == "/" {
			i.partitions[m.Source] = partition{
				fsType:     m.FSType,
				mountpoint: m.Mountpoint,
				major:      uint(m.Major),
				minor:      uint(m.Minor),
			}
			i.labels[LabelSystemRoot] = m.Source
			return
		}
	}
}

// addDockerImagesLabel attempts to determine which device contains the mount for docker images.
func (i *RealFsInfo) addDockerImagesLabel(context Context, mounts []*mount.Info) {
	if context.Docker.Driver != "" {
		i.updateContainerImagesPath(LabelDockerImages, mounts, getDockerImagePaths(context))
	}
}

func (i *RealFsInfo) addCrioImagesLabel(context Context, mounts []*mount.Info) {
	labelCrioImageOrContainers := LabelCrioContainers
	// If imagestore is not specified, let's fall back to the original case.
	// Everything will be stored in crio-images
	if context.Crio.ImageStore == "" {
		labelCrioImageOrContainers = LabelCrioImages
	}
	if context.Crio.Root != "" {
		crioPath := context.Crio.Root
		crioImagePaths := map[string]struct{}{
			"/": {},
		}
		imageOrContainerPath := context.Crio.Driver + "-containers"
		if context.Crio.ImageStore == "" {
			// If ImageStore is not specified then we will assume ImageFs is complete separate.
			// No need to split the image store.
			imageOrContainerPath = context.Crio.Driver + "-images"

		}
		crioImagePaths[path.Join(crioPath, imageOrContainerPath)] = struct{}{}
		for crioPath != "/" && crioPath != "." {
			crioImagePaths[crioPath] = struct{}{}
			crioPath = filepath.Dir(crioPath)
		}
		i.updateContainerImagesPath(labelCrioImageOrContainers, mounts, crioImagePaths)
	}
	if context.Crio.ImageStore != "" {
		crioPath := context.Crio.ImageStore
		crioImagePaths := map[string]struct{}{
			"/": {},
		}
		crioImagePaths[path.Join(crioPath, context.Crio.Driver+"-images")] = struct{}{}
		for crioPath != "/" && crioPath != "." {
			crioImagePaths[crioPath] = struct{}{}
			crioPath = filepath.Dir(crioPath)
		}
		i.updateContainerImagesPath(LabelCrioImages, mounts, crioImagePaths)
	}
}

// Generate a list of possible mount points for docker image management from the docker root directory.
// Right now, we look for each type of supported graph driver directories, but we can do better by parsing
// some of the context from `docker info`.
func getDockerImagePaths(context Context) map[string]struct{} {
	dockerImagePaths := map[string]struct{}{
		"/": {},
	}

	// TODO(rjnagal): Detect docker root and graphdriver directories from docker info.
	dockerRoot := context.Docker.Root
	for _, dir := range []string{"devicemapper", "btrfs", "aufs", "overlay", "overlay2", "zfs"} {
		dockerImagePaths[path.Join(dockerRoot, dir)] = struct{}{}
	}
	for dockerRoot != "/" && dockerRoot != "." {
		dockerImagePaths[dockerRoot] = struct{}{}
		dockerRoot = filepath.Dir(dockerRoot)
	}
	return dockerImagePaths
}

// This method compares the mountpoints with possible container image mount points. If a match is found,
// the label is added to the partition.
func (i *RealFsInfo) updateContainerImagesPath(label string, mounts []*mount.Info, containerImagePaths map[string]struct{}) {
	var useMount *mount.Info
	for _, m := range mounts {
		if _, ok := containerImagePaths[m.Mountpoint]; ok {
			if useMount == nil || (len(useMount.Mountpoint) < len(m.Mountpoint)) {
				useMount = m
			}
		}
	}
	if useMount != nil {
		i.partitions[useMount.Source] = partition{
			fsType:     useMount.FSType,
			mountpoint: useMount.Mountpoint,
			major:      uint(useMount.Major),
			minor:      uint(useMount.Minor),
		}
		i.labels[label] = useMount.Source
	}
}

func (i *RealFsInfo) GetDeviceForLabel(label string) (string, error) {
	dev, ok := i.labels[label]
	if !ok {
		return "", fmt.Errorf("non-existent label %q", label)
	}
	return dev, nil
}

func (i *RealFsInfo) GetLabelsForDevice(device string) ([]string, error) {
	var labels []string
	for label, dev := range i.labels {
		if dev == device {
			labels = append(labels, label)
		}
	}
	return labels, nil
}

func (i *RealFsInfo) GetMountpointForDevice(dev string) (string, error) {
	p, ok := i.partitions[dev]
	if !ok {
		return "", fmt.Errorf("no partition info for device %q", dev)
	}
	return p.mountpoint, nil
}

func (i *RealFsInfo) GetFsInfoForPath(mountSet map[string]struct{}) ([]Fs, error) {
	filesystems := make([]Fs, 0)
	deviceSet := make(map[string]struct{})
	diskStatsMap, err := getDiskStatsMap("/proc/diskstats")
	if err != nil {
		return nil, err
	}
	// statsCache stores cached filesystem stats by cache key for plugins that implement FsCachingPlugin
	statsCache := make(map[string]Fs)
	for device, partition := range i.partitions {
		_, hasMount := mountSet[partition.mountpoint]
		_, hasDevice := deviceSet[device]
		if mountSet == nil || (hasMount && !hasDevice) {
			var (
				statsErr error
				fs       Fs
			)

			// Use plugin system to get filesystem stats
			plugin := GetPluginForFsType(partition.fsType)
			if plugin == nil {
				klog.V(4).Infof("no plugin found for filesystem type: %v", partition.fsType)
				continue
			}

			partInfo := PartitionInfo{
				Mountpoint: partition.mountpoint,
				Major:      partition.major,
				Minor:      partition.minor,
				FsType:     partition.fsType,
				BlockSize:  partition.blockSize,
			}

			// Check if plugin supports caching and if we have a cached value
			var cacheKey string
			if cachingPlugin, ok := plugin.(FsCachingPlugin); ok {
				cacheKey = cachingPlugin.CacheKey(partInfo)
				if cacheKey != "" {
					if cachedFs, found := statsCache[cacheKey]; found {
						fs = cachedFs
						// Skip stats fetching, use cached value
						deviceSet[device] = struct{}{}
						fs.DeviceInfo = DeviceInfo{
							Device: device,
							Major:  uint(partition.major),
							Minor:  uint(partition.minor),
						}
						if val, ok := diskStatsMap[device]; ok {
							fs.DiskStats = val
						} else {
							for k, v := range diskStatsMap {
								if v.MajorNum == uint64(partition.major) && v.MinorNum == uint64(partition.minor) {
									fs.DiskStats = diskStatsMap[k]
									break
								}
							}
						}
						filesystems = append(filesystems, fs)
						continue
					}
				}
			}

			stats, statsErr := plugin.GetStats(device, partInfo)
			if statsErr != nil {
				// Handle fallback to VFS for plugins that request it
				if errors.Is(statsErr, ErrFallbackToVFS) {
					vfsPlugin := GetPluginForFsType("ext4") // VFS handles ext*
					if vfsPlugin != nil {
						stats, statsErr = vfsPlugin.GetStats(device, partInfo)
					}
				}
				if statsErr != nil {
					klog.V(4).Infof("Stat fs failed for %s. Error: %v", partition.fsType, statsErr)
					continue
				}
			}

			if stats == nil {
				klog.V(4).Infof("no stats returned for %s at %s", partition.fsType, partition.mountpoint)
				continue
			}

			fs.Capacity = stats.Capacity
			fs.Free = stats.Free
			fs.Available = stats.Available
			fs.Inodes = stats.Inodes
			fs.InodesFree = stats.InodesFree
			fs.Type = stats.Type

			// Store in cache if plugin supports caching
			if cacheKey != "" {
				statsCache[cacheKey] = fs
			}

			deviceSet[device] = struct{}{}
			fs.DeviceInfo = DeviceInfo{
				Device: device,
				Major:  uint(partition.major),
				Minor:  uint(partition.minor),
			}

			if val, ok := diskStatsMap[device]; ok {
				fs.DiskStats = val
			} else {
				for k, v := range diskStatsMap {
					if v.MajorNum == uint64(partition.major) && v.MinorNum == uint64(partition.minor) {
						fs.DiskStats = diskStatsMap[k]
						break
					}
				}
			}
			filesystems = append(filesystems, fs)
		}
	}
	return filesystems, nil
}

var partitionRegex = regexp.MustCompile(`^(?:(?:s|v|xv)d[a-z]+\d*|dm-\d+)$`)

func getDiskStatsMap(diskStatsFile string) (map[string]DiskStats, error) {
	diskStatsMap := make(map[string]DiskStats)
	file, err := os.Open(diskStatsFile)
	if err != nil {
		if os.IsNotExist(err) {
			klog.Warningf("Not collecting filesystem statistics because file %q was not found", diskStatsFile)
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

		var err error
		devInfo := make([]uint64, 2)
		for i := 0; i < len(devInfo); i++ {
			devInfo[i], err = strconv.ParseUint(words[i], 10, 64)
			if err != nil {
				return nil, err
			}
		}

		wordLength := len(words)
		offset := 3
		var stats = make([]uint64, wordLength-offset)
		if len(stats) < 11 {
			return nil, fmt.Errorf("could not parse all 11 columns of /proc/diskstats")
		}
		for i := offset; i < wordLength; i++ {
			stats[i-offset], err = strconv.ParseUint(words[i], 10, 64)
			if err != nil {
				return nil, err
			}
		}

		major64, err := strconv.ParseUint(words[0], 10, 64)
		if err != nil {
			return nil, err
		}

		minor64, err := strconv.ParseUint(words[1], 10, 64)
		if err != nil {
			return nil, err
		}

		diskStats := DiskStats{
			MajorNum:        devInfo[0],
			MinorNum:        devInfo[1],
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
			Major:           major64,
			Minor:           minor64,
		}
		diskStatsMap[deviceName] = diskStats
	}
	return diskStatsMap, nil
}

func (i *RealFsInfo) GetGlobalFsInfo() ([]Fs, error) {
	return i.GetFsInfoForPath(nil)
}

func major(devNumber uint64) uint {
	return uint((devNumber >> 8) & 0xfff)
}

func minor(devNumber uint64) uint {
	return uint((devNumber & 0xff) | ((devNumber >> 12) & 0xfff00))
}

func (i *RealFsInfo) GetDeviceInfoByFsUUID(uuid string) (*DeviceInfo, error) {
	deviceName, found := i.fsUUIDToDeviceName[uuid]
	if !found {
		return nil, ErrNoSuchDevice
	}
	p, found := i.partitions[deviceName]
	if !found {
		return nil, fmt.Errorf("cannot find device %q in partitions", deviceName)
	}
	return &DeviceInfo{deviceName, p.major, p.minor}, nil
}

func (i *RealFsInfo) mountInfoFromDir(dir string) (*mount.Info, bool) {
	mnt, found := i.mounts[dir]
	// try the parent dir if not found until we reach the root dir
	// this is an issue on btrfs systems where the directory is not
	// the subvolume
	for !found {
		pathdir, _ := filepath.Split(dir)
		// break when we reach root
		if pathdir == "/" {
			mnt, found = i.mounts["/"]
			break
		}
		// trim "/" from the new parent path otherwise the next possible
		// filepath.Split in the loop will not split the string any further
		dir = strings.TrimSuffix(pathdir, "/")
		mnt, found = i.mounts[dir]
	}
	return &mnt, found
}

func (i *RealFsInfo) GetDirFsDevice(dir string) (*DeviceInfo, error) {
	buf := new(syscall.Stat_t)
	err := syscall.Stat(dir, buf)
	if err != nil {
		return nil, fmt.Errorf("stat failed on %s with error: %s", dir, err)
	}

	// The type Dev in Stat_t is 32bit on mips.
	major := major(uint64(buf.Dev)) // nolint: unconvert
	minor := minor(uint64(buf.Dev)) // nolint: unconvert
	for device, partition := range i.partitions {
		if partition.major == major && partition.minor == minor {
			return &DeviceInfo{device, major, minor}, nil
		}
	}

	mnt, found := i.mountInfoFromDir(dir)
	if found && strings.HasPrefix(mnt.Source, "/dev/") {
		major, minor := mnt.Major, mnt.Minor

		if mnt.FSType == "btrfs" && major == 0 {
			major, minor, err = getBtrfsMajorMinorIds(mnt)
			if err != nil {
				klog.Warningf("Unable to get btrfs mountpoint IDs: %v", err)
			}
		}

		return &DeviceInfo{mnt.Source, uint(major), uint(minor)}, nil
	}

	return nil, fmt.Errorf("with major: %d, minor: %d: %w", major, minor, ErrDeviceNotInPartitionsMap)
}

func GetDirUsage(dir string) (UsageInfo, error) {
	var usage UsageInfo

	if dir == "" {
		return usage, fmt.Errorf("invalid directory")
	}

	rootInfo, err := os.Stat(dir)
	if err != nil {
		return usage, fmt.Errorf("could not stat %q to get inode usage: %v", dir, err)
	}

	rootStat, ok := rootInfo.Sys().(*syscall.Stat_t)
	if !ok {
		return usage, fmt.Errorf("unsupported fileinfo for getting inode usage of %q", dir)
	}

	rootDevID := rootStat.Dev

	// dedupedInode stores inodes that could be duplicates (nlink > 1)
	dedupedInodes := make(map[uint64]struct{})

	err = filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if os.IsNotExist(err) {
			// expected if files appear/vanish
			return nil
		}
		if err != nil {
			return fmt.Errorf("unable to count inodes for part of dir %s: %s", dir, err)
		}

		// according to the docs, Sys can be nil
		if info.Sys() == nil {
			return fmt.Errorf("fileinfo Sys is nil")
		}

		s, ok := info.Sys().(*syscall.Stat_t)
		if !ok {
			return fmt.Errorf("unsupported fileinfo; could not convert to stat_t")
		}

		if s.Dev != rootDevID {
			// don't descend into directories on other devices
			return filepath.SkipDir
		}
		if s.Nlink > 1 {
			if _, ok := dedupedInodes[s.Ino]; !ok {
				// Dedupe things that could be hardlinks
				dedupedInodes[s.Ino] = struct{}{}

				usage.Bytes += uint64(s.Blocks) * statBlockSize
				usage.Inodes++
			}
		} else {
			usage.Bytes += uint64(s.Blocks) * statBlockSize
			usage.Inodes++
		}
		return nil
	})

	return usage, err
}

func (i *RealFsInfo) GetDirUsage(dir string) (UsageInfo, error) {
	claimToken()
	defer releaseToken()
	return GetDirUsage(dir)
}

// Get major and minor Ids for a mount point using btrfs as filesystem.
func getBtrfsMajorMinorIds(mount *mount.Info) (int, int, error) {
	// btrfs fix: following workaround fixes wrong btrfs Major and Minor Ids reported in /proc/self/mountinfo.
	// instead of using values from /proc/self/mountinfo we use stat to get Ids from btrfs mount point

	buf := new(syscall.Stat_t)
	err := syscall.Stat(mount.Source, buf)
	if err != nil {
		err = fmt.Errorf("stat failed on %s with error: %s", mount.Source, err)
		return 0, 0, err
	}

	klog.V(4).Infof("btrfs mount %#v", mount)
	if buf.Mode&syscall.S_IFMT == syscall.S_IFBLK {
		err := syscall.Stat(mount.Mountpoint, buf)
		if err != nil {
			err = fmt.Errorf("stat failed on %s with error: %s", mount.Mountpoint, err)
			return 0, 0, err
		}

		// The type Dev and Rdev in Stat_t are 32bit on mips.
		klog.V(4).Infof("btrfs dev major:minor %d:%d\n", int(major(uint64(buf.Dev))), int(minor(uint64(buf.Dev))))    // nolint: unconvert
		klog.V(4).Infof("btrfs rdev major:minor %d:%d\n", int(major(uint64(buf.Rdev))), int(minor(uint64(buf.Rdev)))) // nolint: unconvert

		return int(major(uint64(buf.Dev))), int(minor(uint64(buf.Dev))), nil // nolint: unconvert
	}
	return 0, 0, fmt.Errorf("%s is not a block device", mount.Source)
}
