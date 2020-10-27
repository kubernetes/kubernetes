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

package fs

import (
	"errors"
)

type Context struct {
	// docker root directory.
	Docker DockerContext
	Crio   CrioContext
}

type DockerContext struct {
	Root         string
	Driver       string
	DriverStatus map[string]string
}

type CrioContext struct {
	Root string
}

type DeviceInfo struct {
	Device string
	Major  uint
	Minor  uint
}

type FsType string

func (ft FsType) String() string {
	return string(ft)
}

const (
	ZFS          FsType = "zfs"
	DeviceMapper FsType = "devicemapper"
	VFS          FsType = "vfs"
)

type Fs struct {
	DeviceInfo
	Type       FsType
	Capacity   uint64
	Free       uint64
	Available  uint64
	Inodes     *uint64
	InodesFree *uint64
	DiskStats  DiskStats
}

type DiskStats struct {
	MajorNum        uint64
	MinorNum        uint64
	ReadsCompleted  uint64
	ReadsMerged     uint64
	SectorsRead     uint64
	ReadTime        uint64
	WritesCompleted uint64
	WritesMerged    uint64
	SectorsWritten  uint64
	WriteTime       uint64
	IoInProgress    uint64
	IoTime          uint64
	WeightedIoTime  uint64
}

type UsageInfo struct {
	Bytes  uint64
	Inodes uint64
}

// ErrNoSuchDevice is the error indicating the requested device does not exist.
var ErrNoSuchDevice = errors.New("cadvisor: no such device")

type FsInfo interface {
	// Returns capacity and free space, in bytes, of all the ext2, ext3, ext4 filesystems on the host.
	GetGlobalFsInfo() ([]Fs, error)

	// Returns capacity and free space, in bytes, of the set of mounts passed.
	GetFsInfoForPath(mountSet map[string]struct{}) ([]Fs, error)

	// GetDirUsage returns a usage information for 'dir'.
	GetDirUsage(dir string) (UsageInfo, error)

	// GetDeviceInfoByFsUUID returns the information of the device with the
	// specified filesystem uuid. If no such device exists, this function will
	// return the ErrNoSuchDevice error.
	GetDeviceInfoByFsUUID(uuid string) (*DeviceInfo, error)

	// Returns the block device info of the filesystem on which 'dir' resides.
	GetDirFsDevice(dir string) (*DeviceInfo, error)

	// Returns the device name associated with a particular label.
	GetDeviceForLabel(label string) (string, error)

	// Returns all labels associated with a particular device name.
	GetLabelsForDevice(device string) ([]string, error)

	// Returns the mountpoint associated with a particular device.
	GetMountpointForDevice(device string) (string, error)
}
