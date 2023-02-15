//go:build freebsd
// +build freebsd

/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package freebsdtats provides a client to get node and pod level stats on freebsd
package freebsdstats

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"time"
	"unsafe"

	cadvisorapi "github.com/google/cadvisor/info/v1"
	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
	"golang.org/x/sys/unix"
)

/*
#include <sys/param.h>
#include <sys/ucred.h>
#include <sys/mount.h>
#include <stdio.h>
*/
import "C"

// Client is an interface that is used to get stats information.
type Client interface {
	ContainerInfos() (map[string]cadvisorapiv2.ContainerInfo, error)
	MachineInfo() (*cadvisorapi.MachineInfo, error)
	VersionInfo() (*cadvisorapi.VersionInfo, error)
	GetDirFsInfo(path string) (cadvisorapiv2.FsInfo, error)
}

// StatsClient is a client that implements the Client interface
type StatsClient struct {
}

// newClient constructs a Client.
func newClient() (Client, error) {
	statsClient := new(StatsClient)

	// err := statsClient.client.startMonitoring()
	// if err != nil {
	// 	return nil, err
	// }

	return statsClient, nil
}

func processorCount() (int, error) {
	n, err := unix.SysctlUint32("hw.ncpu")
	return int(n), err
}

func memorySize() (uint64, error) {
	return unix.SysctlUint64("hw.physmem")
}

func getSystemUUID() (string, error) {
	return unix.Sysctl("kern.hostuuid")
}

func getOsRelease() (string, error) {
	return unix.Sysctl("kern.osrelease")
}

func getKernVersion() (string, error) {
	return unix.Sysctl("kern.version")
}

// ContainerInfos returns a map of container infos. The map contains node and
// pod level stats. Analogous to cadvisor GetContainerInfoV2 method.
func (c *StatsClient) ContainerInfos() (map[string]cadvisorapiv2.ContainerInfo, error) {
	infos := make(map[string]cadvisorapiv2.ContainerInfo)
	// rootContainerInfo, err := c.createRootContainerInfo()
	// if err != nil {
	// 	return nil, err
	// }

	// infos["/"] = *rootContainerInfo

	return infos, nil
}

// MachineInfo returns a cadvisorapi.MachineInfo with details about the
// node machine. Analogous to cadvisor MachineInfo method.
func (c *StatsClient) MachineInfo() (*cadvisorapi.MachineInfo, error) {
	hostname, err := os.Hostname()
	if err != nil {
		return nil, err
	}

	systemUUID, err := getSystemUUID()
	if err != nil {
		return nil, err
	}

	// This is not implemented on FreeBSD
	// bootId, err := getBootID()
	// if err != nil {
	// 	return nil, err
	// }

	numCores, err := processorCount()
	if err != nil {
		return nil, err
	}

	memSize, err := memorySize()
	if err != nil {
		return nil, err
	}

	return &cadvisorapi.MachineInfo{
		NumCores:       numCores,
		MemoryCapacity: memSize,
		MachineID:      hostname,
		SystemUUID:     systemUUID,
		// BootID:         bootId,
	}, nil
}

// WinVersionInfo returns a  cadvisorapi.VersionInfo with version info of
// the kernel and docker runtime. Analogous to cadvisor VersionInfo method.
func (c *StatsClient) VersionInfo() (*cadvisorapi.VersionInfo, error) {
	kver, err := getKernVersion()
	if err != nil {
		return nil, err
	}

	osver, err := getOsRelease()
	if err != nil {
		return nil, err
	}

	return &cadvisorapi.VersionInfo{
		KernelVersion:      kver,
		ContainerOsVersion: osver,
	}, nil
}

func getMountPoints() ([]cadvisorapiv2.FsInfo, error) {
	var mntbuf *C.struct_statfs
	count := C.getmntinfo(&mntbuf, C.MNT_NOWAIT)
	if count == 0 {
		return nil, errors.New("failed to run FreeBSD getmntinfo() syscall")
	}

	mnt := (*[1 << 20]C.struct_statfs)(unsafe.Pointer(mntbuf))
	infos := make([]cadvisorapiv2.FsInfo, count)
	for i := 0; i < int(count); i++ {
		inodes := uint64(mnt[i].f_files)
		inodesFree := uint64(mnt[i].f_ffree)
		infos = append(infos, cadvisorapiv2.FsInfo{
			Timestamp:  time.Now(),
			Device:     C.GoString(&mnt[i].f_mntfromname[0]),
			Mountpoint: C.GoString(&mnt[i].f_mntonname[0]),
			Inodes:     &inodes,
			InodesFree: &inodesFree,
			Capacity:   uint64(mnt[i].f_blocks) * uint64(mnt[i].f_bsize),
			Available:  uint64(mnt[i].f_bavail) * uint64(mnt[i].f_bsize),
			Usage: (uint64(mnt[i].f_blocks) - uint64(mnt[i].f_bavail)) * uint64(mnt[i].f_bsize),
		})
	}
	return infos, nil
}

func getMountpoint(mountPath string, mountpoints []cadvisorapiv2.FsInfo) *cadvisorapiv2.FsInfo {
	for _, mp := range mountpoints {
		if mp.Mountpoint == mountPath {
			return &mp
		}
	}
	return nil
}

// GetDirFsInfo returns filesystem capacity and usage information.
func (c *StatsClient) GetDirFsInfo(path string) (cadvisorapiv2.FsInfo, error) {
	// var freeBytesAvailable, totalNumberOfBytes, totalNumberOfFreeBytes int64
	var err error

	mountpoints, err := getMountPoints()
	if err != nil {
		return cadvisorapiv2.FsInfo{}, err
	}

	dir := path
	for {
		pathdir, _ := filepath.Split(dir)
		// break when we reach root
		if pathdir == "/" {
			if mp := getMountpoint(pathdir, mountpoints); mp != nil {
				return *mp, nil
			}

			return cadvisorapiv2.FsInfo{}, fmt.Errorf("unable to find mountpoint for path %s", path)
		}
		// trim "/" from the new parent path otherwise the next possible
		// filepath.Split in the loop will not split the string any further
		dir = strings.TrimSuffix(pathdir, "/")
		if mp := getMountpoint(dir, mountpoints); mp != nil {
			return *mp, nil
		}
	}

	return cadvisorapiv2.FsInfo{}, err
}
