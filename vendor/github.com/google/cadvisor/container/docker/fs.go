// Copyright 2022 Google Inc. All Rights Reserved.
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

package docker

import (
	"fmt"

	"k8s.io/klog/v2"

	"github.com/google/cadvisor/container"
	"github.com/google/cadvisor/container/common"
	"github.com/google/cadvisor/devicemapper"
	"github.com/google/cadvisor/fs"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/zfs"
)

func FsStats(
	stats *info.ContainerStats,
	machineInfoFactory info.MachineInfoFactory,
	metrics container.MetricSet,
	storageDriver StorageDriver,
	fsHandler common.FsHandler,
	globalFsInfo fs.FsInfo,
	poolName string,
	rootfsStorageDir string,
	zfsParent string,
) error {
	mi, err := machineInfoFactory.GetMachineInfo()
	if err != nil {
		return err
	}

	if metrics.Has(container.DiskIOMetrics) {
		common.AssignDeviceNamesToDiskStats((*common.MachineInfoNamer)(mi), &stats.DiskIo)
	}

	if metrics.Has(container.DiskUsageMetrics) {
		var device string
		switch storageDriver {
		case DevicemapperStorageDriver:
			device = poolName
		case AufsStorageDriver, OverlayStorageDriver, Overlay2StorageDriver, VfsStorageDriver:
			deviceInfo, err := globalFsInfo.GetDirFsDevice(rootfsStorageDir)
			if err != nil {
				return fmt.Errorf("unable to determine device info for dir: %v: %v", rootfsStorageDir, err)
			}
			device = deviceInfo.Device
		case ZfsStorageDriver:
			device = zfsParent
		default:
			return nil
		}

		for _, fs := range mi.Filesystems {
			if fs.Device == device {
				usage := fsHandler.Usage()
				fsStat := info.FsStats{
					Device:    device,
					Type:      fs.Type,
					Limit:     fs.Capacity,
					BaseUsage: usage.BaseUsageBytes,
					Usage:     usage.TotalUsageBytes,
					Inodes:    usage.InodeUsage,
				}
				fileSystems, err := globalFsInfo.GetGlobalFsInfo()
				if err != nil {
					return fmt.Errorf("unable to obtain diskstats for filesystem %s: %v", fsStat.Device, err)
				}
				addDiskStats(fileSystems, &fs, &fsStat)
				stats.Filesystem = append(stats.Filesystem, fsStat)
				break
			}
		}
	}

	return nil
}

func addDiskStats(fileSystems []fs.Fs, fsInfo *info.FsInfo, fsStats *info.FsStats) {
	if fsInfo == nil {
		return
	}

	for _, fileSys := range fileSystems {
		if fsInfo.DeviceMajor == fileSys.DiskStats.Major &&
			fsInfo.DeviceMinor == fileSys.DiskStats.Minor {
			fsStats.ReadsCompleted = fileSys.DiskStats.ReadsCompleted
			fsStats.ReadsMerged = fileSys.DiskStats.ReadsMerged
			fsStats.SectorsRead = fileSys.DiskStats.SectorsRead
			fsStats.ReadTime = fileSys.DiskStats.ReadTime
			fsStats.WritesCompleted = fileSys.DiskStats.WritesCompleted
			fsStats.WritesMerged = fileSys.DiskStats.WritesMerged
			fsStats.SectorsWritten = fileSys.DiskStats.SectorsWritten
			fsStats.WriteTime = fileSys.DiskStats.WriteTime
			fsStats.IoInProgress = fileSys.DiskStats.IoInProgress
			fsStats.IoTime = fileSys.DiskStats.IoTime
			fsStats.WeightedIoTime = fileSys.DiskStats.WeightedIoTime
			break
		}
	}
}

// FsHandler is a composite FsHandler implementation the incorporates
// the common fs handler, a devicemapper ThinPoolWatcher, and a zfsWatcher
type FsHandler struct {
	FsHandler common.FsHandler

	// thinPoolWatcher is the devicemapper thin pool watcher
	ThinPoolWatcher *devicemapper.ThinPoolWatcher
	// deviceID is the id of the container's fs device
	DeviceID string

	// zfsWatcher is the zfs filesystem watcher
	ZfsWatcher *zfs.ZfsWatcher
	// zfsFilesystem is the docker zfs filesystem
	ZfsFilesystem string
}

var _ common.FsHandler = &FsHandler{}

func (h *FsHandler) Start() {
	h.FsHandler.Start()
}

func (h *FsHandler) Stop() {
	h.FsHandler.Stop()
}

func (h *FsHandler) Usage() common.FsUsage {
	usage := h.FsHandler.Usage()

	// When devicemapper is the storage driver, the base usage of the container comes from the thin pool.
	// We still need the result of the fsHandler for any extra storage associated with the container.
	// To correctly factor in the thin pool usage, we should:
	// * Usage the thin pool usage as the base usage
	// * Calculate the overall usage by adding the overall usage from the fs handler to the thin pool usage
	if h.ThinPoolWatcher != nil {
		thinPoolUsage, err := h.ThinPoolWatcher.GetUsage(h.DeviceID)
		if err != nil {
			// TODO: ideally we should keep track of how many times we failed to get the usage for this
			// device vs how many refreshes of the cache there have been, and display an error e.g. if we've
			// had at least 1 refresh and we still can't find the device.
			klog.V(5).Infof("unable to get fs usage from thin pool for device %s: %v", h.DeviceID, err)
		} else {
			usage.BaseUsageBytes = thinPoolUsage
			usage.TotalUsageBytes += thinPoolUsage
		}
	}

	if h.ZfsWatcher != nil {
		zfsUsage, err := h.ZfsWatcher.GetUsage(h.ZfsFilesystem)
		if err != nil {
			klog.V(5).Infof("unable to get fs usage from zfs for filesystem %s: %v", h.ZfsFilesystem, err)
		} else {
			usage.BaseUsageBytes = zfsUsage
			usage.TotalUsageBytes += zfsUsage
		}
	}
	return usage
}
