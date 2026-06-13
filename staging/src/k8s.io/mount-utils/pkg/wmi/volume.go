//go:build windows
// +build windows

/*
Copyright 2025 The Kubernetes Authors.

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

package wmi

import (
	"fmt"
	"strconv"

	"github.com/go-ole/go-ole"
	"k8s.io/klog/v2"
)

const (
	MSFTVolumeClass    = "MSFT_Volume"
	MSFTPartitionClass = "MSFT_Partition"

	FileSystemUnknown = 0
)

var (
	VolumeSelectorListForFileSystemType = []string{"FileSystemType"}
	VolumeSelectorListForStats          = []string{"UniqueId", "SizeRemaining", "Size"}
	VolumeSelectorListUniqueID          = []string{"UniqueId"}

	PartitionSelectorListObjectID = []string{"ObjectId"}
)

// QueryVolumeByUniqueID retrieves a specific volume by its unique identifier.
//
// The equivalent WMI query is:
//
//	SELECT [selectors] FROM MSFT_Volume WHERE UniqueId = "<volumeID>"
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-volume
// for the WMI class definition.
func QueryVolumeByUniqueID(scope *Scope, volumeID string, selectorList []string) (*COMDispatchObject, error) {
	q := NewQuery(MSFTVolumeClass).WithNamespace(WMINamespaceStorage).
		Select(selectorList...).
		WithCondition("UniqueId", "=", volumeID)

	result, err := QueryFirstObjectWithBuilder(scope, q)
	if err != nil {
		return nil, fmt.Errorf("failed to query volume %s: %w", volumeID, err)
	}

	return result, nil
}

// ListVolumes retrieves all available volumes on the system.
//
// The equivalent WMI query is:
//
//	SELECT [selectors] FROM MSFT_Volume
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-volume
// for the WMI class definition.
func ListVolumes(scope *Scope, selectorList []string) ([]*COMDispatchObject, error) {
	q := NewQuery(MSFTVolumeClass).WithNamespace(WMINamespaceStorage).Select(selectorList...)
	instances, err := QueryObjectsWithBuilder(scope, q)
	if err != nil {
		return nil, err
	}

	return instances, nil
}

// FormatVolume formats the specified volume.
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/format-msft-volume
// for the WMI method definition.
func FormatVolume(volume *COMDispatchObject, params ...interface{}) error {
	result, err := volume.CallUint32("Format", params...)
	if err != nil {
		return fmt.Errorf("failed to format volume %v. error: %w", volume, err)
	}
	if result != 0 {
		return NewWMIError(MSFTVolumeClass, "Format", volume.Dispatch(), result)
	}
	return nil
}

// FlushVolume flushes the cached data in the volume's file system to disk.
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-volume-flush
// for the WMI method definition.
func FlushVolume(volume *COMDispatchObject) error {
	result, err := volume.CallUint32("Flush")
	if err != nil {
		return fmt.Errorf("failed to flush volume %v. error: %w", volume, err)
	}
	if result != 0 {
		return NewWMIError(MSFTVolumeClass, "Flush", volume.Dispatch(), result)
	}
	return nil
}

// GetVolumeUniqueID returns the unique ID (object ID) of a volume.
func GetVolumeUniqueID(volume *COMDispatchObject) (string, error) {
	return volume.GetStringProperty("UniqueId")
}

// GetVolumeFileSystemType returns the file system type of a volume.
func GetVolumeFileSystemType(volume *COMDispatchObject) (uint16, error) {
	return volume.GetUint16Property("FileSystemType")
}

// GetVolumeSize returns the size of a volume.
func GetVolumeSize(volume *COMDispatchObject) (uint64, error) {
	return volume.GetStringPropertyAsUint64("Size")
}

// GetVolumeSizeRemaining returns the remaining size of a volume.
func GetVolumeSizeRemaining(volume *COMDispatchObject) (uint64, error) {
	return volume.GetStringPropertyAsUint64("SizeRemaining")
}

// ListPartitionsOnDisk retrieves all partitions or a partition with the specified number on a disk.
//
// The equivalent WMI query is:
//
//	SELECT [selectors] FROM MSFT_Partition
//	  WHERE DiskNumber = '<diskNumber>'
//	    AND PartitionNumber = '<partitionNumber>'
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-partition
// for the WMI class definition.
func ListPartitionsOnDisk(scope *Scope, diskNumber, partitionNumber uint32, selectorList []string) ([]*COMDispatchObject, error) {
	filters := []Condition{
		WithCondition("DiskNumber", "=", diskNumber),
	}
	if partitionNumber > 0 {
		filters = append(filters, WithCondition("PartitionNumber", "=", partitionNumber))
	}
	return ListPartitionsWithFilters(scope, selectorList, filters...)
}

// ListPartitionsWithFilters retrieves all partitions matching with the conditions specified by query filters.
//
// The equivalent WMI query is:
//
//	SELECT [selectors] FROM MSFT_Partition
//	  WHERE ...
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-partition
// for the WMI class definition.
func ListPartitionsWithFilters(scope *Scope, selectorList []string, filters ...Condition) ([]*COMDispatchObject, error) {
	q := NewQuery(MSFTPartitionClass).WithNamespace(WMINamespaceStorage).Select(selectorList...)
	q.WithConditions(filters...)
	instances, err := QueryObjectsWithBuilder(scope, q)
	if err != nil {
		return nil, err
	}
	return instances, nil
}

// FindPartitionsByVolume finds all partitions associated with the given volumes
// using MSFT_PartitionToVolume association.
//
// WMI association MSFT_PartitionToVolume:
//
//	Partition                                                               | Volume
//	---------                                                               | ------
//	MSFT_Partition (ObjectId = "{1}\\WIN-8E2EVAQ9QSB\ROOT/Microsoft/Win...) | MSFT_Volume (ObjectId = "{1}\\WIN-8E2EVAQ9QS...
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-partitiontovolume
// for the WMI class definition.
func FindPartitionsByVolume(scope *Scope, volumes []*COMDispatchObject) ([]*COMDispatchObject, error) {
	partitions := make([]*COMDispatchObject, 0)
	err := ForEach(volumes, func(volume *COMDispatchObject) error {
		collection, err := volume.GetAssociated(scope, "MSFT_PartitionToVolume", MSFTPartitionClass, "Partition", "Volume")
		if err != nil {
			return fmt.Errorf("failed to query associated partition for %v. error: %w", volume, err)
		}

		partitions = append(partitions, collection...)
		return nil
	})

	if err != nil {
		return nil, err
	}

	return partitions, nil
}

// FindVolumesByPartition finds all volumes associated with the given partitions
// using MSFT_PartitionToVolume association.
//
// WMI association MSFT_PartitionToVolume:
//
//	Partition                                                               | Volume
//	---------                                                               | ------
//	MSFT_Partition (ObjectId = "{1}\\WIN-8E2EVAQ9QSB\ROOT/Microsoft/Win...) | MSFT_Volume (ObjectId = "{1}\\WIN-8E2EVAQ9QS...
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-partitiontovolume
// for the WMI class definition.
func FindVolumesByPartition(scope *Scope, partitions []*COMDispatchObject) ([]*COMDispatchObject, error) {
	volumes := make([]*COMDispatchObject, 0)
	err := ForEach(partitions, func(part *COMDispatchObject) error {
		collection, err := part.GetAssociated(scope, "MSFT_PartitionToVolume", MSFTVolumeClass, "Volume", "Partition")
		if err != nil {
			return fmt.Errorf("failed to query associated volumes for %v. error: %w", part, err)
		}

		volumes = append(volumes, collection...)
		return nil
	})

	if err != nil {
		return nil, err
	}

	return volumes, nil
}

// GetPartitionByVolumeUniqueID retrieves a specific partition from a volume identified by its unique ID.
func GetPartitionByVolumeUniqueID(scope *Scope, volumeID string) (*COMDispatchObject, error) {
	volume, err := QueryVolumeByUniqueID(scope, volumeID, []string{"ObjectId"})
	if err != nil {
		return nil, err
	}

	partitions, err := FindPartitionsByVolume(scope, []*COMDispatchObject{volume})
	if err != nil {
		return nil, err
	}

	if len(partitions) == 0 {
		return nil, ErrNotFound
	}

	return partitions[0], nil
}

// GetVolumeByDriveLetter retrieves a volume associated with a specific drive letter.
func GetVolumeByDriveLetter(scope *Scope, driveLetter string, partitionSelectorList []string) (*COMDispatchObject, error) {
	var selectorsForPart []string
	selectorsForPart = append(selectorsForPart, partitionSelectorList...)
	selectorsForPart = append(selectorsForPart, "ObjectId")
	partitions, err := ListPartitionsWithFilters(scope, selectorsForPart, WithCondition("DriveLetter", "=", driveLetter))
	if err != nil {
		return nil, err
	}

	volumes, err := FindVolumesByPartition(scope, partitions)
	if err != nil {
		return nil, err
	}

	if len(volumes) == 0 {
		return nil, ErrNotFound
	}

	return volumes[0], nil
}

// GetPartitionDiskNumber retrieves the disk number associated with a given partition.
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-partition
// for the WMI class definitions.
func GetPartitionDiskNumber(part *COMDispatchObject) (uint32, error) {
	return part.GetUint32Property("DiskNumber")
}

// SetPartitionState takes a partition online or offline.
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-partition-online and
// https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-partition-offline
// for the WMI method definition.
func SetPartitionState(part *COMDispatchObject, online bool) (string, error) {
	method := "Offline"
	if online {
		method = "Online"
	}

	var status string
	// MSFT_Partition Online/Offline methods do not take input parameters
	// per https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-partition-online
	// ExtendedStatus is an optional out parameter passed via &status.
	result, err := part.CallUint32(method, &status)
	if err != nil {
		return "", err
	}
	if result != 0 {
		return status, NewWMIError(MSFTPartitionClass, method, part.Dispatch(), result)
	}
	return status, err
}

// GetPartitionSupportedSize retrieves the minimum and maximum sizes that the partition can be resized to using the ResizePartition method.
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-partition-getsupportedsizes
// for the WMI method definition.
func GetPartitionSupportedSize(part *COMDispatchObject) (sizeMin, sizeMax uint64, status string, err error) {
	var sizeMinVar, sizeMaxVar ole.VARIANT
	defer sizeMinVar.Clear()
	defer sizeMaxVar.Clear()
	result, err := part.CallUint32("GetSupportedSize", &sizeMinVar, &sizeMaxVar, &status)
	if err != nil {
		return
	}
	if result != 0 {
		err = NewWMIError(MSFTPartitionClass, "GetSupportedSize", part.Dispatch(), result)
		return
	}

	klog.V(5).Infof("got sizeMin (%v) sizeMax (%v) from partition (%v), status: %s", sizeMinVar, sizeMaxVar, part, status)

	sizeMin, err = strconv.ParseUint(NewSafeVariant(&sizeMinVar).String(), 10, 64)
	if err != nil {
		return
	}

	sizeMax, err = strconv.ParseUint(NewSafeVariant(&sizeMaxVar).String(), 10, 64)
	if err != nil {
		return
	}

	return sizeMin, sizeMax, status, nil
}

// ResizePartition resizes a partition.
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-partition-resize
// for the WMI method definition.
func ResizePartition(part *COMDispatchObject, size uint64) (string, error) {
	var status string
	result, err := part.CallUint32("Resize", strconv.FormatUint(size, 10), &status)
	if err != nil {
		return "", fmt.Errorf("failed to resize partition %v. error: %w", part, err)
	}
	if result != 0 {
		return status, NewWMIError(MSFTPartitionClass, "Resize", part.Dispatch(), result)
	}
	return status, nil
}

// GetPartitionSize returns the size of a partition.
func GetPartitionSize(part *COMDispatchObject) (uint64, error) {
	return part.GetStringPropertyAsUint64("Size")
}

// FilterForPartitionOnDisk creates a WMI query filter to query a disk by its number.
func FilterForPartitionOnDisk(diskNumber uint32) Condition {
	return WithCondition("DiskNumber", "=", diskNumber)
}

// FilterForPartitionsOfTypeNormal creates a WMI query filter for all non-reserved partitions.
func FilterForPartitionsOfTypeNormal() Condition {
	return WithCondition("GptType", "<>", GPTPartitionTypeMicrosoftReserved)
}
