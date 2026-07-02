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
)

const (
	MSFTDiskClass           = "MSFT_Disk"
	MSFTStorageSettingClass = "MSFT_StorageSetting"

	// PartitionStyleUnknown indicates an unknown partition table format
	PartitionStyleUnknown = 0
	// PartitionStyleMBR indicates the disk uses Master Boot Record (MBR) format
	PartitionStyleMBR = 1
	// PartitionStyleGPT indicates the disk uses GUID Partition Table (GPT) format
	PartitionStyleGPT = 2

	// GPTPartitionTypeBasicData is the GUID for basic data partitions in GPT
	// Used for general purpose storage partitions
	GPTPartitionTypeBasicData = "{ebd0a0a2-b9e5-4433-87c0-68b6b72699c7}"
	// GPTPartitionTypeMicrosoftReserved is the GUID for Microsoft Reserved Partition (MSR)
	// Reserved by Windows for system use
	GPTPartitionTypeMicrosoftReserved = "{e3c9e316-0b5c-4db8-817d-f92df00215ae}"

	// ErrorCodeCreatePartitionAccessPathAlreadyInUse is the error code (42002) returned when the driver letter failed to assign after partition created
	ErrorCodeCreatePartitionAccessPathAlreadyInUse = 42002
)

var (
	DiskSelectorListForDiskNumberAndLocation = []string{"Number", "Location"}
	DiskSelectorListForPartitionStyle        = []string{"PartitionStyle"}
	DiskSelectorListForPathAndSerialNumber   = []string{"Path", "SerialNumber"}
	DiskSelectorListForIsOffline             = []string{"IsOffline"}
	DiskSelectorListForSize                  = []string{"Size"}
)

// QueryDiskByNumber retrieves disk information for a specific disk identified by its number.
//
// The equivalent WMI query is:
//
//	SELECT [selectors] FROM MSFT_Disk
//	  WHERE Number = '<diskNumber>'
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-disk
// for the WMI class definition.
func QueryDiskByNumber(scope *Scope, diskNumber uint32, selectorList []string) (*COMDispatchObject, error) {
	q := NewQuery(MSFTDiskClass).
		WithNamespace(WMINamespaceStorage).
		Select(selectorList...).
		WithCondition("Number", "=", diskNumber)

	disk, err := QueryFirstObjectWithBuilder(scope, q)
	if err != nil {
		return nil, fmt.Errorf("failed to query disk %d. error: %w", diskNumber, err)
	}
	return disk, nil
}

// ListDisks retrieves information about all available disks.
//
// The equivalent WMI query is:
//
//	SELECT [selectors] FROM MSFT_Disk
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-disk
// for the WMI class definition.
func ListDisks(scope *Scope, selectorList []string) ([]*COMDispatchObject, error) {
	q := NewQuery(MSFTDiskClass).
		WithNamespace(WMINamespaceStorage).
		Select(selectorList...)

	disks, err := QueryObjectsWithBuilder(scope, q)
	if err != nil {
		return nil, fmt.Errorf("failed to query disks: %w", err)
	}
	return disks, nil
}

// InitializeDisk initializes a RAW disk with a particular partition style.
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/initialize-msft-disk
// for the WMI method definition.
func InitializeDisk(disk *COMDispatchObject, partitionStyle int) error {
	result, err := disk.CallUint32("Initialize", int32(partitionStyle))
	if err != nil {
		return fmt.Errorf("failed to initialize disk: %w", err)
	}
	if result != 0 {
		return NewWMIError(MSFTDiskClass, "Initialize", disk.Dispatch(), result)
	}
	return nil
}

// RefreshDisk Refreshes the cached disk layout information.
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-disk-refresh
// for the WMI method definition.
func RefreshDisk(disk *COMDispatchObject) (string, error) {
	var status string
	result, err := disk.CallUint32("Refresh", &status)
	if err != nil {
		return "", fmt.Errorf("failed to refresh disk: %w", err)
	}
	if result != 0 {
		return "", NewWMIError(MSFTDiskClass, "Refresh", disk.Dispatch(), result)
	}
	return status, nil
}

// CreatePartition creates a partition on a disk.
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/createpartition-msft-disk
// for the WMI method definition.
func CreatePartition(disk *COMDispatchObject, params ...interface{}) error {
	result, err := disk.CallUint32("CreatePartition", params...)
	if err != nil {
		return fmt.Errorf("failed to create partition: %w", err)
	}
	if result != 0 {
		return NewWMIError(MSFTDiskClass, "CreatePartition", disk.Dispatch(), result)
	}
	return nil
}

// SetDiskState takes a disk online or offline.
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-disk-online and
// https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-disk-offline
// for the WMI method definition.
func SetDiskState(disk *COMDispatchObject, online bool) (string, error) {
	method := "Offline"
	if online {
		method = "Online"
	}

	var status string
	// MSFT_Disk Online/Offline methods do not take input parameters
	// per https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-disk-online
	// ExtendedStatus is an optional out parameter passed via &status.
	result, err := disk.CallUint32(method, &status)
	if err != nil {
		return "", fmt.Errorf("failed to set disk state: %w", err)
	}
	if result != 0 {
		return "", NewWMIError(MSFTDiskClass, method, disk.Dispatch(), result)
	}
	return status, nil
}

// RescanDisks rescans all changes by updating the internal cache of software objects (that is, Disks, Partitions, Volumes)
// for the storage setting.
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-storagesetting-updatehoststoragecache
// for the WMI method definition.
func RescanDisks() error {
	result, _, err := CallMethodOnWMIClass(WMINamespaceStorage, MSFTStorageSettingClass, "UpdateHostStorageCache", nil, DiscardOutputParameter)
	if err != nil {
		return fmt.Errorf("failed to update host storage cache: %w", err)
	}
	if result != 0 {
		return NewWMIError(MSFTStorageSettingClass, "UpdateHostStorageCache", nil, result)
	}
	return nil
}

// GetDiskNumber returns the number of a disk.
func GetDiskNumber(disk *COMDispatchObject) (uint32, error) {
	return disk.GetUint32Property("Number")
}

// GetDiskLocation returns the location of a disk.
func GetDiskLocation(disk *COMDispatchObject) (string, error) {
	return disk.GetStringProperty("Location")
}

// GetDiskPartitionStyle returns the partition style of a disk.
func GetDiskPartitionStyle(disk *COMDispatchObject) (uint16, error) {
	return disk.GetUint16Property("PartitionStyle")
}

// IsDiskOffline returns whether a disk is offline.
func IsDiskOffline(disk *COMDispatchObject) (bool, error) {
	return disk.GetBoolProperty("IsOffline")
}

// GetDiskSize returns the size of a disk.
func GetDiskSize(disk *COMDispatchObject) (uint64, error) {
	return disk.GetStringPropertyAsUint64("Size")
}

// GetDiskPath returns the path of a disk.
func GetDiskPath(disk *COMDispatchObject) (string, error) {
	return disk.GetStringProperty("Path")
}

// GetDiskSerialNumber returns the serial number of a disk.
func GetDiskSerialNumber(disk *COMDispatchObject) (string, error) {
	return disk.GetStringProperty("SerialNumber")
}
