//go:build windows
// +build windows

package cim

import (
	"fmt"
	"strconv"

	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/server2019/root/microsoft/windows/storage"
)

const (
	// PartitionStyleUnknown indicates an unknown partition table format
	PartitionStyleUnknown = 0
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
//	  WHERE DiskNumber = '<diskNumber>'
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-disk
// for the WMI class definition.
func QueryDiskByNumber(diskNumber uint32, selectorList []string) (*storage.MSFT_Disk, error) {
	diskQuery := query.NewWmiQueryWithSelectList("MSFT_Disk", selectorList, "Number", strconv.Itoa(int(diskNumber)))
	instances, err := QueryInstances(WMINamespaceStorage, diskQuery)
	if err != nil {
		return nil, err
	}

	disk, err := storage.NewMSFT_DiskEx1(instances[0])
	if err != nil {
		return nil, fmt.Errorf("failed to query disk %d. error: %v", diskNumber, err)
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
func ListDisks(selectorList []string) ([]*storage.MSFT_Disk, error) {
	diskQuery := query.NewWmiQueryWithSelectList("MSFT_Disk", selectorList)
	instances, err := QueryInstances(WMINamespaceStorage, diskQuery)
	if IgnoreNotFound(err) != nil {
		return nil, err
	}

	var disks []*storage.MSFT_Disk
	for _, instance := range instances {
		disk, err := storage.NewMSFT_DiskEx1(instance)
		if err != nil {
			return nil, fmt.Errorf("failed to query disk %v. error: %v", instance, err)
		}

		disks = append(disks, disk)
	}

	return disks, nil
}

// InitializeDisk initializes a RAW disk with a particular partition style.
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/initialize-msft-disk
// for the WMI method definition.
func InitializeDisk(disk *storage.MSFT_Disk, partitionStyle int) (int, error) {
	result, err := disk.InvokeMethodWithReturn("Initialize", int32(partitionStyle))
	return int(result), err
}

// RefreshDisk Refreshes the cached disk layout information.
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-disk-refresh
// for the WMI method definition.
func RefreshDisk(disk *storage.MSFT_Disk) (int, string, error) {
	var status string
	result, err := disk.InvokeMethodWithReturn("Refresh", &status)
	return int(result), status, err
}

// CreatePartition creates a partition on a disk.
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/createpartition-msft-disk
// for the WMI method definition.
func CreatePartition(disk *storage.MSFT_Disk, params ...interface{}) (int, error) {
	result, err := disk.InvokeMethodWithReturn("CreatePartition", params...)
	return int(result), err
}

// SetDiskState takes a disk online or offline.
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-disk-online and
// https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-disk-offline
// for the WMI method definition.
func SetDiskState(disk *storage.MSFT_Disk, online bool) (int, string, error) {
	method := "Offline"
	if online {
		method = "Online"
	}

	var status string
	result, err := disk.InvokeMethodWithReturn(method, &status)
	return int(result), status, err
}

// RescanDisks rescans all changes by updating the internal cache of software objects (that is, Disks, Partitions, Volumes)
// for the storage setting.
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-storagesetting-updatehoststoragecache
// for the WMI method definition.
func RescanDisks() (int, error) {
	result, _, err := InvokeCimMethod(WMINamespaceStorage, "MSFT_StorageSetting", "UpdateHostStorageCache", nil)
	return result, err
}

// GetDiskNumber returns the number of a disk.
func GetDiskNumber(disk *storage.MSFT_Disk) (uint32, error) {
	number, err := disk.GetProperty("Number")
	if err != nil {
		return 0, err
	}
	return uint32(number.(int32)), err
}

// GetDiskLocation returns the location of a disk.
func GetDiskLocation(disk *storage.MSFT_Disk) (string, error) {
	return disk.GetPropertyLocation()
}

// GetDiskPartitionStyle returns the partition style of a disk.
func GetDiskPartitionStyle(disk *storage.MSFT_Disk) (int32, error) {
	retValue, err := disk.GetProperty("PartitionStyle")
	if err != nil {
		return 0, err
	}
	return retValue.(int32), err
}

// IsDiskOffline returns whether a disk is offline.
func IsDiskOffline(disk *storage.MSFT_Disk) (bool, error) {
	return disk.GetPropertyIsOffline()
}

// GetDiskSize returns the size of a disk.
func GetDiskSize(disk *storage.MSFT_Disk) (int64, error) {
	sz, err := disk.GetProperty("Size")
	if err != nil {
		return -1, err
	}
	return strconv.ParseInt(sz.(string), 10, 64)
}

// GetDiskPath returns the path of a disk.
func GetDiskPath(disk *storage.MSFT_Disk) (string, error) {
	return disk.GetPropertyPath()
}

// GetDiskSerialNumber returns the serial number of a disk.
func GetDiskSerialNumber(disk *storage.MSFT_Disk) (string, error) {
	return disk.GetPropertySerialNumber()
}
