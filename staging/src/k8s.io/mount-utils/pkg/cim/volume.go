//go:build windows
// +build windows

package cim

import (
	"fmt"
	"strconv"

	"github.com/go-ole/go-ole"
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	"github.com/microsoft/wmi/server2019/root/microsoft/windows/storage"
	"k8s.io/klog/v2"
)

const (
	FileSystemUnknown = 0
)

var (
	VolumeSelectorListForFileSystemType = []string{"FileSystemType"}
	VolumeSelectorListForStats          = []string{"UniqueId", "SizeRemaining", "Size"}
	VolumeSelectorListUniqueID          = []string{"UniqueId"}

	PartitionSelectorListObjectID = []string{"ObjectId"}
)

// QueryVolumeByUniqueID retrieves a specific volume by its unique identifier,
// returning the first volume that matches the given volume ID.
//
// The equivalent WMI query is:
//
//	SELECT [selectors] FROM MSFT_Volume
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-volume
// for the WMI class definition.
func QueryVolumeByUniqueID(volumeID string, selectorList []string) (*storage.MSFT_Volume, error) {
	var selectors []string
	selectors = append(selectors, selectorList...)
	selectors = append(selectors, "UniqueId")
	volumeQuery := query.NewWmiQueryWithSelectList("MSFT_Volume", selectors)
	instances, err := QueryInstances(WMINamespaceStorage, volumeQuery)
	if err != nil {
		return nil, err
	}

	for _, instance := range instances {
		volume, err := storage.NewMSFT_VolumeEx1(instance)
		if err != nil {
			return nil, fmt.Errorf("failed to query volume (%s). error: %w", volumeID, err)
		}

		uniqueID, err := volume.GetPropertyUniqueId()
		if err != nil {
			return nil, fmt.Errorf("failed to query volume unique ID (%s). error: %w", volumeID, err)
		}

		if uniqueID == volumeID {
			return volume, nil
		}
	}

	return nil, errors.NotFound
}

// ListVolumes retrieves all available volumes on the system.
//
// The equivalent WMI query is:
//
//	SELECT [selectors] FROM MSFT_Volume
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-volume
// for the WMI class definition.
func ListVolumes(selectorList []string) ([]*storage.MSFT_Volume, error) {
	diskQuery := query.NewWmiQueryWithSelectList("MSFT_Volume", selectorList)
	instances, err := QueryInstances(WMINamespaceStorage, diskQuery)
	if IgnoreNotFound(err) != nil {
		return nil, err
	}

	var volumes []*storage.MSFT_Volume
	for _, instance := range instances {
		volume, err := storage.NewMSFT_VolumeEx1(instance)
		if err != nil {
			return nil, fmt.Errorf("failed to query volume %v. error: %v", instance, err)
		}

		volumes = append(volumes, volume)
	}

	return volumes, nil
}

// FormatVolume formats the specified volume.
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/format-msft-volume
// for the WMI method definition.
func FormatVolume(volume *storage.MSFT_Volume, params ...interface{}) (int, error) {
	result, err := volume.InvokeMethodWithReturn("Format", params...)
	return int(result), err
}

// FlushVolume flushes the cached data in the volume's file system to disk.
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-volume-flush
// for the WMI method definition.
func FlushVolume(volume *storage.MSFT_Volume) (int, error) {
	result, err := volume.Flush()
	return int(result), err
}

// GetVolumeUniqueID returns the unique ID (object ID) of a volume.
func GetVolumeUniqueID(volume *storage.MSFT_Volume) (string, error) {
	return volume.GetPropertyUniqueId()
}

// GetVolumeFileSystemType returns the file system type of a volume.
func GetVolumeFileSystemType(volume *storage.MSFT_Volume) (int32, error) {
	fsType, err := volume.GetProperty("FileSystemType")
	if err != nil {
		return 0, err
	}
	return fsType.(int32), nil
}

// GetVolumeSize returns the size of a volume.
func GetVolumeSize(volume *storage.MSFT_Volume) (int64, error) {
	volumeSizeVal, err := volume.GetProperty("Size")
	if err != nil {
		return -1, err
	}

	volumeSize, err := strconv.ParseInt(volumeSizeVal.(string), 10, 64)
	if err != nil {
		return -1, err
	}

	return volumeSize, err
}

// GetVolumeSizeRemaining returns the remaining size of a volume.
func GetVolumeSizeRemaining(volume *storage.MSFT_Volume) (int64, error) {
	volumeSizeRemainingVal, err := volume.GetProperty("SizeRemaining")
	if err != nil {
		return -1, err
	}

	volumeSizeRemaining, err := strconv.ParseInt(volumeSizeRemainingVal.(string), 10, 64)
	if err != nil {
		return -1, err
	}

	return volumeSizeRemaining, err
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
func ListPartitionsOnDisk(diskNumber, partitionNumber uint32, selectorList []string) ([]*storage.MSFT_Partition, error) {
	filters := []*query.WmiQueryFilter{
		query.NewWmiQueryFilter("DiskNumber", strconv.Itoa(int(diskNumber)), query.Equals),
	}
	if partitionNumber > 0 {
		filters = append(filters, query.NewWmiQueryFilter("PartitionNumber", strconv.Itoa(int(partitionNumber)), query.Equals))
	}
	return ListPartitionsWithFilters(selectorList, filters...)
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
func ListPartitionsWithFilters(selectorList []string, filters ...*query.WmiQueryFilter) ([]*storage.MSFT_Partition, error) {
	partitionQuery := query.NewWmiQueryWithSelectList("MSFT_Partition", selectorList)
	partitionQuery.Filters = append(partitionQuery.Filters, filters...)
	instances, err := QueryInstances(WMINamespaceStorage, partitionQuery)
	if IgnoreNotFound(err) != nil {
		return nil, err
	}

	var partitions []*storage.MSFT_Partition
	for _, instance := range instances {
		part, err := storage.NewMSFT_PartitionEx1(instance)
		if err != nil {
			return nil, fmt.Errorf("failed to query partition %v. error: %v", instance, err)
		}

		partitions = append(partitions, part)
	}

	return partitions, nil
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
func FindPartitionsByVolume(volumes []*storage.MSFT_Volume) ([]*storage.MSFT_Partition, error) {
	var result []*storage.MSFT_Partition
	for _, vol := range volumes {
		collection, err := vol.GetAssociated("MSFT_PartitionToVolume", "MSFT_Partition", "Partition", "Volume")
		if err != nil {
			return nil, fmt.Errorf("failed to query associated partition for %v. error: %v", vol, err)
		}

		for _, instance := range collection {
			part, err := storage.NewMSFT_PartitionEx1(instance)
			if err != nil {
				return nil, fmt.Errorf("failed to query partition %v. error: %v", instance, err)
			}

			result = append(result, part)
		}
	}

	return result, nil
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
func FindVolumesByPartition(partitions []*storage.MSFT_Partition) ([]*storage.MSFT_Volume, error) {
	var result []*storage.MSFT_Volume
	for _, part := range partitions {
		collection, err := part.GetAssociated("MSFT_PartitionToVolume", "MSFT_Volume", "Volume", "Partition")
		if err != nil {
			return nil, fmt.Errorf("failed to query associated volumes for %v. error: %v", part, err)
		}

		for _, instance := range collection {
			volume, err := storage.NewMSFT_VolumeEx1(instance)
			if err != nil {
				return nil, fmt.Errorf("failed to query volume %v. error: %v", instance, err)
			}

			result = append(result, volume)
		}
	}

	return result, nil
}

// GetPartitionByVolumeUniqueID retrieves a specific partition from a volume identified by its unique ID.
func GetPartitionByVolumeUniqueID(volumeID string) (*storage.MSFT_Partition, error) {
	volume, err := QueryVolumeByUniqueID(volumeID, []string{"ObjectId"})
	if err != nil {
		return nil, err
	}

	result, err := FindPartitionsByVolume([]*storage.MSFT_Volume{volume})
	if err != nil {
		return nil, err
	}

	if len(result) == 0 {
		return nil, errors.NotFound
	}

	return result[0], nil
}

// GetVolumeByDriveLetter retrieves a volume associated with a specific drive letter.
func GetVolumeByDriveLetter(driveLetter string, partitionSelectorList []string) (*storage.MSFT_Volume, error) {
	var selectorsForPart []string
	selectorsForPart = append(selectorsForPart, partitionSelectorList...)
	selectorsForPart = append(selectorsForPart, "ObjectId")
	partitions, err := ListPartitionsWithFilters(selectorsForPart, query.NewWmiQueryFilter("DriveLetter", driveLetter, query.Equals))
	if err != nil {
		return nil, err
	}

	result, err := FindVolumesByPartition(partitions)
	if err != nil {
		return nil, err
	}

	if len(result) == 0 {
		return nil, errors.NotFound
	}

	return result[0], nil
}

// GetPartitionDiskNumber retrieves the disk number associated with a given partition.
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-partition
// for the WMI class definitions.
func GetPartitionDiskNumber(part *storage.MSFT_Partition) (uint32, error) {
	diskNumber, err := part.GetProperty("DiskNumber")
	if err != nil {
		return 0, err
	}

	return uint32(diskNumber.(int32)), nil
}

// SetPartitionState takes a partition online or offline.
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-partition-online and
// https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-partition-offline
// for the WMI method definition.
func SetPartitionState(part *storage.MSFT_Partition, online bool) (int, string, error) {
	method := "Offline"
	if online {
		method = "Online"
	}

	var status string
	result, err := part.InvokeMethodWithReturn(method, &status)
	return int(result), status, err
}

// GetPartitionSupportedSize retrieves the minimum and maximum sizes that the partition can be resized to using the ResizePartition method.
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-partition-getsupportedsizes
// for the WMI method definition.
func GetPartitionSupportedSize(part *storage.MSFT_Partition) (result int, sizeMin, sizeMax int64, status string, err error) {
	sizeMin = -1
	sizeMax = -1

	var sizeMinVar, sizeMaxVar ole.VARIANT
	invokeResult, err := part.InvokeMethodWithReturn("GetSupportedSize", &sizeMinVar, &sizeMaxVar, &status)
	if invokeResult != 0 || err != nil {
		result = int(invokeResult)
	}
	klog.V(5).Infof("got sizeMin (%v) sizeMax (%v) from partition (%v), status: %s", sizeMinVar, sizeMaxVar, part, status)

	sizeMin, err = strconv.ParseInt(sizeMinVar.ToString(), 10, 64)
	if err != nil {
		return
	}

	sizeMax, err = strconv.ParseInt(sizeMaxVar.ToString(), 10, 64)
	return
}

// ResizePartition resizes a partition.
//
// Refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/msft-partition-resize
// for the WMI method definition.
func ResizePartition(part *storage.MSFT_Partition, size int64) (int, string, error) {
	var status string
	result, err := part.InvokeMethodWithReturn("Resize", strconv.Itoa(int(size)), &status)
	return int(result), status, err
}

// GetPartitionSize returns the size of a partition.
func GetPartitionSize(part *storage.MSFT_Partition) (int64, error) {
	sizeProp, err := part.GetProperty("Size")
	if err != nil {
		return -1, err
	}

	size, err := strconv.ParseInt(sizeProp.(string), 10, 64)
	if err != nil {
		return -1, err
	}

	return size, err
}

// FilterForPartitionOnDisk creates a WMI query filter to query a disk by its number.
func FilterForPartitionOnDisk(diskNumber uint32) *query.WmiQueryFilter {
	return query.NewWmiQueryFilter("DiskNumber", strconv.Itoa(int(diskNumber)), query.Equals)
}

// FilterForPartitionsOfTypeNormal creates a WMI query filter for all non-reserved partitions.
func FilterForPartitionsOfTypeNormal() *query.WmiQueryFilter {
	return query.NewWmiQueryFilter("GptType", GPTPartitionTypeMicrosoftReserved, query.NotEquals)
}
