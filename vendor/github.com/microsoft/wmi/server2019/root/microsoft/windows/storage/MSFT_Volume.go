// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.Microsoft.Windows.Storage
//////////////////////////////////////////////
package storage

import (
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// MSFT_Volume struct
type MSFT_Volume struct {
	*MSFT_StorageObject

	//
	AllocationUnitSize uint32

	//
	DedupMode uint32

	//
	DriveLetter byte

	//
	DriveType uint32

	//
	FileSystem string

	//
	FileSystemLabel string

	//
	FileSystemType uint16

	//
	HealthStatus uint16

	//
	OperationalStatus []uint16

	//
	Path string

	//
	Size uint64

	//
	SizeRemaining uint64
}

func NewMSFT_VolumeEx1(instance *cim.WmiInstance) (newInstance *MSFT_Volume, err error) {
	tmp, err := NewMSFT_StorageObjectEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_Volume{
		MSFT_StorageObject: tmp,
	}
	return
}

func NewMSFT_VolumeEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_Volume, err error) {
	tmp, err := NewMSFT_StorageObjectEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_Volume{
		MSFT_StorageObject: tmp,
	}
	return
}

// SetAllocationUnitSize sets the value of AllocationUnitSize for the instance
func (instance *MSFT_Volume) SetPropertyAllocationUnitSize(value uint32) (err error) {
	return instance.SetProperty("AllocationUnitSize", (value))
}

// GetAllocationUnitSize gets the value of AllocationUnitSize for the instance
func (instance *MSFT_Volume) GetPropertyAllocationUnitSize() (value uint32, err error) {
	retValue, err := instance.GetProperty("AllocationUnitSize")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}

// SetDedupMode sets the value of DedupMode for the instance
func (instance *MSFT_Volume) SetPropertyDedupMode(value uint32) (err error) {
	return instance.SetProperty("DedupMode", (value))
}

// GetDedupMode gets the value of DedupMode for the instance
func (instance *MSFT_Volume) GetPropertyDedupMode() (value uint32, err error) {
	retValue, err := instance.GetProperty("DedupMode")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}

// SetDriveLetter sets the value of DriveLetter for the instance
func (instance *MSFT_Volume) SetPropertyDriveLetter(value byte) (err error) {
	return instance.SetProperty("DriveLetter", (value))
}

// GetDriveLetter gets the value of DriveLetter for the instance
func (instance *MSFT_Volume) GetPropertyDriveLetter() (value byte, err error) {
	retValue, err := instance.GetProperty("DriveLetter")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(byte)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " byte is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = byte(valuetmp)

	return
}

// SetDriveType sets the value of DriveType for the instance
func (instance *MSFT_Volume) SetPropertyDriveType(value uint32) (err error) {
	return instance.SetProperty("DriveType", (value))
}

// GetDriveType gets the value of DriveType for the instance
func (instance *MSFT_Volume) GetPropertyDriveType() (value uint32, err error) {
	retValue, err := instance.GetProperty("DriveType")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}

// SetFileSystem sets the value of FileSystem for the instance
func (instance *MSFT_Volume) SetPropertyFileSystem(value string) (err error) {
	return instance.SetProperty("FileSystem", (value))
}

// GetFileSystem gets the value of FileSystem for the instance
func (instance *MSFT_Volume) GetPropertyFileSystem() (value string, err error) {
	retValue, err := instance.GetProperty("FileSystem")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetFileSystemLabel sets the value of FileSystemLabel for the instance
func (instance *MSFT_Volume) SetPropertyFileSystemLabel(value string) (err error) {
	return instance.SetProperty("FileSystemLabel", (value))
}

// GetFileSystemLabel gets the value of FileSystemLabel for the instance
func (instance *MSFT_Volume) GetPropertyFileSystemLabel() (value string, err error) {
	retValue, err := instance.GetProperty("FileSystemLabel")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetFileSystemType sets the value of FileSystemType for the instance
func (instance *MSFT_Volume) SetPropertyFileSystemType(value uint16) (err error) {
	return instance.SetProperty("FileSystemType", (value))
}

// GetFileSystemType gets the value of FileSystemType for the instance
func (instance *MSFT_Volume) GetPropertyFileSystemType() (value uint16, err error) {
	retValue, err := instance.GetProperty("FileSystemType")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetHealthStatus sets the value of HealthStatus for the instance
func (instance *MSFT_Volume) SetPropertyHealthStatus(value uint16) (err error) {
	return instance.SetProperty("HealthStatus", (value))
}

// GetHealthStatus gets the value of HealthStatus for the instance
func (instance *MSFT_Volume) GetPropertyHealthStatus() (value uint16, err error) {
	retValue, err := instance.GetProperty("HealthStatus")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetOperationalStatus sets the value of OperationalStatus for the instance
func (instance *MSFT_Volume) SetPropertyOperationalStatus(value []uint16) (err error) {
	return instance.SetProperty("OperationalStatus", (value))
}

// GetOperationalStatus gets the value of OperationalStatus for the instance
func (instance *MSFT_Volume) GetPropertyOperationalStatus() (value []uint16, err error) {
	retValue, err := instance.GetProperty("OperationalStatus")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint16)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint16(valuetmp))
	}

	return
}

// SetPath sets the value of Path for the instance
func (instance *MSFT_Volume) SetPropertyPath(value string) (err error) {
	return instance.SetProperty("Path", (value))
}

// GetPath gets the value of Path for the instance
func (instance *MSFT_Volume) GetPropertyPath() (value string, err error) {
	retValue, err := instance.GetProperty("Path")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetSize sets the value of Size for the instance
func (instance *MSFT_Volume) SetPropertySize(value uint64) (err error) {
	return instance.SetProperty("Size", (value))
}

// GetSize gets the value of Size for the instance
func (instance *MSFT_Volume) GetPropertySize() (value uint64, err error) {
	retValue, err := instance.GetProperty("Size")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetSizeRemaining sets the value of SizeRemaining for the instance
func (instance *MSFT_Volume) SetPropertySizeRemaining(value uint64) (err error) {
	return instance.SetProperty("SizeRemaining", (value))
}

// GetSizeRemaining gets the value of SizeRemaining for the instance
func (instance *MSFT_Volume) GetPropertySizeRemaining() (value uint64, err error) {
	retValue, err := instance.GetProperty("SizeRemaining")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

//

// <param name="RunAsJob" type="bool "></param>

// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_Volume) DeleteObject( /* IN */ RunAsJob bool,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("DeleteObject", RunAsJob)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="AllocationUnitSize" type="uint32 "></param>
// <param name="Compress" type="bool "></param>
// <param name="DisableHeatGathering" type="bool "></param>
// <param name="FileSystem" type="string "></param>
// <param name="FileSystemLabel" type="string "></param>
// <param name="Force" type="bool "></param>
// <param name="Full" type="bool "></param>
// <param name="IsDAX" type="bool "></param>
// <param name="RunAsJob" type="bool "></param>
// <param name="SetIntegrityStreams" type="bool "></param>
// <param name="ShortFileNameSupport" type="bool "></param>
// <param name="UseLargeFRS" type="bool "></param>

// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="FormattedVolume" type="MSFT_Volume "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_Volume) Format( /* IN */ FileSystem string,
	/* IN */ FileSystemLabel string,
	/* IN */ AllocationUnitSize uint32,
	/* IN */ Full bool,
	/* IN */ Force bool,
	/* IN */ Compress bool,
	/* IN */ ShortFileNameSupport bool,
	/* IN */ SetIntegrityStreams bool,
	/* IN */ UseLargeFRS bool,
	/* IN */ DisableHeatGathering bool,
	/* IN */ IsDAX bool,
	/* IN */ RunAsJob bool,
	/* OUT */ FormattedVolume MSFT_Volume,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("Format", FileSystem, FileSystemLabel, AllocationUnitSize, Full, Force, Compress, ShortFileNameSupport, SetIntegrityStreams, UseLargeFRS, DisableHeatGathering, IsDAX, RunAsJob)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="OfflineScanAndFix" type="bool "></param>
// <param name="RunAsJob" type="bool "></param>
// <param name="Scan" type="bool "></param>
// <param name="SpotFix" type="bool "></param>

// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="Output" type="uint32 "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_Volume) Repair( /* IN */ OfflineScanAndFix bool,
	/* IN */ Scan bool,
	/* IN */ SpotFix bool,
	/* OUT */ Output uint32,
	/* OPTIONAL IN */ RunAsJob bool,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("Repair", OfflineScanAndFix, Scan, SpotFix, RunAsJob)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="Analyze" type="bool "></param>
// <param name="Defrag" type="bool "></param>
// <param name="NormalPriority" type="bool "></param>
// <param name="ReTrim" type="bool "></param>
// <param name="RunAsJob" type="bool "></param>
// <param name="SlabConsolidate" type="bool "></param>
// <param name="TierOptimize" type="bool "></param>

// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_Volume) Optimize( /* IN */ ReTrim bool,
	/* IN */ Analyze bool,
	/* IN */ Defrag bool,
	/* IN */ SlabConsolidate bool,
	/* IN */ TierOptimize bool,
	/* IN */ NormalPriority bool,
	/* IN */ RunAsJob bool,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("Optimize", ReTrim, Analyze, Defrag, SlabConsolidate, TierOptimize, NormalPriority, RunAsJob)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="FileSystemLabel" type="string "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_Volume) SetFileSystemLabel( /* IN */ FileSystemLabel string,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("SetFileSystemLabel", FileSystemLabel)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
// <param name="SupportedFileSystems" type="string []"></param>
func (instance *MSFT_Volume) GetSupportedFileSystems( /* OUT */ SupportedFileSystems []string,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetSupportedFileSystems")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="FileSystem" type="string "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
// <param name="SupportedClusterSizes" type="uint32 []"></param>
func (instance *MSFT_Volume) GetSupportedClusterSizes( /* IN */ FileSystem string,
	/* OUT */ SupportedClusterSizes []uint32,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetSupportedClusterSizes", FileSystem)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="CorruptionCount" type="uint32 "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_Volume) GetCorruptionCount( /* OUT */ CorruptionCount uint32,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetCorruptionCount")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
// <param name="VolumeScrubEnabled" type="bool "></param>
func (instance *MSFT_Volume) GetAttributes( /* OUT */ VolumeScrubEnabled bool) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetAttributes")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="EnableVolumeScrub" type="bool "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_Volume) SetAttributes( /* IN */ EnableVolumeScrub bool,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("SetAttributes", EnableVolumeScrub)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_Volume) Flush() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Flush")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="RunAsJob" type="bool "></param>
// <param name="Size" type="uint64 "></param>

// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_Volume) Resize( /* IN */ Size uint64,
	/* IN */ RunAsJob bool,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("Resize", Size, RunAsJob)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="DiagnoseResults" type="MSFT_StorageDiagnoseResult []"></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_Volume) Diagnose( /* OUT */ DiagnoseResults []MSFT_StorageDiagnoseResult,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("Diagnose")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="DedupMode" type="uint32 "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_Volume) SetDedupMode( /* IN */ DedupMode uint32,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("SetDedupMode", DedupMode)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="DedupProperties" type="MSFT_DedupProperties "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_Volume) GetDedupProperties( /* OUT */ DedupProperties MSFT_DedupProperties,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetDedupProperties")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="ActionResults" type="MSFT_HealthAction []"></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_Volume) GetActions( /* OUT */ ActionResults []MSFT_HealthAction,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetActions")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}
