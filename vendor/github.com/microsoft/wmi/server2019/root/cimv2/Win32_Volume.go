// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.CIMV2
//////////////////////////////////////////////
package cimv2

import (
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// Win32_Volume struct
type Win32_Volume struct {
	*CIM_StorageVolume

	//
	Automount bool

	//
	BootVolume bool

	//
	Capacity uint64

	//
	Compressed bool

	//
	DirtyBitSet bool

	//
	DriveLetter string

	//
	DriveType uint32

	//
	FileSystem string

	//
	FreeSpace uint64

	//
	IndexingEnabled bool

	//
	Label string

	//
	MaximumFileNameLength uint32

	//
	PageFilePresent bool

	//
	QuotasEnabled bool

	//
	QuotasIncomplete bool

	//
	QuotasRebuilding bool

	//
	SerialNumber uint32

	//
	SupportsDiskQuotas bool

	//
	SupportsFileBasedCompression bool

	//
	SystemVolume bool
}

func NewWin32_VolumeEx1(instance *cim.WmiInstance) (newInstance *Win32_Volume, err error) {
	tmp, err := NewCIM_StorageVolumeEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_Volume{
		CIM_StorageVolume: tmp,
	}
	return
}

func NewWin32_VolumeEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_Volume, err error) {
	tmp, err := NewCIM_StorageVolumeEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_Volume{
		CIM_StorageVolume: tmp,
	}
	return
}

// SetAutomount sets the value of Automount for the instance
func (instance *Win32_Volume) SetPropertyAutomount(value bool) (err error) {
	return instance.SetProperty("Automount", (value))
}

// GetAutomount gets the value of Automount for the instance
func (instance *Win32_Volume) GetPropertyAutomount() (value bool, err error) {
	retValue, err := instance.GetProperty("Automount")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetBootVolume sets the value of BootVolume for the instance
func (instance *Win32_Volume) SetPropertyBootVolume(value bool) (err error) {
	return instance.SetProperty("BootVolume", (value))
}

// GetBootVolume gets the value of BootVolume for the instance
func (instance *Win32_Volume) GetPropertyBootVolume() (value bool, err error) {
	retValue, err := instance.GetProperty("BootVolume")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetCapacity sets the value of Capacity for the instance
func (instance *Win32_Volume) SetPropertyCapacity(value uint64) (err error) {
	return instance.SetProperty("Capacity", (value))
}

// GetCapacity gets the value of Capacity for the instance
func (instance *Win32_Volume) GetPropertyCapacity() (value uint64, err error) {
	retValue, err := instance.GetProperty("Capacity")
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

// SetCompressed sets the value of Compressed for the instance
func (instance *Win32_Volume) SetPropertyCompressed(value bool) (err error) {
	return instance.SetProperty("Compressed", (value))
}

// GetCompressed gets the value of Compressed for the instance
func (instance *Win32_Volume) GetPropertyCompressed() (value bool, err error) {
	retValue, err := instance.GetProperty("Compressed")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetDirtyBitSet sets the value of DirtyBitSet for the instance
func (instance *Win32_Volume) SetPropertyDirtyBitSet(value bool) (err error) {
	return instance.SetProperty("DirtyBitSet", (value))
}

// GetDirtyBitSet gets the value of DirtyBitSet for the instance
func (instance *Win32_Volume) GetPropertyDirtyBitSet() (value bool, err error) {
	retValue, err := instance.GetProperty("DirtyBitSet")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetDriveLetter sets the value of DriveLetter for the instance
func (instance *Win32_Volume) SetPropertyDriveLetter(value string) (err error) {
	return instance.SetProperty("DriveLetter", (value))
}

// GetDriveLetter gets the value of DriveLetter for the instance
func (instance *Win32_Volume) GetPropertyDriveLetter() (value string, err error) {
	retValue, err := instance.GetProperty("DriveLetter")
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

// SetDriveType sets the value of DriveType for the instance
func (instance *Win32_Volume) SetPropertyDriveType(value uint32) (err error) {
	return instance.SetProperty("DriveType", (value))
}

// GetDriveType gets the value of DriveType for the instance
func (instance *Win32_Volume) GetPropertyDriveType() (value uint32, err error) {
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
func (instance *Win32_Volume) SetPropertyFileSystem(value string) (err error) {
	return instance.SetProperty("FileSystem", (value))
}

// GetFileSystem gets the value of FileSystem for the instance
func (instance *Win32_Volume) GetPropertyFileSystem() (value string, err error) {
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

// SetFreeSpace sets the value of FreeSpace for the instance
func (instance *Win32_Volume) SetPropertyFreeSpace(value uint64) (err error) {
	return instance.SetProperty("FreeSpace", (value))
}

// GetFreeSpace gets the value of FreeSpace for the instance
func (instance *Win32_Volume) GetPropertyFreeSpace() (value uint64, err error) {
	retValue, err := instance.GetProperty("FreeSpace")
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

// SetIndexingEnabled sets the value of IndexingEnabled for the instance
func (instance *Win32_Volume) SetPropertyIndexingEnabled(value bool) (err error) {
	return instance.SetProperty("IndexingEnabled", (value))
}

// GetIndexingEnabled gets the value of IndexingEnabled for the instance
func (instance *Win32_Volume) GetPropertyIndexingEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("IndexingEnabled")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetLabel sets the value of Label for the instance
func (instance *Win32_Volume) SetPropertyLabel(value string) (err error) {
	return instance.SetProperty("Label", (value))
}

// GetLabel gets the value of Label for the instance
func (instance *Win32_Volume) GetPropertyLabel() (value string, err error) {
	retValue, err := instance.GetProperty("Label")
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

// SetMaximumFileNameLength sets the value of MaximumFileNameLength for the instance
func (instance *Win32_Volume) SetPropertyMaximumFileNameLength(value uint32) (err error) {
	return instance.SetProperty("MaximumFileNameLength", (value))
}

// GetMaximumFileNameLength gets the value of MaximumFileNameLength for the instance
func (instance *Win32_Volume) GetPropertyMaximumFileNameLength() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaximumFileNameLength")
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

// SetPageFilePresent sets the value of PageFilePresent for the instance
func (instance *Win32_Volume) SetPropertyPageFilePresent(value bool) (err error) {
	return instance.SetProperty("PageFilePresent", (value))
}

// GetPageFilePresent gets the value of PageFilePresent for the instance
func (instance *Win32_Volume) GetPropertyPageFilePresent() (value bool, err error) {
	retValue, err := instance.GetProperty("PageFilePresent")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetQuotasEnabled sets the value of QuotasEnabled for the instance
func (instance *Win32_Volume) SetPropertyQuotasEnabled(value bool) (err error) {
	return instance.SetProperty("QuotasEnabled", (value))
}

// GetQuotasEnabled gets the value of QuotasEnabled for the instance
func (instance *Win32_Volume) GetPropertyQuotasEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("QuotasEnabled")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetQuotasIncomplete sets the value of QuotasIncomplete for the instance
func (instance *Win32_Volume) SetPropertyQuotasIncomplete(value bool) (err error) {
	return instance.SetProperty("QuotasIncomplete", (value))
}

// GetQuotasIncomplete gets the value of QuotasIncomplete for the instance
func (instance *Win32_Volume) GetPropertyQuotasIncomplete() (value bool, err error) {
	retValue, err := instance.GetProperty("QuotasIncomplete")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetQuotasRebuilding sets the value of QuotasRebuilding for the instance
func (instance *Win32_Volume) SetPropertyQuotasRebuilding(value bool) (err error) {
	return instance.SetProperty("QuotasRebuilding", (value))
}

// GetQuotasRebuilding gets the value of QuotasRebuilding for the instance
func (instance *Win32_Volume) GetPropertyQuotasRebuilding() (value bool, err error) {
	retValue, err := instance.GetProperty("QuotasRebuilding")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetSerialNumber sets the value of SerialNumber for the instance
func (instance *Win32_Volume) SetPropertySerialNumber(value uint32) (err error) {
	return instance.SetProperty("SerialNumber", (value))
}

// GetSerialNumber gets the value of SerialNumber for the instance
func (instance *Win32_Volume) GetPropertySerialNumber() (value uint32, err error) {
	retValue, err := instance.GetProperty("SerialNumber")
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

// SetSupportsDiskQuotas sets the value of SupportsDiskQuotas for the instance
func (instance *Win32_Volume) SetPropertySupportsDiskQuotas(value bool) (err error) {
	return instance.SetProperty("SupportsDiskQuotas", (value))
}

// GetSupportsDiskQuotas gets the value of SupportsDiskQuotas for the instance
func (instance *Win32_Volume) GetPropertySupportsDiskQuotas() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsDiskQuotas")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetSupportsFileBasedCompression sets the value of SupportsFileBasedCompression for the instance
func (instance *Win32_Volume) SetPropertySupportsFileBasedCompression(value bool) (err error) {
	return instance.SetProperty("SupportsFileBasedCompression", (value))
}

// GetSupportsFileBasedCompression gets the value of SupportsFileBasedCompression for the instance
func (instance *Win32_Volume) GetPropertySupportsFileBasedCompression() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsFileBasedCompression")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetSystemVolume sets the value of SystemVolume for the instance
func (instance *Win32_Volume) SetPropertySystemVolume(value bool) (err error) {
	return instance.SetProperty("SystemVolume", (value))
}

// GetSystemVolume gets the value of SystemVolume for the instance
func (instance *Win32_Volume) GetPropertySystemVolume() (value bool, err error) {
	retValue, err := instance.GetProperty("SystemVolume")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

//

// <param name="FixErrors" type="bool "></param>
// <param name="ForceDismount" type="bool "></param>
// <param name="OkToRunAtBootUp" type="bool "></param>
// <param name="RecoverBadSectors" type="bool "></param>
// <param name="SkipFolderCycle" type="bool "></param>
// <param name="VigorousIndexCheck" type="bool "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_Volume) Chkdsk( /* IN */ FixErrors bool,
	/* IN */ VigorousIndexCheck bool,
	/* IN */ SkipFolderCycle bool,
	/* IN */ ForceDismount bool,
	/* IN */ RecoverBadSectors bool,
	/* IN */ OkToRunAtBootUp bool) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Chkdsk", FixErrors, VigorousIndexCheck, SkipFolderCycle, ForceDismount, RecoverBadSectors, OkToRunAtBootUp)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="Volume" type="string []"></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_Volume) ScheduleAutoChk( /* IN */ Volume []string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("ScheduleAutoChk", Volume)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="Volume" type="string []"></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_Volume) ExcludeFromAutoChk( /* IN */ Volume []string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("ExcludeFromAutoChk", Volume)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ClusterSize" type="uint32 "></param>
// <param name="EnableCompression" type="bool "></param>
// <param name="FileSystem" type="string "></param>
// <param name="Label" type="string "></param>
// <param name="QuickFormat" type="bool "></param>
// <param name="Version" type="uint32 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_Volume) Format( /* IN */ FileSystem string,
	/* IN */ QuickFormat bool,
	/* IN */ ClusterSize uint32,
	/* IN */ Label string,
	/* IN */ EnableCompression bool,
	/* IN */ Version uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Format", FileSystem, QuickFormat, ClusterSize, Label, EnableCompression, Version)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="Force" type="bool "></param>

// <param name="DefragAnalysis" type="interface{} "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_Volume) Defrag( /* IN */ Force bool,
	/* OUT */ DefragAnalysis interface{}) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("Defrag", Force)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="DefragAnalysis" type="interface{} "></param>
// <param name="DefragRecommended" type="bool "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_Volume) DefragAnalysis( /* OUT */ DefragRecommended bool,
	/* OUT */ DefragAnalysis interface{}) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("DefragAnalysis")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="Directory" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_Volume) AddMountPoint( /* IN */ Directory string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("AddMountPoint", Directory)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_Volume) Mount() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Mount")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="Force" type="bool "></param>
// <param name="Permanent" type="bool "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_Volume) Dismount( /* IN */ Force bool,
	/* IN */ Permanent bool) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Dismount", Force, Permanent)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}
