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

// Win32_LogicalDisk struct
type Win32_LogicalDisk struct {
	*CIM_LogicalDisk

	//
	Compressed bool

	//
	DriveType uint32

	//
	FileSystem string

	//
	MaximumComponentLength uint32

	//
	MediaType uint32

	//
	ProviderName string

	//
	QuotasDisabled bool

	//
	QuotasIncomplete bool

	//
	QuotasRebuilding bool

	//
	SupportsDiskQuotas bool

	//
	SupportsFileBasedCompression bool

	//
	VolumeDirty bool

	//
	VolumeName string

	//
	VolumeSerialNumber string
}

func NewWin32_LogicalDiskEx1(instance *cim.WmiInstance) (newInstance *Win32_LogicalDisk, err error) {
	tmp, err := NewCIM_LogicalDiskEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_LogicalDisk{
		CIM_LogicalDisk: tmp,
	}
	return
}

func NewWin32_LogicalDiskEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_LogicalDisk, err error) {
	tmp, err := NewCIM_LogicalDiskEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_LogicalDisk{
		CIM_LogicalDisk: tmp,
	}
	return
}

// SetCompressed sets the value of Compressed for the instance
func (instance *Win32_LogicalDisk) SetPropertyCompressed(value bool) (err error) {
	return instance.SetProperty("Compressed", (value))
}

// GetCompressed gets the value of Compressed for the instance
func (instance *Win32_LogicalDisk) GetPropertyCompressed() (value bool, err error) {
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

// SetDriveType sets the value of DriveType for the instance
func (instance *Win32_LogicalDisk) SetPropertyDriveType(value uint32) (err error) {
	return instance.SetProperty("DriveType", (value))
}

// GetDriveType gets the value of DriveType for the instance
func (instance *Win32_LogicalDisk) GetPropertyDriveType() (value uint32, err error) {
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
func (instance *Win32_LogicalDisk) SetPropertyFileSystem(value string) (err error) {
	return instance.SetProperty("FileSystem", (value))
}

// GetFileSystem gets the value of FileSystem for the instance
func (instance *Win32_LogicalDisk) GetPropertyFileSystem() (value string, err error) {
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

// SetMaximumComponentLength sets the value of MaximumComponentLength for the instance
func (instance *Win32_LogicalDisk) SetPropertyMaximumComponentLength(value uint32) (err error) {
	return instance.SetProperty("MaximumComponentLength", (value))
}

// GetMaximumComponentLength gets the value of MaximumComponentLength for the instance
func (instance *Win32_LogicalDisk) GetPropertyMaximumComponentLength() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaximumComponentLength")
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

// SetMediaType sets the value of MediaType for the instance
func (instance *Win32_LogicalDisk) SetPropertyMediaType(value uint32) (err error) {
	return instance.SetProperty("MediaType", (value))
}

// GetMediaType gets the value of MediaType for the instance
func (instance *Win32_LogicalDisk) GetPropertyMediaType() (value uint32, err error) {
	retValue, err := instance.GetProperty("MediaType")
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

// SetProviderName sets the value of ProviderName for the instance
func (instance *Win32_LogicalDisk) SetPropertyProviderName(value string) (err error) {
	return instance.SetProperty("ProviderName", (value))
}

// GetProviderName gets the value of ProviderName for the instance
func (instance *Win32_LogicalDisk) GetPropertyProviderName() (value string, err error) {
	retValue, err := instance.GetProperty("ProviderName")
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

// SetQuotasDisabled sets the value of QuotasDisabled for the instance
func (instance *Win32_LogicalDisk) SetPropertyQuotasDisabled(value bool) (err error) {
	return instance.SetProperty("QuotasDisabled", (value))
}

// GetQuotasDisabled gets the value of QuotasDisabled for the instance
func (instance *Win32_LogicalDisk) GetPropertyQuotasDisabled() (value bool, err error) {
	retValue, err := instance.GetProperty("QuotasDisabled")
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
func (instance *Win32_LogicalDisk) SetPropertyQuotasIncomplete(value bool) (err error) {
	return instance.SetProperty("QuotasIncomplete", (value))
}

// GetQuotasIncomplete gets the value of QuotasIncomplete for the instance
func (instance *Win32_LogicalDisk) GetPropertyQuotasIncomplete() (value bool, err error) {
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
func (instance *Win32_LogicalDisk) SetPropertyQuotasRebuilding(value bool) (err error) {
	return instance.SetProperty("QuotasRebuilding", (value))
}

// GetQuotasRebuilding gets the value of QuotasRebuilding for the instance
func (instance *Win32_LogicalDisk) GetPropertyQuotasRebuilding() (value bool, err error) {
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

// SetSupportsDiskQuotas sets the value of SupportsDiskQuotas for the instance
func (instance *Win32_LogicalDisk) SetPropertySupportsDiskQuotas(value bool) (err error) {
	return instance.SetProperty("SupportsDiskQuotas", (value))
}

// GetSupportsDiskQuotas gets the value of SupportsDiskQuotas for the instance
func (instance *Win32_LogicalDisk) GetPropertySupportsDiskQuotas() (value bool, err error) {
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
func (instance *Win32_LogicalDisk) SetPropertySupportsFileBasedCompression(value bool) (err error) {
	return instance.SetProperty("SupportsFileBasedCompression", (value))
}

// GetSupportsFileBasedCompression gets the value of SupportsFileBasedCompression for the instance
func (instance *Win32_LogicalDisk) GetPropertySupportsFileBasedCompression() (value bool, err error) {
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

// SetVolumeDirty sets the value of VolumeDirty for the instance
func (instance *Win32_LogicalDisk) SetPropertyVolumeDirty(value bool) (err error) {
	return instance.SetProperty("VolumeDirty", (value))
}

// GetVolumeDirty gets the value of VolumeDirty for the instance
func (instance *Win32_LogicalDisk) GetPropertyVolumeDirty() (value bool, err error) {
	retValue, err := instance.GetProperty("VolumeDirty")
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

// SetVolumeName sets the value of VolumeName for the instance
func (instance *Win32_LogicalDisk) SetPropertyVolumeName(value string) (err error) {
	return instance.SetProperty("VolumeName", (value))
}

// GetVolumeName gets the value of VolumeName for the instance
func (instance *Win32_LogicalDisk) GetPropertyVolumeName() (value string, err error) {
	retValue, err := instance.GetProperty("VolumeName")
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

// SetVolumeSerialNumber sets the value of VolumeSerialNumber for the instance
func (instance *Win32_LogicalDisk) SetPropertyVolumeSerialNumber(value string) (err error) {
	return instance.SetProperty("VolumeSerialNumber", (value))
}

// GetVolumeSerialNumber gets the value of VolumeSerialNumber for the instance
func (instance *Win32_LogicalDisk) GetPropertyVolumeSerialNumber() (value string, err error) {
	retValue, err := instance.GetProperty("VolumeSerialNumber")
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

//

// <param name="FixErrors" type="bool "></param>
// <param name="ForceDismount" type="bool "></param>
// <param name="OkToRunAtBootUp" type="bool "></param>
// <param name="RecoverBadSectors" type="bool "></param>
// <param name="SkipFolderCycle" type="bool "></param>
// <param name="VigorousIndexCheck" type="bool "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_LogicalDisk) Chkdsk( /* IN */ FixErrors bool,
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

// <param name="LogicalDisk" type="string []"></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_LogicalDisk) ScheduleAutoChk( /* IN */ LogicalDisk []string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("ScheduleAutoChk", LogicalDisk)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="LogicalDisk" type="string []"></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_LogicalDisk) ExcludeFromAutochk( /* IN */ LogicalDisk []string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("ExcludeFromAutochk", LogicalDisk)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}
