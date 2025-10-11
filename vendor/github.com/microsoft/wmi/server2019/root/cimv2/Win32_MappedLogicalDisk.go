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

// Win32_MappedLogicalDisk struct
type Win32_MappedLogicalDisk struct {
	*CIM_LogicalDisk

	//
	Compressed bool

	//
	FileSystem string

	//
	MaximumComponentLength uint32

	//
	ProviderName string

	//
	QuotasDisabled bool

	//
	QuotasIncomplete bool

	//
	QuotasRebuilding bool

	//
	SessionID string

	//
	SupportsDiskQuotas bool

	//
	SupportsFileBasedCompression bool

	//
	VolumeName string

	//
	VolumeSerialNumber string
}

func NewWin32_MappedLogicalDiskEx1(instance *cim.WmiInstance) (newInstance *Win32_MappedLogicalDisk, err error) {
	tmp, err := NewCIM_LogicalDiskEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_MappedLogicalDisk{
		CIM_LogicalDisk: tmp,
	}
	return
}

func NewWin32_MappedLogicalDiskEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_MappedLogicalDisk, err error) {
	tmp, err := NewCIM_LogicalDiskEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_MappedLogicalDisk{
		CIM_LogicalDisk: tmp,
	}
	return
}

// SetCompressed sets the value of Compressed for the instance
func (instance *Win32_MappedLogicalDisk) SetPropertyCompressed(value bool) (err error) {
	return instance.SetProperty("Compressed", (value))
}

// GetCompressed gets the value of Compressed for the instance
func (instance *Win32_MappedLogicalDisk) GetPropertyCompressed() (value bool, err error) {
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

// SetFileSystem sets the value of FileSystem for the instance
func (instance *Win32_MappedLogicalDisk) SetPropertyFileSystem(value string) (err error) {
	return instance.SetProperty("FileSystem", (value))
}

// GetFileSystem gets the value of FileSystem for the instance
func (instance *Win32_MappedLogicalDisk) GetPropertyFileSystem() (value string, err error) {
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
func (instance *Win32_MappedLogicalDisk) SetPropertyMaximumComponentLength(value uint32) (err error) {
	return instance.SetProperty("MaximumComponentLength", (value))
}

// GetMaximumComponentLength gets the value of MaximumComponentLength for the instance
func (instance *Win32_MappedLogicalDisk) GetPropertyMaximumComponentLength() (value uint32, err error) {
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

// SetProviderName sets the value of ProviderName for the instance
func (instance *Win32_MappedLogicalDisk) SetPropertyProviderName(value string) (err error) {
	return instance.SetProperty("ProviderName", (value))
}

// GetProviderName gets the value of ProviderName for the instance
func (instance *Win32_MappedLogicalDisk) GetPropertyProviderName() (value string, err error) {
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
func (instance *Win32_MappedLogicalDisk) SetPropertyQuotasDisabled(value bool) (err error) {
	return instance.SetProperty("QuotasDisabled", (value))
}

// GetQuotasDisabled gets the value of QuotasDisabled for the instance
func (instance *Win32_MappedLogicalDisk) GetPropertyQuotasDisabled() (value bool, err error) {
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
func (instance *Win32_MappedLogicalDisk) SetPropertyQuotasIncomplete(value bool) (err error) {
	return instance.SetProperty("QuotasIncomplete", (value))
}

// GetQuotasIncomplete gets the value of QuotasIncomplete for the instance
func (instance *Win32_MappedLogicalDisk) GetPropertyQuotasIncomplete() (value bool, err error) {
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
func (instance *Win32_MappedLogicalDisk) SetPropertyQuotasRebuilding(value bool) (err error) {
	return instance.SetProperty("QuotasRebuilding", (value))
}

// GetQuotasRebuilding gets the value of QuotasRebuilding for the instance
func (instance *Win32_MappedLogicalDisk) GetPropertyQuotasRebuilding() (value bool, err error) {
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

// SetSessionID sets the value of SessionID for the instance
func (instance *Win32_MappedLogicalDisk) SetPropertySessionID(value string) (err error) {
	return instance.SetProperty("SessionID", (value))
}

// GetSessionID gets the value of SessionID for the instance
func (instance *Win32_MappedLogicalDisk) GetPropertySessionID() (value string, err error) {
	retValue, err := instance.GetProperty("SessionID")
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

// SetSupportsDiskQuotas sets the value of SupportsDiskQuotas for the instance
func (instance *Win32_MappedLogicalDisk) SetPropertySupportsDiskQuotas(value bool) (err error) {
	return instance.SetProperty("SupportsDiskQuotas", (value))
}

// GetSupportsDiskQuotas gets the value of SupportsDiskQuotas for the instance
func (instance *Win32_MappedLogicalDisk) GetPropertySupportsDiskQuotas() (value bool, err error) {
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
func (instance *Win32_MappedLogicalDisk) SetPropertySupportsFileBasedCompression(value bool) (err error) {
	return instance.SetProperty("SupportsFileBasedCompression", (value))
}

// GetSupportsFileBasedCompression gets the value of SupportsFileBasedCompression for the instance
func (instance *Win32_MappedLogicalDisk) GetPropertySupportsFileBasedCompression() (value bool, err error) {
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

// SetVolumeName sets the value of VolumeName for the instance
func (instance *Win32_MappedLogicalDisk) SetPropertyVolumeName(value string) (err error) {
	return instance.SetProperty("VolumeName", (value))
}

// GetVolumeName gets the value of VolumeName for the instance
func (instance *Win32_MappedLogicalDisk) GetPropertyVolumeName() (value string, err error) {
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
func (instance *Win32_MappedLogicalDisk) SetPropertyVolumeSerialNumber(value string) (err error) {
	return instance.SetProperty("VolumeSerialNumber", (value))
}

// GetVolumeSerialNumber gets the value of VolumeSerialNumber for the instance
func (instance *Win32_MappedLogicalDisk) GetPropertyVolumeSerialNumber() (value string, err error) {
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
