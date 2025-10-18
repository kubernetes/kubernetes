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

// Win32_CDROMDrive struct
type Win32_CDROMDrive struct {
	*CIM_CDROMDrive

	//
	Drive string

	//
	DriveIntegrity bool

	//
	FileSystemFlags uint16

	//
	FileSystemFlagsEx uint32

	//
	Id string

	//
	Manufacturer string

	//
	MaximumComponentLength uint32

	//
	MediaLoaded bool

	//
	MediaType string

	//
	MfrAssignedRevisionLevel string

	//
	RevisionLevel string

	//
	SCSIBus uint32

	//
	SCSILogicalUnit uint16

	//
	SCSIPort uint16

	//
	SCSITargetId uint16

	//
	SerialNumber string

	//
	Size uint64

	//
	TransferRate float64

	//
	VolumeName string

	//
	VolumeSerialNumber string
}

func NewWin32_CDROMDriveEx1(instance *cim.WmiInstance) (newInstance *Win32_CDROMDrive, err error) {
	tmp, err := NewCIM_CDROMDriveEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_CDROMDrive{
		CIM_CDROMDrive: tmp,
	}
	return
}

func NewWin32_CDROMDriveEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_CDROMDrive, err error) {
	tmp, err := NewCIM_CDROMDriveEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_CDROMDrive{
		CIM_CDROMDrive: tmp,
	}
	return
}

// SetDrive sets the value of Drive for the instance
func (instance *Win32_CDROMDrive) SetPropertyDrive(value string) (err error) {
	return instance.SetProperty("Drive", (value))
}

// GetDrive gets the value of Drive for the instance
func (instance *Win32_CDROMDrive) GetPropertyDrive() (value string, err error) {
	retValue, err := instance.GetProperty("Drive")
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

// SetDriveIntegrity sets the value of DriveIntegrity for the instance
func (instance *Win32_CDROMDrive) SetPropertyDriveIntegrity(value bool) (err error) {
	return instance.SetProperty("DriveIntegrity", (value))
}

// GetDriveIntegrity gets the value of DriveIntegrity for the instance
func (instance *Win32_CDROMDrive) GetPropertyDriveIntegrity() (value bool, err error) {
	retValue, err := instance.GetProperty("DriveIntegrity")
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

// SetFileSystemFlags sets the value of FileSystemFlags for the instance
func (instance *Win32_CDROMDrive) SetPropertyFileSystemFlags(value uint16) (err error) {
	return instance.SetProperty("FileSystemFlags", (value))
}

// GetFileSystemFlags gets the value of FileSystemFlags for the instance
func (instance *Win32_CDROMDrive) GetPropertyFileSystemFlags() (value uint16, err error) {
	retValue, err := instance.GetProperty("FileSystemFlags")
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

// SetFileSystemFlagsEx sets the value of FileSystemFlagsEx for the instance
func (instance *Win32_CDROMDrive) SetPropertyFileSystemFlagsEx(value uint32) (err error) {
	return instance.SetProperty("FileSystemFlagsEx", (value))
}

// GetFileSystemFlagsEx gets the value of FileSystemFlagsEx for the instance
func (instance *Win32_CDROMDrive) GetPropertyFileSystemFlagsEx() (value uint32, err error) {
	retValue, err := instance.GetProperty("FileSystemFlagsEx")
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

// SetId sets the value of Id for the instance
func (instance *Win32_CDROMDrive) SetPropertyId(value string) (err error) {
	return instance.SetProperty("Id", (value))
}

// GetId gets the value of Id for the instance
func (instance *Win32_CDROMDrive) GetPropertyId() (value string, err error) {
	retValue, err := instance.GetProperty("Id")
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

// SetManufacturer sets the value of Manufacturer for the instance
func (instance *Win32_CDROMDrive) SetPropertyManufacturer(value string) (err error) {
	return instance.SetProperty("Manufacturer", (value))
}

// GetManufacturer gets the value of Manufacturer for the instance
func (instance *Win32_CDROMDrive) GetPropertyManufacturer() (value string, err error) {
	retValue, err := instance.GetProperty("Manufacturer")
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
func (instance *Win32_CDROMDrive) SetPropertyMaximumComponentLength(value uint32) (err error) {
	return instance.SetProperty("MaximumComponentLength", (value))
}

// GetMaximumComponentLength gets the value of MaximumComponentLength for the instance
func (instance *Win32_CDROMDrive) GetPropertyMaximumComponentLength() (value uint32, err error) {
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

// SetMediaLoaded sets the value of MediaLoaded for the instance
func (instance *Win32_CDROMDrive) SetPropertyMediaLoaded(value bool) (err error) {
	return instance.SetProperty("MediaLoaded", (value))
}

// GetMediaLoaded gets the value of MediaLoaded for the instance
func (instance *Win32_CDROMDrive) GetPropertyMediaLoaded() (value bool, err error) {
	retValue, err := instance.GetProperty("MediaLoaded")
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

// SetMediaType sets the value of MediaType for the instance
func (instance *Win32_CDROMDrive) SetPropertyMediaType(value string) (err error) {
	return instance.SetProperty("MediaType", (value))
}

// GetMediaType gets the value of MediaType for the instance
func (instance *Win32_CDROMDrive) GetPropertyMediaType() (value string, err error) {
	retValue, err := instance.GetProperty("MediaType")
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

// SetMfrAssignedRevisionLevel sets the value of MfrAssignedRevisionLevel for the instance
func (instance *Win32_CDROMDrive) SetPropertyMfrAssignedRevisionLevel(value string) (err error) {
	return instance.SetProperty("MfrAssignedRevisionLevel", (value))
}

// GetMfrAssignedRevisionLevel gets the value of MfrAssignedRevisionLevel for the instance
func (instance *Win32_CDROMDrive) GetPropertyMfrAssignedRevisionLevel() (value string, err error) {
	retValue, err := instance.GetProperty("MfrAssignedRevisionLevel")
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

// SetRevisionLevel sets the value of RevisionLevel for the instance
func (instance *Win32_CDROMDrive) SetPropertyRevisionLevel(value string) (err error) {
	return instance.SetProperty("RevisionLevel", (value))
}

// GetRevisionLevel gets the value of RevisionLevel for the instance
func (instance *Win32_CDROMDrive) GetPropertyRevisionLevel() (value string, err error) {
	retValue, err := instance.GetProperty("RevisionLevel")
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

// SetSCSIBus sets the value of SCSIBus for the instance
func (instance *Win32_CDROMDrive) SetPropertySCSIBus(value uint32) (err error) {
	return instance.SetProperty("SCSIBus", (value))
}

// GetSCSIBus gets the value of SCSIBus for the instance
func (instance *Win32_CDROMDrive) GetPropertySCSIBus() (value uint32, err error) {
	retValue, err := instance.GetProperty("SCSIBus")
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

// SetSCSILogicalUnit sets the value of SCSILogicalUnit for the instance
func (instance *Win32_CDROMDrive) SetPropertySCSILogicalUnit(value uint16) (err error) {
	return instance.SetProperty("SCSILogicalUnit", (value))
}

// GetSCSILogicalUnit gets the value of SCSILogicalUnit for the instance
func (instance *Win32_CDROMDrive) GetPropertySCSILogicalUnit() (value uint16, err error) {
	retValue, err := instance.GetProperty("SCSILogicalUnit")
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

// SetSCSIPort sets the value of SCSIPort for the instance
func (instance *Win32_CDROMDrive) SetPropertySCSIPort(value uint16) (err error) {
	return instance.SetProperty("SCSIPort", (value))
}

// GetSCSIPort gets the value of SCSIPort for the instance
func (instance *Win32_CDROMDrive) GetPropertySCSIPort() (value uint16, err error) {
	retValue, err := instance.GetProperty("SCSIPort")
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

// SetSCSITargetId sets the value of SCSITargetId for the instance
func (instance *Win32_CDROMDrive) SetPropertySCSITargetId(value uint16) (err error) {
	return instance.SetProperty("SCSITargetId", (value))
}

// GetSCSITargetId gets the value of SCSITargetId for the instance
func (instance *Win32_CDROMDrive) GetPropertySCSITargetId() (value uint16, err error) {
	retValue, err := instance.GetProperty("SCSITargetId")
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

// SetSerialNumber sets the value of SerialNumber for the instance
func (instance *Win32_CDROMDrive) SetPropertySerialNumber(value string) (err error) {
	return instance.SetProperty("SerialNumber", (value))
}

// GetSerialNumber gets the value of SerialNumber for the instance
func (instance *Win32_CDROMDrive) GetPropertySerialNumber() (value string, err error) {
	retValue, err := instance.GetProperty("SerialNumber")
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
func (instance *Win32_CDROMDrive) SetPropertySize(value uint64) (err error) {
	return instance.SetProperty("Size", (value))
}

// GetSize gets the value of Size for the instance
func (instance *Win32_CDROMDrive) GetPropertySize() (value uint64, err error) {
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

// SetTransferRate sets the value of TransferRate for the instance
func (instance *Win32_CDROMDrive) SetPropertyTransferRate(value float64) (err error) {
	return instance.SetProperty("TransferRate", (value))
}

// GetTransferRate gets the value of TransferRate for the instance
func (instance *Win32_CDROMDrive) GetPropertyTransferRate() (value float64, err error) {
	retValue, err := instance.GetProperty("TransferRate")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(float64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " float64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = float64(valuetmp)

	return
}

// SetVolumeName sets the value of VolumeName for the instance
func (instance *Win32_CDROMDrive) SetPropertyVolumeName(value string) (err error) {
	return instance.SetProperty("VolumeName", (value))
}

// GetVolumeName gets the value of VolumeName for the instance
func (instance *Win32_CDROMDrive) GetPropertyVolumeName() (value string, err error) {
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
func (instance *Win32_CDROMDrive) SetPropertyVolumeSerialNumber(value string) (err error) {
	return instance.SetProperty("VolumeSerialNumber", (value))
}

// GetVolumeSerialNumber gets the value of VolumeSerialNumber for the instance
func (instance *Win32_CDROMDrive) GetPropertyVolumeSerialNumber() (value string, err error) {
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
