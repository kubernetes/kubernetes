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

// Win32_DiskDrive struct
type Win32_DiskDrive struct {
	*CIM_DiskDrive

	//
	BytesPerSector uint32

	//
	FirmwareRevision string

	//
	Index uint32

	//
	InterfaceType string

	//
	Manufacturer string

	//
	MediaLoaded bool

	//
	MediaType string

	//
	Model string

	//
	Partitions uint32

	//
	SCSIBus uint32

	//
	SCSILogicalUnit uint16

	//
	SCSIPort uint16

	//
	SCSITargetId uint16

	//
	SectorsPerTrack uint32

	//
	SerialNumber string

	//
	Signature uint32

	//
	Size uint64

	//
	TotalCylinders uint64

	//
	TotalHeads uint32

	//
	TotalSectors uint64

	//
	TotalTracks uint64

	//
	TracksPerCylinder uint32
}

func NewWin32_DiskDriveEx1(instance *cim.WmiInstance) (newInstance *Win32_DiskDrive, err error) {
	tmp, err := NewCIM_DiskDriveEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_DiskDrive{
		CIM_DiskDrive: tmp,
	}
	return
}

func NewWin32_DiskDriveEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_DiskDrive, err error) {
	tmp, err := NewCIM_DiskDriveEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_DiskDrive{
		CIM_DiskDrive: tmp,
	}
	return
}

// SetBytesPerSector sets the value of BytesPerSector for the instance
func (instance *Win32_DiskDrive) SetPropertyBytesPerSector(value uint32) (err error) {
	return instance.SetProperty("BytesPerSector", (value))
}

// GetBytesPerSector gets the value of BytesPerSector for the instance
func (instance *Win32_DiskDrive) GetPropertyBytesPerSector() (value uint32, err error) {
	retValue, err := instance.GetProperty("BytesPerSector")
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

// SetFirmwareRevision sets the value of FirmwareRevision for the instance
func (instance *Win32_DiskDrive) SetPropertyFirmwareRevision(value string) (err error) {
	return instance.SetProperty("FirmwareRevision", (value))
}

// GetFirmwareRevision gets the value of FirmwareRevision for the instance
func (instance *Win32_DiskDrive) GetPropertyFirmwareRevision() (value string, err error) {
	retValue, err := instance.GetProperty("FirmwareRevision")
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

// SetIndex sets the value of Index for the instance
func (instance *Win32_DiskDrive) SetPropertyIndex(value uint32) (err error) {
	return instance.SetProperty("Index", (value))
}

// GetIndex gets the value of Index for the instance
func (instance *Win32_DiskDrive) GetPropertyIndex() (value uint32, err error) {
	retValue, err := instance.GetProperty("Index")
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

// SetInterfaceType sets the value of InterfaceType for the instance
func (instance *Win32_DiskDrive) SetPropertyInterfaceType(value string) (err error) {
	return instance.SetProperty("InterfaceType", (value))
}

// GetInterfaceType gets the value of InterfaceType for the instance
func (instance *Win32_DiskDrive) GetPropertyInterfaceType() (value string, err error) {
	retValue, err := instance.GetProperty("InterfaceType")
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
func (instance *Win32_DiskDrive) SetPropertyManufacturer(value string) (err error) {
	return instance.SetProperty("Manufacturer", (value))
}

// GetManufacturer gets the value of Manufacturer for the instance
func (instance *Win32_DiskDrive) GetPropertyManufacturer() (value string, err error) {
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

// SetMediaLoaded sets the value of MediaLoaded for the instance
func (instance *Win32_DiskDrive) SetPropertyMediaLoaded(value bool) (err error) {
	return instance.SetProperty("MediaLoaded", (value))
}

// GetMediaLoaded gets the value of MediaLoaded for the instance
func (instance *Win32_DiskDrive) GetPropertyMediaLoaded() (value bool, err error) {
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
func (instance *Win32_DiskDrive) SetPropertyMediaType(value string) (err error) {
	return instance.SetProperty("MediaType", (value))
}

// GetMediaType gets the value of MediaType for the instance
func (instance *Win32_DiskDrive) GetPropertyMediaType() (value string, err error) {
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

// SetModel sets the value of Model for the instance
func (instance *Win32_DiskDrive) SetPropertyModel(value string) (err error) {
	return instance.SetProperty("Model", (value))
}

// GetModel gets the value of Model for the instance
func (instance *Win32_DiskDrive) GetPropertyModel() (value string, err error) {
	retValue, err := instance.GetProperty("Model")
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

// SetPartitions sets the value of Partitions for the instance
func (instance *Win32_DiskDrive) SetPropertyPartitions(value uint32) (err error) {
	return instance.SetProperty("Partitions", (value))
}

// GetPartitions gets the value of Partitions for the instance
func (instance *Win32_DiskDrive) GetPropertyPartitions() (value uint32, err error) {
	retValue, err := instance.GetProperty("Partitions")
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

// SetSCSIBus sets the value of SCSIBus for the instance
func (instance *Win32_DiskDrive) SetPropertySCSIBus(value uint32) (err error) {
	return instance.SetProperty("SCSIBus", (value))
}

// GetSCSIBus gets the value of SCSIBus for the instance
func (instance *Win32_DiskDrive) GetPropertySCSIBus() (value uint32, err error) {
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
func (instance *Win32_DiskDrive) SetPropertySCSILogicalUnit(value uint16) (err error) {
	return instance.SetProperty("SCSILogicalUnit", (value))
}

// GetSCSILogicalUnit gets the value of SCSILogicalUnit for the instance
func (instance *Win32_DiskDrive) GetPropertySCSILogicalUnit() (value uint16, err error) {
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
func (instance *Win32_DiskDrive) SetPropertySCSIPort(value uint16) (err error) {
	return instance.SetProperty("SCSIPort", (value))
}

// GetSCSIPort gets the value of SCSIPort for the instance
func (instance *Win32_DiskDrive) GetPropertySCSIPort() (value uint16, err error) {
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
func (instance *Win32_DiskDrive) SetPropertySCSITargetId(value uint16) (err error) {
	return instance.SetProperty("SCSITargetId", (value))
}

// GetSCSITargetId gets the value of SCSITargetId for the instance
func (instance *Win32_DiskDrive) GetPropertySCSITargetId() (value uint16, err error) {
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

// SetSectorsPerTrack sets the value of SectorsPerTrack for the instance
func (instance *Win32_DiskDrive) SetPropertySectorsPerTrack(value uint32) (err error) {
	return instance.SetProperty("SectorsPerTrack", (value))
}

// GetSectorsPerTrack gets the value of SectorsPerTrack for the instance
func (instance *Win32_DiskDrive) GetPropertySectorsPerTrack() (value uint32, err error) {
	retValue, err := instance.GetProperty("SectorsPerTrack")
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

// SetSerialNumber sets the value of SerialNumber for the instance
func (instance *Win32_DiskDrive) SetPropertySerialNumber(value string) (err error) {
	return instance.SetProperty("SerialNumber", (value))
}

// GetSerialNumber gets the value of SerialNumber for the instance
func (instance *Win32_DiskDrive) GetPropertySerialNumber() (value string, err error) {
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

// SetSignature sets the value of Signature for the instance
func (instance *Win32_DiskDrive) SetPropertySignature(value uint32) (err error) {
	return instance.SetProperty("Signature", (value))
}

// GetSignature gets the value of Signature for the instance
func (instance *Win32_DiskDrive) GetPropertySignature() (value uint32, err error) {
	retValue, err := instance.GetProperty("Signature")
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

// SetSize sets the value of Size for the instance
func (instance *Win32_DiskDrive) SetPropertySize(value uint64) (err error) {
	return instance.SetProperty("Size", (value))
}

// GetSize gets the value of Size for the instance
func (instance *Win32_DiskDrive) GetPropertySize() (value uint64, err error) {
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

// SetTotalCylinders sets the value of TotalCylinders for the instance
func (instance *Win32_DiskDrive) SetPropertyTotalCylinders(value uint64) (err error) {
	return instance.SetProperty("TotalCylinders", (value))
}

// GetTotalCylinders gets the value of TotalCylinders for the instance
func (instance *Win32_DiskDrive) GetPropertyTotalCylinders() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalCylinders")
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

// SetTotalHeads sets the value of TotalHeads for the instance
func (instance *Win32_DiskDrive) SetPropertyTotalHeads(value uint32) (err error) {
	return instance.SetProperty("TotalHeads", (value))
}

// GetTotalHeads gets the value of TotalHeads for the instance
func (instance *Win32_DiskDrive) GetPropertyTotalHeads() (value uint32, err error) {
	retValue, err := instance.GetProperty("TotalHeads")
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

// SetTotalSectors sets the value of TotalSectors for the instance
func (instance *Win32_DiskDrive) SetPropertyTotalSectors(value uint64) (err error) {
	return instance.SetProperty("TotalSectors", (value))
}

// GetTotalSectors gets the value of TotalSectors for the instance
func (instance *Win32_DiskDrive) GetPropertyTotalSectors() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalSectors")
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

// SetTotalTracks sets the value of TotalTracks for the instance
func (instance *Win32_DiskDrive) SetPropertyTotalTracks(value uint64) (err error) {
	return instance.SetProperty("TotalTracks", (value))
}

// GetTotalTracks gets the value of TotalTracks for the instance
func (instance *Win32_DiskDrive) GetPropertyTotalTracks() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalTracks")
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

// SetTracksPerCylinder sets the value of TracksPerCylinder for the instance
func (instance *Win32_DiskDrive) SetPropertyTracksPerCylinder(value uint32) (err error) {
	return instance.SetProperty("TracksPerCylinder", (value))
}

// GetTracksPerCylinder gets the value of TracksPerCylinder for the instance
func (instance *Win32_DiskDrive) GetPropertyTracksPerCylinder() (value uint32, err error) {
	retValue, err := instance.GetProperty("TracksPerCylinder")
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
