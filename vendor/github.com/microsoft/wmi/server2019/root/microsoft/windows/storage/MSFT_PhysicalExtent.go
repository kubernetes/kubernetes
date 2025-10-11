// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.Microsoft.Windows.Storage
//////////////////////////////////////////////
package storage

import (
	"github.com/microsoft/wmi/pkg/base/instance"
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// MSFT_PhysicalExtent struct
type MSFT_PhysicalExtent struct {
	*cim.WmiInstance

	//
	ColumnNumber uint16

	//
	CopyNumber uint16

	//
	Flags uint64

	//
	OperationalDetails []string

	//
	OperationalStatus []uint16

	//
	PhysicalDiskOffset uint64

	//
	PhysicalDiskUniqueId string

	//
	ReplacementCopyNumber uint16

	//
	Size uint64

	//
	StorageTierUniqueId string

	//
	VirtualDiskOffset uint64

	//
	VirtualDiskUniqueId string
}

func NewMSFT_PhysicalExtentEx1(instance *cim.WmiInstance) (newInstance *MSFT_PhysicalExtent, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_PhysicalExtent{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_PhysicalExtentEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_PhysicalExtent, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_PhysicalExtent{
		WmiInstance: tmp,
	}
	return
}

// SetColumnNumber sets the value of ColumnNumber for the instance
func (instance *MSFT_PhysicalExtent) SetPropertyColumnNumber(value uint16) (err error) {
	return instance.SetProperty("ColumnNumber", (value))
}

// GetColumnNumber gets the value of ColumnNumber for the instance
func (instance *MSFT_PhysicalExtent) GetPropertyColumnNumber() (value uint16, err error) {
	retValue, err := instance.GetProperty("ColumnNumber")
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

// SetCopyNumber sets the value of CopyNumber for the instance
func (instance *MSFT_PhysicalExtent) SetPropertyCopyNumber(value uint16) (err error) {
	return instance.SetProperty("CopyNumber", (value))
}

// GetCopyNumber gets the value of CopyNumber for the instance
func (instance *MSFT_PhysicalExtent) GetPropertyCopyNumber() (value uint16, err error) {
	retValue, err := instance.GetProperty("CopyNumber")
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

// SetFlags sets the value of Flags for the instance
func (instance *MSFT_PhysicalExtent) SetPropertyFlags(value uint64) (err error) {
	return instance.SetProperty("Flags", (value))
}

// GetFlags gets the value of Flags for the instance
func (instance *MSFT_PhysicalExtent) GetPropertyFlags() (value uint64, err error) {
	retValue, err := instance.GetProperty("Flags")
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

// SetOperationalDetails sets the value of OperationalDetails for the instance
func (instance *MSFT_PhysicalExtent) SetPropertyOperationalDetails(value []string) (err error) {
	return instance.SetProperty("OperationalDetails", (value))
}

// GetOperationalDetails gets the value of OperationalDetails for the instance
func (instance *MSFT_PhysicalExtent) GetPropertyOperationalDetails() (value []string, err error) {
	retValue, err := instance.GetProperty("OperationalDetails")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetOperationalStatus sets the value of OperationalStatus for the instance
func (instance *MSFT_PhysicalExtent) SetPropertyOperationalStatus(value []uint16) (err error) {
	return instance.SetProperty("OperationalStatus", (value))
}

// GetOperationalStatus gets the value of OperationalStatus for the instance
func (instance *MSFT_PhysicalExtent) GetPropertyOperationalStatus() (value []uint16, err error) {
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

// SetPhysicalDiskOffset sets the value of PhysicalDiskOffset for the instance
func (instance *MSFT_PhysicalExtent) SetPropertyPhysicalDiskOffset(value uint64) (err error) {
	return instance.SetProperty("PhysicalDiskOffset", (value))
}

// GetPhysicalDiskOffset gets the value of PhysicalDiskOffset for the instance
func (instance *MSFT_PhysicalExtent) GetPropertyPhysicalDiskOffset() (value uint64, err error) {
	retValue, err := instance.GetProperty("PhysicalDiskOffset")
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

// SetPhysicalDiskUniqueId sets the value of PhysicalDiskUniqueId for the instance
func (instance *MSFT_PhysicalExtent) SetPropertyPhysicalDiskUniqueId(value string) (err error) {
	return instance.SetProperty("PhysicalDiskUniqueId", (value))
}

// GetPhysicalDiskUniqueId gets the value of PhysicalDiskUniqueId for the instance
func (instance *MSFT_PhysicalExtent) GetPropertyPhysicalDiskUniqueId() (value string, err error) {
	retValue, err := instance.GetProperty("PhysicalDiskUniqueId")
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

// SetReplacementCopyNumber sets the value of ReplacementCopyNumber for the instance
func (instance *MSFT_PhysicalExtent) SetPropertyReplacementCopyNumber(value uint16) (err error) {
	return instance.SetProperty("ReplacementCopyNumber", (value))
}

// GetReplacementCopyNumber gets the value of ReplacementCopyNumber for the instance
func (instance *MSFT_PhysicalExtent) GetPropertyReplacementCopyNumber() (value uint16, err error) {
	retValue, err := instance.GetProperty("ReplacementCopyNumber")
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

// SetSize sets the value of Size for the instance
func (instance *MSFT_PhysicalExtent) SetPropertySize(value uint64) (err error) {
	return instance.SetProperty("Size", (value))
}

// GetSize gets the value of Size for the instance
func (instance *MSFT_PhysicalExtent) GetPropertySize() (value uint64, err error) {
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

// SetStorageTierUniqueId sets the value of StorageTierUniqueId for the instance
func (instance *MSFT_PhysicalExtent) SetPropertyStorageTierUniqueId(value string) (err error) {
	return instance.SetProperty("StorageTierUniqueId", (value))
}

// GetStorageTierUniqueId gets the value of StorageTierUniqueId for the instance
func (instance *MSFT_PhysicalExtent) GetPropertyStorageTierUniqueId() (value string, err error) {
	retValue, err := instance.GetProperty("StorageTierUniqueId")
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

// SetVirtualDiskOffset sets the value of VirtualDiskOffset for the instance
func (instance *MSFT_PhysicalExtent) SetPropertyVirtualDiskOffset(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskOffset", (value))
}

// GetVirtualDiskOffset gets the value of VirtualDiskOffset for the instance
func (instance *MSFT_PhysicalExtent) GetPropertyVirtualDiskOffset() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskOffset")
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

// SetVirtualDiskUniqueId sets the value of VirtualDiskUniqueId for the instance
func (instance *MSFT_PhysicalExtent) SetPropertyVirtualDiskUniqueId(value string) (err error) {
	return instance.SetProperty("VirtualDiskUniqueId", (value))
}

// GetVirtualDiskUniqueId gets the value of VirtualDiskUniqueId for the instance
func (instance *MSFT_PhysicalExtent) GetPropertyVirtualDiskUniqueId() (value string, err error) {
	retValue, err := instance.GetProperty("VirtualDiskUniqueId")
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
