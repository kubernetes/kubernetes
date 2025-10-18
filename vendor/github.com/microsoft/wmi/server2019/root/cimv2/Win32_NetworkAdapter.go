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

// Win32_NetworkAdapter struct
type Win32_NetworkAdapter struct {
	*CIM_NetworkAdapter

	//
	AdapterType string

	//
	AdapterTypeId uint16

	//
	GUID string

	//
	Index uint32

	//
	Installed bool

	//
	InterfaceIndex uint32

	//
	MACAddress string

	//
	Manufacturer string

	//
	MaxNumberControlled uint32

	//
	NetConnectionID string

	//
	NetConnectionStatus uint16

	//
	NetEnabled bool

	//
	PhysicalAdapter bool

	//
	ProductName string

	//
	ServiceName string

	//
	TimeOfLastReset string
}

func NewWin32_NetworkAdapterEx1(instance *cim.WmiInstance) (newInstance *Win32_NetworkAdapter, err error) {
	tmp, err := NewCIM_NetworkAdapterEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_NetworkAdapter{
		CIM_NetworkAdapter: tmp,
	}
	return
}

func NewWin32_NetworkAdapterEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_NetworkAdapter, err error) {
	tmp, err := NewCIM_NetworkAdapterEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_NetworkAdapter{
		CIM_NetworkAdapter: tmp,
	}
	return
}

// SetAdapterType sets the value of AdapterType for the instance
func (instance *Win32_NetworkAdapter) SetPropertyAdapterType(value string) (err error) {
	return instance.SetProperty("AdapterType", (value))
}

// GetAdapterType gets the value of AdapterType for the instance
func (instance *Win32_NetworkAdapter) GetPropertyAdapterType() (value string, err error) {
	retValue, err := instance.GetProperty("AdapterType")
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

// SetAdapterTypeId sets the value of AdapterTypeId for the instance
func (instance *Win32_NetworkAdapter) SetPropertyAdapterTypeId(value uint16) (err error) {
	return instance.SetProperty("AdapterTypeId", (value))
}

// GetAdapterTypeId gets the value of AdapterTypeId for the instance
func (instance *Win32_NetworkAdapter) GetPropertyAdapterTypeId() (value uint16, err error) {
	retValue, err := instance.GetProperty("AdapterTypeId")
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

// SetGUID sets the value of GUID for the instance
func (instance *Win32_NetworkAdapter) SetPropertyGUID(value string) (err error) {
	return instance.SetProperty("GUID", (value))
}

// GetGUID gets the value of GUID for the instance
func (instance *Win32_NetworkAdapter) GetPropertyGUID() (value string, err error) {
	retValue, err := instance.GetProperty("GUID")
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
func (instance *Win32_NetworkAdapter) SetPropertyIndex(value uint32) (err error) {
	return instance.SetProperty("Index", (value))
}

// GetIndex gets the value of Index for the instance
func (instance *Win32_NetworkAdapter) GetPropertyIndex() (value uint32, err error) {
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

// SetInstalled sets the value of Installed for the instance
func (instance *Win32_NetworkAdapter) SetPropertyInstalled(value bool) (err error) {
	return instance.SetProperty("Installed", (value))
}

// GetInstalled gets the value of Installed for the instance
func (instance *Win32_NetworkAdapter) GetPropertyInstalled() (value bool, err error) {
	retValue, err := instance.GetProperty("Installed")
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

// SetInterfaceIndex sets the value of InterfaceIndex for the instance
func (instance *Win32_NetworkAdapter) SetPropertyInterfaceIndex(value uint32) (err error) {
	return instance.SetProperty("InterfaceIndex", (value))
}

// GetInterfaceIndex gets the value of InterfaceIndex for the instance
func (instance *Win32_NetworkAdapter) GetPropertyInterfaceIndex() (value uint32, err error) {
	retValue, err := instance.GetProperty("InterfaceIndex")
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

// SetMACAddress sets the value of MACAddress for the instance
func (instance *Win32_NetworkAdapter) SetPropertyMACAddress(value string) (err error) {
	return instance.SetProperty("MACAddress", (value))
}

// GetMACAddress gets the value of MACAddress for the instance
func (instance *Win32_NetworkAdapter) GetPropertyMACAddress() (value string, err error) {
	retValue, err := instance.GetProperty("MACAddress")
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
func (instance *Win32_NetworkAdapter) SetPropertyManufacturer(value string) (err error) {
	return instance.SetProperty("Manufacturer", (value))
}

// GetManufacturer gets the value of Manufacturer for the instance
func (instance *Win32_NetworkAdapter) GetPropertyManufacturer() (value string, err error) {
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

// SetMaxNumberControlled sets the value of MaxNumberControlled for the instance
func (instance *Win32_NetworkAdapter) SetPropertyMaxNumberControlled(value uint32) (err error) {
	return instance.SetProperty("MaxNumberControlled", (value))
}

// GetMaxNumberControlled gets the value of MaxNumberControlled for the instance
func (instance *Win32_NetworkAdapter) GetPropertyMaxNumberControlled() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaxNumberControlled")
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

// SetNetConnectionID sets the value of NetConnectionID for the instance
func (instance *Win32_NetworkAdapter) SetPropertyNetConnectionID(value string) (err error) {
	return instance.SetProperty("NetConnectionID", (value))
}

// GetNetConnectionID gets the value of NetConnectionID for the instance
func (instance *Win32_NetworkAdapter) GetPropertyNetConnectionID() (value string, err error) {
	retValue, err := instance.GetProperty("NetConnectionID")
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

// SetNetConnectionStatus sets the value of NetConnectionStatus for the instance
func (instance *Win32_NetworkAdapter) SetPropertyNetConnectionStatus(value uint16) (err error) {
	return instance.SetProperty("NetConnectionStatus", (value))
}

// GetNetConnectionStatus gets the value of NetConnectionStatus for the instance
func (instance *Win32_NetworkAdapter) GetPropertyNetConnectionStatus() (value uint16, err error) {
	retValue, err := instance.GetProperty("NetConnectionStatus")
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

// SetNetEnabled sets the value of NetEnabled for the instance
func (instance *Win32_NetworkAdapter) SetPropertyNetEnabled(value bool) (err error) {
	return instance.SetProperty("NetEnabled", (value))
}

// GetNetEnabled gets the value of NetEnabled for the instance
func (instance *Win32_NetworkAdapter) GetPropertyNetEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("NetEnabled")
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

// SetPhysicalAdapter sets the value of PhysicalAdapter for the instance
func (instance *Win32_NetworkAdapter) SetPropertyPhysicalAdapter(value bool) (err error) {
	return instance.SetProperty("PhysicalAdapter", (value))
}

// GetPhysicalAdapter gets the value of PhysicalAdapter for the instance
func (instance *Win32_NetworkAdapter) GetPropertyPhysicalAdapter() (value bool, err error) {
	retValue, err := instance.GetProperty("PhysicalAdapter")
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

// SetProductName sets the value of ProductName for the instance
func (instance *Win32_NetworkAdapter) SetPropertyProductName(value string) (err error) {
	return instance.SetProperty("ProductName", (value))
}

// GetProductName gets the value of ProductName for the instance
func (instance *Win32_NetworkAdapter) GetPropertyProductName() (value string, err error) {
	retValue, err := instance.GetProperty("ProductName")
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

// SetServiceName sets the value of ServiceName for the instance
func (instance *Win32_NetworkAdapter) SetPropertyServiceName(value string) (err error) {
	return instance.SetProperty("ServiceName", (value))
}

// GetServiceName gets the value of ServiceName for the instance
func (instance *Win32_NetworkAdapter) GetPropertyServiceName() (value string, err error) {
	retValue, err := instance.GetProperty("ServiceName")
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

// SetTimeOfLastReset sets the value of TimeOfLastReset for the instance
func (instance *Win32_NetworkAdapter) SetPropertyTimeOfLastReset(value string) (err error) {
	return instance.SetProperty("TimeOfLastReset", (value))
}

// GetTimeOfLastReset gets the value of TimeOfLastReset for the instance
func (instance *Win32_NetworkAdapter) GetPropertyTimeOfLastReset() (value string, err error) {
	retValue, err := instance.GetProperty("TimeOfLastReset")
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

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapter) Enable() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Enable")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapter) Disable() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Disable")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}
