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

// Win32_PnPSignedDriver struct
type Win32_PnPSignedDriver struct {
	*CIM_Service

	//
	ClassGuid string

	//
	CompatID string

	//
	DeviceClass string

	//
	DeviceID string

	//
	DeviceName string

	//
	DevLoader string

	//
	DriverDate string

	//
	DriverName string

	//
	DriverProviderName string

	//
	DriverVersion string

	//
	FriendlyName string

	//
	HardWareID string

	//
	InfName string

	//
	IsSigned bool

	//
	Location string

	//
	Manufacturer string

	//
	PDO string

	//
	Signer string
}

func NewWin32_PnPSignedDriverEx1(instance *cim.WmiInstance) (newInstance *Win32_PnPSignedDriver, err error) {
	tmp, err := NewCIM_ServiceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PnPSignedDriver{
		CIM_Service: tmp,
	}
	return
}

func NewWin32_PnPSignedDriverEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PnPSignedDriver, err error) {
	tmp, err := NewCIM_ServiceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PnPSignedDriver{
		CIM_Service: tmp,
	}
	return
}

// SetClassGuid sets the value of ClassGuid for the instance
func (instance *Win32_PnPSignedDriver) SetPropertyClassGuid(value string) (err error) {
	return instance.SetProperty("ClassGuid", (value))
}

// GetClassGuid gets the value of ClassGuid for the instance
func (instance *Win32_PnPSignedDriver) GetPropertyClassGuid() (value string, err error) {
	retValue, err := instance.GetProperty("ClassGuid")
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

// SetCompatID sets the value of CompatID for the instance
func (instance *Win32_PnPSignedDriver) SetPropertyCompatID(value string) (err error) {
	return instance.SetProperty("CompatID", (value))
}

// GetCompatID gets the value of CompatID for the instance
func (instance *Win32_PnPSignedDriver) GetPropertyCompatID() (value string, err error) {
	retValue, err := instance.GetProperty("CompatID")
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

// SetDeviceClass sets the value of DeviceClass for the instance
func (instance *Win32_PnPSignedDriver) SetPropertyDeviceClass(value string) (err error) {
	return instance.SetProperty("DeviceClass", (value))
}

// GetDeviceClass gets the value of DeviceClass for the instance
func (instance *Win32_PnPSignedDriver) GetPropertyDeviceClass() (value string, err error) {
	retValue, err := instance.GetProperty("DeviceClass")
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

// SetDeviceID sets the value of DeviceID for the instance
func (instance *Win32_PnPSignedDriver) SetPropertyDeviceID(value string) (err error) {
	return instance.SetProperty("DeviceID", (value))
}

// GetDeviceID gets the value of DeviceID for the instance
func (instance *Win32_PnPSignedDriver) GetPropertyDeviceID() (value string, err error) {
	retValue, err := instance.GetProperty("DeviceID")
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

// SetDeviceName sets the value of DeviceName for the instance
func (instance *Win32_PnPSignedDriver) SetPropertyDeviceName(value string) (err error) {
	return instance.SetProperty("DeviceName", (value))
}

// GetDeviceName gets the value of DeviceName for the instance
func (instance *Win32_PnPSignedDriver) GetPropertyDeviceName() (value string, err error) {
	retValue, err := instance.GetProperty("DeviceName")
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

// SetDevLoader sets the value of DevLoader for the instance
func (instance *Win32_PnPSignedDriver) SetPropertyDevLoader(value string) (err error) {
	return instance.SetProperty("DevLoader", (value))
}

// GetDevLoader gets the value of DevLoader for the instance
func (instance *Win32_PnPSignedDriver) GetPropertyDevLoader() (value string, err error) {
	retValue, err := instance.GetProperty("DevLoader")
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

// SetDriverDate sets the value of DriverDate for the instance
func (instance *Win32_PnPSignedDriver) SetPropertyDriverDate(value string) (err error) {
	return instance.SetProperty("DriverDate", (value))
}

// GetDriverDate gets the value of DriverDate for the instance
func (instance *Win32_PnPSignedDriver) GetPropertyDriverDate() (value string, err error) {
	retValue, err := instance.GetProperty("DriverDate")
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

// SetDriverName sets the value of DriverName for the instance
func (instance *Win32_PnPSignedDriver) SetPropertyDriverName(value string) (err error) {
	return instance.SetProperty("DriverName", (value))
}

// GetDriverName gets the value of DriverName for the instance
func (instance *Win32_PnPSignedDriver) GetPropertyDriverName() (value string, err error) {
	retValue, err := instance.GetProperty("DriverName")
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

// SetDriverProviderName sets the value of DriverProviderName for the instance
func (instance *Win32_PnPSignedDriver) SetPropertyDriverProviderName(value string) (err error) {
	return instance.SetProperty("DriverProviderName", (value))
}

// GetDriverProviderName gets the value of DriverProviderName for the instance
func (instance *Win32_PnPSignedDriver) GetPropertyDriverProviderName() (value string, err error) {
	retValue, err := instance.GetProperty("DriverProviderName")
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

// SetDriverVersion sets the value of DriverVersion for the instance
func (instance *Win32_PnPSignedDriver) SetPropertyDriverVersion(value string) (err error) {
	return instance.SetProperty("DriverVersion", (value))
}

// GetDriverVersion gets the value of DriverVersion for the instance
func (instance *Win32_PnPSignedDriver) GetPropertyDriverVersion() (value string, err error) {
	retValue, err := instance.GetProperty("DriverVersion")
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

// SetFriendlyName sets the value of FriendlyName for the instance
func (instance *Win32_PnPSignedDriver) SetPropertyFriendlyName(value string) (err error) {
	return instance.SetProperty("FriendlyName", (value))
}

// GetFriendlyName gets the value of FriendlyName for the instance
func (instance *Win32_PnPSignedDriver) GetPropertyFriendlyName() (value string, err error) {
	retValue, err := instance.GetProperty("FriendlyName")
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

// SetHardWareID sets the value of HardWareID for the instance
func (instance *Win32_PnPSignedDriver) SetPropertyHardWareID(value string) (err error) {
	return instance.SetProperty("HardWareID", (value))
}

// GetHardWareID gets the value of HardWareID for the instance
func (instance *Win32_PnPSignedDriver) GetPropertyHardWareID() (value string, err error) {
	retValue, err := instance.GetProperty("HardWareID")
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

// SetInfName sets the value of InfName for the instance
func (instance *Win32_PnPSignedDriver) SetPropertyInfName(value string) (err error) {
	return instance.SetProperty("InfName", (value))
}

// GetInfName gets the value of InfName for the instance
func (instance *Win32_PnPSignedDriver) GetPropertyInfName() (value string, err error) {
	retValue, err := instance.GetProperty("InfName")
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

// SetIsSigned sets the value of IsSigned for the instance
func (instance *Win32_PnPSignedDriver) SetPropertyIsSigned(value bool) (err error) {
	return instance.SetProperty("IsSigned", (value))
}

// GetIsSigned gets the value of IsSigned for the instance
func (instance *Win32_PnPSignedDriver) GetPropertyIsSigned() (value bool, err error) {
	retValue, err := instance.GetProperty("IsSigned")
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

// SetLocation sets the value of Location for the instance
func (instance *Win32_PnPSignedDriver) SetPropertyLocation(value string) (err error) {
	return instance.SetProperty("Location", (value))
}

// GetLocation gets the value of Location for the instance
func (instance *Win32_PnPSignedDriver) GetPropertyLocation() (value string, err error) {
	retValue, err := instance.GetProperty("Location")
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
func (instance *Win32_PnPSignedDriver) SetPropertyManufacturer(value string) (err error) {
	return instance.SetProperty("Manufacturer", (value))
}

// GetManufacturer gets the value of Manufacturer for the instance
func (instance *Win32_PnPSignedDriver) GetPropertyManufacturer() (value string, err error) {
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

// SetPDO sets the value of PDO for the instance
func (instance *Win32_PnPSignedDriver) SetPropertyPDO(value string) (err error) {
	return instance.SetProperty("PDO", (value))
}

// GetPDO gets the value of PDO for the instance
func (instance *Win32_PnPSignedDriver) GetPropertyPDO() (value string, err error) {
	retValue, err := instance.GetProperty("PDO")
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

// SetSigner sets the value of Signer for the instance
func (instance *Win32_PnPSignedDriver) SetPropertySigner(value string) (err error) {
	return instance.SetProperty("Signer", (value))
}

// GetSigner gets the value of Signer for the instance
func (instance *Win32_PnPSignedDriver) GetPropertySigner() (value string, err error) {
	retValue, err := instance.GetProperty("Signer")
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
