// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.CIMV2
//////////////////////////////////////////////
package cimv2

import (
	"github.com/microsoft/wmi/pkg/base/instance"
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// Win32_ProtocolBinding struct
type Win32_ProtocolBinding struct {
	*cim.WmiInstance

	//
	Antecedent Win32_NetworkProtocol

	//
	Dependent Win32_SystemDriver

	//
	Device Win32_NetworkAdapter
}

func NewWin32_ProtocolBindingEx1(instance *cim.WmiInstance) (newInstance *Win32_ProtocolBinding, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_ProtocolBinding{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_ProtocolBindingEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ProtocolBinding, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ProtocolBinding{
		WmiInstance: tmp,
	}
	return
}

// SetAntecedent sets the value of Antecedent for the instance
func (instance *Win32_ProtocolBinding) SetPropertyAntecedent(value Win32_NetworkProtocol) (err error) {
	return instance.SetProperty("Antecedent", (value))
}

// GetAntecedent gets the value of Antecedent for the instance
func (instance *Win32_ProtocolBinding) GetPropertyAntecedent() (value Win32_NetworkProtocol, err error) {
	retValue, err := instance.GetProperty("Antecedent")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_NetworkProtocol)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_NetworkProtocol is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_NetworkProtocol(valuetmp)

	return
}

// SetDependent sets the value of Dependent for the instance
func (instance *Win32_ProtocolBinding) SetPropertyDependent(value Win32_SystemDriver) (err error) {
	return instance.SetProperty("Dependent", (value))
}

// GetDependent gets the value of Dependent for the instance
func (instance *Win32_ProtocolBinding) GetPropertyDependent() (value Win32_SystemDriver, err error) {
	retValue, err := instance.GetProperty("Dependent")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_SystemDriver)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_SystemDriver is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_SystemDriver(valuetmp)

	return
}

// SetDevice sets the value of Device for the instance
func (instance *Win32_ProtocolBinding) SetPropertyDevice(value Win32_NetworkAdapter) (err error) {
	return instance.SetProperty("Device", (value))
}

// GetDevice gets the value of Device for the instance
func (instance *Win32_ProtocolBinding) GetPropertyDevice() (value Win32_NetworkAdapter, err error) {
	retValue, err := instance.GetProperty("Device")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_NetworkAdapter)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_NetworkAdapter is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_NetworkAdapter(valuetmp)

	return
}
