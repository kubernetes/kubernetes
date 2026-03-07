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

// Win32_Account struct
type Win32_Account struct {
	*CIM_LogicalElement

	//
	Domain string

	//
	LocalAccount bool

	//
	SID string

	//
	SIDType uint8
}

func NewWin32_AccountEx1(instance *cim.WmiInstance) (newInstance *Win32_Account, err error) {
	tmp, err := NewCIM_LogicalElementEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_Account{
		CIM_LogicalElement: tmp,
	}
	return
}

func NewWin32_AccountEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_Account, err error) {
	tmp, err := NewCIM_LogicalElementEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_Account{
		CIM_LogicalElement: tmp,
	}
	return
}

// SetDomain sets the value of Domain for the instance
func (instance *Win32_Account) SetPropertyDomain(value string) (err error) {
	return instance.SetProperty("Domain", (value))
}

// GetDomain gets the value of Domain for the instance
func (instance *Win32_Account) GetPropertyDomain() (value string, err error) {
	retValue, err := instance.GetProperty("Domain")
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

// SetLocalAccount sets the value of LocalAccount for the instance
func (instance *Win32_Account) SetPropertyLocalAccount(value bool) (err error) {
	return instance.SetProperty("LocalAccount", (value))
}

// GetLocalAccount gets the value of LocalAccount for the instance
func (instance *Win32_Account) GetPropertyLocalAccount() (value bool, err error) {
	retValue, err := instance.GetProperty("LocalAccount")
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

// SetSID sets the value of SID for the instance
func (instance *Win32_Account) SetPropertySID(value string) (err error) {
	return instance.SetProperty("SID", (value))
}

// GetSID gets the value of SID for the instance
func (instance *Win32_Account) GetPropertySID() (value string, err error) {
	retValue, err := instance.GetProperty("SID")
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

// SetSIDType sets the value of SIDType for the instance
func (instance *Win32_Account) SetPropertySIDType(value uint8) (err error) {
	return instance.SetProperty("SIDType", (value))
}

// GetSIDType gets the value of SIDType for the instance
func (instance *Win32_Account) GetPropertySIDType() (value uint8, err error) {
	retValue, err := instance.GetProperty("SIDType")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint8)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint8 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint8(valuetmp)

	return
}
