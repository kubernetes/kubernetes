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

// Win32_UserAccount struct
type Win32_UserAccount struct {
	*Win32_Account

	//
	AccountType uint32

	//
	Disabled bool

	//
	FullName string

	//
	Lockout bool

	//
	PasswordChangeable bool

	//
	PasswordExpires bool

	//
	PasswordRequired bool
}

func NewWin32_UserAccountEx1(instance *cim.WmiInstance) (newInstance *Win32_UserAccount, err error) {
	tmp, err := NewWin32_AccountEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_UserAccount{
		Win32_Account: tmp,
	}
	return
}

func NewWin32_UserAccountEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_UserAccount, err error) {
	tmp, err := NewWin32_AccountEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_UserAccount{
		Win32_Account: tmp,
	}
	return
}

// SetAccountType sets the value of AccountType for the instance
func (instance *Win32_UserAccount) SetPropertyAccountType(value uint32) (err error) {
	return instance.SetProperty("AccountType", (value))
}

// GetAccountType gets the value of AccountType for the instance
func (instance *Win32_UserAccount) GetPropertyAccountType() (value uint32, err error) {
	retValue, err := instance.GetProperty("AccountType")
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

// SetDisabled sets the value of Disabled for the instance
func (instance *Win32_UserAccount) SetPropertyDisabled(value bool) (err error) {
	return instance.SetProperty("Disabled", (value))
}

// GetDisabled gets the value of Disabled for the instance
func (instance *Win32_UserAccount) GetPropertyDisabled() (value bool, err error) {
	retValue, err := instance.GetProperty("Disabled")
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

// SetFullName sets the value of FullName for the instance
func (instance *Win32_UserAccount) SetPropertyFullName(value string) (err error) {
	return instance.SetProperty("FullName", (value))
}

// GetFullName gets the value of FullName for the instance
func (instance *Win32_UserAccount) GetPropertyFullName() (value string, err error) {
	retValue, err := instance.GetProperty("FullName")
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

// SetLockout sets the value of Lockout for the instance
func (instance *Win32_UserAccount) SetPropertyLockout(value bool) (err error) {
	return instance.SetProperty("Lockout", (value))
}

// GetLockout gets the value of Lockout for the instance
func (instance *Win32_UserAccount) GetPropertyLockout() (value bool, err error) {
	retValue, err := instance.GetProperty("Lockout")
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

// SetPasswordChangeable sets the value of PasswordChangeable for the instance
func (instance *Win32_UserAccount) SetPropertyPasswordChangeable(value bool) (err error) {
	return instance.SetProperty("PasswordChangeable", (value))
}

// GetPasswordChangeable gets the value of PasswordChangeable for the instance
func (instance *Win32_UserAccount) GetPropertyPasswordChangeable() (value bool, err error) {
	retValue, err := instance.GetProperty("PasswordChangeable")
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

// SetPasswordExpires sets the value of PasswordExpires for the instance
func (instance *Win32_UserAccount) SetPropertyPasswordExpires(value bool) (err error) {
	return instance.SetProperty("PasswordExpires", (value))
}

// GetPasswordExpires gets the value of PasswordExpires for the instance
func (instance *Win32_UserAccount) GetPropertyPasswordExpires() (value bool, err error) {
	retValue, err := instance.GetProperty("PasswordExpires")
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

// SetPasswordRequired sets the value of PasswordRequired for the instance
func (instance *Win32_UserAccount) SetPropertyPasswordRequired(value bool) (err error) {
	return instance.SetProperty("PasswordRequired", (value))
}

// GetPasswordRequired gets the value of PasswordRequired for the instance
func (instance *Win32_UserAccount) GetPropertyPasswordRequired() (value bool, err error) {
	retValue, err := instance.GetProperty("PasswordRequired")
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

// <param name="Name" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_UserAccount) Rename( /* IN */ Name string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Rename", Name)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}
