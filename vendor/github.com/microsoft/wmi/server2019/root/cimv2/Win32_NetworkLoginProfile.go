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

// Win32_NetworkLoginProfile struct
type Win32_NetworkLoginProfile struct {
	*CIM_Setting

	//
	AccountExpires string

	//
	AuthorizationFlags uint32

	//
	BadPasswordCount uint32

	//
	CodePage uint32

	//
	Comment string

	//
	CountryCode uint32

	//
	Flags uint32

	//
	FullName string

	//
	HomeDirectory string

	//
	HomeDirectoryDrive string

	//
	LastLogoff string

	//
	LastLogon string

	//
	LogonHours string

	//
	LogonServer string

	//
	MaximumStorage uint64

	//
	Name string

	//
	NumberOfLogons uint32

	//
	Parameters string

	//
	PasswordAge string

	//
	PasswordExpires string

	//
	PrimaryGroupId uint32

	//
	Privileges uint32

	//
	Profile string

	//
	ScriptPath string

	//
	UnitsPerWeek uint32

	//
	UserComment string

	//
	UserId uint32

	//
	UserType string

	//
	Workstations string
}

func NewWin32_NetworkLoginProfileEx1(instance *cim.WmiInstance) (newInstance *Win32_NetworkLoginProfile, err error) {
	tmp, err := NewCIM_SettingEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_NetworkLoginProfile{
		CIM_Setting: tmp,
	}
	return
}

func NewWin32_NetworkLoginProfileEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_NetworkLoginProfile, err error) {
	tmp, err := NewCIM_SettingEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_NetworkLoginProfile{
		CIM_Setting: tmp,
	}
	return
}

// SetAccountExpires sets the value of AccountExpires for the instance
func (instance *Win32_NetworkLoginProfile) SetPropertyAccountExpires(value string) (err error) {
	return instance.SetProperty("AccountExpires", (value))
}

// GetAccountExpires gets the value of AccountExpires for the instance
func (instance *Win32_NetworkLoginProfile) GetPropertyAccountExpires() (value string, err error) {
	retValue, err := instance.GetProperty("AccountExpires")
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

// SetAuthorizationFlags sets the value of AuthorizationFlags for the instance
func (instance *Win32_NetworkLoginProfile) SetPropertyAuthorizationFlags(value uint32) (err error) {
	return instance.SetProperty("AuthorizationFlags", (value))
}

// GetAuthorizationFlags gets the value of AuthorizationFlags for the instance
func (instance *Win32_NetworkLoginProfile) GetPropertyAuthorizationFlags() (value uint32, err error) {
	retValue, err := instance.GetProperty("AuthorizationFlags")
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

// SetBadPasswordCount sets the value of BadPasswordCount for the instance
func (instance *Win32_NetworkLoginProfile) SetPropertyBadPasswordCount(value uint32) (err error) {
	return instance.SetProperty("BadPasswordCount", (value))
}

// GetBadPasswordCount gets the value of BadPasswordCount for the instance
func (instance *Win32_NetworkLoginProfile) GetPropertyBadPasswordCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("BadPasswordCount")
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

// SetCodePage sets the value of CodePage for the instance
func (instance *Win32_NetworkLoginProfile) SetPropertyCodePage(value uint32) (err error) {
	return instance.SetProperty("CodePage", (value))
}

// GetCodePage gets the value of CodePage for the instance
func (instance *Win32_NetworkLoginProfile) GetPropertyCodePage() (value uint32, err error) {
	retValue, err := instance.GetProperty("CodePage")
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

// SetComment sets the value of Comment for the instance
func (instance *Win32_NetworkLoginProfile) SetPropertyComment(value string) (err error) {
	return instance.SetProperty("Comment", (value))
}

// GetComment gets the value of Comment for the instance
func (instance *Win32_NetworkLoginProfile) GetPropertyComment() (value string, err error) {
	retValue, err := instance.GetProperty("Comment")
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

// SetCountryCode sets the value of CountryCode for the instance
func (instance *Win32_NetworkLoginProfile) SetPropertyCountryCode(value uint32) (err error) {
	return instance.SetProperty("CountryCode", (value))
}

// GetCountryCode gets the value of CountryCode for the instance
func (instance *Win32_NetworkLoginProfile) GetPropertyCountryCode() (value uint32, err error) {
	retValue, err := instance.GetProperty("CountryCode")
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

// SetFlags sets the value of Flags for the instance
func (instance *Win32_NetworkLoginProfile) SetPropertyFlags(value uint32) (err error) {
	return instance.SetProperty("Flags", (value))
}

// GetFlags gets the value of Flags for the instance
func (instance *Win32_NetworkLoginProfile) GetPropertyFlags() (value uint32, err error) {
	retValue, err := instance.GetProperty("Flags")
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

// SetFullName sets the value of FullName for the instance
func (instance *Win32_NetworkLoginProfile) SetPropertyFullName(value string) (err error) {
	return instance.SetProperty("FullName", (value))
}

// GetFullName gets the value of FullName for the instance
func (instance *Win32_NetworkLoginProfile) GetPropertyFullName() (value string, err error) {
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

// SetHomeDirectory sets the value of HomeDirectory for the instance
func (instance *Win32_NetworkLoginProfile) SetPropertyHomeDirectory(value string) (err error) {
	return instance.SetProperty("HomeDirectory", (value))
}

// GetHomeDirectory gets the value of HomeDirectory for the instance
func (instance *Win32_NetworkLoginProfile) GetPropertyHomeDirectory() (value string, err error) {
	retValue, err := instance.GetProperty("HomeDirectory")
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

// SetHomeDirectoryDrive sets the value of HomeDirectoryDrive for the instance
func (instance *Win32_NetworkLoginProfile) SetPropertyHomeDirectoryDrive(value string) (err error) {
	return instance.SetProperty("HomeDirectoryDrive", (value))
}

// GetHomeDirectoryDrive gets the value of HomeDirectoryDrive for the instance
func (instance *Win32_NetworkLoginProfile) GetPropertyHomeDirectoryDrive() (value string, err error) {
	retValue, err := instance.GetProperty("HomeDirectoryDrive")
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

// SetLastLogoff sets the value of LastLogoff for the instance
func (instance *Win32_NetworkLoginProfile) SetPropertyLastLogoff(value string) (err error) {
	return instance.SetProperty("LastLogoff", (value))
}

// GetLastLogoff gets the value of LastLogoff for the instance
func (instance *Win32_NetworkLoginProfile) GetPropertyLastLogoff() (value string, err error) {
	retValue, err := instance.GetProperty("LastLogoff")
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

// SetLastLogon sets the value of LastLogon for the instance
func (instance *Win32_NetworkLoginProfile) SetPropertyLastLogon(value string) (err error) {
	return instance.SetProperty("LastLogon", (value))
}

// GetLastLogon gets the value of LastLogon for the instance
func (instance *Win32_NetworkLoginProfile) GetPropertyLastLogon() (value string, err error) {
	retValue, err := instance.GetProperty("LastLogon")
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

// SetLogonHours sets the value of LogonHours for the instance
func (instance *Win32_NetworkLoginProfile) SetPropertyLogonHours(value string) (err error) {
	return instance.SetProperty("LogonHours", (value))
}

// GetLogonHours gets the value of LogonHours for the instance
func (instance *Win32_NetworkLoginProfile) GetPropertyLogonHours() (value string, err error) {
	retValue, err := instance.GetProperty("LogonHours")
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

// SetLogonServer sets the value of LogonServer for the instance
func (instance *Win32_NetworkLoginProfile) SetPropertyLogonServer(value string) (err error) {
	return instance.SetProperty("LogonServer", (value))
}

// GetLogonServer gets the value of LogonServer for the instance
func (instance *Win32_NetworkLoginProfile) GetPropertyLogonServer() (value string, err error) {
	retValue, err := instance.GetProperty("LogonServer")
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

// SetMaximumStorage sets the value of MaximumStorage for the instance
func (instance *Win32_NetworkLoginProfile) SetPropertyMaximumStorage(value uint64) (err error) {
	return instance.SetProperty("MaximumStorage", (value))
}

// GetMaximumStorage gets the value of MaximumStorage for the instance
func (instance *Win32_NetworkLoginProfile) GetPropertyMaximumStorage() (value uint64, err error) {
	retValue, err := instance.GetProperty("MaximumStorage")
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

// SetName sets the value of Name for the instance
func (instance *Win32_NetworkLoginProfile) SetPropertyName(value string) (err error) {
	return instance.SetProperty("Name", (value))
}

// GetName gets the value of Name for the instance
func (instance *Win32_NetworkLoginProfile) GetPropertyName() (value string, err error) {
	retValue, err := instance.GetProperty("Name")
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

// SetNumberOfLogons sets the value of NumberOfLogons for the instance
func (instance *Win32_NetworkLoginProfile) SetPropertyNumberOfLogons(value uint32) (err error) {
	return instance.SetProperty("NumberOfLogons", (value))
}

// GetNumberOfLogons gets the value of NumberOfLogons for the instance
func (instance *Win32_NetworkLoginProfile) GetPropertyNumberOfLogons() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfLogons")
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

// SetParameters sets the value of Parameters for the instance
func (instance *Win32_NetworkLoginProfile) SetPropertyParameters(value string) (err error) {
	return instance.SetProperty("Parameters", (value))
}

// GetParameters gets the value of Parameters for the instance
func (instance *Win32_NetworkLoginProfile) GetPropertyParameters() (value string, err error) {
	retValue, err := instance.GetProperty("Parameters")
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

// SetPasswordAge sets the value of PasswordAge for the instance
func (instance *Win32_NetworkLoginProfile) SetPropertyPasswordAge(value string) (err error) {
	return instance.SetProperty("PasswordAge", (value))
}

// GetPasswordAge gets the value of PasswordAge for the instance
func (instance *Win32_NetworkLoginProfile) GetPropertyPasswordAge() (value string, err error) {
	retValue, err := instance.GetProperty("PasswordAge")
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

// SetPasswordExpires sets the value of PasswordExpires for the instance
func (instance *Win32_NetworkLoginProfile) SetPropertyPasswordExpires(value string) (err error) {
	return instance.SetProperty("PasswordExpires", (value))
}

// GetPasswordExpires gets the value of PasswordExpires for the instance
func (instance *Win32_NetworkLoginProfile) GetPropertyPasswordExpires() (value string, err error) {
	retValue, err := instance.GetProperty("PasswordExpires")
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

// SetPrimaryGroupId sets the value of PrimaryGroupId for the instance
func (instance *Win32_NetworkLoginProfile) SetPropertyPrimaryGroupId(value uint32) (err error) {
	return instance.SetProperty("PrimaryGroupId", (value))
}

// GetPrimaryGroupId gets the value of PrimaryGroupId for the instance
func (instance *Win32_NetworkLoginProfile) GetPropertyPrimaryGroupId() (value uint32, err error) {
	retValue, err := instance.GetProperty("PrimaryGroupId")
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

// SetPrivileges sets the value of Privileges for the instance
func (instance *Win32_NetworkLoginProfile) SetPropertyPrivileges(value uint32) (err error) {
	return instance.SetProperty("Privileges", (value))
}

// GetPrivileges gets the value of Privileges for the instance
func (instance *Win32_NetworkLoginProfile) GetPropertyPrivileges() (value uint32, err error) {
	retValue, err := instance.GetProperty("Privileges")
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

// SetProfile sets the value of Profile for the instance
func (instance *Win32_NetworkLoginProfile) SetPropertyProfile(value string) (err error) {
	return instance.SetProperty("Profile", (value))
}

// GetProfile gets the value of Profile for the instance
func (instance *Win32_NetworkLoginProfile) GetPropertyProfile() (value string, err error) {
	retValue, err := instance.GetProperty("Profile")
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

// SetScriptPath sets the value of ScriptPath for the instance
func (instance *Win32_NetworkLoginProfile) SetPropertyScriptPath(value string) (err error) {
	return instance.SetProperty("ScriptPath", (value))
}

// GetScriptPath gets the value of ScriptPath for the instance
func (instance *Win32_NetworkLoginProfile) GetPropertyScriptPath() (value string, err error) {
	retValue, err := instance.GetProperty("ScriptPath")
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

// SetUnitsPerWeek sets the value of UnitsPerWeek for the instance
func (instance *Win32_NetworkLoginProfile) SetPropertyUnitsPerWeek(value uint32) (err error) {
	return instance.SetProperty("UnitsPerWeek", (value))
}

// GetUnitsPerWeek gets the value of UnitsPerWeek for the instance
func (instance *Win32_NetworkLoginProfile) GetPropertyUnitsPerWeek() (value uint32, err error) {
	retValue, err := instance.GetProperty("UnitsPerWeek")
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

// SetUserComment sets the value of UserComment for the instance
func (instance *Win32_NetworkLoginProfile) SetPropertyUserComment(value string) (err error) {
	return instance.SetProperty("UserComment", (value))
}

// GetUserComment gets the value of UserComment for the instance
func (instance *Win32_NetworkLoginProfile) GetPropertyUserComment() (value string, err error) {
	retValue, err := instance.GetProperty("UserComment")
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

// SetUserId sets the value of UserId for the instance
func (instance *Win32_NetworkLoginProfile) SetPropertyUserId(value uint32) (err error) {
	return instance.SetProperty("UserId", (value))
}

// GetUserId gets the value of UserId for the instance
func (instance *Win32_NetworkLoginProfile) GetPropertyUserId() (value uint32, err error) {
	retValue, err := instance.GetProperty("UserId")
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

// SetUserType sets the value of UserType for the instance
func (instance *Win32_NetworkLoginProfile) SetPropertyUserType(value string) (err error) {
	return instance.SetProperty("UserType", (value))
}

// GetUserType gets the value of UserType for the instance
func (instance *Win32_NetworkLoginProfile) GetPropertyUserType() (value string, err error) {
	retValue, err := instance.GetProperty("UserType")
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

// SetWorkstations sets the value of Workstations for the instance
func (instance *Win32_NetworkLoginProfile) SetPropertyWorkstations(value string) (err error) {
	return instance.SetProperty("Workstations", (value))
}

// GetWorkstations gets the value of Workstations for the instance
func (instance *Win32_NetworkLoginProfile) GetPropertyWorkstations() (value string, err error) {
	retValue, err := instance.GetProperty("Workstations")
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
