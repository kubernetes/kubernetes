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

// Win32_OSRecoveryConfiguration struct
type Win32_OSRecoveryConfiguration struct {
	*CIM_Setting

	//
	AutoReboot bool

	//
	DebugFilePath string

	//
	DebugInfoType uint32

	//
	ExpandedDebugFilePath string

	//
	ExpandedMiniDumpDirectory string

	//
	KernelDumpOnly bool

	//
	MiniDumpDirectory string

	//
	Name string

	//
	OverwriteExistingDebugFile bool

	//
	SendAdminAlert bool

	//
	WriteDebugInfo bool

	//
	WriteToSystemLog bool
}

func NewWin32_OSRecoveryConfigurationEx1(instance *cim.WmiInstance) (newInstance *Win32_OSRecoveryConfiguration, err error) {
	tmp, err := NewCIM_SettingEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_OSRecoveryConfiguration{
		CIM_Setting: tmp,
	}
	return
}

func NewWin32_OSRecoveryConfigurationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_OSRecoveryConfiguration, err error) {
	tmp, err := NewCIM_SettingEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_OSRecoveryConfiguration{
		CIM_Setting: tmp,
	}
	return
}

// SetAutoReboot sets the value of AutoReboot for the instance
func (instance *Win32_OSRecoveryConfiguration) SetPropertyAutoReboot(value bool) (err error) {
	return instance.SetProperty("AutoReboot", (value))
}

// GetAutoReboot gets the value of AutoReboot for the instance
func (instance *Win32_OSRecoveryConfiguration) GetPropertyAutoReboot() (value bool, err error) {
	retValue, err := instance.GetProperty("AutoReboot")
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

// SetDebugFilePath sets the value of DebugFilePath for the instance
func (instance *Win32_OSRecoveryConfiguration) SetPropertyDebugFilePath(value string) (err error) {
	return instance.SetProperty("DebugFilePath", (value))
}

// GetDebugFilePath gets the value of DebugFilePath for the instance
func (instance *Win32_OSRecoveryConfiguration) GetPropertyDebugFilePath() (value string, err error) {
	retValue, err := instance.GetProperty("DebugFilePath")
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

// SetDebugInfoType sets the value of DebugInfoType for the instance
func (instance *Win32_OSRecoveryConfiguration) SetPropertyDebugInfoType(value uint32) (err error) {
	return instance.SetProperty("DebugInfoType", (value))
}

// GetDebugInfoType gets the value of DebugInfoType for the instance
func (instance *Win32_OSRecoveryConfiguration) GetPropertyDebugInfoType() (value uint32, err error) {
	retValue, err := instance.GetProperty("DebugInfoType")
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

// SetExpandedDebugFilePath sets the value of ExpandedDebugFilePath for the instance
func (instance *Win32_OSRecoveryConfiguration) SetPropertyExpandedDebugFilePath(value string) (err error) {
	return instance.SetProperty("ExpandedDebugFilePath", (value))
}

// GetExpandedDebugFilePath gets the value of ExpandedDebugFilePath for the instance
func (instance *Win32_OSRecoveryConfiguration) GetPropertyExpandedDebugFilePath() (value string, err error) {
	retValue, err := instance.GetProperty("ExpandedDebugFilePath")
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

// SetExpandedMiniDumpDirectory sets the value of ExpandedMiniDumpDirectory for the instance
func (instance *Win32_OSRecoveryConfiguration) SetPropertyExpandedMiniDumpDirectory(value string) (err error) {
	return instance.SetProperty("ExpandedMiniDumpDirectory", (value))
}

// GetExpandedMiniDumpDirectory gets the value of ExpandedMiniDumpDirectory for the instance
func (instance *Win32_OSRecoveryConfiguration) GetPropertyExpandedMiniDumpDirectory() (value string, err error) {
	retValue, err := instance.GetProperty("ExpandedMiniDumpDirectory")
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

// SetKernelDumpOnly sets the value of KernelDumpOnly for the instance
func (instance *Win32_OSRecoveryConfiguration) SetPropertyKernelDumpOnly(value bool) (err error) {
	return instance.SetProperty("KernelDumpOnly", (value))
}

// GetKernelDumpOnly gets the value of KernelDumpOnly for the instance
func (instance *Win32_OSRecoveryConfiguration) GetPropertyKernelDumpOnly() (value bool, err error) {
	retValue, err := instance.GetProperty("KernelDumpOnly")
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

// SetMiniDumpDirectory sets the value of MiniDumpDirectory for the instance
func (instance *Win32_OSRecoveryConfiguration) SetPropertyMiniDumpDirectory(value string) (err error) {
	return instance.SetProperty("MiniDumpDirectory", (value))
}

// GetMiniDumpDirectory gets the value of MiniDumpDirectory for the instance
func (instance *Win32_OSRecoveryConfiguration) GetPropertyMiniDumpDirectory() (value string, err error) {
	retValue, err := instance.GetProperty("MiniDumpDirectory")
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

// SetName sets the value of Name for the instance
func (instance *Win32_OSRecoveryConfiguration) SetPropertyName(value string) (err error) {
	return instance.SetProperty("Name", (value))
}

// GetName gets the value of Name for the instance
func (instance *Win32_OSRecoveryConfiguration) GetPropertyName() (value string, err error) {
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

// SetOverwriteExistingDebugFile sets the value of OverwriteExistingDebugFile for the instance
func (instance *Win32_OSRecoveryConfiguration) SetPropertyOverwriteExistingDebugFile(value bool) (err error) {
	return instance.SetProperty("OverwriteExistingDebugFile", (value))
}

// GetOverwriteExistingDebugFile gets the value of OverwriteExistingDebugFile for the instance
func (instance *Win32_OSRecoveryConfiguration) GetPropertyOverwriteExistingDebugFile() (value bool, err error) {
	retValue, err := instance.GetProperty("OverwriteExistingDebugFile")
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

// SetSendAdminAlert sets the value of SendAdminAlert for the instance
func (instance *Win32_OSRecoveryConfiguration) SetPropertySendAdminAlert(value bool) (err error) {
	return instance.SetProperty("SendAdminAlert", (value))
}

// GetSendAdminAlert gets the value of SendAdminAlert for the instance
func (instance *Win32_OSRecoveryConfiguration) GetPropertySendAdminAlert() (value bool, err error) {
	retValue, err := instance.GetProperty("SendAdminAlert")
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

// SetWriteDebugInfo sets the value of WriteDebugInfo for the instance
func (instance *Win32_OSRecoveryConfiguration) SetPropertyWriteDebugInfo(value bool) (err error) {
	return instance.SetProperty("WriteDebugInfo", (value))
}

// GetWriteDebugInfo gets the value of WriteDebugInfo for the instance
func (instance *Win32_OSRecoveryConfiguration) GetPropertyWriteDebugInfo() (value bool, err error) {
	retValue, err := instance.GetProperty("WriteDebugInfo")
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

// SetWriteToSystemLog sets the value of WriteToSystemLog for the instance
func (instance *Win32_OSRecoveryConfiguration) SetPropertyWriteToSystemLog(value bool) (err error) {
	return instance.SetProperty("WriteToSystemLog", (value))
}

// GetWriteToSystemLog gets the value of WriteToSystemLog for the instance
func (instance *Win32_OSRecoveryConfiguration) GetPropertyWriteToSystemLog() (value bool, err error) {
	retValue, err := instance.GetProperty("WriteToSystemLog")
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
