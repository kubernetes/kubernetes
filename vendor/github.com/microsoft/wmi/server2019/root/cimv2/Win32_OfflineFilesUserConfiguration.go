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

// Win32_OfflineFilesUserConfiguration struct
type Win32_OfflineFilesUserConfiguration struct {
	*cim.WmiInstance

	//
	AssignedOfflineFiles []string

	//
	IsConfiguredByWMI bool

	//
	MakeAvailableOfflineButtonRemoved bool

	//
	WorkOfflineButtonRemoved bool
}

func NewWin32_OfflineFilesUserConfigurationEx1(instance *cim.WmiInstance) (newInstance *Win32_OfflineFilesUserConfiguration, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_OfflineFilesUserConfiguration{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_OfflineFilesUserConfigurationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_OfflineFilesUserConfiguration, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_OfflineFilesUserConfiguration{
		WmiInstance: tmp,
	}
	return
}

// SetAssignedOfflineFiles sets the value of AssignedOfflineFiles for the instance
func (instance *Win32_OfflineFilesUserConfiguration) SetPropertyAssignedOfflineFiles(value []string) (err error) {
	return instance.SetProperty("AssignedOfflineFiles", (value))
}

// GetAssignedOfflineFiles gets the value of AssignedOfflineFiles for the instance
func (instance *Win32_OfflineFilesUserConfiguration) GetPropertyAssignedOfflineFiles() (value []string, err error) {
	retValue, err := instance.GetProperty("AssignedOfflineFiles")
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

// SetIsConfiguredByWMI sets the value of IsConfiguredByWMI for the instance
func (instance *Win32_OfflineFilesUserConfiguration) SetPropertyIsConfiguredByWMI(value bool) (err error) {
	return instance.SetProperty("IsConfiguredByWMI", (value))
}

// GetIsConfiguredByWMI gets the value of IsConfiguredByWMI for the instance
func (instance *Win32_OfflineFilesUserConfiguration) GetPropertyIsConfiguredByWMI() (value bool, err error) {
	retValue, err := instance.GetProperty("IsConfiguredByWMI")
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

// SetMakeAvailableOfflineButtonRemoved sets the value of MakeAvailableOfflineButtonRemoved for the instance
func (instance *Win32_OfflineFilesUserConfiguration) SetPropertyMakeAvailableOfflineButtonRemoved(value bool) (err error) {
	return instance.SetProperty("MakeAvailableOfflineButtonRemoved", (value))
}

// GetMakeAvailableOfflineButtonRemoved gets the value of MakeAvailableOfflineButtonRemoved for the instance
func (instance *Win32_OfflineFilesUserConfiguration) GetPropertyMakeAvailableOfflineButtonRemoved() (value bool, err error) {
	retValue, err := instance.GetProperty("MakeAvailableOfflineButtonRemoved")
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

// SetWorkOfflineButtonRemoved sets the value of WorkOfflineButtonRemoved for the instance
func (instance *Win32_OfflineFilesUserConfiguration) SetPropertyWorkOfflineButtonRemoved(value bool) (err error) {
	return instance.SetProperty("WorkOfflineButtonRemoved", (value))
}

// GetWorkOfflineButtonRemoved gets the value of WorkOfflineButtonRemoved for the instance
func (instance *Win32_OfflineFilesUserConfiguration) GetPropertyWorkOfflineButtonRemoved() (value bool, err error) {
	retValue, err := instance.GetProperty("WorkOfflineButtonRemoved")
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
