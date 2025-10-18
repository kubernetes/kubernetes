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

// Win32_RoamingProfileUserConfiguration struct
type Win32_RoamingProfileUserConfiguration struct {
	*cim.WmiInstance

	// An array of strings containing network directories to synchronize at when the user logs on to or off of a local computer.
	DirectoriesToSyncAtLogonLogoff []string

	// An array of strings containing directories to exclude from the roaming profile.
	ExcludedProfileDirs []string

	// Indicates if the settings configured through this WMI class are taking affect.
	IsConfiguredByWMI bool
}

func NewWin32_RoamingProfileUserConfigurationEx1(instance *cim.WmiInstance) (newInstance *Win32_RoamingProfileUserConfiguration, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_RoamingProfileUserConfiguration{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_RoamingProfileUserConfigurationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_RoamingProfileUserConfiguration, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_RoamingProfileUserConfiguration{
		WmiInstance: tmp,
	}
	return
}

// SetDirectoriesToSyncAtLogonLogoff sets the value of DirectoriesToSyncAtLogonLogoff for the instance
func (instance *Win32_RoamingProfileUserConfiguration) SetPropertyDirectoriesToSyncAtLogonLogoff(value []string) (err error) {
	return instance.SetProperty("DirectoriesToSyncAtLogonLogoff", (value))
}

// GetDirectoriesToSyncAtLogonLogoff gets the value of DirectoriesToSyncAtLogonLogoff for the instance
func (instance *Win32_RoamingProfileUserConfiguration) GetPropertyDirectoriesToSyncAtLogonLogoff() (value []string, err error) {
	retValue, err := instance.GetProperty("DirectoriesToSyncAtLogonLogoff")
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

// SetExcludedProfileDirs sets the value of ExcludedProfileDirs for the instance
func (instance *Win32_RoamingProfileUserConfiguration) SetPropertyExcludedProfileDirs(value []string) (err error) {
	return instance.SetProperty("ExcludedProfileDirs", (value))
}

// GetExcludedProfileDirs gets the value of ExcludedProfileDirs for the instance
func (instance *Win32_RoamingProfileUserConfiguration) GetPropertyExcludedProfileDirs() (value []string, err error) {
	retValue, err := instance.GetProperty("ExcludedProfileDirs")
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
func (instance *Win32_RoamingProfileUserConfiguration) SetPropertyIsConfiguredByWMI(value bool) (err error) {
	return instance.SetProperty("IsConfiguredByWMI", (value))
}

// GetIsConfiguredByWMI gets the value of IsConfiguredByWMI for the instance
func (instance *Win32_RoamingProfileUserConfiguration) GetPropertyIsConfiguredByWMI() (value bool, err error) {
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
