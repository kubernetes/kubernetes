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

// Win32_CIMLogicalDeviceCIMDataFile struct
type Win32_CIMLogicalDeviceCIMDataFile struct {
	*CIM_Dependency

	//
	Purpose uint16

	//
	PurposeDescription string
}

func NewWin32_CIMLogicalDeviceCIMDataFileEx1(instance *cim.WmiInstance) (newInstance *Win32_CIMLogicalDeviceCIMDataFile, err error) {
	tmp, err := NewCIM_DependencyEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_CIMLogicalDeviceCIMDataFile{
		CIM_Dependency: tmp,
	}
	return
}

func NewWin32_CIMLogicalDeviceCIMDataFileEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_CIMLogicalDeviceCIMDataFile, err error) {
	tmp, err := NewCIM_DependencyEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_CIMLogicalDeviceCIMDataFile{
		CIM_Dependency: tmp,
	}
	return
}

// SetPurpose sets the value of Purpose for the instance
func (instance *Win32_CIMLogicalDeviceCIMDataFile) SetPropertyPurpose(value uint16) (err error) {
	return instance.SetProperty("Purpose", (value))
}

// GetPurpose gets the value of Purpose for the instance
func (instance *Win32_CIMLogicalDeviceCIMDataFile) GetPropertyPurpose() (value uint16, err error) {
	retValue, err := instance.GetProperty("Purpose")
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

// SetPurposeDescription sets the value of PurposeDescription for the instance
func (instance *Win32_CIMLogicalDeviceCIMDataFile) SetPropertyPurposeDescription(value string) (err error) {
	return instance.SetProperty("PurposeDescription", (value))
}

// GetPurposeDescription gets the value of PurposeDescription for the instance
func (instance *Win32_CIMLogicalDeviceCIMDataFile) GetPropertyPurposeDescription() (value string, err error) {
	retValue, err := instance.GetProperty("PurposeDescription")
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
