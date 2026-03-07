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

// Win32_TokenGroups struct
type Win32_TokenGroups struct {
	*cim.WmiInstance

	//
	GroupCount uint32

	//
	Groups []Win32_SIDandAttributes
}

func NewWin32_TokenGroupsEx1(instance *cim.WmiInstance) (newInstance *Win32_TokenGroups, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_TokenGroups{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_TokenGroupsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_TokenGroups, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_TokenGroups{
		WmiInstance: tmp,
	}
	return
}

// SetGroupCount sets the value of GroupCount for the instance
func (instance *Win32_TokenGroups) SetPropertyGroupCount(value uint32) (err error) {
	return instance.SetProperty("GroupCount", (value))
}

// GetGroupCount gets the value of GroupCount for the instance
func (instance *Win32_TokenGroups) GetPropertyGroupCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("GroupCount")
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

// SetGroups sets the value of Groups for the instance
func (instance *Win32_TokenGroups) SetPropertyGroups(value []Win32_SIDandAttributes) (err error) {
	return instance.SetProperty("Groups", (value))
}

// GetGroups gets the value of Groups for the instance
func (instance *Win32_TokenGroups) GetPropertyGroups() (value []Win32_SIDandAttributes, err error) {
	retValue, err := instance.GetProperty("Groups")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(Win32_SIDandAttributes)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " Win32_SIDandAttributes is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, Win32_SIDandAttributes(valuetmp))
	}

	return
}
