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

// CIM_OSVersionCheck struct
type CIM_OSVersionCheck struct {
	*CIM_Check

	//
	MaximumVersion string

	//
	MinimumVersion string
}

func NewCIM_OSVersionCheckEx1(instance *cim.WmiInstance) (newInstance *CIM_OSVersionCheck, err error) {
	tmp, err := NewCIM_CheckEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_OSVersionCheck{
		CIM_Check: tmp,
	}
	return
}

func NewCIM_OSVersionCheckEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_OSVersionCheck, err error) {
	tmp, err := NewCIM_CheckEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_OSVersionCheck{
		CIM_Check: tmp,
	}
	return
}

// SetMaximumVersion sets the value of MaximumVersion for the instance
func (instance *CIM_OSVersionCheck) SetPropertyMaximumVersion(value string) (err error) {
	return instance.SetProperty("MaximumVersion", (value))
}

// GetMaximumVersion gets the value of MaximumVersion for the instance
func (instance *CIM_OSVersionCheck) GetPropertyMaximumVersion() (value string, err error) {
	retValue, err := instance.GetProperty("MaximumVersion")
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

// SetMinimumVersion sets the value of MinimumVersion for the instance
func (instance *CIM_OSVersionCheck) SetPropertyMinimumVersion(value string) (err error) {
	return instance.SetProperty("MinimumVersion", (value))
}

// GetMinimumVersion gets the value of MinimumVersion for the instance
func (instance *CIM_OSVersionCheck) GetPropertyMinimumVersion() (value string, err error) {
	retValue, err := instance.GetProperty("MinimumVersion")
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
