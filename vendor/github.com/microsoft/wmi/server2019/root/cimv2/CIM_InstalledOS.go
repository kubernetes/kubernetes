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

// CIM_InstalledOS struct
type CIM_InstalledOS struct {
	*CIM_SystemComponent

	//
	PrimaryOS bool
}

func NewCIM_InstalledOSEx1(instance *cim.WmiInstance) (newInstance *CIM_InstalledOS, err error) {
	tmp, err := NewCIM_SystemComponentEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_InstalledOS{
		CIM_SystemComponent: tmp,
	}
	return
}

func NewCIM_InstalledOSEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_InstalledOS, err error) {
	tmp, err := NewCIM_SystemComponentEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_InstalledOS{
		CIM_SystemComponent: tmp,
	}
	return
}

// SetPrimaryOS sets the value of PrimaryOS for the instance
func (instance *CIM_InstalledOS) SetPropertyPrimaryOS(value bool) (err error) {
	return instance.SetProperty("PrimaryOS", (value))
}

// GetPrimaryOS gets the value of PrimaryOS for the instance
func (instance *CIM_InstalledOS) GetPropertyPrimaryOS() (value bool, err error) {
	retValue, err := instance.GetProperty("PrimaryOS")
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
