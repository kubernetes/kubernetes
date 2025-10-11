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

// Win32_ProgIDSpecification struct
type Win32_ProgIDSpecification struct {
	*CIM_Check

	//
	Parent string

	//
	ProgID string
}

func NewWin32_ProgIDSpecificationEx1(instance *cim.WmiInstance) (newInstance *Win32_ProgIDSpecification, err error) {
	tmp, err := NewCIM_CheckEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_ProgIDSpecification{
		CIM_Check: tmp,
	}
	return
}

func NewWin32_ProgIDSpecificationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ProgIDSpecification, err error) {
	tmp, err := NewCIM_CheckEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ProgIDSpecification{
		CIM_Check: tmp,
	}
	return
}

// SetParent sets the value of Parent for the instance
func (instance *Win32_ProgIDSpecification) SetPropertyParent(value string) (err error) {
	return instance.SetProperty("Parent", (value))
}

// GetParent gets the value of Parent for the instance
func (instance *Win32_ProgIDSpecification) GetPropertyParent() (value string, err error) {
	retValue, err := instance.GetProperty("Parent")
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

// SetProgID sets the value of ProgID for the instance
func (instance *Win32_ProgIDSpecification) SetPropertyProgID(value string) (err error) {
	return instance.SetProperty("ProgID", (value))
}

// GetProgID gets the value of ProgID for the instance
func (instance *Win32_ProgIDSpecification) GetPropertyProgID() (value string, err error) {
	retValue, err := instance.GetProperty("ProgID")
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
