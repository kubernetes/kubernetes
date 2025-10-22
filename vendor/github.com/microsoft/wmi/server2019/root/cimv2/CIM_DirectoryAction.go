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

// CIM_DirectoryAction struct
type CIM_DirectoryAction struct {
	*CIM_Action

	//
	DirectoryName string
}

func NewCIM_DirectoryActionEx1(instance *cim.WmiInstance) (newInstance *CIM_DirectoryAction, err error) {
	tmp, err := NewCIM_ActionEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_DirectoryAction{
		CIM_Action: tmp,
	}
	return
}

func NewCIM_DirectoryActionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_DirectoryAction, err error) {
	tmp, err := NewCIM_ActionEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_DirectoryAction{
		CIM_Action: tmp,
	}
	return
}

// SetDirectoryName sets the value of DirectoryName for the instance
func (instance *CIM_DirectoryAction) SetPropertyDirectoryName(value string) (err error) {
	return instance.SetProperty("DirectoryName", (value))
}

// GetDirectoryName gets the value of DirectoryName for the instance
func (instance *CIM_DirectoryAction) GetPropertyDirectoryName() (value string, err error) {
	retValue, err := instance.GetProperty("DirectoryName")
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
