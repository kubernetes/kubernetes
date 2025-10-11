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

// CIM_RemoveDirectoryAction struct
type CIM_RemoveDirectoryAction struct {
	*CIM_DirectoryAction

	//
	MustBeEmpty bool
}

func NewCIM_RemoveDirectoryActionEx1(instance *cim.WmiInstance) (newInstance *CIM_RemoveDirectoryAction, err error) {
	tmp, err := NewCIM_DirectoryActionEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_RemoveDirectoryAction{
		CIM_DirectoryAction: tmp,
	}
	return
}

func NewCIM_RemoveDirectoryActionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_RemoveDirectoryAction, err error) {
	tmp, err := NewCIM_DirectoryActionEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_RemoveDirectoryAction{
		CIM_DirectoryAction: tmp,
	}
	return
}

// SetMustBeEmpty sets the value of MustBeEmpty for the instance
func (instance *CIM_RemoveDirectoryAction) SetPropertyMustBeEmpty(value bool) (err error) {
	return instance.SetProperty("MustBeEmpty", (value))
}

// GetMustBeEmpty gets the value of MustBeEmpty for the instance
func (instance *CIM_RemoveDirectoryAction) GetPropertyMustBeEmpty() (value bool, err error) {
	retValue, err := instance.GetProperty("MustBeEmpty")
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
