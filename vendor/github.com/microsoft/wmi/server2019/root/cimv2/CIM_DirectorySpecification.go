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

// CIM_DirectorySpecification struct
type CIM_DirectorySpecification struct {
	*CIM_Check

	//
	DirectoryPath string

	//
	DirectoryType uint16
}

func NewCIM_DirectorySpecificationEx1(instance *cim.WmiInstance) (newInstance *CIM_DirectorySpecification, err error) {
	tmp, err := NewCIM_CheckEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_DirectorySpecification{
		CIM_Check: tmp,
	}
	return
}

func NewCIM_DirectorySpecificationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_DirectorySpecification, err error) {
	tmp, err := NewCIM_CheckEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_DirectorySpecification{
		CIM_Check: tmp,
	}
	return
}

// SetDirectoryPath sets the value of DirectoryPath for the instance
func (instance *CIM_DirectorySpecification) SetPropertyDirectoryPath(value string) (err error) {
	return instance.SetProperty("DirectoryPath", (value))
}

// GetDirectoryPath gets the value of DirectoryPath for the instance
func (instance *CIM_DirectorySpecification) GetPropertyDirectoryPath() (value string, err error) {
	retValue, err := instance.GetProperty("DirectoryPath")
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

// SetDirectoryType sets the value of DirectoryType for the instance
func (instance *CIM_DirectorySpecification) SetPropertyDirectoryType(value uint16) (err error) {
	return instance.SetProperty("DirectoryType", (value))
}

// GetDirectoryType gets the value of DirectoryType for the instance
func (instance *CIM_DirectorySpecification) GetPropertyDirectoryType() (value uint16, err error) {
	retValue, err := instance.GetProperty("DirectoryType")
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
