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

// CIM_FromDirectorySpecification struct
type CIM_FromDirectorySpecification struct {
	*cim.WmiInstance

	//
	FileName CIM_FileAction

	//
	SourceDirectory CIM_DirectorySpecification
}

func NewCIM_FromDirectorySpecificationEx1(instance *cim.WmiInstance) (newInstance *CIM_FromDirectorySpecification, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &CIM_FromDirectorySpecification{
		WmiInstance: tmp,
	}
	return
}

func NewCIM_FromDirectorySpecificationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_FromDirectorySpecification, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_FromDirectorySpecification{
		WmiInstance: tmp,
	}
	return
}

// SetFileName sets the value of FileName for the instance
func (instance *CIM_FromDirectorySpecification) SetPropertyFileName(value CIM_FileAction) (err error) {
	return instance.SetProperty("FileName", (value))
}

// GetFileName gets the value of FileName for the instance
func (instance *CIM_FromDirectorySpecification) GetPropertyFileName() (value CIM_FileAction, err error) {
	retValue, err := instance.GetProperty("FileName")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_FileAction)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_FileAction is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_FileAction(valuetmp)

	return
}

// SetSourceDirectory sets the value of SourceDirectory for the instance
func (instance *CIM_FromDirectorySpecification) SetPropertySourceDirectory(value CIM_DirectorySpecification) (err error) {
	return instance.SetProperty("SourceDirectory", (value))
}

// GetSourceDirectory gets the value of SourceDirectory for the instance
func (instance *CIM_FromDirectorySpecification) GetPropertySourceDirectory() (value CIM_DirectorySpecification, err error) {
	retValue, err := instance.GetProperty("SourceDirectory")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_DirectorySpecification)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_DirectorySpecification is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_DirectorySpecification(valuetmp)

	return
}
