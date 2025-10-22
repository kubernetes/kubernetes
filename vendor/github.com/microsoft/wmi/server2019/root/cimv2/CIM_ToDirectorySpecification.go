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

// CIM_ToDirectorySpecification struct
type CIM_ToDirectorySpecification struct {
	*cim.WmiInstance

	//
	DestinationDirectory CIM_DirectorySpecification

	//
	FileName CIM_CopyFileAction
}

func NewCIM_ToDirectorySpecificationEx1(instance *cim.WmiInstance) (newInstance *CIM_ToDirectorySpecification, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &CIM_ToDirectorySpecification{
		WmiInstance: tmp,
	}
	return
}

func NewCIM_ToDirectorySpecificationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_ToDirectorySpecification, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_ToDirectorySpecification{
		WmiInstance: tmp,
	}
	return
}

// SetDestinationDirectory sets the value of DestinationDirectory for the instance
func (instance *CIM_ToDirectorySpecification) SetPropertyDestinationDirectory(value CIM_DirectorySpecification) (err error) {
	return instance.SetProperty("DestinationDirectory", (value))
}

// GetDestinationDirectory gets the value of DestinationDirectory for the instance
func (instance *CIM_ToDirectorySpecification) GetPropertyDestinationDirectory() (value CIM_DirectorySpecification, err error) {
	retValue, err := instance.GetProperty("DestinationDirectory")
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

// SetFileName sets the value of FileName for the instance
func (instance *CIM_ToDirectorySpecification) SetPropertyFileName(value CIM_CopyFileAction) (err error) {
	return instance.SetProperty("FileName", (value))
}

// GetFileName gets the value of FileName for the instance
func (instance *CIM_ToDirectorySpecification) GetPropertyFileName() (value CIM_CopyFileAction, err error) {
	retValue, err := instance.GetProperty("FileName")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_CopyFileAction)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_CopyFileAction is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_CopyFileAction(valuetmp)

	return
}
