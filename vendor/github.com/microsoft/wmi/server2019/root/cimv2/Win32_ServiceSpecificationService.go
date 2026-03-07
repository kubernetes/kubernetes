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

// Win32_ServiceSpecificationService struct
type Win32_ServiceSpecificationService struct {
	*cim.WmiInstance

	//
	Check Win32_ServiceSpecification

	//
	Element Win32_Service
}

func NewWin32_ServiceSpecificationServiceEx1(instance *cim.WmiInstance) (newInstance *Win32_ServiceSpecificationService, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_ServiceSpecificationService{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_ServiceSpecificationServiceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ServiceSpecificationService, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ServiceSpecificationService{
		WmiInstance: tmp,
	}
	return
}

// SetCheck sets the value of Check for the instance
func (instance *Win32_ServiceSpecificationService) SetPropertyCheck(value Win32_ServiceSpecification) (err error) {
	return instance.SetProperty("Check", (value))
}

// GetCheck gets the value of Check for the instance
func (instance *Win32_ServiceSpecificationService) GetPropertyCheck() (value Win32_ServiceSpecification, err error) {
	retValue, err := instance.GetProperty("Check")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_ServiceSpecification)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_ServiceSpecification is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_ServiceSpecification(valuetmp)

	return
}

// SetElement sets the value of Element for the instance
func (instance *Win32_ServiceSpecificationService) SetPropertyElement(value Win32_Service) (err error) {
	return instance.SetProperty("Element", (value))
}

// GetElement gets the value of Element for the instance
func (instance *Win32_ServiceSpecificationService) GetPropertyElement() (value Win32_Service, err error) {
	retValue, err := instance.GetProperty("Element")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_Service)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_Service is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_Service(valuetmp)

	return
}
