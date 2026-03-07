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

// CIM_ElementConfiguration struct
type CIM_ElementConfiguration struct {
	*cim.WmiInstance

	//
	Configuration CIM_Configuration

	//
	Element CIM_ManagedSystemElement
}

func NewCIM_ElementConfigurationEx1(instance *cim.WmiInstance) (newInstance *CIM_ElementConfiguration, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &CIM_ElementConfiguration{
		WmiInstance: tmp,
	}
	return
}

func NewCIM_ElementConfigurationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_ElementConfiguration, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_ElementConfiguration{
		WmiInstance: tmp,
	}
	return
}

// SetConfiguration sets the value of Configuration for the instance
func (instance *CIM_ElementConfiguration) SetPropertyConfiguration(value CIM_Configuration) (err error) {
	return instance.SetProperty("Configuration", (value))
}

// GetConfiguration gets the value of Configuration for the instance
func (instance *CIM_ElementConfiguration) GetPropertyConfiguration() (value CIM_Configuration, err error) {
	retValue, err := instance.GetProperty("Configuration")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_Configuration)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_Configuration is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_Configuration(valuetmp)

	return
}

// SetElement sets the value of Element for the instance
func (instance *CIM_ElementConfiguration) SetPropertyElement(value CIM_ManagedSystemElement) (err error) {
	return instance.SetProperty("Element", (value))
}

// GetElement gets the value of Element for the instance
func (instance *CIM_ElementConfiguration) GetPropertyElement() (value CIM_ManagedSystemElement, err error) {
	retValue, err := instance.GetProperty("Element")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_ManagedSystemElement)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_ManagedSystemElement is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_ManagedSystemElement(valuetmp)

	return
}
