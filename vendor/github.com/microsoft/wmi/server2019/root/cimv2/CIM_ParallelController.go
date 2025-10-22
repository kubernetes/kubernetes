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

// CIM_ParallelController struct
type CIM_ParallelController struct {
	*CIM_Controller

	//
	Capabilities []uint16

	//
	CapabilityDescriptions []string

	//
	DMASupport bool
}

func NewCIM_ParallelControllerEx1(instance *cim.WmiInstance) (newInstance *CIM_ParallelController, err error) {
	tmp, err := NewCIM_ControllerEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_ParallelController{
		CIM_Controller: tmp,
	}
	return
}

func NewCIM_ParallelControllerEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_ParallelController, err error) {
	tmp, err := NewCIM_ControllerEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_ParallelController{
		CIM_Controller: tmp,
	}
	return
}

// SetCapabilities sets the value of Capabilities for the instance
func (instance *CIM_ParallelController) SetPropertyCapabilities(value []uint16) (err error) {
	return instance.SetProperty("Capabilities", (value))
}

// GetCapabilities gets the value of Capabilities for the instance
func (instance *CIM_ParallelController) GetPropertyCapabilities() (value []uint16, err error) {
	retValue, err := instance.GetProperty("Capabilities")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint16)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint16(valuetmp))
	}

	return
}

// SetCapabilityDescriptions sets the value of CapabilityDescriptions for the instance
func (instance *CIM_ParallelController) SetPropertyCapabilityDescriptions(value []string) (err error) {
	return instance.SetProperty("CapabilityDescriptions", (value))
}

// GetCapabilityDescriptions gets the value of CapabilityDescriptions for the instance
func (instance *CIM_ParallelController) GetPropertyCapabilityDescriptions() (value []string, err error) {
	retValue, err := instance.GetProperty("CapabilityDescriptions")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetDMASupport sets the value of DMASupport for the instance
func (instance *CIM_ParallelController) SetPropertyDMASupport(value bool) (err error) {
	return instance.SetProperty("DMASupport", (value))
}

// GetDMASupport gets the value of DMASupport for the instance
func (instance *CIM_ParallelController) GetPropertyDMASupport() (value bool, err error) {
	retValue, err := instance.GetProperty("DMASupport")
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
