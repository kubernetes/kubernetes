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

// CIM_IRQ struct
type CIM_IRQ struct {
	*CIM_SystemResource

	//
	Availability uint16

	//
	CreationClassName string

	//
	CSCreationClassName string

	//
	CSName string

	//
	IRQNumber uint32

	//
	Shareable bool

	//
	TriggerLevel uint16

	//
	TriggerType uint16
}

func NewCIM_IRQEx1(instance *cim.WmiInstance) (newInstance *CIM_IRQ, err error) {
	tmp, err := NewCIM_SystemResourceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_IRQ{
		CIM_SystemResource: tmp,
	}
	return
}

func NewCIM_IRQEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_IRQ, err error) {
	tmp, err := NewCIM_SystemResourceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_IRQ{
		CIM_SystemResource: tmp,
	}
	return
}

// SetAvailability sets the value of Availability for the instance
func (instance *CIM_IRQ) SetPropertyAvailability(value uint16) (err error) {
	return instance.SetProperty("Availability", (value))
}

// GetAvailability gets the value of Availability for the instance
func (instance *CIM_IRQ) GetPropertyAvailability() (value uint16, err error) {
	retValue, err := instance.GetProperty("Availability")
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

// SetCreationClassName sets the value of CreationClassName for the instance
func (instance *CIM_IRQ) SetPropertyCreationClassName(value string) (err error) {
	return instance.SetProperty("CreationClassName", (value))
}

// GetCreationClassName gets the value of CreationClassName for the instance
func (instance *CIM_IRQ) GetPropertyCreationClassName() (value string, err error) {
	retValue, err := instance.GetProperty("CreationClassName")
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

// SetCSCreationClassName sets the value of CSCreationClassName for the instance
func (instance *CIM_IRQ) SetPropertyCSCreationClassName(value string) (err error) {
	return instance.SetProperty("CSCreationClassName", (value))
}

// GetCSCreationClassName gets the value of CSCreationClassName for the instance
func (instance *CIM_IRQ) GetPropertyCSCreationClassName() (value string, err error) {
	retValue, err := instance.GetProperty("CSCreationClassName")
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

// SetCSName sets the value of CSName for the instance
func (instance *CIM_IRQ) SetPropertyCSName(value string) (err error) {
	return instance.SetProperty("CSName", (value))
}

// GetCSName gets the value of CSName for the instance
func (instance *CIM_IRQ) GetPropertyCSName() (value string, err error) {
	retValue, err := instance.GetProperty("CSName")
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

// SetIRQNumber sets the value of IRQNumber for the instance
func (instance *CIM_IRQ) SetPropertyIRQNumber(value uint32) (err error) {
	return instance.SetProperty("IRQNumber", (value))
}

// GetIRQNumber gets the value of IRQNumber for the instance
func (instance *CIM_IRQ) GetPropertyIRQNumber() (value uint32, err error) {
	retValue, err := instance.GetProperty("IRQNumber")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}

// SetShareable sets the value of Shareable for the instance
func (instance *CIM_IRQ) SetPropertyShareable(value bool) (err error) {
	return instance.SetProperty("Shareable", (value))
}

// GetShareable gets the value of Shareable for the instance
func (instance *CIM_IRQ) GetPropertyShareable() (value bool, err error) {
	retValue, err := instance.GetProperty("Shareable")
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

// SetTriggerLevel sets the value of TriggerLevel for the instance
func (instance *CIM_IRQ) SetPropertyTriggerLevel(value uint16) (err error) {
	return instance.SetProperty("TriggerLevel", (value))
}

// GetTriggerLevel gets the value of TriggerLevel for the instance
func (instance *CIM_IRQ) GetPropertyTriggerLevel() (value uint16, err error) {
	retValue, err := instance.GetProperty("TriggerLevel")
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

// SetTriggerType sets the value of TriggerType for the instance
func (instance *CIM_IRQ) SetPropertyTriggerType(value uint16) (err error) {
	return instance.SetProperty("TriggerType", (value))
}

// GetTriggerType gets the value of TriggerType for the instance
func (instance *CIM_IRQ) GetPropertyTriggerType() (value uint16, err error) {
	retValue, err := instance.GetProperty("TriggerType")
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
