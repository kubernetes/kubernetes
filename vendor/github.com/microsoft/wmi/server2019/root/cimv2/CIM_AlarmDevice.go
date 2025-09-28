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

// CIM_AlarmDevice struct
type CIM_AlarmDevice struct {
	*CIM_LogicalDevice

	//
	AudibleAlarm bool

	//
	Urgency uint16

	//
	VisibleAlarm bool
}

func NewCIM_AlarmDeviceEx1(instance *cim.WmiInstance) (newInstance *CIM_AlarmDevice, err error) {
	tmp, err := NewCIM_LogicalDeviceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_AlarmDevice{
		CIM_LogicalDevice: tmp,
	}
	return
}

func NewCIM_AlarmDeviceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_AlarmDevice, err error) {
	tmp, err := NewCIM_LogicalDeviceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_AlarmDevice{
		CIM_LogicalDevice: tmp,
	}
	return
}

// SetAudibleAlarm sets the value of AudibleAlarm for the instance
func (instance *CIM_AlarmDevice) SetPropertyAudibleAlarm(value bool) (err error) {
	return instance.SetProperty("AudibleAlarm", (value))
}

// GetAudibleAlarm gets the value of AudibleAlarm for the instance
func (instance *CIM_AlarmDevice) GetPropertyAudibleAlarm() (value bool, err error) {
	retValue, err := instance.GetProperty("AudibleAlarm")
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

// SetUrgency sets the value of Urgency for the instance
func (instance *CIM_AlarmDevice) SetPropertyUrgency(value uint16) (err error) {
	return instance.SetProperty("Urgency", (value))
}

// GetUrgency gets the value of Urgency for the instance
func (instance *CIM_AlarmDevice) GetPropertyUrgency() (value uint16, err error) {
	retValue, err := instance.GetProperty("Urgency")
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

// SetVisibleAlarm sets the value of VisibleAlarm for the instance
func (instance *CIM_AlarmDevice) SetPropertyVisibleAlarm(value bool) (err error) {
	return instance.SetProperty("VisibleAlarm", (value))
}

// GetVisibleAlarm gets the value of VisibleAlarm for the instance
func (instance *CIM_AlarmDevice) GetPropertyVisibleAlarm() (value bool, err error) {
	retValue, err := instance.GetProperty("VisibleAlarm")
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

//

// <param name="RequestedUrgency" type="uint16 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *CIM_AlarmDevice) SetUrgency( /* IN */ RequestedUrgency uint16) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetUrgency", RequestedUrgency)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}
