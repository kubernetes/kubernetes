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

// CIM_Fan struct
type CIM_Fan struct {
	*CIM_CoolingDevice

	//
	DesiredSpeed uint64

	//
	VariableSpeed bool
}

func NewCIM_FanEx1(instance *cim.WmiInstance) (newInstance *CIM_Fan, err error) {
	tmp, err := NewCIM_CoolingDeviceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_Fan{
		CIM_CoolingDevice: tmp,
	}
	return
}

func NewCIM_FanEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_Fan, err error) {
	tmp, err := NewCIM_CoolingDeviceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_Fan{
		CIM_CoolingDevice: tmp,
	}
	return
}

// SetDesiredSpeed sets the value of DesiredSpeed for the instance
func (instance *CIM_Fan) SetPropertyDesiredSpeed(value uint64) (err error) {
	return instance.SetProperty("DesiredSpeed", (value))
}

// GetDesiredSpeed gets the value of DesiredSpeed for the instance
func (instance *CIM_Fan) GetPropertyDesiredSpeed() (value uint64, err error) {
	retValue, err := instance.GetProperty("DesiredSpeed")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetVariableSpeed sets the value of VariableSpeed for the instance
func (instance *CIM_Fan) SetPropertyVariableSpeed(value bool) (err error) {
	return instance.SetProperty("VariableSpeed", (value))
}

// GetVariableSpeed gets the value of VariableSpeed for the instance
func (instance *CIM_Fan) GetPropertyVariableSpeed() (value bool, err error) {
	retValue, err := instance.GetProperty("VariableSpeed")
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

// <param name="DesiredSpeed" type="uint64 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *CIM_Fan) SetSpeed( /* IN */ DesiredSpeed uint64) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetSpeed", DesiredSpeed)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}
