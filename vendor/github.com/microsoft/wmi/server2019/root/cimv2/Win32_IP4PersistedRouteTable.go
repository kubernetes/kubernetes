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

// Win32_IP4PersistedRouteTable struct
type Win32_IP4PersistedRouteTable struct {
	*CIM_LogicalElement

	//
	Destination string

	//
	Mask string

	//
	Metric1 int32

	//
	NextHop string
}

func NewWin32_IP4PersistedRouteTableEx1(instance *cim.WmiInstance) (newInstance *Win32_IP4PersistedRouteTable, err error) {
	tmp, err := NewCIM_LogicalElementEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_IP4PersistedRouteTable{
		CIM_LogicalElement: tmp,
	}
	return
}

func NewWin32_IP4PersistedRouteTableEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_IP4PersistedRouteTable, err error) {
	tmp, err := NewCIM_LogicalElementEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_IP4PersistedRouteTable{
		CIM_LogicalElement: tmp,
	}
	return
}

// SetDestination sets the value of Destination for the instance
func (instance *Win32_IP4PersistedRouteTable) SetPropertyDestination(value string) (err error) {
	return instance.SetProperty("Destination", (value))
}

// GetDestination gets the value of Destination for the instance
func (instance *Win32_IP4PersistedRouteTable) GetPropertyDestination() (value string, err error) {
	retValue, err := instance.GetProperty("Destination")
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

// SetMask sets the value of Mask for the instance
func (instance *Win32_IP4PersistedRouteTable) SetPropertyMask(value string) (err error) {
	return instance.SetProperty("Mask", (value))
}

// GetMask gets the value of Mask for the instance
func (instance *Win32_IP4PersistedRouteTable) GetPropertyMask() (value string, err error) {
	retValue, err := instance.GetProperty("Mask")
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

// SetMetric1 sets the value of Metric1 for the instance
func (instance *Win32_IP4PersistedRouteTable) SetPropertyMetric1(value int32) (err error) {
	return instance.SetProperty("Metric1", (value))
}

// GetMetric1 gets the value of Metric1 for the instance
func (instance *Win32_IP4PersistedRouteTable) GetPropertyMetric1() (value int32, err error) {
	retValue, err := instance.GetProperty("Metric1")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = int32(valuetmp)

	return
}

// SetNextHop sets the value of NextHop for the instance
func (instance *Win32_IP4PersistedRouteTable) SetPropertyNextHop(value string) (err error) {
	return instance.SetProperty("NextHop", (value))
}

// GetNextHop gets the value of NextHop for the instance
func (instance *Win32_IP4PersistedRouteTable) GetPropertyNextHop() (value string, err error) {
	retValue, err := instance.GetProperty("NextHop")
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
