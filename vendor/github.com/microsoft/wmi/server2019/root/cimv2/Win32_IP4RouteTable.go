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

// Win32_IP4RouteTable struct
type Win32_IP4RouteTable struct {
	*CIM_LogicalElement

	//
	Age uint32

	//
	Destination string

	//
	Information string

	//
	InterfaceIndex int32

	//
	Mask string

	//
	Metric1 int32

	//
	Metric2 int32

	//
	Metric3 int32

	//
	Metric4 int32

	//
	Metric5 int32

	//
	NextHop string

	//
	Protocol uint32

	//
	Type uint32
}

func NewWin32_IP4RouteTableEx1(instance *cim.WmiInstance) (newInstance *Win32_IP4RouteTable, err error) {
	tmp, err := NewCIM_LogicalElementEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_IP4RouteTable{
		CIM_LogicalElement: tmp,
	}
	return
}

func NewWin32_IP4RouteTableEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_IP4RouteTable, err error) {
	tmp, err := NewCIM_LogicalElementEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_IP4RouteTable{
		CIM_LogicalElement: tmp,
	}
	return
}

// SetAge sets the value of Age for the instance
func (instance *Win32_IP4RouteTable) SetPropertyAge(value uint32) (err error) {
	return instance.SetProperty("Age", (value))
}

// GetAge gets the value of Age for the instance
func (instance *Win32_IP4RouteTable) GetPropertyAge() (value uint32, err error) {
	retValue, err := instance.GetProperty("Age")
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

// SetDestination sets the value of Destination for the instance
func (instance *Win32_IP4RouteTable) SetPropertyDestination(value string) (err error) {
	return instance.SetProperty("Destination", (value))
}

// GetDestination gets the value of Destination for the instance
func (instance *Win32_IP4RouteTable) GetPropertyDestination() (value string, err error) {
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

// SetInformation sets the value of Information for the instance
func (instance *Win32_IP4RouteTable) SetPropertyInformation(value string) (err error) {
	return instance.SetProperty("Information", (value))
}

// GetInformation gets the value of Information for the instance
func (instance *Win32_IP4RouteTable) GetPropertyInformation() (value string, err error) {
	retValue, err := instance.GetProperty("Information")
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

// SetInterfaceIndex sets the value of InterfaceIndex for the instance
func (instance *Win32_IP4RouteTable) SetPropertyInterfaceIndex(value int32) (err error) {
	return instance.SetProperty("InterfaceIndex", (value))
}

// GetInterfaceIndex gets the value of InterfaceIndex for the instance
func (instance *Win32_IP4RouteTable) GetPropertyInterfaceIndex() (value int32, err error) {
	retValue, err := instance.GetProperty("InterfaceIndex")
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

// SetMask sets the value of Mask for the instance
func (instance *Win32_IP4RouteTable) SetPropertyMask(value string) (err error) {
	return instance.SetProperty("Mask", (value))
}

// GetMask gets the value of Mask for the instance
func (instance *Win32_IP4RouteTable) GetPropertyMask() (value string, err error) {
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
func (instance *Win32_IP4RouteTable) SetPropertyMetric1(value int32) (err error) {
	return instance.SetProperty("Metric1", (value))
}

// GetMetric1 gets the value of Metric1 for the instance
func (instance *Win32_IP4RouteTable) GetPropertyMetric1() (value int32, err error) {
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

// SetMetric2 sets the value of Metric2 for the instance
func (instance *Win32_IP4RouteTable) SetPropertyMetric2(value int32) (err error) {
	return instance.SetProperty("Metric2", (value))
}

// GetMetric2 gets the value of Metric2 for the instance
func (instance *Win32_IP4RouteTable) GetPropertyMetric2() (value int32, err error) {
	retValue, err := instance.GetProperty("Metric2")
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

// SetMetric3 sets the value of Metric3 for the instance
func (instance *Win32_IP4RouteTable) SetPropertyMetric3(value int32) (err error) {
	return instance.SetProperty("Metric3", (value))
}

// GetMetric3 gets the value of Metric3 for the instance
func (instance *Win32_IP4RouteTable) GetPropertyMetric3() (value int32, err error) {
	retValue, err := instance.GetProperty("Metric3")
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

// SetMetric4 sets the value of Metric4 for the instance
func (instance *Win32_IP4RouteTable) SetPropertyMetric4(value int32) (err error) {
	return instance.SetProperty("Metric4", (value))
}

// GetMetric4 gets the value of Metric4 for the instance
func (instance *Win32_IP4RouteTable) GetPropertyMetric4() (value int32, err error) {
	retValue, err := instance.GetProperty("Metric4")
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

// SetMetric5 sets the value of Metric5 for the instance
func (instance *Win32_IP4RouteTable) SetPropertyMetric5(value int32) (err error) {
	return instance.SetProperty("Metric5", (value))
}

// GetMetric5 gets the value of Metric5 for the instance
func (instance *Win32_IP4RouteTable) GetPropertyMetric5() (value int32, err error) {
	retValue, err := instance.GetProperty("Metric5")
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
func (instance *Win32_IP4RouteTable) SetPropertyNextHop(value string) (err error) {
	return instance.SetProperty("NextHop", (value))
}

// GetNextHop gets the value of NextHop for the instance
func (instance *Win32_IP4RouteTable) GetPropertyNextHop() (value string, err error) {
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

// SetProtocol sets the value of Protocol for the instance
func (instance *Win32_IP4RouteTable) SetPropertyProtocol(value uint32) (err error) {
	return instance.SetProperty("Protocol", (value))
}

// GetProtocol gets the value of Protocol for the instance
func (instance *Win32_IP4RouteTable) GetPropertyProtocol() (value uint32, err error) {
	retValue, err := instance.GetProperty("Protocol")
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

// SetType sets the value of Type for the instance
func (instance *Win32_IP4RouteTable) SetPropertyType(value uint32) (err error) {
	return instance.SetProperty("Type", (value))
}

// GetType gets the value of Type for the instance
func (instance *Win32_IP4RouteTable) GetPropertyType() (value uint32, err error) {
	retValue, err := instance.GetProperty("Type")
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
