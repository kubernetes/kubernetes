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

// Win32_CurrentTime struct
type Win32_CurrentTime struct {
	*cim.WmiInstance

	//
	Day uint32

	//
	DayOfWeek uint32

	//
	Hour uint32

	//
	Milliseconds uint32

	//
	Minute uint32

	//
	Month uint32

	//
	Quarter uint32

	//
	Second uint32

	//
	WeekInMonth uint32

	//
	Year uint32
}

func NewWin32_CurrentTimeEx1(instance *cim.WmiInstance) (newInstance *Win32_CurrentTime, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_CurrentTime{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_CurrentTimeEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_CurrentTime, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_CurrentTime{
		WmiInstance: tmp,
	}
	return
}

// SetDay sets the value of Day for the instance
func (instance *Win32_CurrentTime) SetPropertyDay(value uint32) (err error) {
	return instance.SetProperty("Day", (value))
}

// GetDay gets the value of Day for the instance
func (instance *Win32_CurrentTime) GetPropertyDay() (value uint32, err error) {
	retValue, err := instance.GetProperty("Day")
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

// SetDayOfWeek sets the value of DayOfWeek for the instance
func (instance *Win32_CurrentTime) SetPropertyDayOfWeek(value uint32) (err error) {
	return instance.SetProperty("DayOfWeek", (value))
}

// GetDayOfWeek gets the value of DayOfWeek for the instance
func (instance *Win32_CurrentTime) GetPropertyDayOfWeek() (value uint32, err error) {
	retValue, err := instance.GetProperty("DayOfWeek")
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

// SetHour sets the value of Hour for the instance
func (instance *Win32_CurrentTime) SetPropertyHour(value uint32) (err error) {
	return instance.SetProperty("Hour", (value))
}

// GetHour gets the value of Hour for the instance
func (instance *Win32_CurrentTime) GetPropertyHour() (value uint32, err error) {
	retValue, err := instance.GetProperty("Hour")
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

// SetMilliseconds sets the value of Milliseconds for the instance
func (instance *Win32_CurrentTime) SetPropertyMilliseconds(value uint32) (err error) {
	return instance.SetProperty("Milliseconds", (value))
}

// GetMilliseconds gets the value of Milliseconds for the instance
func (instance *Win32_CurrentTime) GetPropertyMilliseconds() (value uint32, err error) {
	retValue, err := instance.GetProperty("Milliseconds")
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

// SetMinute sets the value of Minute for the instance
func (instance *Win32_CurrentTime) SetPropertyMinute(value uint32) (err error) {
	return instance.SetProperty("Minute", (value))
}

// GetMinute gets the value of Minute for the instance
func (instance *Win32_CurrentTime) GetPropertyMinute() (value uint32, err error) {
	retValue, err := instance.GetProperty("Minute")
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

// SetMonth sets the value of Month for the instance
func (instance *Win32_CurrentTime) SetPropertyMonth(value uint32) (err error) {
	return instance.SetProperty("Month", (value))
}

// GetMonth gets the value of Month for the instance
func (instance *Win32_CurrentTime) GetPropertyMonth() (value uint32, err error) {
	retValue, err := instance.GetProperty("Month")
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

// SetQuarter sets the value of Quarter for the instance
func (instance *Win32_CurrentTime) SetPropertyQuarter(value uint32) (err error) {
	return instance.SetProperty("Quarter", (value))
}

// GetQuarter gets the value of Quarter for the instance
func (instance *Win32_CurrentTime) GetPropertyQuarter() (value uint32, err error) {
	retValue, err := instance.GetProperty("Quarter")
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

// SetSecond sets the value of Second for the instance
func (instance *Win32_CurrentTime) SetPropertySecond(value uint32) (err error) {
	return instance.SetProperty("Second", (value))
}

// GetSecond gets the value of Second for the instance
func (instance *Win32_CurrentTime) GetPropertySecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("Second")
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

// SetWeekInMonth sets the value of WeekInMonth for the instance
func (instance *Win32_CurrentTime) SetPropertyWeekInMonth(value uint32) (err error) {
	return instance.SetProperty("WeekInMonth", (value))
}

// GetWeekInMonth gets the value of WeekInMonth for the instance
func (instance *Win32_CurrentTime) GetPropertyWeekInMonth() (value uint32, err error) {
	retValue, err := instance.GetProperty("WeekInMonth")
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

// SetYear sets the value of Year for the instance
func (instance *Win32_CurrentTime) SetPropertyYear(value uint32) (err error) {
	return instance.SetProperty("Year", (value))
}

// GetYear gets the value of Year for the instance
func (instance *Win32_CurrentTime) GetPropertyYear() (value uint32, err error) {
	retValue, err := instance.GetProperty("Year")
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
