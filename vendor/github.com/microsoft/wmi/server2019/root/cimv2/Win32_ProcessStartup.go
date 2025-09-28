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

// Win32_ProcessStartup struct
type Win32_ProcessStartup struct {
	*Win32_MethodParameterClass

	//
	CreateFlags uint32

	//
	EnvironmentVariables []string

	//
	ErrorMode uint16

	//
	FillAttribute uint32

	//
	PriorityClass uint32

	//
	ShowWindow uint16

	//
	Title string

	//
	WinstationDesktop string

	//
	X uint32

	//
	XCountChars uint32

	//
	XSize uint32

	//
	Y uint32

	//
	YCountChars uint32

	//
	YSize uint32
}

func NewWin32_ProcessStartupEx1(instance *cim.WmiInstance) (newInstance *Win32_ProcessStartup, err error) {
	tmp, err := NewWin32_MethodParameterClassEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_ProcessStartup{
		Win32_MethodParameterClass: tmp,
	}
	return
}

func NewWin32_ProcessStartupEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ProcessStartup, err error) {
	tmp, err := NewWin32_MethodParameterClassEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ProcessStartup{
		Win32_MethodParameterClass: tmp,
	}
	return
}

// SetCreateFlags sets the value of CreateFlags for the instance
func (instance *Win32_ProcessStartup) SetPropertyCreateFlags(value uint32) (err error) {
	return instance.SetProperty("CreateFlags", (value))
}

// GetCreateFlags gets the value of CreateFlags for the instance
func (instance *Win32_ProcessStartup) GetPropertyCreateFlags() (value uint32, err error) {
	retValue, err := instance.GetProperty("CreateFlags")
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

// SetEnvironmentVariables sets the value of EnvironmentVariables for the instance
func (instance *Win32_ProcessStartup) SetPropertyEnvironmentVariables(value []string) (err error) {
	return instance.SetProperty("EnvironmentVariables", (value))
}

// GetEnvironmentVariables gets the value of EnvironmentVariables for the instance
func (instance *Win32_ProcessStartup) GetPropertyEnvironmentVariables() (value []string, err error) {
	retValue, err := instance.GetProperty("EnvironmentVariables")
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

// SetErrorMode sets the value of ErrorMode for the instance
func (instance *Win32_ProcessStartup) SetPropertyErrorMode(value uint16) (err error) {
	return instance.SetProperty("ErrorMode", (value))
}

// GetErrorMode gets the value of ErrorMode for the instance
func (instance *Win32_ProcessStartup) GetPropertyErrorMode() (value uint16, err error) {
	retValue, err := instance.GetProperty("ErrorMode")
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

// SetFillAttribute sets the value of FillAttribute for the instance
func (instance *Win32_ProcessStartup) SetPropertyFillAttribute(value uint32) (err error) {
	return instance.SetProperty("FillAttribute", (value))
}

// GetFillAttribute gets the value of FillAttribute for the instance
func (instance *Win32_ProcessStartup) GetPropertyFillAttribute() (value uint32, err error) {
	retValue, err := instance.GetProperty("FillAttribute")
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

// SetPriorityClass sets the value of PriorityClass for the instance
func (instance *Win32_ProcessStartup) SetPropertyPriorityClass(value uint32) (err error) {
	return instance.SetProperty("PriorityClass", (value))
}

// GetPriorityClass gets the value of PriorityClass for the instance
func (instance *Win32_ProcessStartup) GetPropertyPriorityClass() (value uint32, err error) {
	retValue, err := instance.GetProperty("PriorityClass")
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

// SetShowWindow sets the value of ShowWindow for the instance
func (instance *Win32_ProcessStartup) SetPropertyShowWindow(value uint16) (err error) {
	return instance.SetProperty("ShowWindow", (value))
}

// GetShowWindow gets the value of ShowWindow for the instance
func (instance *Win32_ProcessStartup) GetPropertyShowWindow() (value uint16, err error) {
	retValue, err := instance.GetProperty("ShowWindow")
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

// SetTitle sets the value of Title for the instance
func (instance *Win32_ProcessStartup) SetPropertyTitle(value string) (err error) {
	return instance.SetProperty("Title", (value))
}

// GetTitle gets the value of Title for the instance
func (instance *Win32_ProcessStartup) GetPropertyTitle() (value string, err error) {
	retValue, err := instance.GetProperty("Title")
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

// SetWinstationDesktop sets the value of WinstationDesktop for the instance
func (instance *Win32_ProcessStartup) SetPropertyWinstationDesktop(value string) (err error) {
	return instance.SetProperty("WinstationDesktop", (value))
}

// GetWinstationDesktop gets the value of WinstationDesktop for the instance
func (instance *Win32_ProcessStartup) GetPropertyWinstationDesktop() (value string, err error) {
	retValue, err := instance.GetProperty("WinstationDesktop")
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

// SetX sets the value of X for the instance
func (instance *Win32_ProcessStartup) SetPropertyX(value uint32) (err error) {
	return instance.SetProperty("X", (value))
}

// GetX gets the value of X for the instance
func (instance *Win32_ProcessStartup) GetPropertyX() (value uint32, err error) {
	retValue, err := instance.GetProperty("X")
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

// SetXCountChars sets the value of XCountChars for the instance
func (instance *Win32_ProcessStartup) SetPropertyXCountChars(value uint32) (err error) {
	return instance.SetProperty("XCountChars", (value))
}

// GetXCountChars gets the value of XCountChars for the instance
func (instance *Win32_ProcessStartup) GetPropertyXCountChars() (value uint32, err error) {
	retValue, err := instance.GetProperty("XCountChars")
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

// SetXSize sets the value of XSize for the instance
func (instance *Win32_ProcessStartup) SetPropertyXSize(value uint32) (err error) {
	return instance.SetProperty("XSize", (value))
}

// GetXSize gets the value of XSize for the instance
func (instance *Win32_ProcessStartup) GetPropertyXSize() (value uint32, err error) {
	retValue, err := instance.GetProperty("XSize")
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

// SetY sets the value of Y for the instance
func (instance *Win32_ProcessStartup) SetPropertyY(value uint32) (err error) {
	return instance.SetProperty("Y", (value))
}

// GetY gets the value of Y for the instance
func (instance *Win32_ProcessStartup) GetPropertyY() (value uint32, err error) {
	retValue, err := instance.GetProperty("Y")
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

// SetYCountChars sets the value of YCountChars for the instance
func (instance *Win32_ProcessStartup) SetPropertyYCountChars(value uint32) (err error) {
	return instance.SetProperty("YCountChars", (value))
}

// GetYCountChars gets the value of YCountChars for the instance
func (instance *Win32_ProcessStartup) GetPropertyYCountChars() (value uint32, err error) {
	retValue, err := instance.GetProperty("YCountChars")
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

// SetYSize sets the value of YSize for the instance
func (instance *Win32_ProcessStartup) SetPropertyYSize(value uint32) (err error) {
	return instance.SetProperty("YSize", (value))
}

// GetYSize gets the value of YSize for the instance
func (instance *Win32_ProcessStartup) GetPropertyYSize() (value uint32, err error) {
	retValue, err := instance.GetProperty("YSize")
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
