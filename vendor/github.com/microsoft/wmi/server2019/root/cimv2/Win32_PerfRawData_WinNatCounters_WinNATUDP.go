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

// Win32_PerfRawData_WinNatCounters_WinNATUDP struct
type Win32_PerfRawData_WinNatCounters_WinNATUDP struct {
	*Win32_PerfRawData

	//
	NumberOfBindings uint32

	//
	NumberOfSessions uint32

	//
	NumExtToIntTranslations uint32

	//
	NumIntToExtTranslations uint32

	//
	NumPacketsDropped uint32

	//
	NumSessionsTimedOut uint32
}

func NewWin32_PerfRawData_WinNatCounters_WinNATUDPEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_WinNatCounters_WinNATUDP, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_WinNatCounters_WinNATUDP{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_WinNatCounters_WinNATUDPEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_WinNatCounters_WinNATUDP, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_WinNatCounters_WinNATUDP{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetNumberOfBindings sets the value of NumberOfBindings for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNATUDP) SetPropertyNumberOfBindings(value uint32) (err error) {
	return instance.SetProperty("NumberOfBindings", (value))
}

// GetNumberOfBindings gets the value of NumberOfBindings for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNATUDP) GetPropertyNumberOfBindings() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfBindings")
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

// SetNumberOfSessions sets the value of NumberOfSessions for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNATUDP) SetPropertyNumberOfSessions(value uint32) (err error) {
	return instance.SetProperty("NumberOfSessions", (value))
}

// GetNumberOfSessions gets the value of NumberOfSessions for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNATUDP) GetPropertyNumberOfSessions() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfSessions")
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

// SetNumExtToIntTranslations sets the value of NumExtToIntTranslations for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNATUDP) SetPropertyNumExtToIntTranslations(value uint32) (err error) {
	return instance.SetProperty("NumExtToIntTranslations", (value))
}

// GetNumExtToIntTranslations gets the value of NumExtToIntTranslations for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNATUDP) GetPropertyNumExtToIntTranslations() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumExtToIntTranslations")
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

// SetNumIntToExtTranslations sets the value of NumIntToExtTranslations for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNATUDP) SetPropertyNumIntToExtTranslations(value uint32) (err error) {
	return instance.SetProperty("NumIntToExtTranslations", (value))
}

// GetNumIntToExtTranslations gets the value of NumIntToExtTranslations for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNATUDP) GetPropertyNumIntToExtTranslations() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumIntToExtTranslations")
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

// SetNumPacketsDropped sets the value of NumPacketsDropped for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNATUDP) SetPropertyNumPacketsDropped(value uint32) (err error) {
	return instance.SetProperty("NumPacketsDropped", (value))
}

// GetNumPacketsDropped gets the value of NumPacketsDropped for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNATUDP) GetPropertyNumPacketsDropped() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumPacketsDropped")
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

// SetNumSessionsTimedOut sets the value of NumSessionsTimedOut for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNATUDP) SetPropertyNumSessionsTimedOut(value uint32) (err error) {
	return instance.SetProperty("NumSessionsTimedOut", (value))
}

// GetNumSessionsTimedOut gets the value of NumSessionsTimedOut for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNATUDP) GetPropertyNumSessionsTimedOut() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumSessionsTimedOut")
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
