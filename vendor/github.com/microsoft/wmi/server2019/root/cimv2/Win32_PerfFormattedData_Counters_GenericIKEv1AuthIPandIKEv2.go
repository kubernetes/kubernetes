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

// Win32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2 struct
type Win32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2 struct {
	*Win32_PerfFormattedData

	//
	AuthIPMainModeNegotiationTime uint32

	//
	AuthIPQuickModeNegotiationTime uint32

	//
	ExtendedModeNegotiationTime uint32

	//
	FailedNegotiations uint32

	//
	FailedNegotiationsPersec uint32

	//
	IKEv1MainModeNegotiationTime uint32

	//
	IKEv1QuickModeNegotiationTime uint32

	//
	IKEv2MainModeNegotiationTime uint32

	//
	IKEv2QuickModeNegotiationTime uint32

	//
	InvalidPacketsReceivedPersec uint32

	//
	PacketsReceivedPersec uint32

	//
	SuccessfulNegotiations uint32

	//
	SuccessfulNegotiationsPersec uint32
}

func NewWin32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2Ex1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2Ex6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetAuthIPMainModeNegotiationTime sets the value of AuthIPMainModeNegotiationTime for the instance
func (instance *Win32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2) SetPropertyAuthIPMainModeNegotiationTime(value uint32) (err error) {
	return instance.SetProperty("AuthIPMainModeNegotiationTime", (value))
}

// GetAuthIPMainModeNegotiationTime gets the value of AuthIPMainModeNegotiationTime for the instance
func (instance *Win32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2) GetPropertyAuthIPMainModeNegotiationTime() (value uint32, err error) {
	retValue, err := instance.GetProperty("AuthIPMainModeNegotiationTime")
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

// SetAuthIPQuickModeNegotiationTime sets the value of AuthIPQuickModeNegotiationTime for the instance
func (instance *Win32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2) SetPropertyAuthIPQuickModeNegotiationTime(value uint32) (err error) {
	return instance.SetProperty("AuthIPQuickModeNegotiationTime", (value))
}

// GetAuthIPQuickModeNegotiationTime gets the value of AuthIPQuickModeNegotiationTime for the instance
func (instance *Win32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2) GetPropertyAuthIPQuickModeNegotiationTime() (value uint32, err error) {
	retValue, err := instance.GetProperty("AuthIPQuickModeNegotiationTime")
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

// SetExtendedModeNegotiationTime sets the value of ExtendedModeNegotiationTime for the instance
func (instance *Win32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2) SetPropertyExtendedModeNegotiationTime(value uint32) (err error) {
	return instance.SetProperty("ExtendedModeNegotiationTime", (value))
}

// GetExtendedModeNegotiationTime gets the value of ExtendedModeNegotiationTime for the instance
func (instance *Win32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2) GetPropertyExtendedModeNegotiationTime() (value uint32, err error) {
	retValue, err := instance.GetProperty("ExtendedModeNegotiationTime")
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

// SetFailedNegotiations sets the value of FailedNegotiations for the instance
func (instance *Win32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2) SetPropertyFailedNegotiations(value uint32) (err error) {
	return instance.SetProperty("FailedNegotiations", (value))
}

// GetFailedNegotiations gets the value of FailedNegotiations for the instance
func (instance *Win32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2) GetPropertyFailedNegotiations() (value uint32, err error) {
	retValue, err := instance.GetProperty("FailedNegotiations")
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

// SetFailedNegotiationsPersec sets the value of FailedNegotiationsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2) SetPropertyFailedNegotiationsPersec(value uint32) (err error) {
	return instance.SetProperty("FailedNegotiationsPersec", (value))
}

// GetFailedNegotiationsPersec gets the value of FailedNegotiationsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2) GetPropertyFailedNegotiationsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("FailedNegotiationsPersec")
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

// SetIKEv1MainModeNegotiationTime sets the value of IKEv1MainModeNegotiationTime for the instance
func (instance *Win32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2) SetPropertyIKEv1MainModeNegotiationTime(value uint32) (err error) {
	return instance.SetProperty("IKEv1MainModeNegotiationTime", (value))
}

// GetIKEv1MainModeNegotiationTime gets the value of IKEv1MainModeNegotiationTime for the instance
func (instance *Win32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2) GetPropertyIKEv1MainModeNegotiationTime() (value uint32, err error) {
	retValue, err := instance.GetProperty("IKEv1MainModeNegotiationTime")
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

// SetIKEv1QuickModeNegotiationTime sets the value of IKEv1QuickModeNegotiationTime for the instance
func (instance *Win32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2) SetPropertyIKEv1QuickModeNegotiationTime(value uint32) (err error) {
	return instance.SetProperty("IKEv1QuickModeNegotiationTime", (value))
}

// GetIKEv1QuickModeNegotiationTime gets the value of IKEv1QuickModeNegotiationTime for the instance
func (instance *Win32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2) GetPropertyIKEv1QuickModeNegotiationTime() (value uint32, err error) {
	retValue, err := instance.GetProperty("IKEv1QuickModeNegotiationTime")
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

// SetIKEv2MainModeNegotiationTime sets the value of IKEv2MainModeNegotiationTime for the instance
func (instance *Win32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2) SetPropertyIKEv2MainModeNegotiationTime(value uint32) (err error) {
	return instance.SetProperty("IKEv2MainModeNegotiationTime", (value))
}

// GetIKEv2MainModeNegotiationTime gets the value of IKEv2MainModeNegotiationTime for the instance
func (instance *Win32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2) GetPropertyIKEv2MainModeNegotiationTime() (value uint32, err error) {
	retValue, err := instance.GetProperty("IKEv2MainModeNegotiationTime")
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

// SetIKEv2QuickModeNegotiationTime sets the value of IKEv2QuickModeNegotiationTime for the instance
func (instance *Win32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2) SetPropertyIKEv2QuickModeNegotiationTime(value uint32) (err error) {
	return instance.SetProperty("IKEv2QuickModeNegotiationTime", (value))
}

// GetIKEv2QuickModeNegotiationTime gets the value of IKEv2QuickModeNegotiationTime for the instance
func (instance *Win32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2) GetPropertyIKEv2QuickModeNegotiationTime() (value uint32, err error) {
	retValue, err := instance.GetProperty("IKEv2QuickModeNegotiationTime")
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

// SetInvalidPacketsReceivedPersec sets the value of InvalidPacketsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2) SetPropertyInvalidPacketsReceivedPersec(value uint32) (err error) {
	return instance.SetProperty("InvalidPacketsReceivedPersec", (value))
}

// GetInvalidPacketsReceivedPersec gets the value of InvalidPacketsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2) GetPropertyInvalidPacketsReceivedPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("InvalidPacketsReceivedPersec")
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

// SetPacketsReceivedPersec sets the value of PacketsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2) SetPropertyPacketsReceivedPersec(value uint32) (err error) {
	return instance.SetProperty("PacketsReceivedPersec", (value))
}

// GetPacketsReceivedPersec gets the value of PacketsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2) GetPropertyPacketsReceivedPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("PacketsReceivedPersec")
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

// SetSuccessfulNegotiations sets the value of SuccessfulNegotiations for the instance
func (instance *Win32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2) SetPropertySuccessfulNegotiations(value uint32) (err error) {
	return instance.SetProperty("SuccessfulNegotiations", (value))
}

// GetSuccessfulNegotiations gets the value of SuccessfulNegotiations for the instance
func (instance *Win32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2) GetPropertySuccessfulNegotiations() (value uint32, err error) {
	retValue, err := instance.GetProperty("SuccessfulNegotiations")
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

// SetSuccessfulNegotiationsPersec sets the value of SuccessfulNegotiationsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2) SetPropertySuccessfulNegotiationsPersec(value uint32) (err error) {
	return instance.SetProperty("SuccessfulNegotiationsPersec", (value))
}

// GetSuccessfulNegotiationsPersec gets the value of SuccessfulNegotiationsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_GenericIKEv1AuthIPandIKEv2) GetPropertySuccessfulNegotiationsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("SuccessfulNegotiationsPersec")
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
