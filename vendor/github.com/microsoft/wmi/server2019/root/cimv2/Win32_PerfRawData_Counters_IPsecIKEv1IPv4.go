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

// Win32_PerfRawData_Counters_IPsecIKEv1IPv4 struct
type Win32_PerfRawData_Counters_IPsecIKEv1IPv4 struct {
	*Win32_PerfRawData

	//
	ActiveMainModeSAs uint32

	//
	ActiveQuickModeSAs uint32

	//
	FailedMainModeNegotiations uint32

	//
	FailedMainModeNegotiationsPersec uint32

	//
	FailedQuickModeNegotiations uint32

	//
	FailedQuickModeNegotiationsPersec uint32

	//
	MainModeNegotiationRequestsReceived uint32

	//
	MainModeNegotiationRequestsReceivedPersec uint32

	//
	MainModeNegotiations uint32

	//
	MainModeNegotiationsPersec uint32

	//
	PendingMainModeNegotiations uint32

	//
	PendingQuickModeNegotiations uint32

	//
	QuickModeNegotiations uint32

	//
	QuickModeNegotiationsPersec uint32

	//
	SuccessfulMainModeNegotiations uint32

	//
	SuccessfulMainModeNegotiationsPersec uint32

	//
	SuccessfulQuickModeNegotiations uint32

	//
	SuccessfulQuickModeNegotiationsPersec uint32
}

func NewWin32_PerfRawData_Counters_IPsecIKEv1IPv4Ex1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_IPsecIKEv1IPv4{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_Counters_IPsecIKEv1IPv4Ex6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_IPsecIKEv1IPv4{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetActiveMainModeSAs sets the value of ActiveMainModeSAs for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) SetPropertyActiveMainModeSAs(value uint32) (err error) {
	return instance.SetProperty("ActiveMainModeSAs", (value))
}

// GetActiveMainModeSAs gets the value of ActiveMainModeSAs for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) GetPropertyActiveMainModeSAs() (value uint32, err error) {
	retValue, err := instance.GetProperty("ActiveMainModeSAs")
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

// SetActiveQuickModeSAs sets the value of ActiveQuickModeSAs for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) SetPropertyActiveQuickModeSAs(value uint32) (err error) {
	return instance.SetProperty("ActiveQuickModeSAs", (value))
}

// GetActiveQuickModeSAs gets the value of ActiveQuickModeSAs for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) GetPropertyActiveQuickModeSAs() (value uint32, err error) {
	retValue, err := instance.GetProperty("ActiveQuickModeSAs")
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

// SetFailedMainModeNegotiations sets the value of FailedMainModeNegotiations for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) SetPropertyFailedMainModeNegotiations(value uint32) (err error) {
	return instance.SetProperty("FailedMainModeNegotiations", (value))
}

// GetFailedMainModeNegotiations gets the value of FailedMainModeNegotiations for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) GetPropertyFailedMainModeNegotiations() (value uint32, err error) {
	retValue, err := instance.GetProperty("FailedMainModeNegotiations")
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

// SetFailedMainModeNegotiationsPersec sets the value of FailedMainModeNegotiationsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) SetPropertyFailedMainModeNegotiationsPersec(value uint32) (err error) {
	return instance.SetProperty("FailedMainModeNegotiationsPersec", (value))
}

// GetFailedMainModeNegotiationsPersec gets the value of FailedMainModeNegotiationsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) GetPropertyFailedMainModeNegotiationsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("FailedMainModeNegotiationsPersec")
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

// SetFailedQuickModeNegotiations sets the value of FailedQuickModeNegotiations for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) SetPropertyFailedQuickModeNegotiations(value uint32) (err error) {
	return instance.SetProperty("FailedQuickModeNegotiations", (value))
}

// GetFailedQuickModeNegotiations gets the value of FailedQuickModeNegotiations for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) GetPropertyFailedQuickModeNegotiations() (value uint32, err error) {
	retValue, err := instance.GetProperty("FailedQuickModeNegotiations")
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

// SetFailedQuickModeNegotiationsPersec sets the value of FailedQuickModeNegotiationsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) SetPropertyFailedQuickModeNegotiationsPersec(value uint32) (err error) {
	return instance.SetProperty("FailedQuickModeNegotiationsPersec", (value))
}

// GetFailedQuickModeNegotiationsPersec gets the value of FailedQuickModeNegotiationsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) GetPropertyFailedQuickModeNegotiationsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("FailedQuickModeNegotiationsPersec")
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

// SetMainModeNegotiationRequestsReceived sets the value of MainModeNegotiationRequestsReceived for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) SetPropertyMainModeNegotiationRequestsReceived(value uint32) (err error) {
	return instance.SetProperty("MainModeNegotiationRequestsReceived", (value))
}

// GetMainModeNegotiationRequestsReceived gets the value of MainModeNegotiationRequestsReceived for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) GetPropertyMainModeNegotiationRequestsReceived() (value uint32, err error) {
	retValue, err := instance.GetProperty("MainModeNegotiationRequestsReceived")
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

// SetMainModeNegotiationRequestsReceivedPersec sets the value of MainModeNegotiationRequestsReceivedPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) SetPropertyMainModeNegotiationRequestsReceivedPersec(value uint32) (err error) {
	return instance.SetProperty("MainModeNegotiationRequestsReceivedPersec", (value))
}

// GetMainModeNegotiationRequestsReceivedPersec gets the value of MainModeNegotiationRequestsReceivedPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) GetPropertyMainModeNegotiationRequestsReceivedPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("MainModeNegotiationRequestsReceivedPersec")
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

// SetMainModeNegotiations sets the value of MainModeNegotiations for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) SetPropertyMainModeNegotiations(value uint32) (err error) {
	return instance.SetProperty("MainModeNegotiations", (value))
}

// GetMainModeNegotiations gets the value of MainModeNegotiations for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) GetPropertyMainModeNegotiations() (value uint32, err error) {
	retValue, err := instance.GetProperty("MainModeNegotiations")
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

// SetMainModeNegotiationsPersec sets the value of MainModeNegotiationsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) SetPropertyMainModeNegotiationsPersec(value uint32) (err error) {
	return instance.SetProperty("MainModeNegotiationsPersec", (value))
}

// GetMainModeNegotiationsPersec gets the value of MainModeNegotiationsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) GetPropertyMainModeNegotiationsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("MainModeNegotiationsPersec")
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

// SetPendingMainModeNegotiations sets the value of PendingMainModeNegotiations for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) SetPropertyPendingMainModeNegotiations(value uint32) (err error) {
	return instance.SetProperty("PendingMainModeNegotiations", (value))
}

// GetPendingMainModeNegotiations gets the value of PendingMainModeNegotiations for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) GetPropertyPendingMainModeNegotiations() (value uint32, err error) {
	retValue, err := instance.GetProperty("PendingMainModeNegotiations")
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

// SetPendingQuickModeNegotiations sets the value of PendingQuickModeNegotiations for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) SetPropertyPendingQuickModeNegotiations(value uint32) (err error) {
	return instance.SetProperty("PendingQuickModeNegotiations", (value))
}

// GetPendingQuickModeNegotiations gets the value of PendingQuickModeNegotiations for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) GetPropertyPendingQuickModeNegotiations() (value uint32, err error) {
	retValue, err := instance.GetProperty("PendingQuickModeNegotiations")
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

// SetQuickModeNegotiations sets the value of QuickModeNegotiations for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) SetPropertyQuickModeNegotiations(value uint32) (err error) {
	return instance.SetProperty("QuickModeNegotiations", (value))
}

// GetQuickModeNegotiations gets the value of QuickModeNegotiations for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) GetPropertyQuickModeNegotiations() (value uint32, err error) {
	retValue, err := instance.GetProperty("QuickModeNegotiations")
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

// SetQuickModeNegotiationsPersec sets the value of QuickModeNegotiationsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) SetPropertyQuickModeNegotiationsPersec(value uint32) (err error) {
	return instance.SetProperty("QuickModeNegotiationsPersec", (value))
}

// GetQuickModeNegotiationsPersec gets the value of QuickModeNegotiationsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) GetPropertyQuickModeNegotiationsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("QuickModeNegotiationsPersec")
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

// SetSuccessfulMainModeNegotiations sets the value of SuccessfulMainModeNegotiations for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) SetPropertySuccessfulMainModeNegotiations(value uint32) (err error) {
	return instance.SetProperty("SuccessfulMainModeNegotiations", (value))
}

// GetSuccessfulMainModeNegotiations gets the value of SuccessfulMainModeNegotiations for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) GetPropertySuccessfulMainModeNegotiations() (value uint32, err error) {
	retValue, err := instance.GetProperty("SuccessfulMainModeNegotiations")
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

// SetSuccessfulMainModeNegotiationsPersec sets the value of SuccessfulMainModeNegotiationsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) SetPropertySuccessfulMainModeNegotiationsPersec(value uint32) (err error) {
	return instance.SetProperty("SuccessfulMainModeNegotiationsPersec", (value))
}

// GetSuccessfulMainModeNegotiationsPersec gets the value of SuccessfulMainModeNegotiationsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) GetPropertySuccessfulMainModeNegotiationsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("SuccessfulMainModeNegotiationsPersec")
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

// SetSuccessfulQuickModeNegotiations sets the value of SuccessfulQuickModeNegotiations for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) SetPropertySuccessfulQuickModeNegotiations(value uint32) (err error) {
	return instance.SetProperty("SuccessfulQuickModeNegotiations", (value))
}

// GetSuccessfulQuickModeNegotiations gets the value of SuccessfulQuickModeNegotiations for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) GetPropertySuccessfulQuickModeNegotiations() (value uint32, err error) {
	retValue, err := instance.GetProperty("SuccessfulQuickModeNegotiations")
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

// SetSuccessfulQuickModeNegotiationsPersec sets the value of SuccessfulQuickModeNegotiationsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) SetPropertySuccessfulQuickModeNegotiationsPersec(value uint32) (err error) {
	return instance.SetProperty("SuccessfulQuickModeNegotiationsPersec", (value))
}

// GetSuccessfulQuickModeNegotiationsPersec gets the value of SuccessfulQuickModeNegotiationsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecIKEv1IPv4) GetPropertySuccessfulQuickModeNegotiationsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("SuccessfulQuickModeNegotiationsPersec")
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
