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

// Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000 struct
type Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000 struct {
	*Win32_PerfRawData

	//
	Calls uint32

	//
	CallsDuration uint32

	//
	CallsDuration_Base uint32

	//
	CallsFailed uint32

	//
	CallsFailedPerSecond uint32

	//
	CallsFaulted uint32

	//
	CallsFaultedPerSecond uint32

	//
	CallsOutstanding uint32

	//
	CallsPerSecond uint32

	//
	ReliableMessagingMessagesDropped uint32

	//
	ReliableMessagingMessagesDroppedPerSecond uint32

	//
	ReliableMessagingSessionsFaulted uint32

	//
	ReliableMessagingSessionsFaultedPerSecond uint32

	//
	SecurityCallsNotAuthorized uint32

	//
	SecurityCallsNotAuthorizedPerSecond uint32

	//
	SecurityValidationandAuthenticationFailures uint32

	//
	SecurityValidationandAuthenticationFailuresPerSecond uint32

	//
	TransactionsFlowed uint32

	//
	TransactionsFlowedPerSecond uint32
}

func NewWin32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000Ex1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000Ex6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetCalls sets the value of Calls for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) SetPropertyCalls(value uint32) (err error) {
	return instance.SetProperty("Calls", (value))
}

// GetCalls gets the value of Calls for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) GetPropertyCalls() (value uint32, err error) {
	retValue, err := instance.GetProperty("Calls")
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

// SetCallsDuration sets the value of CallsDuration for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) SetPropertyCallsDuration(value uint32) (err error) {
	return instance.SetProperty("CallsDuration", (value))
}

// GetCallsDuration gets the value of CallsDuration for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) GetPropertyCallsDuration() (value uint32, err error) {
	retValue, err := instance.GetProperty("CallsDuration")
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

// SetCallsDuration_Base sets the value of CallsDuration_Base for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) SetPropertyCallsDuration_Base(value uint32) (err error) {
	return instance.SetProperty("CallsDuration_Base", (value))
}

// GetCallsDuration_Base gets the value of CallsDuration_Base for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) GetPropertyCallsDuration_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("CallsDuration_Base")
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

// SetCallsFailed sets the value of CallsFailed for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) SetPropertyCallsFailed(value uint32) (err error) {
	return instance.SetProperty("CallsFailed", (value))
}

// GetCallsFailed gets the value of CallsFailed for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) GetPropertyCallsFailed() (value uint32, err error) {
	retValue, err := instance.GetProperty("CallsFailed")
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

// SetCallsFailedPerSecond sets the value of CallsFailedPerSecond for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) SetPropertyCallsFailedPerSecond(value uint32) (err error) {
	return instance.SetProperty("CallsFailedPerSecond", (value))
}

// GetCallsFailedPerSecond gets the value of CallsFailedPerSecond for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) GetPropertyCallsFailedPerSecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("CallsFailedPerSecond")
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

// SetCallsFaulted sets the value of CallsFaulted for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) SetPropertyCallsFaulted(value uint32) (err error) {
	return instance.SetProperty("CallsFaulted", (value))
}

// GetCallsFaulted gets the value of CallsFaulted for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) GetPropertyCallsFaulted() (value uint32, err error) {
	retValue, err := instance.GetProperty("CallsFaulted")
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

// SetCallsFaultedPerSecond sets the value of CallsFaultedPerSecond for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) SetPropertyCallsFaultedPerSecond(value uint32) (err error) {
	return instance.SetProperty("CallsFaultedPerSecond", (value))
}

// GetCallsFaultedPerSecond gets the value of CallsFaultedPerSecond for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) GetPropertyCallsFaultedPerSecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("CallsFaultedPerSecond")
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

// SetCallsOutstanding sets the value of CallsOutstanding for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) SetPropertyCallsOutstanding(value uint32) (err error) {
	return instance.SetProperty("CallsOutstanding", (value))
}

// GetCallsOutstanding gets the value of CallsOutstanding for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) GetPropertyCallsOutstanding() (value uint32, err error) {
	retValue, err := instance.GetProperty("CallsOutstanding")
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

// SetCallsPerSecond sets the value of CallsPerSecond for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) SetPropertyCallsPerSecond(value uint32) (err error) {
	return instance.SetProperty("CallsPerSecond", (value))
}

// GetCallsPerSecond gets the value of CallsPerSecond for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) GetPropertyCallsPerSecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("CallsPerSecond")
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

// SetReliableMessagingMessagesDropped sets the value of ReliableMessagingMessagesDropped for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) SetPropertyReliableMessagingMessagesDropped(value uint32) (err error) {
	return instance.SetProperty("ReliableMessagingMessagesDropped", (value))
}

// GetReliableMessagingMessagesDropped gets the value of ReliableMessagingMessagesDropped for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) GetPropertyReliableMessagingMessagesDropped() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReliableMessagingMessagesDropped")
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

// SetReliableMessagingMessagesDroppedPerSecond sets the value of ReliableMessagingMessagesDroppedPerSecond for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) SetPropertyReliableMessagingMessagesDroppedPerSecond(value uint32) (err error) {
	return instance.SetProperty("ReliableMessagingMessagesDroppedPerSecond", (value))
}

// GetReliableMessagingMessagesDroppedPerSecond gets the value of ReliableMessagingMessagesDroppedPerSecond for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) GetPropertyReliableMessagingMessagesDroppedPerSecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReliableMessagingMessagesDroppedPerSecond")
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

// SetReliableMessagingSessionsFaulted sets the value of ReliableMessagingSessionsFaulted for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) SetPropertyReliableMessagingSessionsFaulted(value uint32) (err error) {
	return instance.SetProperty("ReliableMessagingSessionsFaulted", (value))
}

// GetReliableMessagingSessionsFaulted gets the value of ReliableMessagingSessionsFaulted for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) GetPropertyReliableMessagingSessionsFaulted() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReliableMessagingSessionsFaulted")
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

// SetReliableMessagingSessionsFaultedPerSecond sets the value of ReliableMessagingSessionsFaultedPerSecond for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) SetPropertyReliableMessagingSessionsFaultedPerSecond(value uint32) (err error) {
	return instance.SetProperty("ReliableMessagingSessionsFaultedPerSecond", (value))
}

// GetReliableMessagingSessionsFaultedPerSecond gets the value of ReliableMessagingSessionsFaultedPerSecond for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) GetPropertyReliableMessagingSessionsFaultedPerSecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReliableMessagingSessionsFaultedPerSecond")
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

// SetSecurityCallsNotAuthorized sets the value of SecurityCallsNotAuthorized for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) SetPropertySecurityCallsNotAuthorized(value uint32) (err error) {
	return instance.SetProperty("SecurityCallsNotAuthorized", (value))
}

// GetSecurityCallsNotAuthorized gets the value of SecurityCallsNotAuthorized for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) GetPropertySecurityCallsNotAuthorized() (value uint32, err error) {
	retValue, err := instance.GetProperty("SecurityCallsNotAuthorized")
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

// SetSecurityCallsNotAuthorizedPerSecond sets the value of SecurityCallsNotAuthorizedPerSecond for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) SetPropertySecurityCallsNotAuthorizedPerSecond(value uint32) (err error) {
	return instance.SetProperty("SecurityCallsNotAuthorizedPerSecond", (value))
}

// GetSecurityCallsNotAuthorizedPerSecond gets the value of SecurityCallsNotAuthorizedPerSecond for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) GetPropertySecurityCallsNotAuthorizedPerSecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("SecurityCallsNotAuthorizedPerSecond")
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

// SetSecurityValidationandAuthenticationFailures sets the value of SecurityValidationandAuthenticationFailures for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) SetPropertySecurityValidationandAuthenticationFailures(value uint32) (err error) {
	return instance.SetProperty("SecurityValidationandAuthenticationFailures", (value))
}

// GetSecurityValidationandAuthenticationFailures gets the value of SecurityValidationandAuthenticationFailures for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) GetPropertySecurityValidationandAuthenticationFailures() (value uint32, err error) {
	retValue, err := instance.GetProperty("SecurityValidationandAuthenticationFailures")
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

// SetSecurityValidationandAuthenticationFailuresPerSecond sets the value of SecurityValidationandAuthenticationFailuresPerSecond for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) SetPropertySecurityValidationandAuthenticationFailuresPerSecond(value uint32) (err error) {
	return instance.SetProperty("SecurityValidationandAuthenticationFailuresPerSecond", (value))
}

// GetSecurityValidationandAuthenticationFailuresPerSecond gets the value of SecurityValidationandAuthenticationFailuresPerSecond for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) GetPropertySecurityValidationandAuthenticationFailuresPerSecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("SecurityValidationandAuthenticationFailuresPerSecond")
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

// SetTransactionsFlowed sets the value of TransactionsFlowed for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) SetPropertyTransactionsFlowed(value uint32) (err error) {
	return instance.SetProperty("TransactionsFlowed", (value))
}

// GetTransactionsFlowed gets the value of TransactionsFlowed for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) GetPropertyTransactionsFlowed() (value uint32, err error) {
	retValue, err := instance.GetProperty("TransactionsFlowed")
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

// SetTransactionsFlowedPerSecond sets the value of TransactionsFlowedPerSecond for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) SetPropertyTransactionsFlowedPerSecond(value uint32) (err error) {
	return instance.SetProperty("TransactionsFlowedPerSecond", (value))
}

// GetTransactionsFlowedPerSecond gets the value of TransactionsFlowedPerSecond for the instance
func (instance *Win32_PerfRawData_ServiceModel4000_ServiceModelEndpoint4000) GetPropertyTransactionsFlowedPerSecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("TransactionsFlowedPerSecond")
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
