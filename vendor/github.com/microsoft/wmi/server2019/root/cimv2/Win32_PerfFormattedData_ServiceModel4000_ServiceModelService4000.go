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

// Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000 struct
type Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000 struct {
	*Win32_PerfFormattedData

	//
	Calls uint32

	//
	CallsDuration uint32

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
	Instances uint32

	//
	InstancesCreatedPerSecond uint32

	//
	PercentOfMaxConcurrentCalls uint32

	//
	PercentOfMaxConcurrentInstances uint32

	//
	PercentOfMaxConcurrentSessions uint32

	//
	QueuedMessagesDropped uint32

	//
	QueuedMessagesDroppedPerSecond uint32

	//
	QueuedMessagesRejected uint32

	//
	QueuedMessagesRejectedPerSecond uint32

	//
	QueuedPoisonMessages uint32

	//
	QueuedPoisonMessagesPerSecond uint32

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
	TransactedOperationsAborted uint32

	//
	TransactedOperationsAbortedPerSecond uint32

	//
	TransactedOperationsCommitted uint32

	//
	TransactedOperationsCommittedPerSecond uint32

	//
	TransactedOperationsInDoubt uint32

	//
	TransactedOperationsInDoubtPerSecond uint32

	//
	TransactionsFlowed uint32

	//
	TransactionsFlowedPerSecond uint32
}

func NewWin32_PerfFormattedData_ServiceModel4000_ServiceModelService4000Ex1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_ServiceModel4000_ServiceModelService4000Ex6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetCalls sets the value of Calls for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertyCalls(value uint32) (err error) {
	return instance.SetProperty("Calls", (value))
}

// GetCalls gets the value of Calls for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertyCalls() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertyCallsDuration(value uint32) (err error) {
	return instance.SetProperty("CallsDuration", (value))
}

// GetCallsDuration gets the value of CallsDuration for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertyCallsDuration() (value uint32, err error) {
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

// SetCallsFailed sets the value of CallsFailed for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertyCallsFailed(value uint32) (err error) {
	return instance.SetProperty("CallsFailed", (value))
}

// GetCallsFailed gets the value of CallsFailed for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertyCallsFailed() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertyCallsFailedPerSecond(value uint32) (err error) {
	return instance.SetProperty("CallsFailedPerSecond", (value))
}

// GetCallsFailedPerSecond gets the value of CallsFailedPerSecond for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertyCallsFailedPerSecond() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertyCallsFaulted(value uint32) (err error) {
	return instance.SetProperty("CallsFaulted", (value))
}

// GetCallsFaulted gets the value of CallsFaulted for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertyCallsFaulted() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertyCallsFaultedPerSecond(value uint32) (err error) {
	return instance.SetProperty("CallsFaultedPerSecond", (value))
}

// GetCallsFaultedPerSecond gets the value of CallsFaultedPerSecond for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertyCallsFaultedPerSecond() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertyCallsOutstanding(value uint32) (err error) {
	return instance.SetProperty("CallsOutstanding", (value))
}

// GetCallsOutstanding gets the value of CallsOutstanding for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertyCallsOutstanding() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertyCallsPerSecond(value uint32) (err error) {
	return instance.SetProperty("CallsPerSecond", (value))
}

// GetCallsPerSecond gets the value of CallsPerSecond for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertyCallsPerSecond() (value uint32, err error) {
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

// SetInstances sets the value of Instances for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertyInstances(value uint32) (err error) {
	return instance.SetProperty("Instances", (value))
}

// GetInstances gets the value of Instances for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertyInstances() (value uint32, err error) {
	retValue, err := instance.GetProperty("Instances")
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

// SetInstancesCreatedPerSecond sets the value of InstancesCreatedPerSecond for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertyInstancesCreatedPerSecond(value uint32) (err error) {
	return instance.SetProperty("InstancesCreatedPerSecond", (value))
}

// GetInstancesCreatedPerSecond gets the value of InstancesCreatedPerSecond for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertyInstancesCreatedPerSecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("InstancesCreatedPerSecond")
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

// SetPercentOfMaxConcurrentCalls sets the value of PercentOfMaxConcurrentCalls for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertyPercentOfMaxConcurrentCalls(value uint32) (err error) {
	return instance.SetProperty("PercentOfMaxConcurrentCalls", (value))
}

// GetPercentOfMaxConcurrentCalls gets the value of PercentOfMaxConcurrentCalls for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertyPercentOfMaxConcurrentCalls() (value uint32, err error) {
	retValue, err := instance.GetProperty("PercentOfMaxConcurrentCalls")
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

// SetPercentOfMaxConcurrentInstances sets the value of PercentOfMaxConcurrentInstances for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertyPercentOfMaxConcurrentInstances(value uint32) (err error) {
	return instance.SetProperty("PercentOfMaxConcurrentInstances", (value))
}

// GetPercentOfMaxConcurrentInstances gets the value of PercentOfMaxConcurrentInstances for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertyPercentOfMaxConcurrentInstances() (value uint32, err error) {
	retValue, err := instance.GetProperty("PercentOfMaxConcurrentInstances")
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

// SetPercentOfMaxConcurrentSessions sets the value of PercentOfMaxConcurrentSessions for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertyPercentOfMaxConcurrentSessions(value uint32) (err error) {
	return instance.SetProperty("PercentOfMaxConcurrentSessions", (value))
}

// GetPercentOfMaxConcurrentSessions gets the value of PercentOfMaxConcurrentSessions for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertyPercentOfMaxConcurrentSessions() (value uint32, err error) {
	retValue, err := instance.GetProperty("PercentOfMaxConcurrentSessions")
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

// SetQueuedMessagesDropped sets the value of QueuedMessagesDropped for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertyQueuedMessagesDropped(value uint32) (err error) {
	return instance.SetProperty("QueuedMessagesDropped", (value))
}

// GetQueuedMessagesDropped gets the value of QueuedMessagesDropped for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertyQueuedMessagesDropped() (value uint32, err error) {
	retValue, err := instance.GetProperty("QueuedMessagesDropped")
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

// SetQueuedMessagesDroppedPerSecond sets the value of QueuedMessagesDroppedPerSecond for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertyQueuedMessagesDroppedPerSecond(value uint32) (err error) {
	return instance.SetProperty("QueuedMessagesDroppedPerSecond", (value))
}

// GetQueuedMessagesDroppedPerSecond gets the value of QueuedMessagesDroppedPerSecond for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertyQueuedMessagesDroppedPerSecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("QueuedMessagesDroppedPerSecond")
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

// SetQueuedMessagesRejected sets the value of QueuedMessagesRejected for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertyQueuedMessagesRejected(value uint32) (err error) {
	return instance.SetProperty("QueuedMessagesRejected", (value))
}

// GetQueuedMessagesRejected gets the value of QueuedMessagesRejected for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertyQueuedMessagesRejected() (value uint32, err error) {
	retValue, err := instance.GetProperty("QueuedMessagesRejected")
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

// SetQueuedMessagesRejectedPerSecond sets the value of QueuedMessagesRejectedPerSecond for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertyQueuedMessagesRejectedPerSecond(value uint32) (err error) {
	return instance.SetProperty("QueuedMessagesRejectedPerSecond", (value))
}

// GetQueuedMessagesRejectedPerSecond gets the value of QueuedMessagesRejectedPerSecond for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertyQueuedMessagesRejectedPerSecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("QueuedMessagesRejectedPerSecond")
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

// SetQueuedPoisonMessages sets the value of QueuedPoisonMessages for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertyQueuedPoisonMessages(value uint32) (err error) {
	return instance.SetProperty("QueuedPoisonMessages", (value))
}

// GetQueuedPoisonMessages gets the value of QueuedPoisonMessages for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertyQueuedPoisonMessages() (value uint32, err error) {
	retValue, err := instance.GetProperty("QueuedPoisonMessages")
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

// SetQueuedPoisonMessagesPerSecond sets the value of QueuedPoisonMessagesPerSecond for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertyQueuedPoisonMessagesPerSecond(value uint32) (err error) {
	return instance.SetProperty("QueuedPoisonMessagesPerSecond", (value))
}

// GetQueuedPoisonMessagesPerSecond gets the value of QueuedPoisonMessagesPerSecond for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertyQueuedPoisonMessagesPerSecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("QueuedPoisonMessagesPerSecond")
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
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertyReliableMessagingMessagesDropped(value uint32) (err error) {
	return instance.SetProperty("ReliableMessagingMessagesDropped", (value))
}

// GetReliableMessagingMessagesDropped gets the value of ReliableMessagingMessagesDropped for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertyReliableMessagingMessagesDropped() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertyReliableMessagingMessagesDroppedPerSecond(value uint32) (err error) {
	return instance.SetProperty("ReliableMessagingMessagesDroppedPerSecond", (value))
}

// GetReliableMessagingMessagesDroppedPerSecond gets the value of ReliableMessagingMessagesDroppedPerSecond for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertyReliableMessagingMessagesDroppedPerSecond() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertyReliableMessagingSessionsFaulted(value uint32) (err error) {
	return instance.SetProperty("ReliableMessagingSessionsFaulted", (value))
}

// GetReliableMessagingSessionsFaulted gets the value of ReliableMessagingSessionsFaulted for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertyReliableMessagingSessionsFaulted() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertyReliableMessagingSessionsFaultedPerSecond(value uint32) (err error) {
	return instance.SetProperty("ReliableMessagingSessionsFaultedPerSecond", (value))
}

// GetReliableMessagingSessionsFaultedPerSecond gets the value of ReliableMessagingSessionsFaultedPerSecond for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertyReliableMessagingSessionsFaultedPerSecond() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertySecurityCallsNotAuthorized(value uint32) (err error) {
	return instance.SetProperty("SecurityCallsNotAuthorized", (value))
}

// GetSecurityCallsNotAuthorized gets the value of SecurityCallsNotAuthorized for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertySecurityCallsNotAuthorized() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertySecurityCallsNotAuthorizedPerSecond(value uint32) (err error) {
	return instance.SetProperty("SecurityCallsNotAuthorizedPerSecond", (value))
}

// GetSecurityCallsNotAuthorizedPerSecond gets the value of SecurityCallsNotAuthorizedPerSecond for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertySecurityCallsNotAuthorizedPerSecond() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertySecurityValidationandAuthenticationFailures(value uint32) (err error) {
	return instance.SetProperty("SecurityValidationandAuthenticationFailures", (value))
}

// GetSecurityValidationandAuthenticationFailures gets the value of SecurityValidationandAuthenticationFailures for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertySecurityValidationandAuthenticationFailures() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertySecurityValidationandAuthenticationFailuresPerSecond(value uint32) (err error) {
	return instance.SetProperty("SecurityValidationandAuthenticationFailuresPerSecond", (value))
}

// GetSecurityValidationandAuthenticationFailuresPerSecond gets the value of SecurityValidationandAuthenticationFailuresPerSecond for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertySecurityValidationandAuthenticationFailuresPerSecond() (value uint32, err error) {
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

// SetTransactedOperationsAborted sets the value of TransactedOperationsAborted for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertyTransactedOperationsAborted(value uint32) (err error) {
	return instance.SetProperty("TransactedOperationsAborted", (value))
}

// GetTransactedOperationsAborted gets the value of TransactedOperationsAborted for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertyTransactedOperationsAborted() (value uint32, err error) {
	retValue, err := instance.GetProperty("TransactedOperationsAborted")
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

// SetTransactedOperationsAbortedPerSecond sets the value of TransactedOperationsAbortedPerSecond for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertyTransactedOperationsAbortedPerSecond(value uint32) (err error) {
	return instance.SetProperty("TransactedOperationsAbortedPerSecond", (value))
}

// GetTransactedOperationsAbortedPerSecond gets the value of TransactedOperationsAbortedPerSecond for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertyTransactedOperationsAbortedPerSecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("TransactedOperationsAbortedPerSecond")
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

// SetTransactedOperationsCommitted sets the value of TransactedOperationsCommitted for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertyTransactedOperationsCommitted(value uint32) (err error) {
	return instance.SetProperty("TransactedOperationsCommitted", (value))
}

// GetTransactedOperationsCommitted gets the value of TransactedOperationsCommitted for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertyTransactedOperationsCommitted() (value uint32, err error) {
	retValue, err := instance.GetProperty("TransactedOperationsCommitted")
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

// SetTransactedOperationsCommittedPerSecond sets the value of TransactedOperationsCommittedPerSecond for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertyTransactedOperationsCommittedPerSecond(value uint32) (err error) {
	return instance.SetProperty("TransactedOperationsCommittedPerSecond", (value))
}

// GetTransactedOperationsCommittedPerSecond gets the value of TransactedOperationsCommittedPerSecond for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertyTransactedOperationsCommittedPerSecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("TransactedOperationsCommittedPerSecond")
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

// SetTransactedOperationsInDoubt sets the value of TransactedOperationsInDoubt for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertyTransactedOperationsInDoubt(value uint32) (err error) {
	return instance.SetProperty("TransactedOperationsInDoubt", (value))
}

// GetTransactedOperationsInDoubt gets the value of TransactedOperationsInDoubt for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertyTransactedOperationsInDoubt() (value uint32, err error) {
	retValue, err := instance.GetProperty("TransactedOperationsInDoubt")
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

// SetTransactedOperationsInDoubtPerSecond sets the value of TransactedOperationsInDoubtPerSecond for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertyTransactedOperationsInDoubtPerSecond(value uint32) (err error) {
	return instance.SetProperty("TransactedOperationsInDoubtPerSecond", (value))
}

// GetTransactedOperationsInDoubtPerSecond gets the value of TransactedOperationsInDoubtPerSecond for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertyTransactedOperationsInDoubtPerSecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("TransactedOperationsInDoubtPerSecond")
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
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertyTransactionsFlowed(value uint32) (err error) {
	return instance.SetProperty("TransactionsFlowed", (value))
}

// GetTransactionsFlowed gets the value of TransactionsFlowed for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertyTransactionsFlowed() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) SetPropertyTransactionsFlowedPerSecond(value uint32) (err error) {
	return instance.SetProperty("TransactionsFlowedPerSecond", (value))
}

// GetTransactionsFlowedPerSecond gets the value of TransactionsFlowedPerSecond for the instance
func (instance *Win32_PerfFormattedData_ServiceModel4000_ServiceModelService4000) GetPropertyTransactionsFlowedPerSecond() (value uint32, err error) {
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
