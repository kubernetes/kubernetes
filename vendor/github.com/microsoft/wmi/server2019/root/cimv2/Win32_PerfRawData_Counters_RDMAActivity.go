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

// Win32_PerfRawData_Counters_RDMAActivity struct
type Win32_PerfRawData_Counters_RDMAActivity struct {
	*Win32_PerfRawData

	//
	RDMAAcceptedConnections uint32

	//
	RDMAActiveConnections uint32

	//
	RDMACompletionQueueErrors uint32

	//
	RDMAConnectionErrors uint32

	//
	RDMAFailedConnectionAttempts uint32

	//
	RDMAInboundBytesPersec uint64

	//
	RDMAInboundFramesPersec uint64

	//
	RDMAInitiatedConnections uint32

	//
	RDMAOutboundBytesPersec uint64

	//
	RDMAOutboundFramesPersec uint64
}

func NewWin32_PerfRawData_Counters_RDMAActivityEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_Counters_RDMAActivity, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_RDMAActivity{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_Counters_RDMAActivityEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_Counters_RDMAActivity, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_RDMAActivity{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetRDMAAcceptedConnections sets the value of RDMAAcceptedConnections for the instance
func (instance *Win32_PerfRawData_Counters_RDMAActivity) SetPropertyRDMAAcceptedConnections(value uint32) (err error) {
	return instance.SetProperty("RDMAAcceptedConnections", (value))
}

// GetRDMAAcceptedConnections gets the value of RDMAAcceptedConnections for the instance
func (instance *Win32_PerfRawData_Counters_RDMAActivity) GetPropertyRDMAAcceptedConnections() (value uint32, err error) {
	retValue, err := instance.GetProperty("RDMAAcceptedConnections")
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

// SetRDMAActiveConnections sets the value of RDMAActiveConnections for the instance
func (instance *Win32_PerfRawData_Counters_RDMAActivity) SetPropertyRDMAActiveConnections(value uint32) (err error) {
	return instance.SetProperty("RDMAActiveConnections", (value))
}

// GetRDMAActiveConnections gets the value of RDMAActiveConnections for the instance
func (instance *Win32_PerfRawData_Counters_RDMAActivity) GetPropertyRDMAActiveConnections() (value uint32, err error) {
	retValue, err := instance.GetProperty("RDMAActiveConnections")
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

// SetRDMACompletionQueueErrors sets the value of RDMACompletionQueueErrors for the instance
func (instance *Win32_PerfRawData_Counters_RDMAActivity) SetPropertyRDMACompletionQueueErrors(value uint32) (err error) {
	return instance.SetProperty("RDMACompletionQueueErrors", (value))
}

// GetRDMACompletionQueueErrors gets the value of RDMACompletionQueueErrors for the instance
func (instance *Win32_PerfRawData_Counters_RDMAActivity) GetPropertyRDMACompletionQueueErrors() (value uint32, err error) {
	retValue, err := instance.GetProperty("RDMACompletionQueueErrors")
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

// SetRDMAConnectionErrors sets the value of RDMAConnectionErrors for the instance
func (instance *Win32_PerfRawData_Counters_RDMAActivity) SetPropertyRDMAConnectionErrors(value uint32) (err error) {
	return instance.SetProperty("RDMAConnectionErrors", (value))
}

// GetRDMAConnectionErrors gets the value of RDMAConnectionErrors for the instance
func (instance *Win32_PerfRawData_Counters_RDMAActivity) GetPropertyRDMAConnectionErrors() (value uint32, err error) {
	retValue, err := instance.GetProperty("RDMAConnectionErrors")
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

// SetRDMAFailedConnectionAttempts sets the value of RDMAFailedConnectionAttempts for the instance
func (instance *Win32_PerfRawData_Counters_RDMAActivity) SetPropertyRDMAFailedConnectionAttempts(value uint32) (err error) {
	return instance.SetProperty("RDMAFailedConnectionAttempts", (value))
}

// GetRDMAFailedConnectionAttempts gets the value of RDMAFailedConnectionAttempts for the instance
func (instance *Win32_PerfRawData_Counters_RDMAActivity) GetPropertyRDMAFailedConnectionAttempts() (value uint32, err error) {
	retValue, err := instance.GetProperty("RDMAFailedConnectionAttempts")
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

// SetRDMAInboundBytesPersec sets the value of RDMAInboundBytesPersec for the instance
func (instance *Win32_PerfRawData_Counters_RDMAActivity) SetPropertyRDMAInboundBytesPersec(value uint64) (err error) {
	return instance.SetProperty("RDMAInboundBytesPersec", (value))
}

// GetRDMAInboundBytesPersec gets the value of RDMAInboundBytesPersec for the instance
func (instance *Win32_PerfRawData_Counters_RDMAActivity) GetPropertyRDMAInboundBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("RDMAInboundBytesPersec")
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

// SetRDMAInboundFramesPersec sets the value of RDMAInboundFramesPersec for the instance
func (instance *Win32_PerfRawData_Counters_RDMAActivity) SetPropertyRDMAInboundFramesPersec(value uint64) (err error) {
	return instance.SetProperty("RDMAInboundFramesPersec", (value))
}

// GetRDMAInboundFramesPersec gets the value of RDMAInboundFramesPersec for the instance
func (instance *Win32_PerfRawData_Counters_RDMAActivity) GetPropertyRDMAInboundFramesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("RDMAInboundFramesPersec")
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

// SetRDMAInitiatedConnections sets the value of RDMAInitiatedConnections for the instance
func (instance *Win32_PerfRawData_Counters_RDMAActivity) SetPropertyRDMAInitiatedConnections(value uint32) (err error) {
	return instance.SetProperty("RDMAInitiatedConnections", (value))
}

// GetRDMAInitiatedConnections gets the value of RDMAInitiatedConnections for the instance
func (instance *Win32_PerfRawData_Counters_RDMAActivity) GetPropertyRDMAInitiatedConnections() (value uint32, err error) {
	retValue, err := instance.GetProperty("RDMAInitiatedConnections")
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

// SetRDMAOutboundBytesPersec sets the value of RDMAOutboundBytesPersec for the instance
func (instance *Win32_PerfRawData_Counters_RDMAActivity) SetPropertyRDMAOutboundBytesPersec(value uint64) (err error) {
	return instance.SetProperty("RDMAOutboundBytesPersec", (value))
}

// GetRDMAOutboundBytesPersec gets the value of RDMAOutboundBytesPersec for the instance
func (instance *Win32_PerfRawData_Counters_RDMAActivity) GetPropertyRDMAOutboundBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("RDMAOutboundBytesPersec")
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

// SetRDMAOutboundFramesPersec sets the value of RDMAOutboundFramesPersec for the instance
func (instance *Win32_PerfRawData_Counters_RDMAActivity) SetPropertyRDMAOutboundFramesPersec(value uint64) (err error) {
	return instance.SetProperty("RDMAOutboundFramesPersec", (value))
}

// GetRDMAOutboundFramesPersec gets the value of RDMAOutboundFramesPersec for the instance
func (instance *Win32_PerfRawData_Counters_RDMAActivity) GetPropertyRDMAOutboundFramesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("RDMAOutboundFramesPersec")
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
