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

// Win32_PerfRawData_Counters_SMBDirectConnection struct
type Win32_PerfRawData_Counters_SMBDirectConnection struct {
	*Win32_PerfRawData

	//
	BytesRDMAReadPersec uint64

	//
	BytesRDMAWrittenPersec uint64

	//
	BytesReceivedPersec uint64

	//
	BytesSentPersec uint64

	//
	MemoryRegions uint32

	//
	RCQNotificationEventsPersec uint32

	//
	RDMARegistrationsPersec uint32

	//
	ReceivesPersec uint32

	//
	RemoteInvalidationsPersec uint32

	//
	SCQNotificationEventsPersec uint32

	//
	SendsPersec uint32

	//
	StallsRDMAReadPersec uint32

	//
	StallsRDMARegistrationsPersec uint32

	//
	StallsSendCreditPersec uint32

	//
	StallsSendQueuePersec uint32
}

func NewWin32_PerfRawData_Counters_SMBDirectConnectionEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_Counters_SMBDirectConnection, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_SMBDirectConnection{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_Counters_SMBDirectConnectionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_Counters_SMBDirectConnection, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_SMBDirectConnection{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetBytesRDMAReadPersec sets the value of BytesRDMAReadPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBDirectConnection) SetPropertyBytesRDMAReadPersec(value uint64) (err error) {
	return instance.SetProperty("BytesRDMAReadPersec", (value))
}

// GetBytesRDMAReadPersec gets the value of BytesRDMAReadPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBDirectConnection) GetPropertyBytesRDMAReadPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesRDMAReadPersec")
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

// SetBytesRDMAWrittenPersec sets the value of BytesRDMAWrittenPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBDirectConnection) SetPropertyBytesRDMAWrittenPersec(value uint64) (err error) {
	return instance.SetProperty("BytesRDMAWrittenPersec", (value))
}

// GetBytesRDMAWrittenPersec gets the value of BytesRDMAWrittenPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBDirectConnection) GetPropertyBytesRDMAWrittenPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesRDMAWrittenPersec")
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

// SetBytesReceivedPersec sets the value of BytesReceivedPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBDirectConnection) SetPropertyBytesReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("BytesReceivedPersec", (value))
}

// GetBytesReceivedPersec gets the value of BytesReceivedPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBDirectConnection) GetPropertyBytesReceivedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesReceivedPersec")
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

// SetBytesSentPersec sets the value of BytesSentPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBDirectConnection) SetPropertyBytesSentPersec(value uint64) (err error) {
	return instance.SetProperty("BytesSentPersec", (value))
}

// GetBytesSentPersec gets the value of BytesSentPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBDirectConnection) GetPropertyBytesSentPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesSentPersec")
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

// SetMemoryRegions sets the value of MemoryRegions for the instance
func (instance *Win32_PerfRawData_Counters_SMBDirectConnection) SetPropertyMemoryRegions(value uint32) (err error) {
	return instance.SetProperty("MemoryRegions", (value))
}

// GetMemoryRegions gets the value of MemoryRegions for the instance
func (instance *Win32_PerfRawData_Counters_SMBDirectConnection) GetPropertyMemoryRegions() (value uint32, err error) {
	retValue, err := instance.GetProperty("MemoryRegions")
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

// SetRCQNotificationEventsPersec sets the value of RCQNotificationEventsPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBDirectConnection) SetPropertyRCQNotificationEventsPersec(value uint32) (err error) {
	return instance.SetProperty("RCQNotificationEventsPersec", (value))
}

// GetRCQNotificationEventsPersec gets the value of RCQNotificationEventsPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBDirectConnection) GetPropertyRCQNotificationEventsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("RCQNotificationEventsPersec")
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

// SetRDMARegistrationsPersec sets the value of RDMARegistrationsPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBDirectConnection) SetPropertyRDMARegistrationsPersec(value uint32) (err error) {
	return instance.SetProperty("RDMARegistrationsPersec", (value))
}

// GetRDMARegistrationsPersec gets the value of RDMARegistrationsPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBDirectConnection) GetPropertyRDMARegistrationsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("RDMARegistrationsPersec")
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

// SetReceivesPersec sets the value of ReceivesPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBDirectConnection) SetPropertyReceivesPersec(value uint32) (err error) {
	return instance.SetProperty("ReceivesPersec", (value))
}

// GetReceivesPersec gets the value of ReceivesPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBDirectConnection) GetPropertyReceivesPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReceivesPersec")
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

// SetRemoteInvalidationsPersec sets the value of RemoteInvalidationsPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBDirectConnection) SetPropertyRemoteInvalidationsPersec(value uint32) (err error) {
	return instance.SetProperty("RemoteInvalidationsPersec", (value))
}

// GetRemoteInvalidationsPersec gets the value of RemoteInvalidationsPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBDirectConnection) GetPropertyRemoteInvalidationsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("RemoteInvalidationsPersec")
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

// SetSCQNotificationEventsPersec sets the value of SCQNotificationEventsPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBDirectConnection) SetPropertySCQNotificationEventsPersec(value uint32) (err error) {
	return instance.SetProperty("SCQNotificationEventsPersec", (value))
}

// GetSCQNotificationEventsPersec gets the value of SCQNotificationEventsPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBDirectConnection) GetPropertySCQNotificationEventsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("SCQNotificationEventsPersec")
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

// SetSendsPersec sets the value of SendsPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBDirectConnection) SetPropertySendsPersec(value uint32) (err error) {
	return instance.SetProperty("SendsPersec", (value))
}

// GetSendsPersec gets the value of SendsPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBDirectConnection) GetPropertySendsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("SendsPersec")
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

// SetStallsRDMAReadPersec sets the value of StallsRDMAReadPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBDirectConnection) SetPropertyStallsRDMAReadPersec(value uint32) (err error) {
	return instance.SetProperty("StallsRDMAReadPersec", (value))
}

// GetStallsRDMAReadPersec gets the value of StallsRDMAReadPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBDirectConnection) GetPropertyStallsRDMAReadPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("StallsRDMAReadPersec")
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

// SetStallsRDMARegistrationsPersec sets the value of StallsRDMARegistrationsPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBDirectConnection) SetPropertyStallsRDMARegistrationsPersec(value uint32) (err error) {
	return instance.SetProperty("StallsRDMARegistrationsPersec", (value))
}

// GetStallsRDMARegistrationsPersec gets the value of StallsRDMARegistrationsPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBDirectConnection) GetPropertyStallsRDMARegistrationsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("StallsRDMARegistrationsPersec")
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

// SetStallsSendCreditPersec sets the value of StallsSendCreditPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBDirectConnection) SetPropertyStallsSendCreditPersec(value uint32) (err error) {
	return instance.SetProperty("StallsSendCreditPersec", (value))
}

// GetStallsSendCreditPersec gets the value of StallsSendCreditPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBDirectConnection) GetPropertyStallsSendCreditPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("StallsSendCreditPersec")
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

// SetStallsSendQueuePersec sets the value of StallsSendQueuePersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBDirectConnection) SetPropertyStallsSendQueuePersec(value uint32) (err error) {
	return instance.SetProperty("StallsSendQueuePersec", (value))
}

// GetStallsSendQueuePersec gets the value of StallsSendQueuePersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBDirectConnection) GetPropertyStallsSendQueuePersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("StallsSendQueuePersec")
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
