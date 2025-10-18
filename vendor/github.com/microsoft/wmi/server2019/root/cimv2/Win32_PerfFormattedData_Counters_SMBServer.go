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

// Win32_PerfFormattedData_Counters_SMBServer struct
type Win32_PerfFormattedData_Counters_SMBServer struct {
	*Win32_PerfFormattedData

	//
	ReadBytesPersec uint64

	//
	ReadRequestsPersec uint64

	//
	ReceiveBytesPersec uint64

	//
	SendBytesPersec uint64

	//
	WriteBytesPersec uint64

	//
	WriteRequestsPersec uint64
}

func NewWin32_PerfFormattedData_Counters_SMBServerEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_SMBServer, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_SMBServer{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_SMBServerEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_SMBServer, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_SMBServer{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetReadBytesPersec sets the value of ReadBytesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_SMBServer) SetPropertyReadBytesPersec(value uint64) (err error) {
	return instance.SetProperty("ReadBytesPersec", (value))
}

// GetReadBytesPersec gets the value of ReadBytesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_SMBServer) GetPropertyReadBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadBytesPersec")
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

// SetReadRequestsPersec sets the value of ReadRequestsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_SMBServer) SetPropertyReadRequestsPersec(value uint64) (err error) {
	return instance.SetProperty("ReadRequestsPersec", (value))
}

// GetReadRequestsPersec gets the value of ReadRequestsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_SMBServer) GetPropertyReadRequestsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadRequestsPersec")
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

// SetReceiveBytesPersec sets the value of ReceiveBytesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_SMBServer) SetPropertyReceiveBytesPersec(value uint64) (err error) {
	return instance.SetProperty("ReceiveBytesPersec", (value))
}

// GetReceiveBytesPersec gets the value of ReceiveBytesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_SMBServer) GetPropertyReceiveBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReceiveBytesPersec")
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

// SetSendBytesPersec sets the value of SendBytesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_SMBServer) SetPropertySendBytesPersec(value uint64) (err error) {
	return instance.SetProperty("SendBytesPersec", (value))
}

// GetSendBytesPersec gets the value of SendBytesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_SMBServer) GetPropertySendBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("SendBytesPersec")
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

// SetWriteBytesPersec sets the value of WriteBytesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_SMBServer) SetPropertyWriteBytesPersec(value uint64) (err error) {
	return instance.SetProperty("WriteBytesPersec", (value))
}

// GetWriteBytesPersec gets the value of WriteBytesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_SMBServer) GetPropertyWriteBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteBytesPersec")
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

// SetWriteRequestsPersec sets the value of WriteRequestsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_SMBServer) SetPropertyWriteRequestsPersec(value uint64) (err error) {
	return instance.SetProperty("WriteRequestsPersec", (value))
}

// GetWriteRequestsPersec gets the value of WriteRequestsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_SMBServer) GetPropertyWriteRequestsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteRequestsPersec")
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
