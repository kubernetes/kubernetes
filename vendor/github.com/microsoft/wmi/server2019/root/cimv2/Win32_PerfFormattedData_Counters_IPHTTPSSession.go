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

// Win32_PerfFormattedData_Counters_IPHTTPSSession struct
type Win32_PerfFormattedData_Counters_IPHTTPSSession struct {
	*Win32_PerfFormattedData

	//
	Bytesreceivedonthissession uint64

	//
	Bytessentonthissession uint64

	//
	DurationDurationofthesessionSeconds uint64

	//
	ErrorsReceiveerrorsonthissession uint64

	//
	ErrorsTransmiterrorsonthissession uint64

	//
	Packetsreceivedonthissession uint64

	//
	Packetssentonthissession uint64
}

func NewWin32_PerfFormattedData_Counters_IPHTTPSSessionEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_IPHTTPSSession, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_IPHTTPSSession{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_IPHTTPSSessionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_IPHTTPSSession, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_IPHTTPSSession{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetBytesreceivedonthissession sets the value of Bytesreceivedonthissession for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSSession) SetPropertyBytesreceivedonthissession(value uint64) (err error) {
	return instance.SetProperty("Bytesreceivedonthissession", (value))
}

// GetBytesreceivedonthissession gets the value of Bytesreceivedonthissession for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSSession) GetPropertyBytesreceivedonthissession() (value uint64, err error) {
	retValue, err := instance.GetProperty("Bytesreceivedonthissession")
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

// SetBytessentonthissession sets the value of Bytessentonthissession for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSSession) SetPropertyBytessentonthissession(value uint64) (err error) {
	return instance.SetProperty("Bytessentonthissession", (value))
}

// GetBytessentonthissession gets the value of Bytessentonthissession for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSSession) GetPropertyBytessentonthissession() (value uint64, err error) {
	retValue, err := instance.GetProperty("Bytessentonthissession")
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

// SetDurationDurationofthesessionSeconds sets the value of DurationDurationofthesessionSeconds for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSSession) SetPropertyDurationDurationofthesessionSeconds(value uint64) (err error) {
	return instance.SetProperty("DurationDurationofthesessionSeconds", (value))
}

// GetDurationDurationofthesessionSeconds gets the value of DurationDurationofthesessionSeconds for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSSession) GetPropertyDurationDurationofthesessionSeconds() (value uint64, err error) {
	retValue, err := instance.GetProperty("DurationDurationofthesessionSeconds")
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

// SetErrorsReceiveerrorsonthissession sets the value of ErrorsReceiveerrorsonthissession for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSSession) SetPropertyErrorsReceiveerrorsonthissession(value uint64) (err error) {
	return instance.SetProperty("ErrorsReceiveerrorsonthissession", (value))
}

// GetErrorsReceiveerrorsonthissession gets the value of ErrorsReceiveerrorsonthissession for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSSession) GetPropertyErrorsReceiveerrorsonthissession() (value uint64, err error) {
	retValue, err := instance.GetProperty("ErrorsReceiveerrorsonthissession")
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

// SetErrorsTransmiterrorsonthissession sets the value of ErrorsTransmiterrorsonthissession for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSSession) SetPropertyErrorsTransmiterrorsonthissession(value uint64) (err error) {
	return instance.SetProperty("ErrorsTransmiterrorsonthissession", (value))
}

// GetErrorsTransmiterrorsonthissession gets the value of ErrorsTransmiterrorsonthissession for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSSession) GetPropertyErrorsTransmiterrorsonthissession() (value uint64, err error) {
	retValue, err := instance.GetProperty("ErrorsTransmiterrorsonthissession")
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

// SetPacketsreceivedonthissession sets the value of Packetsreceivedonthissession for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSSession) SetPropertyPacketsreceivedonthissession(value uint64) (err error) {
	return instance.SetProperty("Packetsreceivedonthissession", (value))
}

// GetPacketsreceivedonthissession gets the value of Packetsreceivedonthissession for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSSession) GetPropertyPacketsreceivedonthissession() (value uint64, err error) {
	retValue, err := instance.GetProperty("Packetsreceivedonthissession")
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

// SetPacketssentonthissession sets the value of Packetssentonthissession for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSSession) SetPropertyPacketssentonthissession(value uint64) (err error) {
	return instance.SetProperty("Packetssentonthissession", (value))
}

// GetPacketssentonthissession gets the value of Packetssentonthissession for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSSession) GetPropertyPacketssentonthissession() (value uint64, err error) {
	retValue, err := instance.GetProperty("Packetssentonthissession")
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
