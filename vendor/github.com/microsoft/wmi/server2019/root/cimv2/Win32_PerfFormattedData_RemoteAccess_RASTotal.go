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

// Win32_PerfFormattedData_RemoteAccess_RASTotal struct
type Win32_PerfFormattedData_RemoteAccess_RASTotal struct {
	*Win32_PerfFormattedData

	//
	AlignmentErrors uint32

	//
	BufferOverrunErrors uint32

	//
	BytesReceived uint64

	//
	BytesReceivedPerSec uint32

	//
	BytesTransmitted uint64

	//
	BytesTransmittedPerSec uint32

	//
	CRCErrors uint32

	//
	FramesReceived uint32

	//
	FramesReceivedPerSec uint32

	//
	FramesTransmitted uint32

	//
	FramesTransmittedPerSec uint32

	//
	PercentCompressionIn uint32

	//
	PercentCompressionOut uint32

	//
	SerialOverrunErrors uint32

	//
	TimeoutErrors uint32

	//
	TotalConnections uint32

	//
	TotalErrors uint32

	//
	TotalErrorsPerSec uint32
}

func NewWin32_PerfFormattedData_RemoteAccess_RASTotalEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_RemoteAccess_RASTotal, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_RemoteAccess_RASTotal{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_RemoteAccess_RASTotalEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_RemoteAccess_RASTotal, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_RemoteAccess_RASTotal{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetAlignmentErrors sets the value of AlignmentErrors for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) SetPropertyAlignmentErrors(value uint32) (err error) {
	return instance.SetProperty("AlignmentErrors", (value))
}

// GetAlignmentErrors gets the value of AlignmentErrors for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) GetPropertyAlignmentErrors() (value uint32, err error) {
	retValue, err := instance.GetProperty("AlignmentErrors")
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

// SetBufferOverrunErrors sets the value of BufferOverrunErrors for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) SetPropertyBufferOverrunErrors(value uint32) (err error) {
	return instance.SetProperty("BufferOverrunErrors", (value))
}

// GetBufferOverrunErrors gets the value of BufferOverrunErrors for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) GetPropertyBufferOverrunErrors() (value uint32, err error) {
	retValue, err := instance.GetProperty("BufferOverrunErrors")
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

// SetBytesReceived sets the value of BytesReceived for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) SetPropertyBytesReceived(value uint64) (err error) {
	return instance.SetProperty("BytesReceived", (value))
}

// GetBytesReceived gets the value of BytesReceived for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) GetPropertyBytesReceived() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesReceived")
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

// SetBytesReceivedPerSec sets the value of BytesReceivedPerSec for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) SetPropertyBytesReceivedPerSec(value uint32) (err error) {
	return instance.SetProperty("BytesReceivedPerSec", (value))
}

// GetBytesReceivedPerSec gets the value of BytesReceivedPerSec for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) GetPropertyBytesReceivedPerSec() (value uint32, err error) {
	retValue, err := instance.GetProperty("BytesReceivedPerSec")
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

// SetBytesTransmitted sets the value of BytesTransmitted for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) SetPropertyBytesTransmitted(value uint64) (err error) {
	return instance.SetProperty("BytesTransmitted", (value))
}

// GetBytesTransmitted gets the value of BytesTransmitted for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) GetPropertyBytesTransmitted() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesTransmitted")
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

// SetBytesTransmittedPerSec sets the value of BytesTransmittedPerSec for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) SetPropertyBytesTransmittedPerSec(value uint32) (err error) {
	return instance.SetProperty("BytesTransmittedPerSec", (value))
}

// GetBytesTransmittedPerSec gets the value of BytesTransmittedPerSec for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) GetPropertyBytesTransmittedPerSec() (value uint32, err error) {
	retValue, err := instance.GetProperty("BytesTransmittedPerSec")
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

// SetCRCErrors sets the value of CRCErrors for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) SetPropertyCRCErrors(value uint32) (err error) {
	return instance.SetProperty("CRCErrors", (value))
}

// GetCRCErrors gets the value of CRCErrors for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) GetPropertyCRCErrors() (value uint32, err error) {
	retValue, err := instance.GetProperty("CRCErrors")
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

// SetFramesReceived sets the value of FramesReceived for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) SetPropertyFramesReceived(value uint32) (err error) {
	return instance.SetProperty("FramesReceived", (value))
}

// GetFramesReceived gets the value of FramesReceived for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) GetPropertyFramesReceived() (value uint32, err error) {
	retValue, err := instance.GetProperty("FramesReceived")
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

// SetFramesReceivedPerSec sets the value of FramesReceivedPerSec for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) SetPropertyFramesReceivedPerSec(value uint32) (err error) {
	return instance.SetProperty("FramesReceivedPerSec", (value))
}

// GetFramesReceivedPerSec gets the value of FramesReceivedPerSec for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) GetPropertyFramesReceivedPerSec() (value uint32, err error) {
	retValue, err := instance.GetProperty("FramesReceivedPerSec")
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

// SetFramesTransmitted sets the value of FramesTransmitted for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) SetPropertyFramesTransmitted(value uint32) (err error) {
	return instance.SetProperty("FramesTransmitted", (value))
}

// GetFramesTransmitted gets the value of FramesTransmitted for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) GetPropertyFramesTransmitted() (value uint32, err error) {
	retValue, err := instance.GetProperty("FramesTransmitted")
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

// SetFramesTransmittedPerSec sets the value of FramesTransmittedPerSec for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) SetPropertyFramesTransmittedPerSec(value uint32) (err error) {
	return instance.SetProperty("FramesTransmittedPerSec", (value))
}

// GetFramesTransmittedPerSec gets the value of FramesTransmittedPerSec for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) GetPropertyFramesTransmittedPerSec() (value uint32, err error) {
	retValue, err := instance.GetProperty("FramesTransmittedPerSec")
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

// SetPercentCompressionIn sets the value of PercentCompressionIn for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) SetPropertyPercentCompressionIn(value uint32) (err error) {
	return instance.SetProperty("PercentCompressionIn", (value))
}

// GetPercentCompressionIn gets the value of PercentCompressionIn for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) GetPropertyPercentCompressionIn() (value uint32, err error) {
	retValue, err := instance.GetProperty("PercentCompressionIn")
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

// SetPercentCompressionOut sets the value of PercentCompressionOut for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) SetPropertyPercentCompressionOut(value uint32) (err error) {
	return instance.SetProperty("PercentCompressionOut", (value))
}

// GetPercentCompressionOut gets the value of PercentCompressionOut for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) GetPropertyPercentCompressionOut() (value uint32, err error) {
	retValue, err := instance.GetProperty("PercentCompressionOut")
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

// SetSerialOverrunErrors sets the value of SerialOverrunErrors for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) SetPropertySerialOverrunErrors(value uint32) (err error) {
	return instance.SetProperty("SerialOverrunErrors", (value))
}

// GetSerialOverrunErrors gets the value of SerialOverrunErrors for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) GetPropertySerialOverrunErrors() (value uint32, err error) {
	retValue, err := instance.GetProperty("SerialOverrunErrors")
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

// SetTimeoutErrors sets the value of TimeoutErrors for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) SetPropertyTimeoutErrors(value uint32) (err error) {
	return instance.SetProperty("TimeoutErrors", (value))
}

// GetTimeoutErrors gets the value of TimeoutErrors for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) GetPropertyTimeoutErrors() (value uint32, err error) {
	retValue, err := instance.GetProperty("TimeoutErrors")
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

// SetTotalConnections sets the value of TotalConnections for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) SetPropertyTotalConnections(value uint32) (err error) {
	return instance.SetProperty("TotalConnections", (value))
}

// GetTotalConnections gets the value of TotalConnections for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) GetPropertyTotalConnections() (value uint32, err error) {
	retValue, err := instance.GetProperty("TotalConnections")
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

// SetTotalErrors sets the value of TotalErrors for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) SetPropertyTotalErrors(value uint32) (err error) {
	return instance.SetProperty("TotalErrors", (value))
}

// GetTotalErrors gets the value of TotalErrors for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) GetPropertyTotalErrors() (value uint32, err error) {
	retValue, err := instance.GetProperty("TotalErrors")
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

// SetTotalErrorsPerSec sets the value of TotalErrorsPerSec for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) SetPropertyTotalErrorsPerSec(value uint32) (err error) {
	return instance.SetProperty("TotalErrorsPerSec", (value))
}

// GetTotalErrorsPerSec gets the value of TotalErrorsPerSec for the instance
func (instance *Win32_PerfFormattedData_RemoteAccess_RASTotal) GetPropertyTotalErrorsPerSec() (value uint32, err error) {
	retValue, err := instance.GetProperty("TotalErrorsPerSec")
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
