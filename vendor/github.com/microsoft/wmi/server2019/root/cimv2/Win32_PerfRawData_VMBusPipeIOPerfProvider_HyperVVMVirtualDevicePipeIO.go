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

// Win32_PerfRawData_VMBusPipeIOPerfProvider_HyperVVMVirtualDevicePipeIO struct
type Win32_PerfRawData_VMBusPipeIOPerfProvider_HyperVVMVirtualDevicePipeIO struct {
	*Win32_PerfRawData

	//
	ReceiveMessageQuotaExceeded uint64

	//
	ReceiveQoSConformantMessagesPersec uint64

	//
	ReceiveQoSExemptMessagesPersec uint64

	//
	ReceiveQoSNonConformantMessagesPersec uint64

	//
	ReceiveQoSTotalMessageDelayTime100ns uint64
}

func NewWin32_PerfRawData_VMBusPipeIOPerfProvider_HyperVVMVirtualDevicePipeIOEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_VMBusPipeIOPerfProvider_HyperVVMVirtualDevicePipeIO, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_VMBusPipeIOPerfProvider_HyperVVMVirtualDevicePipeIO{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_VMBusPipeIOPerfProvider_HyperVVMVirtualDevicePipeIOEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_VMBusPipeIOPerfProvider_HyperVVMVirtualDevicePipeIO, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_VMBusPipeIOPerfProvider_HyperVVMVirtualDevicePipeIO{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetReceiveMessageQuotaExceeded sets the value of ReceiveMessageQuotaExceeded for the instance
func (instance *Win32_PerfRawData_VMBusPipeIOPerfProvider_HyperVVMVirtualDevicePipeIO) SetPropertyReceiveMessageQuotaExceeded(value uint64) (err error) {
	return instance.SetProperty("ReceiveMessageQuotaExceeded", (value))
}

// GetReceiveMessageQuotaExceeded gets the value of ReceiveMessageQuotaExceeded for the instance
func (instance *Win32_PerfRawData_VMBusPipeIOPerfProvider_HyperVVMVirtualDevicePipeIO) GetPropertyReceiveMessageQuotaExceeded() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReceiveMessageQuotaExceeded")
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

// SetReceiveQoSConformantMessagesPersec sets the value of ReceiveQoSConformantMessagesPersec for the instance
func (instance *Win32_PerfRawData_VMBusPipeIOPerfProvider_HyperVVMVirtualDevicePipeIO) SetPropertyReceiveQoSConformantMessagesPersec(value uint64) (err error) {
	return instance.SetProperty("ReceiveQoSConformantMessagesPersec", (value))
}

// GetReceiveQoSConformantMessagesPersec gets the value of ReceiveQoSConformantMessagesPersec for the instance
func (instance *Win32_PerfRawData_VMBusPipeIOPerfProvider_HyperVVMVirtualDevicePipeIO) GetPropertyReceiveQoSConformantMessagesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReceiveQoSConformantMessagesPersec")
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

// SetReceiveQoSExemptMessagesPersec sets the value of ReceiveQoSExemptMessagesPersec for the instance
func (instance *Win32_PerfRawData_VMBusPipeIOPerfProvider_HyperVVMVirtualDevicePipeIO) SetPropertyReceiveQoSExemptMessagesPersec(value uint64) (err error) {
	return instance.SetProperty("ReceiveQoSExemptMessagesPersec", (value))
}

// GetReceiveQoSExemptMessagesPersec gets the value of ReceiveQoSExemptMessagesPersec for the instance
func (instance *Win32_PerfRawData_VMBusPipeIOPerfProvider_HyperVVMVirtualDevicePipeIO) GetPropertyReceiveQoSExemptMessagesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReceiveQoSExemptMessagesPersec")
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

// SetReceiveQoSNonConformantMessagesPersec sets the value of ReceiveQoSNonConformantMessagesPersec for the instance
func (instance *Win32_PerfRawData_VMBusPipeIOPerfProvider_HyperVVMVirtualDevicePipeIO) SetPropertyReceiveQoSNonConformantMessagesPersec(value uint64) (err error) {
	return instance.SetProperty("ReceiveQoSNonConformantMessagesPersec", (value))
}

// GetReceiveQoSNonConformantMessagesPersec gets the value of ReceiveQoSNonConformantMessagesPersec for the instance
func (instance *Win32_PerfRawData_VMBusPipeIOPerfProvider_HyperVVMVirtualDevicePipeIO) GetPropertyReceiveQoSNonConformantMessagesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReceiveQoSNonConformantMessagesPersec")
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

// SetReceiveQoSTotalMessageDelayTime100ns sets the value of ReceiveQoSTotalMessageDelayTime100ns for the instance
func (instance *Win32_PerfRawData_VMBusPipeIOPerfProvider_HyperVVMVirtualDevicePipeIO) SetPropertyReceiveQoSTotalMessageDelayTime100ns(value uint64) (err error) {
	return instance.SetProperty("ReceiveQoSTotalMessageDelayTime100ns", (value))
}

// GetReceiveQoSTotalMessageDelayTime100ns gets the value of ReceiveQoSTotalMessageDelayTime100ns for the instance
func (instance *Win32_PerfRawData_VMBusPipeIOPerfProvider_HyperVVMVirtualDevicePipeIO) GetPropertyReceiveQoSTotalMessageDelayTime100ns() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReceiveQoSTotalMessageDelayTime100ns")
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
