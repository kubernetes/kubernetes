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

// Win32_PerfRawData_OfflineFiles_OfflineFiles struct
type Win32_PerfRawData_OfflineFiles_OfflineFiles struct {
	*Win32_PerfRawData

	//
	BytesReceived uint64

	//
	BytesReceivedPersec uint64

	//
	BytesReceivedPersec_Base uint32

	//
	BytesTransmitted uint64

	//
	BytesTransmittedPersec uint64

	//
	BytesTransmittedPersec_Base uint32
}

func NewWin32_PerfRawData_OfflineFiles_OfflineFilesEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_OfflineFiles_OfflineFiles, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_OfflineFiles_OfflineFiles{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_OfflineFiles_OfflineFilesEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_OfflineFiles_OfflineFiles, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_OfflineFiles_OfflineFiles{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetBytesReceived sets the value of BytesReceived for the instance
func (instance *Win32_PerfRawData_OfflineFiles_OfflineFiles) SetPropertyBytesReceived(value uint64) (err error) {
	return instance.SetProperty("BytesReceived", (value))
}

// GetBytesReceived gets the value of BytesReceived for the instance
func (instance *Win32_PerfRawData_OfflineFiles_OfflineFiles) GetPropertyBytesReceived() (value uint64, err error) {
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

// SetBytesReceivedPersec sets the value of BytesReceivedPersec for the instance
func (instance *Win32_PerfRawData_OfflineFiles_OfflineFiles) SetPropertyBytesReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("BytesReceivedPersec", (value))
}

// GetBytesReceivedPersec gets the value of BytesReceivedPersec for the instance
func (instance *Win32_PerfRawData_OfflineFiles_OfflineFiles) GetPropertyBytesReceivedPersec() (value uint64, err error) {
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

// SetBytesReceivedPersec_Base sets the value of BytesReceivedPersec_Base for the instance
func (instance *Win32_PerfRawData_OfflineFiles_OfflineFiles) SetPropertyBytesReceivedPersec_Base(value uint32) (err error) {
	return instance.SetProperty("BytesReceivedPersec_Base", (value))
}

// GetBytesReceivedPersec_Base gets the value of BytesReceivedPersec_Base for the instance
func (instance *Win32_PerfRawData_OfflineFiles_OfflineFiles) GetPropertyBytesReceivedPersec_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("BytesReceivedPersec_Base")
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
func (instance *Win32_PerfRawData_OfflineFiles_OfflineFiles) SetPropertyBytesTransmitted(value uint64) (err error) {
	return instance.SetProperty("BytesTransmitted", (value))
}

// GetBytesTransmitted gets the value of BytesTransmitted for the instance
func (instance *Win32_PerfRawData_OfflineFiles_OfflineFiles) GetPropertyBytesTransmitted() (value uint64, err error) {
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

// SetBytesTransmittedPersec sets the value of BytesTransmittedPersec for the instance
func (instance *Win32_PerfRawData_OfflineFiles_OfflineFiles) SetPropertyBytesTransmittedPersec(value uint64) (err error) {
	return instance.SetProperty("BytesTransmittedPersec", (value))
}

// GetBytesTransmittedPersec gets the value of BytesTransmittedPersec for the instance
func (instance *Win32_PerfRawData_OfflineFiles_OfflineFiles) GetPropertyBytesTransmittedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesTransmittedPersec")
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

// SetBytesTransmittedPersec_Base sets the value of BytesTransmittedPersec_Base for the instance
func (instance *Win32_PerfRawData_OfflineFiles_OfflineFiles) SetPropertyBytesTransmittedPersec_Base(value uint32) (err error) {
	return instance.SetProperty("BytesTransmittedPersec_Base", (value))
}

// GetBytesTransmittedPersec_Base gets the value of BytesTransmittedPersec_Base for the instance
func (instance *Win32_PerfRawData_OfflineFiles_OfflineFiles) GetPropertyBytesTransmittedPersec_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("BytesTransmittedPersec_Base")
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
