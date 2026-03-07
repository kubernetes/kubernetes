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

// Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB struct
type Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB struct {
	*Win32_PerfRawData

	//
	AvgsecPerRequest uint32

	//
	AvgsecPerRequest_Base uint32

	//
	CurrentOpenFileCount uint32

	//
	CurrentPendingRequests uint32

	//
	DirectMappedPages uint64

	//
	DirectMappedSections uint32

	//
	FlushRequestsPersec uint32

	//
	ReadBytesPersec uint64

	//
	ReadBytesPersecRDMA uint64

	//
	ReadRequestsPersec uint32

	//
	ReadRequestsPersecRDMA uint32

	//
	ReceivedBytesPersec uint64

	//
	RequestsPersec uint32

	//
	SentBytesPersec uint64

	//
	TreeConnectCount uint32

	//
	WriteBytesPersec uint64

	//
	WriteBytesPersecRDMA uint64

	//
	WriteRequestsPersec uint32

	//
	WriteRequestsPersecRDMA uint32
}

func NewWin32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMBEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMBEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetAvgsecPerRequest sets the value of AvgsecPerRequest for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) SetPropertyAvgsecPerRequest(value uint32) (err error) {
	return instance.SetProperty("AvgsecPerRequest", (value))
}

// GetAvgsecPerRequest gets the value of AvgsecPerRequest for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) GetPropertyAvgsecPerRequest() (value uint32, err error) {
	retValue, err := instance.GetProperty("AvgsecPerRequest")
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

// SetAvgsecPerRequest_Base sets the value of AvgsecPerRequest_Base for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) SetPropertyAvgsecPerRequest_Base(value uint32) (err error) {
	return instance.SetProperty("AvgsecPerRequest_Base", (value))
}

// GetAvgsecPerRequest_Base gets the value of AvgsecPerRequest_Base for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) GetPropertyAvgsecPerRequest_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("AvgsecPerRequest_Base")
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

// SetCurrentOpenFileCount sets the value of CurrentOpenFileCount for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) SetPropertyCurrentOpenFileCount(value uint32) (err error) {
	return instance.SetProperty("CurrentOpenFileCount", (value))
}

// GetCurrentOpenFileCount gets the value of CurrentOpenFileCount for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) GetPropertyCurrentOpenFileCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("CurrentOpenFileCount")
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

// SetCurrentPendingRequests sets the value of CurrentPendingRequests for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) SetPropertyCurrentPendingRequests(value uint32) (err error) {
	return instance.SetProperty("CurrentPendingRequests", (value))
}

// GetCurrentPendingRequests gets the value of CurrentPendingRequests for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) GetPropertyCurrentPendingRequests() (value uint32, err error) {
	retValue, err := instance.GetProperty("CurrentPendingRequests")
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

// SetDirectMappedPages sets the value of DirectMappedPages for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) SetPropertyDirectMappedPages(value uint64) (err error) {
	return instance.SetProperty("DirectMappedPages", (value))
}

// GetDirectMappedPages gets the value of DirectMappedPages for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) GetPropertyDirectMappedPages() (value uint64, err error) {
	retValue, err := instance.GetProperty("DirectMappedPages")
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

// SetDirectMappedSections sets the value of DirectMappedSections for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) SetPropertyDirectMappedSections(value uint32) (err error) {
	return instance.SetProperty("DirectMappedSections", (value))
}

// GetDirectMappedSections gets the value of DirectMappedSections for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) GetPropertyDirectMappedSections() (value uint32, err error) {
	retValue, err := instance.GetProperty("DirectMappedSections")
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

// SetFlushRequestsPersec sets the value of FlushRequestsPersec for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) SetPropertyFlushRequestsPersec(value uint32) (err error) {
	return instance.SetProperty("FlushRequestsPersec", (value))
}

// GetFlushRequestsPersec gets the value of FlushRequestsPersec for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) GetPropertyFlushRequestsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("FlushRequestsPersec")
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

// SetReadBytesPersec sets the value of ReadBytesPersec for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) SetPropertyReadBytesPersec(value uint64) (err error) {
	return instance.SetProperty("ReadBytesPersec", (value))
}

// GetReadBytesPersec gets the value of ReadBytesPersec for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) GetPropertyReadBytesPersec() (value uint64, err error) {
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

// SetReadBytesPersecRDMA sets the value of ReadBytesPersecRDMA for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) SetPropertyReadBytesPersecRDMA(value uint64) (err error) {
	return instance.SetProperty("ReadBytesPersecRDMA", (value))
}

// GetReadBytesPersecRDMA gets the value of ReadBytesPersecRDMA for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) GetPropertyReadBytesPersecRDMA() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadBytesPersecRDMA")
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
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) SetPropertyReadRequestsPersec(value uint32) (err error) {
	return instance.SetProperty("ReadRequestsPersec", (value))
}

// GetReadRequestsPersec gets the value of ReadRequestsPersec for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) GetPropertyReadRequestsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReadRequestsPersec")
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

// SetReadRequestsPersecRDMA sets the value of ReadRequestsPersecRDMA for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) SetPropertyReadRequestsPersecRDMA(value uint32) (err error) {
	return instance.SetProperty("ReadRequestsPersecRDMA", (value))
}

// GetReadRequestsPersecRDMA gets the value of ReadRequestsPersecRDMA for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) GetPropertyReadRequestsPersecRDMA() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReadRequestsPersecRDMA")
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

// SetReceivedBytesPersec sets the value of ReceivedBytesPersec for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) SetPropertyReceivedBytesPersec(value uint64) (err error) {
	return instance.SetProperty("ReceivedBytesPersec", (value))
}

// GetReceivedBytesPersec gets the value of ReceivedBytesPersec for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) GetPropertyReceivedBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReceivedBytesPersec")
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

// SetRequestsPersec sets the value of RequestsPersec for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) SetPropertyRequestsPersec(value uint32) (err error) {
	return instance.SetProperty("RequestsPersec", (value))
}

// GetRequestsPersec gets the value of RequestsPersec for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) GetPropertyRequestsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("RequestsPersec")
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

// SetSentBytesPersec sets the value of SentBytesPersec for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) SetPropertySentBytesPersec(value uint64) (err error) {
	return instance.SetProperty("SentBytesPersec", (value))
}

// GetSentBytesPersec gets the value of SentBytesPersec for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) GetPropertySentBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("SentBytesPersec")
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

// SetTreeConnectCount sets the value of TreeConnectCount for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) SetPropertyTreeConnectCount(value uint32) (err error) {
	return instance.SetProperty("TreeConnectCount", (value))
}

// GetTreeConnectCount gets the value of TreeConnectCount for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) GetPropertyTreeConnectCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("TreeConnectCount")
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

// SetWriteBytesPersec sets the value of WriteBytesPersec for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) SetPropertyWriteBytesPersec(value uint64) (err error) {
	return instance.SetProperty("WriteBytesPersec", (value))
}

// GetWriteBytesPersec gets the value of WriteBytesPersec for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) GetPropertyWriteBytesPersec() (value uint64, err error) {
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

// SetWriteBytesPersecRDMA sets the value of WriteBytesPersecRDMA for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) SetPropertyWriteBytesPersecRDMA(value uint64) (err error) {
	return instance.SetProperty("WriteBytesPersecRDMA", (value))
}

// GetWriteBytesPersecRDMA gets the value of WriteBytesPersecRDMA for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) GetPropertyWriteBytesPersecRDMA() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteBytesPersecRDMA")
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
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) SetPropertyWriteRequestsPersec(value uint32) (err error) {
	return instance.SetProperty("WriteRequestsPersec", (value))
}

// GetWriteRequestsPersec gets the value of WriteRequestsPersec for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) GetPropertyWriteRequestsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("WriteRequestsPersec")
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

// SetWriteRequestsPersecRDMA sets the value of WriteRequestsPersecRDMA for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) SetPropertyWriteRequestsPersecRDMA(value uint32) (err error) {
	return instance.SetProperty("WriteRequestsPersecRDMA", (value))
}

// GetWriteRequestsPersecRDMA gets the value of WriteRequestsPersecRDMA for the instance
func (instance *Win32_PerfRawData_VSmbPerfProvider_HyperVVirtualSMB) GetPropertyWriteRequestsPersecRDMA() (value uint32, err error) {
	retValue, err := instance.GetProperty("WriteRequestsPersecRDMA")
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
