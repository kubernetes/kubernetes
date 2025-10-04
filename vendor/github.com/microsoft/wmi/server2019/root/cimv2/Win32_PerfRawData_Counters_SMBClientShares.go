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

// Win32_PerfRawData_Counters_SMBClientShares struct
type Win32_PerfRawData_Counters_SMBClientShares struct {
	*Win32_PerfRawData

	//
	AvgBytesPerRead uint64

	//
	AvgBytesPerRead_Base uint32

	//
	AvgBytesPerWrite uint64

	//
	AvgBytesPerWrite_Base uint32

	//
	AvgDataBytesPerRequest uint64

	//
	AvgDataBytesPerRequest_Base uint32

	//
	AvgDataQueueLength uint64

	//
	AvgReadQueueLength uint64

	//
	AvgsecPerDataRequest uint32

	//
	AvgsecPerDataRequest_Base uint32

	//
	AvgsecPerRead uint32

	//
	AvgsecPerRead_Base uint32

	//
	AvgsecPerWrite uint32

	//
	AvgsecPerWrite_Base uint32

	//
	AvgWriteQueueLength uint64

	//
	CreditStallsPersec uint32

	//
	CurrentDataQueueLength uint32

	//
	DataBytesPersec uint64

	//
	DataRequestsPersec uint32

	//
	MetadataRequestsPersec uint32

	//
	ReadBytesPersec uint64

	//
	ReadBytestransmittedviaSMBDirectPersec uint64

	//
	ReadRequestsPersec uint32

	//
	ReadRequeststransmittedviaSMBDirectPersec uint32

	//
	TurboIOReadsPersec uint32

	//
	TurboIOWritesPersec uint32

	//
	WriteBytesPersec uint64

	//
	WriteBytestransmittedviaSMBDirectPersec uint64

	//
	WriteRequestsPersec uint32

	//
	WriteRequeststransmittedviaSMBDirectPersec uint32
}

func NewWin32_PerfRawData_Counters_SMBClientSharesEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_Counters_SMBClientShares, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_SMBClientShares{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_Counters_SMBClientSharesEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_Counters_SMBClientShares, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_SMBClientShares{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetAvgBytesPerRead sets the value of AvgBytesPerRead for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) SetPropertyAvgBytesPerRead(value uint64) (err error) {
	return instance.SetProperty("AvgBytesPerRead", (value))
}

// GetAvgBytesPerRead gets the value of AvgBytesPerRead for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) GetPropertyAvgBytesPerRead() (value uint64, err error) {
	retValue, err := instance.GetProperty("AvgBytesPerRead")
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

// SetAvgBytesPerRead_Base sets the value of AvgBytesPerRead_Base for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) SetPropertyAvgBytesPerRead_Base(value uint32) (err error) {
	return instance.SetProperty("AvgBytesPerRead_Base", (value))
}

// GetAvgBytesPerRead_Base gets the value of AvgBytesPerRead_Base for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) GetPropertyAvgBytesPerRead_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("AvgBytesPerRead_Base")
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

// SetAvgBytesPerWrite sets the value of AvgBytesPerWrite for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) SetPropertyAvgBytesPerWrite(value uint64) (err error) {
	return instance.SetProperty("AvgBytesPerWrite", (value))
}

// GetAvgBytesPerWrite gets the value of AvgBytesPerWrite for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) GetPropertyAvgBytesPerWrite() (value uint64, err error) {
	retValue, err := instance.GetProperty("AvgBytesPerWrite")
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

// SetAvgBytesPerWrite_Base sets the value of AvgBytesPerWrite_Base for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) SetPropertyAvgBytesPerWrite_Base(value uint32) (err error) {
	return instance.SetProperty("AvgBytesPerWrite_Base", (value))
}

// GetAvgBytesPerWrite_Base gets the value of AvgBytesPerWrite_Base for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) GetPropertyAvgBytesPerWrite_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("AvgBytesPerWrite_Base")
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

// SetAvgDataBytesPerRequest sets the value of AvgDataBytesPerRequest for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) SetPropertyAvgDataBytesPerRequest(value uint64) (err error) {
	return instance.SetProperty("AvgDataBytesPerRequest", (value))
}

// GetAvgDataBytesPerRequest gets the value of AvgDataBytesPerRequest for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) GetPropertyAvgDataBytesPerRequest() (value uint64, err error) {
	retValue, err := instance.GetProperty("AvgDataBytesPerRequest")
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

// SetAvgDataBytesPerRequest_Base sets the value of AvgDataBytesPerRequest_Base for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) SetPropertyAvgDataBytesPerRequest_Base(value uint32) (err error) {
	return instance.SetProperty("AvgDataBytesPerRequest_Base", (value))
}

// GetAvgDataBytesPerRequest_Base gets the value of AvgDataBytesPerRequest_Base for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) GetPropertyAvgDataBytesPerRequest_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("AvgDataBytesPerRequest_Base")
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

// SetAvgDataQueueLength sets the value of AvgDataQueueLength for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) SetPropertyAvgDataQueueLength(value uint64) (err error) {
	return instance.SetProperty("AvgDataQueueLength", (value))
}

// GetAvgDataQueueLength gets the value of AvgDataQueueLength for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) GetPropertyAvgDataQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("AvgDataQueueLength")
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

// SetAvgReadQueueLength sets the value of AvgReadQueueLength for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) SetPropertyAvgReadQueueLength(value uint64) (err error) {
	return instance.SetProperty("AvgReadQueueLength", (value))
}

// GetAvgReadQueueLength gets the value of AvgReadQueueLength for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) GetPropertyAvgReadQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("AvgReadQueueLength")
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

// SetAvgsecPerDataRequest sets the value of AvgsecPerDataRequest for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) SetPropertyAvgsecPerDataRequest(value uint32) (err error) {
	return instance.SetProperty("AvgsecPerDataRequest", (value))
}

// GetAvgsecPerDataRequest gets the value of AvgsecPerDataRequest for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) GetPropertyAvgsecPerDataRequest() (value uint32, err error) {
	retValue, err := instance.GetProperty("AvgsecPerDataRequest")
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

// SetAvgsecPerDataRequest_Base sets the value of AvgsecPerDataRequest_Base for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) SetPropertyAvgsecPerDataRequest_Base(value uint32) (err error) {
	return instance.SetProperty("AvgsecPerDataRequest_Base", (value))
}

// GetAvgsecPerDataRequest_Base gets the value of AvgsecPerDataRequest_Base for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) GetPropertyAvgsecPerDataRequest_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("AvgsecPerDataRequest_Base")
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

// SetAvgsecPerRead sets the value of AvgsecPerRead for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) SetPropertyAvgsecPerRead(value uint32) (err error) {
	return instance.SetProperty("AvgsecPerRead", (value))
}

// GetAvgsecPerRead gets the value of AvgsecPerRead for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) GetPropertyAvgsecPerRead() (value uint32, err error) {
	retValue, err := instance.GetProperty("AvgsecPerRead")
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

// SetAvgsecPerRead_Base sets the value of AvgsecPerRead_Base for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) SetPropertyAvgsecPerRead_Base(value uint32) (err error) {
	return instance.SetProperty("AvgsecPerRead_Base", (value))
}

// GetAvgsecPerRead_Base gets the value of AvgsecPerRead_Base for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) GetPropertyAvgsecPerRead_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("AvgsecPerRead_Base")
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

// SetAvgsecPerWrite sets the value of AvgsecPerWrite for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) SetPropertyAvgsecPerWrite(value uint32) (err error) {
	return instance.SetProperty("AvgsecPerWrite", (value))
}

// GetAvgsecPerWrite gets the value of AvgsecPerWrite for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) GetPropertyAvgsecPerWrite() (value uint32, err error) {
	retValue, err := instance.GetProperty("AvgsecPerWrite")
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

// SetAvgsecPerWrite_Base sets the value of AvgsecPerWrite_Base for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) SetPropertyAvgsecPerWrite_Base(value uint32) (err error) {
	return instance.SetProperty("AvgsecPerWrite_Base", (value))
}

// GetAvgsecPerWrite_Base gets the value of AvgsecPerWrite_Base for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) GetPropertyAvgsecPerWrite_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("AvgsecPerWrite_Base")
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

// SetAvgWriteQueueLength sets the value of AvgWriteQueueLength for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) SetPropertyAvgWriteQueueLength(value uint64) (err error) {
	return instance.SetProperty("AvgWriteQueueLength", (value))
}

// GetAvgWriteQueueLength gets the value of AvgWriteQueueLength for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) GetPropertyAvgWriteQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("AvgWriteQueueLength")
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

// SetCreditStallsPersec sets the value of CreditStallsPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) SetPropertyCreditStallsPersec(value uint32) (err error) {
	return instance.SetProperty("CreditStallsPersec", (value))
}

// GetCreditStallsPersec gets the value of CreditStallsPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) GetPropertyCreditStallsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("CreditStallsPersec")
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

// SetCurrentDataQueueLength sets the value of CurrentDataQueueLength for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) SetPropertyCurrentDataQueueLength(value uint32) (err error) {
	return instance.SetProperty("CurrentDataQueueLength", (value))
}

// GetCurrentDataQueueLength gets the value of CurrentDataQueueLength for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) GetPropertyCurrentDataQueueLength() (value uint32, err error) {
	retValue, err := instance.GetProperty("CurrentDataQueueLength")
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

// SetDataBytesPersec sets the value of DataBytesPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) SetPropertyDataBytesPersec(value uint64) (err error) {
	return instance.SetProperty("DataBytesPersec", (value))
}

// GetDataBytesPersec gets the value of DataBytesPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) GetPropertyDataBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("DataBytesPersec")
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

// SetDataRequestsPersec sets the value of DataRequestsPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) SetPropertyDataRequestsPersec(value uint32) (err error) {
	return instance.SetProperty("DataRequestsPersec", (value))
}

// GetDataRequestsPersec gets the value of DataRequestsPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) GetPropertyDataRequestsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("DataRequestsPersec")
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

// SetMetadataRequestsPersec sets the value of MetadataRequestsPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) SetPropertyMetadataRequestsPersec(value uint32) (err error) {
	return instance.SetProperty("MetadataRequestsPersec", (value))
}

// GetMetadataRequestsPersec gets the value of MetadataRequestsPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) GetPropertyMetadataRequestsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("MetadataRequestsPersec")
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
func (instance *Win32_PerfRawData_Counters_SMBClientShares) SetPropertyReadBytesPersec(value uint64) (err error) {
	return instance.SetProperty("ReadBytesPersec", (value))
}

// GetReadBytesPersec gets the value of ReadBytesPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) GetPropertyReadBytesPersec() (value uint64, err error) {
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

// SetReadBytestransmittedviaSMBDirectPersec sets the value of ReadBytestransmittedviaSMBDirectPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) SetPropertyReadBytestransmittedviaSMBDirectPersec(value uint64) (err error) {
	return instance.SetProperty("ReadBytestransmittedviaSMBDirectPersec", (value))
}

// GetReadBytestransmittedviaSMBDirectPersec gets the value of ReadBytestransmittedviaSMBDirectPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) GetPropertyReadBytestransmittedviaSMBDirectPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadBytestransmittedviaSMBDirectPersec")
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
func (instance *Win32_PerfRawData_Counters_SMBClientShares) SetPropertyReadRequestsPersec(value uint32) (err error) {
	return instance.SetProperty("ReadRequestsPersec", (value))
}

// GetReadRequestsPersec gets the value of ReadRequestsPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) GetPropertyReadRequestsPersec() (value uint32, err error) {
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

// SetReadRequeststransmittedviaSMBDirectPersec sets the value of ReadRequeststransmittedviaSMBDirectPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) SetPropertyReadRequeststransmittedviaSMBDirectPersec(value uint32) (err error) {
	return instance.SetProperty("ReadRequeststransmittedviaSMBDirectPersec", (value))
}

// GetReadRequeststransmittedviaSMBDirectPersec gets the value of ReadRequeststransmittedviaSMBDirectPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) GetPropertyReadRequeststransmittedviaSMBDirectPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReadRequeststransmittedviaSMBDirectPersec")
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

// SetTurboIOReadsPersec sets the value of TurboIOReadsPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) SetPropertyTurboIOReadsPersec(value uint32) (err error) {
	return instance.SetProperty("TurboIOReadsPersec", (value))
}

// GetTurboIOReadsPersec gets the value of TurboIOReadsPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) GetPropertyTurboIOReadsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("TurboIOReadsPersec")
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

// SetTurboIOWritesPersec sets the value of TurboIOWritesPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) SetPropertyTurboIOWritesPersec(value uint32) (err error) {
	return instance.SetProperty("TurboIOWritesPersec", (value))
}

// GetTurboIOWritesPersec gets the value of TurboIOWritesPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) GetPropertyTurboIOWritesPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("TurboIOWritesPersec")
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
func (instance *Win32_PerfRawData_Counters_SMBClientShares) SetPropertyWriteBytesPersec(value uint64) (err error) {
	return instance.SetProperty("WriteBytesPersec", (value))
}

// GetWriteBytesPersec gets the value of WriteBytesPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) GetPropertyWriteBytesPersec() (value uint64, err error) {
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

// SetWriteBytestransmittedviaSMBDirectPersec sets the value of WriteBytestransmittedviaSMBDirectPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) SetPropertyWriteBytestransmittedviaSMBDirectPersec(value uint64) (err error) {
	return instance.SetProperty("WriteBytestransmittedviaSMBDirectPersec", (value))
}

// GetWriteBytestransmittedviaSMBDirectPersec gets the value of WriteBytestransmittedviaSMBDirectPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) GetPropertyWriteBytestransmittedviaSMBDirectPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteBytestransmittedviaSMBDirectPersec")
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
func (instance *Win32_PerfRawData_Counters_SMBClientShares) SetPropertyWriteRequestsPersec(value uint32) (err error) {
	return instance.SetProperty("WriteRequestsPersec", (value))
}

// GetWriteRequestsPersec gets the value of WriteRequestsPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) GetPropertyWriteRequestsPersec() (value uint32, err error) {
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

// SetWriteRequeststransmittedviaSMBDirectPersec sets the value of WriteRequeststransmittedviaSMBDirectPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) SetPropertyWriteRequeststransmittedviaSMBDirectPersec(value uint32) (err error) {
	return instance.SetProperty("WriteRequeststransmittedviaSMBDirectPersec", (value))
}

// GetWriteRequeststransmittedviaSMBDirectPersec gets the value of WriteRequeststransmittedviaSMBDirectPersec for the instance
func (instance *Win32_PerfRawData_Counters_SMBClientShares) GetPropertyWriteRequeststransmittedviaSMBDirectPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("WriteRequeststransmittedviaSMBDirectPersec")
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
