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

// Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO struct
type Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO struct {
	*Win32_PerfFormattedData

	//
	AvgBytesPerRead uint64

	//
	AvgBytesPerWrite uint64

	//
	AvgReadsQueueLength uint64

	//
	AvgsecPerRead uint32

	//
	AvgsecPerWrite uint32

	//
	AvgTrimQueueLength uint64

	//
	AvgWritesQueueLength uint64

	//
	CurrentReadQueueLength uint64

	//
	CurrentTrimQueueLength uint64

	//
	CurrentWriteQueueLength uint64

	//
	ReadBytes uint64

	//
	ReadBytesPersec uint64

	//
	Reads uint64

	//
	ReadsPersec uint64

	//
	TrimLatency uint32

	//
	TrimPersec uint64

	//
	WriteBytes uint64

	//
	WriteBytesPersec uint64

	//
	WritesPersec uint64
}

func NewWin32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIOEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIOEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetAvgBytesPerRead sets the value of AvgBytesPerRead for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) SetPropertyAvgBytesPerRead(value uint64) (err error) {
	return instance.SetProperty("AvgBytesPerRead", (value))
}

// GetAvgBytesPerRead gets the value of AvgBytesPerRead for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) GetPropertyAvgBytesPerRead() (value uint64, err error) {
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

// SetAvgBytesPerWrite sets the value of AvgBytesPerWrite for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) SetPropertyAvgBytesPerWrite(value uint64) (err error) {
	return instance.SetProperty("AvgBytesPerWrite", (value))
}

// GetAvgBytesPerWrite gets the value of AvgBytesPerWrite for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) GetPropertyAvgBytesPerWrite() (value uint64, err error) {
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

// SetAvgReadsQueueLength sets the value of AvgReadsQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) SetPropertyAvgReadsQueueLength(value uint64) (err error) {
	return instance.SetProperty("AvgReadsQueueLength", (value))
}

// GetAvgReadsQueueLength gets the value of AvgReadsQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) GetPropertyAvgReadsQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("AvgReadsQueueLength")
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

// SetAvgsecPerRead sets the value of AvgsecPerRead for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) SetPropertyAvgsecPerRead(value uint32) (err error) {
	return instance.SetProperty("AvgsecPerRead", (value))
}

// GetAvgsecPerRead gets the value of AvgsecPerRead for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) GetPropertyAvgsecPerRead() (value uint32, err error) {
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

// SetAvgsecPerWrite sets the value of AvgsecPerWrite for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) SetPropertyAvgsecPerWrite(value uint32) (err error) {
	return instance.SetProperty("AvgsecPerWrite", (value))
}

// GetAvgsecPerWrite gets the value of AvgsecPerWrite for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) GetPropertyAvgsecPerWrite() (value uint32, err error) {
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

// SetAvgTrimQueueLength sets the value of AvgTrimQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) SetPropertyAvgTrimQueueLength(value uint64) (err error) {
	return instance.SetProperty("AvgTrimQueueLength", (value))
}

// GetAvgTrimQueueLength gets the value of AvgTrimQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) GetPropertyAvgTrimQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("AvgTrimQueueLength")
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

// SetAvgWritesQueueLength sets the value of AvgWritesQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) SetPropertyAvgWritesQueueLength(value uint64) (err error) {
	return instance.SetProperty("AvgWritesQueueLength", (value))
}

// GetAvgWritesQueueLength gets the value of AvgWritesQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) GetPropertyAvgWritesQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("AvgWritesQueueLength")
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

// SetCurrentReadQueueLength sets the value of CurrentReadQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) SetPropertyCurrentReadQueueLength(value uint64) (err error) {
	return instance.SetProperty("CurrentReadQueueLength", (value))
}

// GetCurrentReadQueueLength gets the value of CurrentReadQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) GetPropertyCurrentReadQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("CurrentReadQueueLength")
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

// SetCurrentTrimQueueLength sets the value of CurrentTrimQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) SetPropertyCurrentTrimQueueLength(value uint64) (err error) {
	return instance.SetProperty("CurrentTrimQueueLength", (value))
}

// GetCurrentTrimQueueLength gets the value of CurrentTrimQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) GetPropertyCurrentTrimQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("CurrentTrimQueueLength")
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

// SetCurrentWriteQueueLength sets the value of CurrentWriteQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) SetPropertyCurrentWriteQueueLength(value uint64) (err error) {
	return instance.SetProperty("CurrentWriteQueueLength", (value))
}

// GetCurrentWriteQueueLength gets the value of CurrentWriteQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) GetPropertyCurrentWriteQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("CurrentWriteQueueLength")
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

// SetReadBytes sets the value of ReadBytes for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) SetPropertyReadBytes(value uint64) (err error) {
	return instance.SetProperty("ReadBytes", (value))
}

// GetReadBytes gets the value of ReadBytes for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) GetPropertyReadBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadBytes")
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

// SetReadBytesPersec sets the value of ReadBytesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) SetPropertyReadBytesPersec(value uint64) (err error) {
	return instance.SetProperty("ReadBytesPersec", (value))
}

// GetReadBytesPersec gets the value of ReadBytesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) GetPropertyReadBytesPersec() (value uint64, err error) {
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

// SetReads sets the value of Reads for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) SetPropertyReads(value uint64) (err error) {
	return instance.SetProperty("Reads", (value))
}

// GetReads gets the value of Reads for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) GetPropertyReads() (value uint64, err error) {
	retValue, err := instance.GetProperty("Reads")
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

// SetReadsPersec sets the value of ReadsPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) SetPropertyReadsPersec(value uint64) (err error) {
	return instance.SetProperty("ReadsPersec", (value))
}

// GetReadsPersec gets the value of ReadsPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) GetPropertyReadsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadsPersec")
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

// SetTrimLatency sets the value of TrimLatency for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) SetPropertyTrimLatency(value uint32) (err error) {
	return instance.SetProperty("TrimLatency", (value))
}

// GetTrimLatency gets the value of TrimLatency for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) GetPropertyTrimLatency() (value uint32, err error) {
	retValue, err := instance.GetProperty("TrimLatency")
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

// SetTrimPersec sets the value of TrimPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) SetPropertyTrimPersec(value uint64) (err error) {
	return instance.SetProperty("TrimPersec", (value))
}

// GetTrimPersec gets the value of TrimPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) GetPropertyTrimPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("TrimPersec")
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

// SetWriteBytes sets the value of WriteBytes for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) SetPropertyWriteBytes(value uint64) (err error) {
	return instance.SetProperty("WriteBytes", (value))
}

// GetWriteBytes gets the value of WriteBytes for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) GetPropertyWriteBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteBytes")
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
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) SetPropertyWriteBytesPersec(value uint64) (err error) {
	return instance.SetProperty("WriteBytesPersec", (value))
}

// GetWriteBytesPersec gets the value of WriteBytesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) GetPropertyWriteBytesPersec() (value uint64, err error) {
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

// SetWritesPersec sets the value of WritesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) SetPropertyWritesPersec(value uint64) (err error) {
	return instance.SetProperty("WritesPersec", (value))
}

// GetWritesPersec gets the value of WritesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSRedirectedIO) GetPropertyWritesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("WritesPersec")
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
