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

// Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO struct
type Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO struct {
	*Win32_PerfFormattedData

	//
	AvgBytesPerRead uint64

	//
	AvgBytesPerWrite uint64

	//
	AvgReadQueueLength uint64

	//
	AvgsecPerRead uint32

	//
	AvgsecPerWrite uint32

	//
	AvgWriteQueueLength uint64

	//
	CurrentReadQueueLength uint64

	//
	CurrentWriteQueueLength uint64

	//
	NonSplitReads uint64

	//
	NonSplitReadsPersec uint64

	//
	NonSplitWrites uint64

	//
	NonSplitWritesPersec uint64

	//
	ReadBytes uint64

	//
	ReadBytesPersec uint64

	//
	Reads uint64

	//
	ReadsPersec uint64

	//
	SplitReads uint64

	//
	SplitReadsPersec uint64

	//
	SplitWrites uint64

	//
	SplitWritesPersec uint64

	//
	WriteBytes uint64

	//
	WriteBytesPersec uint64

	//
	Writes uint64

	//
	WritesPersec uint64
}

func NewWin32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIOEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIOEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetAvgBytesPerRead sets the value of AvgBytesPerRead for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) SetPropertyAvgBytesPerRead(value uint64) (err error) {
	return instance.SetProperty("AvgBytesPerRead", (value))
}

// GetAvgBytesPerRead gets the value of AvgBytesPerRead for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) GetPropertyAvgBytesPerRead() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) SetPropertyAvgBytesPerWrite(value uint64) (err error) {
	return instance.SetProperty("AvgBytesPerWrite", (value))
}

// GetAvgBytesPerWrite gets the value of AvgBytesPerWrite for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) GetPropertyAvgBytesPerWrite() (value uint64, err error) {
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

// SetAvgReadQueueLength sets the value of AvgReadQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) SetPropertyAvgReadQueueLength(value uint64) (err error) {
	return instance.SetProperty("AvgReadQueueLength", (value))
}

// GetAvgReadQueueLength gets the value of AvgReadQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) GetPropertyAvgReadQueueLength() (value uint64, err error) {
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

// SetAvgsecPerRead sets the value of AvgsecPerRead for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) SetPropertyAvgsecPerRead(value uint32) (err error) {
	return instance.SetProperty("AvgsecPerRead", (value))
}

// GetAvgsecPerRead gets the value of AvgsecPerRead for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) GetPropertyAvgsecPerRead() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) SetPropertyAvgsecPerWrite(value uint32) (err error) {
	return instance.SetProperty("AvgsecPerWrite", (value))
}

// GetAvgsecPerWrite gets the value of AvgsecPerWrite for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) GetPropertyAvgsecPerWrite() (value uint32, err error) {
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

// SetAvgWriteQueueLength sets the value of AvgWriteQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) SetPropertyAvgWriteQueueLength(value uint64) (err error) {
	return instance.SetProperty("AvgWriteQueueLength", (value))
}

// GetAvgWriteQueueLength gets the value of AvgWriteQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) GetPropertyAvgWriteQueueLength() (value uint64, err error) {
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

// SetCurrentReadQueueLength sets the value of CurrentReadQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) SetPropertyCurrentReadQueueLength(value uint64) (err error) {
	return instance.SetProperty("CurrentReadQueueLength", (value))
}

// GetCurrentReadQueueLength gets the value of CurrentReadQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) GetPropertyCurrentReadQueueLength() (value uint64, err error) {
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

// SetCurrentWriteQueueLength sets the value of CurrentWriteQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) SetPropertyCurrentWriteQueueLength(value uint64) (err error) {
	return instance.SetProperty("CurrentWriteQueueLength", (value))
}

// GetCurrentWriteQueueLength gets the value of CurrentWriteQueueLength for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) GetPropertyCurrentWriteQueueLength() (value uint64, err error) {
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

// SetNonSplitReads sets the value of NonSplitReads for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) SetPropertyNonSplitReads(value uint64) (err error) {
	return instance.SetProperty("NonSplitReads", (value))
}

// GetNonSplitReads gets the value of NonSplitReads for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) GetPropertyNonSplitReads() (value uint64, err error) {
	retValue, err := instance.GetProperty("NonSplitReads")
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

// SetNonSplitReadsPersec sets the value of NonSplitReadsPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) SetPropertyNonSplitReadsPersec(value uint64) (err error) {
	return instance.SetProperty("NonSplitReadsPersec", (value))
}

// GetNonSplitReadsPersec gets the value of NonSplitReadsPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) GetPropertyNonSplitReadsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("NonSplitReadsPersec")
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

// SetNonSplitWrites sets the value of NonSplitWrites for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) SetPropertyNonSplitWrites(value uint64) (err error) {
	return instance.SetProperty("NonSplitWrites", (value))
}

// GetNonSplitWrites gets the value of NonSplitWrites for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) GetPropertyNonSplitWrites() (value uint64, err error) {
	retValue, err := instance.GetProperty("NonSplitWrites")
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

// SetNonSplitWritesPersec sets the value of NonSplitWritesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) SetPropertyNonSplitWritesPersec(value uint64) (err error) {
	return instance.SetProperty("NonSplitWritesPersec", (value))
}

// GetNonSplitWritesPersec gets the value of NonSplitWritesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) GetPropertyNonSplitWritesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("NonSplitWritesPersec")
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
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) SetPropertyReadBytes(value uint64) (err error) {
	return instance.SetProperty("ReadBytes", (value))
}

// GetReadBytes gets the value of ReadBytes for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) GetPropertyReadBytes() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) SetPropertyReadBytesPersec(value uint64) (err error) {
	return instance.SetProperty("ReadBytesPersec", (value))
}

// GetReadBytesPersec gets the value of ReadBytesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) GetPropertyReadBytesPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) SetPropertyReads(value uint64) (err error) {
	return instance.SetProperty("Reads", (value))
}

// GetReads gets the value of Reads for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) GetPropertyReads() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) SetPropertyReadsPersec(value uint64) (err error) {
	return instance.SetProperty("ReadsPersec", (value))
}

// GetReadsPersec gets the value of ReadsPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) GetPropertyReadsPersec() (value uint64, err error) {
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

// SetSplitReads sets the value of SplitReads for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) SetPropertySplitReads(value uint64) (err error) {
	return instance.SetProperty("SplitReads", (value))
}

// GetSplitReads gets the value of SplitReads for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) GetPropertySplitReads() (value uint64, err error) {
	retValue, err := instance.GetProperty("SplitReads")
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

// SetSplitReadsPersec sets the value of SplitReadsPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) SetPropertySplitReadsPersec(value uint64) (err error) {
	return instance.SetProperty("SplitReadsPersec", (value))
}

// GetSplitReadsPersec gets the value of SplitReadsPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) GetPropertySplitReadsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("SplitReadsPersec")
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

// SetSplitWrites sets the value of SplitWrites for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) SetPropertySplitWrites(value uint64) (err error) {
	return instance.SetProperty("SplitWrites", (value))
}

// GetSplitWrites gets the value of SplitWrites for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) GetPropertySplitWrites() (value uint64, err error) {
	retValue, err := instance.GetProperty("SplitWrites")
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

// SetSplitWritesPersec sets the value of SplitWritesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) SetPropertySplitWritesPersec(value uint64) (err error) {
	return instance.SetProperty("SplitWritesPersec", (value))
}

// GetSplitWritesPersec gets the value of SplitWritesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) GetPropertySplitWritesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("SplitWritesPersec")
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
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) SetPropertyWriteBytes(value uint64) (err error) {
	return instance.SetProperty("WriteBytes", (value))
}

// GetWriteBytes gets the value of WriteBytes for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) GetPropertyWriteBytes() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) SetPropertyWriteBytesPersec(value uint64) (err error) {
	return instance.SetProperty("WriteBytesPersec", (value))
}

// GetWriteBytesPersec gets the value of WriteBytesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) GetPropertyWriteBytesPersec() (value uint64, err error) {
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

// SetWrites sets the value of Writes for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) SetPropertyWrites(value uint64) (err error) {
	return instance.SetProperty("Writes", (value))
}

// GetWrites gets the value of Writes for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) GetPropertyWrites() (value uint64, err error) {
	retValue, err := instance.GetProperty("Writes")
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
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) SetPropertyWritesPersec(value uint64) (err error) {
	return instance.SetProperty("WritesPersec", (value))
}

// GetWritesPersec gets the value of WritesPersec for the instance
func (instance *Win32_PerfFormattedData_CsvFsPerfProvider_ClusterCSVFSDirectIO) GetPropertyWritesPersec() (value uint64, err error) {
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
