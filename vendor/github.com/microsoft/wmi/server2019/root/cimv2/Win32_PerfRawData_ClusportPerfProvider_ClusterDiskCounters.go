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

// Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters struct
type Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters struct {
	*Win32_PerfRawData

	//
	ExceededLatencyLimit uint64

	//
	ExceededLatencyLimitPersec uint32

	//
	IO10000msPersec uint32

	//
	IO1000msPersec uint32

	//
	IO100msPersec uint32

	//
	IO10msPersec uint32

	//
	IO1msPersec uint32

	//
	IO5msPersec uint32

	//
	LocalReadAvgQueueLength uint64

	//
	LocalReadBytes uint64

	//
	LocalReadBytesPersec uint64

	//
	LocalReadLatency uint32

	//
	LocalReadLatency_Base uint32

	//
	LocalReadPersec uint32

	//
	LocalReadQueueLength uint64

	//
	LocalReads uint64

	//
	LocalWriteAvgQueueLength uint64

	//
	LocalWriteBytes uint64

	//
	LocalWriteBytesPersec uint64

	//
	LocalWriteLatency uint32

	//
	LocalWriteLatency_Base uint32

	//
	LocalWriteQueueLength uint64

	//
	LocalWrites uint64

	//
	LocalWritesPersec uint32

	//
	ReadAvgQueueLength uint64

	//
	ReadBytes uint64

	//
	ReadBytesPersec uint64

	//
	ReadLatency uint32

	//
	ReadLatency_Base uint32

	//
	ReadPersec uint32

	//
	ReadQueueLength uint64

	//
	Reads uint64

	//
	RemoteReadAvgQueueLength uint64

	//
	RemoteReadBytes uint64

	//
	RemoteReadBytesPersec uint64

	//
	RemoteReadLatency uint32

	//
	RemoteReadLatency_Base uint32

	//
	RemoteReadPersec uint32

	//
	RemoteReadQueueLength uint64

	//
	RemoteReads uint64

	//
	RemoteWriteAvgQueueLength uint64

	//
	RemoteWriteBytes uint64

	//
	RemoteWriteBytesPersec uint64

	//
	RemoteWriteLatency uint32

	//
	RemoteWriteLatency_Base uint32

	//
	RemoteWriteQueueLength uint64

	//
	RemoteWrites uint64

	//
	RemoteWritesPersec uint32

	//
	WriteAvgQueueLength uint64

	//
	WriteBytes uint64

	//
	WriteBytesPersec uint32

	//
	WriteLatency uint32

	//
	WriteLatency_Base uint32

	//
	WriteQueueLength uint64

	//
	Writes uint64

	//
	WritesPersec uint32
}

func NewWin32_PerfRawData_ClusportPerfProvider_ClusterDiskCountersEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_ClusportPerfProvider_ClusterDiskCountersEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetExceededLatencyLimit sets the value of ExceededLatencyLimit for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyExceededLatencyLimit(value uint64) (err error) {
	return instance.SetProperty("ExceededLatencyLimit", (value))
}

// GetExceededLatencyLimit gets the value of ExceededLatencyLimit for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyExceededLatencyLimit() (value uint64, err error) {
	retValue, err := instance.GetProperty("ExceededLatencyLimit")
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

// SetExceededLatencyLimitPersec sets the value of ExceededLatencyLimitPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyExceededLatencyLimitPersec(value uint32) (err error) {
	return instance.SetProperty("ExceededLatencyLimitPersec", (value))
}

// GetExceededLatencyLimitPersec gets the value of ExceededLatencyLimitPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyExceededLatencyLimitPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ExceededLatencyLimitPersec")
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

// SetIO10000msPersec sets the value of IO10000msPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyIO10000msPersec(value uint32) (err error) {
	return instance.SetProperty("IO10000msPersec", (value))
}

// GetIO10000msPersec gets the value of IO10000msPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyIO10000msPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("IO10000msPersec")
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

// SetIO1000msPersec sets the value of IO1000msPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyIO1000msPersec(value uint32) (err error) {
	return instance.SetProperty("IO1000msPersec", (value))
}

// GetIO1000msPersec gets the value of IO1000msPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyIO1000msPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("IO1000msPersec")
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

// SetIO100msPersec sets the value of IO100msPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyIO100msPersec(value uint32) (err error) {
	return instance.SetProperty("IO100msPersec", (value))
}

// GetIO100msPersec gets the value of IO100msPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyIO100msPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("IO100msPersec")
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

// SetIO10msPersec sets the value of IO10msPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyIO10msPersec(value uint32) (err error) {
	return instance.SetProperty("IO10msPersec", (value))
}

// GetIO10msPersec gets the value of IO10msPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyIO10msPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("IO10msPersec")
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

// SetIO1msPersec sets the value of IO1msPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyIO1msPersec(value uint32) (err error) {
	return instance.SetProperty("IO1msPersec", (value))
}

// GetIO1msPersec gets the value of IO1msPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyIO1msPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("IO1msPersec")
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

// SetIO5msPersec sets the value of IO5msPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyIO5msPersec(value uint32) (err error) {
	return instance.SetProperty("IO5msPersec", (value))
}

// GetIO5msPersec gets the value of IO5msPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyIO5msPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("IO5msPersec")
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

// SetLocalReadAvgQueueLength sets the value of LocalReadAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyLocalReadAvgQueueLength(value uint64) (err error) {
	return instance.SetProperty("LocalReadAvgQueueLength", (value))
}

// GetLocalReadAvgQueueLength gets the value of LocalReadAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyLocalReadAvgQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("LocalReadAvgQueueLength")
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

// SetLocalReadBytes sets the value of LocalReadBytes for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyLocalReadBytes(value uint64) (err error) {
	return instance.SetProperty("LocalReadBytes", (value))
}

// GetLocalReadBytes gets the value of LocalReadBytes for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyLocalReadBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("LocalReadBytes")
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

// SetLocalReadBytesPersec sets the value of LocalReadBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyLocalReadBytesPersec(value uint64) (err error) {
	return instance.SetProperty("LocalReadBytesPersec", (value))
}

// GetLocalReadBytesPersec gets the value of LocalReadBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyLocalReadBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("LocalReadBytesPersec")
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

// SetLocalReadLatency sets the value of LocalReadLatency for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyLocalReadLatency(value uint32) (err error) {
	return instance.SetProperty("LocalReadLatency", (value))
}

// GetLocalReadLatency gets the value of LocalReadLatency for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyLocalReadLatency() (value uint32, err error) {
	retValue, err := instance.GetProperty("LocalReadLatency")
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

// SetLocalReadLatency_Base sets the value of LocalReadLatency_Base for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyLocalReadLatency_Base(value uint32) (err error) {
	return instance.SetProperty("LocalReadLatency_Base", (value))
}

// GetLocalReadLatency_Base gets the value of LocalReadLatency_Base for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyLocalReadLatency_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("LocalReadLatency_Base")
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

// SetLocalReadPersec sets the value of LocalReadPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyLocalReadPersec(value uint32) (err error) {
	return instance.SetProperty("LocalReadPersec", (value))
}

// GetLocalReadPersec gets the value of LocalReadPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyLocalReadPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("LocalReadPersec")
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

// SetLocalReadQueueLength sets the value of LocalReadQueueLength for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyLocalReadQueueLength(value uint64) (err error) {
	return instance.SetProperty("LocalReadQueueLength", (value))
}

// GetLocalReadQueueLength gets the value of LocalReadQueueLength for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyLocalReadQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("LocalReadQueueLength")
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

// SetLocalReads sets the value of LocalReads for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyLocalReads(value uint64) (err error) {
	return instance.SetProperty("LocalReads", (value))
}

// GetLocalReads gets the value of LocalReads for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyLocalReads() (value uint64, err error) {
	retValue, err := instance.GetProperty("LocalReads")
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

// SetLocalWriteAvgQueueLength sets the value of LocalWriteAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyLocalWriteAvgQueueLength(value uint64) (err error) {
	return instance.SetProperty("LocalWriteAvgQueueLength", (value))
}

// GetLocalWriteAvgQueueLength gets the value of LocalWriteAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyLocalWriteAvgQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("LocalWriteAvgQueueLength")
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

// SetLocalWriteBytes sets the value of LocalWriteBytes for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyLocalWriteBytes(value uint64) (err error) {
	return instance.SetProperty("LocalWriteBytes", (value))
}

// GetLocalWriteBytes gets the value of LocalWriteBytes for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyLocalWriteBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("LocalWriteBytes")
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

// SetLocalWriteBytesPersec sets the value of LocalWriteBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyLocalWriteBytesPersec(value uint64) (err error) {
	return instance.SetProperty("LocalWriteBytesPersec", (value))
}

// GetLocalWriteBytesPersec gets the value of LocalWriteBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyLocalWriteBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("LocalWriteBytesPersec")
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

// SetLocalWriteLatency sets the value of LocalWriteLatency for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyLocalWriteLatency(value uint32) (err error) {
	return instance.SetProperty("LocalWriteLatency", (value))
}

// GetLocalWriteLatency gets the value of LocalWriteLatency for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyLocalWriteLatency() (value uint32, err error) {
	retValue, err := instance.GetProperty("LocalWriteLatency")
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

// SetLocalWriteLatency_Base sets the value of LocalWriteLatency_Base for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyLocalWriteLatency_Base(value uint32) (err error) {
	return instance.SetProperty("LocalWriteLatency_Base", (value))
}

// GetLocalWriteLatency_Base gets the value of LocalWriteLatency_Base for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyLocalWriteLatency_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("LocalWriteLatency_Base")
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

// SetLocalWriteQueueLength sets the value of LocalWriteQueueLength for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyLocalWriteQueueLength(value uint64) (err error) {
	return instance.SetProperty("LocalWriteQueueLength", (value))
}

// GetLocalWriteQueueLength gets the value of LocalWriteQueueLength for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyLocalWriteQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("LocalWriteQueueLength")
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

// SetLocalWrites sets the value of LocalWrites for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyLocalWrites(value uint64) (err error) {
	return instance.SetProperty("LocalWrites", (value))
}

// GetLocalWrites gets the value of LocalWrites for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyLocalWrites() (value uint64, err error) {
	retValue, err := instance.GetProperty("LocalWrites")
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

// SetLocalWritesPersec sets the value of LocalWritesPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyLocalWritesPersec(value uint32) (err error) {
	return instance.SetProperty("LocalWritesPersec", (value))
}

// GetLocalWritesPersec gets the value of LocalWritesPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyLocalWritesPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("LocalWritesPersec")
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

// SetReadAvgQueueLength sets the value of ReadAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyReadAvgQueueLength(value uint64) (err error) {
	return instance.SetProperty("ReadAvgQueueLength", (value))
}

// GetReadAvgQueueLength gets the value of ReadAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyReadAvgQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadAvgQueueLength")
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
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyReadBytes(value uint64) (err error) {
	return instance.SetProperty("ReadBytes", (value))
}

// GetReadBytes gets the value of ReadBytes for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyReadBytes() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyReadBytesPersec(value uint64) (err error) {
	return instance.SetProperty("ReadBytesPersec", (value))
}

// GetReadBytesPersec gets the value of ReadBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyReadBytesPersec() (value uint64, err error) {
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

// SetReadLatency sets the value of ReadLatency for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyReadLatency(value uint32) (err error) {
	return instance.SetProperty("ReadLatency", (value))
}

// GetReadLatency gets the value of ReadLatency for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyReadLatency() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReadLatency")
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

// SetReadLatency_Base sets the value of ReadLatency_Base for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyReadLatency_Base(value uint32) (err error) {
	return instance.SetProperty("ReadLatency_Base", (value))
}

// GetReadLatency_Base gets the value of ReadLatency_Base for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyReadLatency_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReadLatency_Base")
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

// SetReadPersec sets the value of ReadPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyReadPersec(value uint32) (err error) {
	return instance.SetProperty("ReadPersec", (value))
}

// GetReadPersec gets the value of ReadPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyReadPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReadPersec")
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

// SetReadQueueLength sets the value of ReadQueueLength for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyReadQueueLength(value uint64) (err error) {
	return instance.SetProperty("ReadQueueLength", (value))
}

// GetReadQueueLength gets the value of ReadQueueLength for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyReadQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadQueueLength")
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
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyReads(value uint64) (err error) {
	return instance.SetProperty("Reads", (value))
}

// GetReads gets the value of Reads for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyReads() (value uint64, err error) {
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

// SetRemoteReadAvgQueueLength sets the value of RemoteReadAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyRemoteReadAvgQueueLength(value uint64) (err error) {
	return instance.SetProperty("RemoteReadAvgQueueLength", (value))
}

// GetRemoteReadAvgQueueLength gets the value of RemoteReadAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyRemoteReadAvgQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("RemoteReadAvgQueueLength")
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

// SetRemoteReadBytes sets the value of RemoteReadBytes for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyRemoteReadBytes(value uint64) (err error) {
	return instance.SetProperty("RemoteReadBytes", (value))
}

// GetRemoteReadBytes gets the value of RemoteReadBytes for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyRemoteReadBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("RemoteReadBytes")
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

// SetRemoteReadBytesPersec sets the value of RemoteReadBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyRemoteReadBytesPersec(value uint64) (err error) {
	return instance.SetProperty("RemoteReadBytesPersec", (value))
}

// GetRemoteReadBytesPersec gets the value of RemoteReadBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyRemoteReadBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("RemoteReadBytesPersec")
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

// SetRemoteReadLatency sets the value of RemoteReadLatency for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyRemoteReadLatency(value uint32) (err error) {
	return instance.SetProperty("RemoteReadLatency", (value))
}

// GetRemoteReadLatency gets the value of RemoteReadLatency for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyRemoteReadLatency() (value uint32, err error) {
	retValue, err := instance.GetProperty("RemoteReadLatency")
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

// SetRemoteReadLatency_Base sets the value of RemoteReadLatency_Base for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyRemoteReadLatency_Base(value uint32) (err error) {
	return instance.SetProperty("RemoteReadLatency_Base", (value))
}

// GetRemoteReadLatency_Base gets the value of RemoteReadLatency_Base for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyRemoteReadLatency_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("RemoteReadLatency_Base")
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

// SetRemoteReadPersec sets the value of RemoteReadPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyRemoteReadPersec(value uint32) (err error) {
	return instance.SetProperty("RemoteReadPersec", (value))
}

// GetRemoteReadPersec gets the value of RemoteReadPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyRemoteReadPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("RemoteReadPersec")
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

// SetRemoteReadQueueLength sets the value of RemoteReadQueueLength for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyRemoteReadQueueLength(value uint64) (err error) {
	return instance.SetProperty("RemoteReadQueueLength", (value))
}

// GetRemoteReadQueueLength gets the value of RemoteReadQueueLength for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyRemoteReadQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("RemoteReadQueueLength")
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

// SetRemoteReads sets the value of RemoteReads for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyRemoteReads(value uint64) (err error) {
	return instance.SetProperty("RemoteReads", (value))
}

// GetRemoteReads gets the value of RemoteReads for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyRemoteReads() (value uint64, err error) {
	retValue, err := instance.GetProperty("RemoteReads")
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

// SetRemoteWriteAvgQueueLength sets the value of RemoteWriteAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyRemoteWriteAvgQueueLength(value uint64) (err error) {
	return instance.SetProperty("RemoteWriteAvgQueueLength", (value))
}

// GetRemoteWriteAvgQueueLength gets the value of RemoteWriteAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyRemoteWriteAvgQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("RemoteWriteAvgQueueLength")
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

// SetRemoteWriteBytes sets the value of RemoteWriteBytes for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyRemoteWriteBytes(value uint64) (err error) {
	return instance.SetProperty("RemoteWriteBytes", (value))
}

// GetRemoteWriteBytes gets the value of RemoteWriteBytes for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyRemoteWriteBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("RemoteWriteBytes")
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

// SetRemoteWriteBytesPersec sets the value of RemoteWriteBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyRemoteWriteBytesPersec(value uint64) (err error) {
	return instance.SetProperty("RemoteWriteBytesPersec", (value))
}

// GetRemoteWriteBytesPersec gets the value of RemoteWriteBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyRemoteWriteBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("RemoteWriteBytesPersec")
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

// SetRemoteWriteLatency sets the value of RemoteWriteLatency for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyRemoteWriteLatency(value uint32) (err error) {
	return instance.SetProperty("RemoteWriteLatency", (value))
}

// GetRemoteWriteLatency gets the value of RemoteWriteLatency for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyRemoteWriteLatency() (value uint32, err error) {
	retValue, err := instance.GetProperty("RemoteWriteLatency")
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

// SetRemoteWriteLatency_Base sets the value of RemoteWriteLatency_Base for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyRemoteWriteLatency_Base(value uint32) (err error) {
	return instance.SetProperty("RemoteWriteLatency_Base", (value))
}

// GetRemoteWriteLatency_Base gets the value of RemoteWriteLatency_Base for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyRemoteWriteLatency_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("RemoteWriteLatency_Base")
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

// SetRemoteWriteQueueLength sets the value of RemoteWriteQueueLength for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyRemoteWriteQueueLength(value uint64) (err error) {
	return instance.SetProperty("RemoteWriteQueueLength", (value))
}

// GetRemoteWriteQueueLength gets the value of RemoteWriteQueueLength for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyRemoteWriteQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("RemoteWriteQueueLength")
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

// SetRemoteWrites sets the value of RemoteWrites for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyRemoteWrites(value uint64) (err error) {
	return instance.SetProperty("RemoteWrites", (value))
}

// GetRemoteWrites gets the value of RemoteWrites for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyRemoteWrites() (value uint64, err error) {
	retValue, err := instance.GetProperty("RemoteWrites")
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

// SetRemoteWritesPersec sets the value of RemoteWritesPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyRemoteWritesPersec(value uint32) (err error) {
	return instance.SetProperty("RemoteWritesPersec", (value))
}

// GetRemoteWritesPersec gets the value of RemoteWritesPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyRemoteWritesPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("RemoteWritesPersec")
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

// SetWriteAvgQueueLength sets the value of WriteAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyWriteAvgQueueLength(value uint64) (err error) {
	return instance.SetProperty("WriteAvgQueueLength", (value))
}

// GetWriteAvgQueueLength gets the value of WriteAvgQueueLength for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyWriteAvgQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteAvgQueueLength")
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
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyWriteBytes(value uint64) (err error) {
	return instance.SetProperty("WriteBytes", (value))
}

// GetWriteBytes gets the value of WriteBytes for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyWriteBytes() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyWriteBytesPersec(value uint32) (err error) {
	return instance.SetProperty("WriteBytesPersec", (value))
}

// GetWriteBytesPersec gets the value of WriteBytesPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyWriteBytesPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("WriteBytesPersec")
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

// SetWriteLatency sets the value of WriteLatency for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyWriteLatency(value uint32) (err error) {
	return instance.SetProperty("WriteLatency", (value))
}

// GetWriteLatency gets the value of WriteLatency for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyWriteLatency() (value uint32, err error) {
	retValue, err := instance.GetProperty("WriteLatency")
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

// SetWriteLatency_Base sets the value of WriteLatency_Base for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyWriteLatency_Base(value uint32) (err error) {
	return instance.SetProperty("WriteLatency_Base", (value))
}

// GetWriteLatency_Base gets the value of WriteLatency_Base for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyWriteLatency_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("WriteLatency_Base")
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

// SetWriteQueueLength sets the value of WriteQueueLength for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyWriteQueueLength(value uint64) (err error) {
	return instance.SetProperty("WriteQueueLength", (value))
}

// GetWriteQueueLength gets the value of WriteQueueLength for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyWriteQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteQueueLength")
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
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyWrites(value uint64) (err error) {
	return instance.SetProperty("Writes", (value))
}

// GetWrites gets the value of Writes for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyWrites() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) SetPropertyWritesPersec(value uint32) (err error) {
	return instance.SetProperty("WritesPersec", (value))
}

// GetWritesPersec gets the value of WritesPersec for the instance
func (instance *Win32_PerfRawData_ClusportPerfProvider_ClusterDiskCounters) GetPropertyWritesPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("WritesPersec")
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
