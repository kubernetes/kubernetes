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

// Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice struct
type Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice struct {
	*Win32_PerfFormattedData

	//
	AdapterOpenChannelCount uint32

	//
	ByteQuotaReplenishmentRate uint64

	//
	ErrorCount uint32

	//
	FlushCount uint32

	//
	IoQuotaReplenishmentRate uint64

	//
	Latency uint32

	//
	LowerLatency uint32

	//
	LowerQueueLength uint64

	//
	MaximumAdapterWorkerCount uint32

	//
	MaximumBandwidth uint64

	//
	MaximumIORate uint64

	//
	MinimumIORate uint64

	//
	NormalizedThroughput uint64

	//
	QueueLength uint64

	//
	ReadBytesPersec uint64

	//
	ReadCount uint32

	//
	ReadOperationsPerSec uint32

	//
	Throughput uint32

	//
	WriteBytesPersec uint64

	//
	WriteCount uint32

	//
	WriteOperationsPerSec uint32
}

func NewWin32_PerfFormattedData_Counters_HyperVVirtualStorageDeviceEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_HyperVVirtualStorageDeviceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetAdapterOpenChannelCount sets the value of AdapterOpenChannelCount for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) SetPropertyAdapterOpenChannelCount(value uint32) (err error) {
	return instance.SetProperty("AdapterOpenChannelCount", (value))
}

// GetAdapterOpenChannelCount gets the value of AdapterOpenChannelCount for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) GetPropertyAdapterOpenChannelCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("AdapterOpenChannelCount")
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

// SetByteQuotaReplenishmentRate sets the value of ByteQuotaReplenishmentRate for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) SetPropertyByteQuotaReplenishmentRate(value uint64) (err error) {
	return instance.SetProperty("ByteQuotaReplenishmentRate", (value))
}

// GetByteQuotaReplenishmentRate gets the value of ByteQuotaReplenishmentRate for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) GetPropertyByteQuotaReplenishmentRate() (value uint64, err error) {
	retValue, err := instance.GetProperty("ByteQuotaReplenishmentRate")
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

// SetErrorCount sets the value of ErrorCount for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) SetPropertyErrorCount(value uint32) (err error) {
	return instance.SetProperty("ErrorCount", (value))
}

// GetErrorCount gets the value of ErrorCount for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) GetPropertyErrorCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("ErrorCount")
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

// SetFlushCount sets the value of FlushCount for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) SetPropertyFlushCount(value uint32) (err error) {
	return instance.SetProperty("FlushCount", (value))
}

// GetFlushCount gets the value of FlushCount for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) GetPropertyFlushCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("FlushCount")
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

// SetIoQuotaReplenishmentRate sets the value of IoQuotaReplenishmentRate for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) SetPropertyIoQuotaReplenishmentRate(value uint64) (err error) {
	return instance.SetProperty("IoQuotaReplenishmentRate", (value))
}

// GetIoQuotaReplenishmentRate gets the value of IoQuotaReplenishmentRate for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) GetPropertyIoQuotaReplenishmentRate() (value uint64, err error) {
	retValue, err := instance.GetProperty("IoQuotaReplenishmentRate")
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

// SetLatency sets the value of Latency for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) SetPropertyLatency(value uint32) (err error) {
	return instance.SetProperty("Latency", (value))
}

// GetLatency gets the value of Latency for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) GetPropertyLatency() (value uint32, err error) {
	retValue, err := instance.GetProperty("Latency")
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

// SetLowerLatency sets the value of LowerLatency for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) SetPropertyLowerLatency(value uint32) (err error) {
	return instance.SetProperty("LowerLatency", (value))
}

// GetLowerLatency gets the value of LowerLatency for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) GetPropertyLowerLatency() (value uint32, err error) {
	retValue, err := instance.GetProperty("LowerLatency")
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

// SetLowerQueueLength sets the value of LowerQueueLength for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) SetPropertyLowerQueueLength(value uint64) (err error) {
	return instance.SetProperty("LowerQueueLength", (value))
}

// GetLowerQueueLength gets the value of LowerQueueLength for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) GetPropertyLowerQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("LowerQueueLength")
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

// SetMaximumAdapterWorkerCount sets the value of MaximumAdapterWorkerCount for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) SetPropertyMaximumAdapterWorkerCount(value uint32) (err error) {
	return instance.SetProperty("MaximumAdapterWorkerCount", (value))
}

// GetMaximumAdapterWorkerCount gets the value of MaximumAdapterWorkerCount for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) GetPropertyMaximumAdapterWorkerCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaximumAdapterWorkerCount")
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

// SetMaximumBandwidth sets the value of MaximumBandwidth for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) SetPropertyMaximumBandwidth(value uint64) (err error) {
	return instance.SetProperty("MaximumBandwidth", (value))
}

// GetMaximumBandwidth gets the value of MaximumBandwidth for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) GetPropertyMaximumBandwidth() (value uint64, err error) {
	retValue, err := instance.GetProperty("MaximumBandwidth")
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

// SetMaximumIORate sets the value of MaximumIORate for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) SetPropertyMaximumIORate(value uint64) (err error) {
	return instance.SetProperty("MaximumIORate", (value))
}

// GetMaximumIORate gets the value of MaximumIORate for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) GetPropertyMaximumIORate() (value uint64, err error) {
	retValue, err := instance.GetProperty("MaximumIORate")
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

// SetMinimumIORate sets the value of MinimumIORate for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) SetPropertyMinimumIORate(value uint64) (err error) {
	return instance.SetProperty("MinimumIORate", (value))
}

// GetMinimumIORate gets the value of MinimumIORate for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) GetPropertyMinimumIORate() (value uint64, err error) {
	retValue, err := instance.GetProperty("MinimumIORate")
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

// SetNormalizedThroughput sets the value of NormalizedThroughput for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) SetPropertyNormalizedThroughput(value uint64) (err error) {
	return instance.SetProperty("NormalizedThroughput", (value))
}

// GetNormalizedThroughput gets the value of NormalizedThroughput for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) GetPropertyNormalizedThroughput() (value uint64, err error) {
	retValue, err := instance.GetProperty("NormalizedThroughput")
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

// SetQueueLength sets the value of QueueLength for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) SetPropertyQueueLength(value uint64) (err error) {
	return instance.SetProperty("QueueLength", (value))
}

// GetQueueLength gets the value of QueueLength for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) GetPropertyQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("QueueLength")
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
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) SetPropertyReadBytesPersec(value uint64) (err error) {
	return instance.SetProperty("ReadBytesPersec", (value))
}

// GetReadBytesPersec gets the value of ReadBytesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) GetPropertyReadBytesPersec() (value uint64, err error) {
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

// SetReadCount sets the value of ReadCount for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) SetPropertyReadCount(value uint32) (err error) {
	return instance.SetProperty("ReadCount", (value))
}

// GetReadCount gets the value of ReadCount for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) GetPropertyReadCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReadCount")
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

// SetReadOperationsPerSec sets the value of ReadOperationsPerSec for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) SetPropertyReadOperationsPerSec(value uint32) (err error) {
	return instance.SetProperty("ReadOperationsPerSec", (value))
}

// GetReadOperationsPerSec gets the value of ReadOperationsPerSec for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) GetPropertyReadOperationsPerSec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReadOperationsPerSec")
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

// SetThroughput sets the value of Throughput for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) SetPropertyThroughput(value uint32) (err error) {
	return instance.SetProperty("Throughput", (value))
}

// GetThroughput gets the value of Throughput for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) GetPropertyThroughput() (value uint32, err error) {
	retValue, err := instance.GetProperty("Throughput")
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
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) SetPropertyWriteBytesPersec(value uint64) (err error) {
	return instance.SetProperty("WriteBytesPersec", (value))
}

// GetWriteBytesPersec gets the value of WriteBytesPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) GetPropertyWriteBytesPersec() (value uint64, err error) {
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

// SetWriteCount sets the value of WriteCount for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) SetPropertyWriteCount(value uint32) (err error) {
	return instance.SetProperty("WriteCount", (value))
}

// GetWriteCount gets the value of WriteCount for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) GetPropertyWriteCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("WriteCount")
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

// SetWriteOperationsPerSec sets the value of WriteOperationsPerSec for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) SetPropertyWriteOperationsPerSec(value uint32) (err error) {
	return instance.SetProperty("WriteOperationsPerSec", (value))
}

// GetWriteOperationsPerSec gets the value of WriteOperationsPerSec for the instance
func (instance *Win32_PerfFormattedData_Counters_HyperVVirtualStorageDevice) GetPropertyWriteOperationsPerSec() (value uint32, err error) {
	retValue, err := instance.GetProperty("WriteOperationsPerSec")
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
