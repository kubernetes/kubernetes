// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.Microsoft.Windows.Storage
//////////////////////////////////////////////
package storage

import (
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// MSFT_StorageReliabilityCounter struct
type MSFT_StorageReliabilityCounter struct {
	*MSFT_StorageObject

	//
	DeviceId string

	//
	FlushLatencyMax uint64

	//
	LoadUnloadCycleCount uint32

	//
	LoadUnloadCycleCountMax uint32

	//
	ManufactureDate string

	//
	PowerOnHours uint32

	//
	ReadErrorsCorrected uint64

	//
	ReadErrorsTotal uint64

	//
	ReadErrorsUncorrected uint64

	//
	ReadLatencyMax uint64

	//
	StartStopCycleCount uint32

	//
	StartStopCycleCountMax uint32

	//
	Temperature uint8

	//
	TemperatureMax uint8

	//
	Wear uint8

	//
	WriteErrorsCorrected uint64

	//
	WriteErrorsTotal uint64

	//
	WriteErrorsUncorrected uint64

	//
	WriteLatencyMax uint64
}

func NewMSFT_StorageReliabilityCounterEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageReliabilityCounter, err error) {
	tmp, err := NewMSFT_StorageObjectEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageReliabilityCounter{
		MSFT_StorageObject: tmp,
	}
	return
}

func NewMSFT_StorageReliabilityCounterEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageReliabilityCounter, err error) {
	tmp, err := NewMSFT_StorageObjectEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageReliabilityCounter{
		MSFT_StorageObject: tmp,
	}
	return
}

// SetDeviceId sets the value of DeviceId for the instance
func (instance *MSFT_StorageReliabilityCounter) SetPropertyDeviceId(value string) (err error) {
	return instance.SetProperty("DeviceId", (value))
}

// GetDeviceId gets the value of DeviceId for the instance
func (instance *MSFT_StorageReliabilityCounter) GetPropertyDeviceId() (value string, err error) {
	retValue, err := instance.GetProperty("DeviceId")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetFlushLatencyMax sets the value of FlushLatencyMax for the instance
func (instance *MSFT_StorageReliabilityCounter) SetPropertyFlushLatencyMax(value uint64) (err error) {
	return instance.SetProperty("FlushLatencyMax", (value))
}

// GetFlushLatencyMax gets the value of FlushLatencyMax for the instance
func (instance *MSFT_StorageReliabilityCounter) GetPropertyFlushLatencyMax() (value uint64, err error) {
	retValue, err := instance.GetProperty("FlushLatencyMax")
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

// SetLoadUnloadCycleCount sets the value of LoadUnloadCycleCount for the instance
func (instance *MSFT_StorageReliabilityCounter) SetPropertyLoadUnloadCycleCount(value uint32) (err error) {
	return instance.SetProperty("LoadUnloadCycleCount", (value))
}

// GetLoadUnloadCycleCount gets the value of LoadUnloadCycleCount for the instance
func (instance *MSFT_StorageReliabilityCounter) GetPropertyLoadUnloadCycleCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("LoadUnloadCycleCount")
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

// SetLoadUnloadCycleCountMax sets the value of LoadUnloadCycleCountMax for the instance
func (instance *MSFT_StorageReliabilityCounter) SetPropertyLoadUnloadCycleCountMax(value uint32) (err error) {
	return instance.SetProperty("LoadUnloadCycleCountMax", (value))
}

// GetLoadUnloadCycleCountMax gets the value of LoadUnloadCycleCountMax for the instance
func (instance *MSFT_StorageReliabilityCounter) GetPropertyLoadUnloadCycleCountMax() (value uint32, err error) {
	retValue, err := instance.GetProperty("LoadUnloadCycleCountMax")
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

// SetManufactureDate sets the value of ManufactureDate for the instance
func (instance *MSFT_StorageReliabilityCounter) SetPropertyManufactureDate(value string) (err error) {
	return instance.SetProperty("ManufactureDate", (value))
}

// GetManufactureDate gets the value of ManufactureDate for the instance
func (instance *MSFT_StorageReliabilityCounter) GetPropertyManufactureDate() (value string, err error) {
	retValue, err := instance.GetProperty("ManufactureDate")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetPowerOnHours sets the value of PowerOnHours for the instance
func (instance *MSFT_StorageReliabilityCounter) SetPropertyPowerOnHours(value uint32) (err error) {
	return instance.SetProperty("PowerOnHours", (value))
}

// GetPowerOnHours gets the value of PowerOnHours for the instance
func (instance *MSFT_StorageReliabilityCounter) GetPropertyPowerOnHours() (value uint32, err error) {
	retValue, err := instance.GetProperty("PowerOnHours")
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

// SetReadErrorsCorrected sets the value of ReadErrorsCorrected for the instance
func (instance *MSFT_StorageReliabilityCounter) SetPropertyReadErrorsCorrected(value uint64) (err error) {
	return instance.SetProperty("ReadErrorsCorrected", (value))
}

// GetReadErrorsCorrected gets the value of ReadErrorsCorrected for the instance
func (instance *MSFT_StorageReliabilityCounter) GetPropertyReadErrorsCorrected() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadErrorsCorrected")
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

// SetReadErrorsTotal sets the value of ReadErrorsTotal for the instance
func (instance *MSFT_StorageReliabilityCounter) SetPropertyReadErrorsTotal(value uint64) (err error) {
	return instance.SetProperty("ReadErrorsTotal", (value))
}

// GetReadErrorsTotal gets the value of ReadErrorsTotal for the instance
func (instance *MSFT_StorageReliabilityCounter) GetPropertyReadErrorsTotal() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadErrorsTotal")
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

// SetReadErrorsUncorrected sets the value of ReadErrorsUncorrected for the instance
func (instance *MSFT_StorageReliabilityCounter) SetPropertyReadErrorsUncorrected(value uint64) (err error) {
	return instance.SetProperty("ReadErrorsUncorrected", (value))
}

// GetReadErrorsUncorrected gets the value of ReadErrorsUncorrected for the instance
func (instance *MSFT_StorageReliabilityCounter) GetPropertyReadErrorsUncorrected() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadErrorsUncorrected")
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

// SetReadLatencyMax sets the value of ReadLatencyMax for the instance
func (instance *MSFT_StorageReliabilityCounter) SetPropertyReadLatencyMax(value uint64) (err error) {
	return instance.SetProperty("ReadLatencyMax", (value))
}

// GetReadLatencyMax gets the value of ReadLatencyMax for the instance
func (instance *MSFT_StorageReliabilityCounter) GetPropertyReadLatencyMax() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadLatencyMax")
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

// SetStartStopCycleCount sets the value of StartStopCycleCount for the instance
func (instance *MSFT_StorageReliabilityCounter) SetPropertyStartStopCycleCount(value uint32) (err error) {
	return instance.SetProperty("StartStopCycleCount", (value))
}

// GetStartStopCycleCount gets the value of StartStopCycleCount for the instance
func (instance *MSFT_StorageReliabilityCounter) GetPropertyStartStopCycleCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("StartStopCycleCount")
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

// SetStartStopCycleCountMax sets the value of StartStopCycleCountMax for the instance
func (instance *MSFT_StorageReliabilityCounter) SetPropertyStartStopCycleCountMax(value uint32) (err error) {
	return instance.SetProperty("StartStopCycleCountMax", (value))
}

// GetStartStopCycleCountMax gets the value of StartStopCycleCountMax for the instance
func (instance *MSFT_StorageReliabilityCounter) GetPropertyStartStopCycleCountMax() (value uint32, err error) {
	retValue, err := instance.GetProperty("StartStopCycleCountMax")
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

// SetTemperature sets the value of Temperature for the instance
func (instance *MSFT_StorageReliabilityCounter) SetPropertyTemperature(value uint8) (err error) {
	return instance.SetProperty("Temperature", (value))
}

// GetTemperature gets the value of Temperature for the instance
func (instance *MSFT_StorageReliabilityCounter) GetPropertyTemperature() (value uint8, err error) {
	retValue, err := instance.GetProperty("Temperature")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint8)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint8 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint8(valuetmp)

	return
}

// SetTemperatureMax sets the value of TemperatureMax for the instance
func (instance *MSFT_StorageReliabilityCounter) SetPropertyTemperatureMax(value uint8) (err error) {
	return instance.SetProperty("TemperatureMax", (value))
}

// GetTemperatureMax gets the value of TemperatureMax for the instance
func (instance *MSFT_StorageReliabilityCounter) GetPropertyTemperatureMax() (value uint8, err error) {
	retValue, err := instance.GetProperty("TemperatureMax")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint8)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint8 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint8(valuetmp)

	return
}

// SetWear sets the value of Wear for the instance
func (instance *MSFT_StorageReliabilityCounter) SetPropertyWear(value uint8) (err error) {
	return instance.SetProperty("Wear", (value))
}

// GetWear gets the value of Wear for the instance
func (instance *MSFT_StorageReliabilityCounter) GetPropertyWear() (value uint8, err error) {
	retValue, err := instance.GetProperty("Wear")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint8)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint8 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint8(valuetmp)

	return
}

// SetWriteErrorsCorrected sets the value of WriteErrorsCorrected for the instance
func (instance *MSFT_StorageReliabilityCounter) SetPropertyWriteErrorsCorrected(value uint64) (err error) {
	return instance.SetProperty("WriteErrorsCorrected", (value))
}

// GetWriteErrorsCorrected gets the value of WriteErrorsCorrected for the instance
func (instance *MSFT_StorageReliabilityCounter) GetPropertyWriteErrorsCorrected() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteErrorsCorrected")
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

// SetWriteErrorsTotal sets the value of WriteErrorsTotal for the instance
func (instance *MSFT_StorageReliabilityCounter) SetPropertyWriteErrorsTotal(value uint64) (err error) {
	return instance.SetProperty("WriteErrorsTotal", (value))
}

// GetWriteErrorsTotal gets the value of WriteErrorsTotal for the instance
func (instance *MSFT_StorageReliabilityCounter) GetPropertyWriteErrorsTotal() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteErrorsTotal")
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

// SetWriteErrorsUncorrected sets the value of WriteErrorsUncorrected for the instance
func (instance *MSFT_StorageReliabilityCounter) SetPropertyWriteErrorsUncorrected(value uint64) (err error) {
	return instance.SetProperty("WriteErrorsUncorrected", (value))
}

// GetWriteErrorsUncorrected gets the value of WriteErrorsUncorrected for the instance
func (instance *MSFT_StorageReliabilityCounter) GetPropertyWriteErrorsUncorrected() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteErrorsUncorrected")
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

// SetWriteLatencyMax sets the value of WriteLatencyMax for the instance
func (instance *MSFT_StorageReliabilityCounter) SetPropertyWriteLatencyMax(value uint64) (err error) {
	return instance.SetProperty("WriteLatencyMax", (value))
}

// GetWriteLatencyMax gets the value of WriteLatencyMax for the instance
func (instance *MSFT_StorageReliabilityCounter) GetPropertyWriteLatencyMax() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteLatencyMax")
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

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageReliabilityCounter) Reset() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Reset")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}
