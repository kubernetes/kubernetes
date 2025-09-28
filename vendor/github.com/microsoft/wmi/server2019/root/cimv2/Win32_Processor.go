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

// Win32_Processor struct
type Win32_Processor struct {
	*CIM_Processor

	//
	Architecture uint16

	//
	AssetTag string

	//
	Characteristics uint32

	//
	CpuStatus uint16

	//
	CurrentVoltage uint16

	//
	ExtClock uint32

	//
	L2CacheSize uint32

	//
	L2CacheSpeed uint32

	//
	L3CacheSize uint32

	//
	L3CacheSpeed uint32

	//
	Level uint16

	//
	Manufacturer string

	//
	NumberOfCores uint32

	//
	NumberOfEnabledCore uint32

	//
	NumberOfLogicalProcessors uint32

	//
	PartNumber string

	//
	ProcessorId string

	//
	ProcessorType uint16

	//
	Revision uint16

	//
	SecondLevelAddressTranslationExtensions bool

	//
	SerialNumber string

	//
	SocketDesignation string

	//
	ThreadCount uint32

	//
	Version string

	//
	VirtualizationFirmwareEnabled bool

	//
	VMMonitorModeExtensions bool

	//
	VoltageCaps uint32
}

func NewWin32_ProcessorEx1(instance *cim.WmiInstance) (newInstance *Win32_Processor, err error) {
	tmp, err := NewCIM_ProcessorEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_Processor{
		CIM_Processor: tmp,
	}
	return
}

func NewWin32_ProcessorEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_Processor, err error) {
	tmp, err := NewCIM_ProcessorEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_Processor{
		CIM_Processor: tmp,
	}
	return
}

// SetArchitecture sets the value of Architecture for the instance
func (instance *Win32_Processor) SetPropertyArchitecture(value uint16) (err error) {
	return instance.SetProperty("Architecture", (value))
}

// GetArchitecture gets the value of Architecture for the instance
func (instance *Win32_Processor) GetPropertyArchitecture() (value uint16, err error) {
	retValue, err := instance.GetProperty("Architecture")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetAssetTag sets the value of AssetTag for the instance
func (instance *Win32_Processor) SetPropertyAssetTag(value string) (err error) {
	return instance.SetProperty("AssetTag", (value))
}

// GetAssetTag gets the value of AssetTag for the instance
func (instance *Win32_Processor) GetPropertyAssetTag() (value string, err error) {
	retValue, err := instance.GetProperty("AssetTag")
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

// SetCharacteristics sets the value of Characteristics for the instance
func (instance *Win32_Processor) SetPropertyCharacteristics(value uint32) (err error) {
	return instance.SetProperty("Characteristics", (value))
}

// GetCharacteristics gets the value of Characteristics for the instance
func (instance *Win32_Processor) GetPropertyCharacteristics() (value uint32, err error) {
	retValue, err := instance.GetProperty("Characteristics")
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

// SetCpuStatus sets the value of CpuStatus for the instance
func (instance *Win32_Processor) SetPropertyCpuStatus(value uint16) (err error) {
	return instance.SetProperty("CpuStatus", (value))
}

// GetCpuStatus gets the value of CpuStatus for the instance
func (instance *Win32_Processor) GetPropertyCpuStatus() (value uint16, err error) {
	retValue, err := instance.GetProperty("CpuStatus")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetCurrentVoltage sets the value of CurrentVoltage for the instance
func (instance *Win32_Processor) SetPropertyCurrentVoltage(value uint16) (err error) {
	return instance.SetProperty("CurrentVoltage", (value))
}

// GetCurrentVoltage gets the value of CurrentVoltage for the instance
func (instance *Win32_Processor) GetPropertyCurrentVoltage() (value uint16, err error) {
	retValue, err := instance.GetProperty("CurrentVoltage")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetExtClock sets the value of ExtClock for the instance
func (instance *Win32_Processor) SetPropertyExtClock(value uint32) (err error) {
	return instance.SetProperty("ExtClock", (value))
}

// GetExtClock gets the value of ExtClock for the instance
func (instance *Win32_Processor) GetPropertyExtClock() (value uint32, err error) {
	retValue, err := instance.GetProperty("ExtClock")
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

// SetL2CacheSize sets the value of L2CacheSize for the instance
func (instance *Win32_Processor) SetPropertyL2CacheSize(value uint32) (err error) {
	return instance.SetProperty("L2CacheSize", (value))
}

// GetL2CacheSize gets the value of L2CacheSize for the instance
func (instance *Win32_Processor) GetPropertyL2CacheSize() (value uint32, err error) {
	retValue, err := instance.GetProperty("L2CacheSize")
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

// SetL2CacheSpeed sets the value of L2CacheSpeed for the instance
func (instance *Win32_Processor) SetPropertyL2CacheSpeed(value uint32) (err error) {
	return instance.SetProperty("L2CacheSpeed", (value))
}

// GetL2CacheSpeed gets the value of L2CacheSpeed for the instance
func (instance *Win32_Processor) GetPropertyL2CacheSpeed() (value uint32, err error) {
	retValue, err := instance.GetProperty("L2CacheSpeed")
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

// SetL3CacheSize sets the value of L3CacheSize for the instance
func (instance *Win32_Processor) SetPropertyL3CacheSize(value uint32) (err error) {
	return instance.SetProperty("L3CacheSize", (value))
}

// GetL3CacheSize gets the value of L3CacheSize for the instance
func (instance *Win32_Processor) GetPropertyL3CacheSize() (value uint32, err error) {
	retValue, err := instance.GetProperty("L3CacheSize")
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

// SetL3CacheSpeed sets the value of L3CacheSpeed for the instance
func (instance *Win32_Processor) SetPropertyL3CacheSpeed(value uint32) (err error) {
	return instance.SetProperty("L3CacheSpeed", (value))
}

// GetL3CacheSpeed gets the value of L3CacheSpeed for the instance
func (instance *Win32_Processor) GetPropertyL3CacheSpeed() (value uint32, err error) {
	retValue, err := instance.GetProperty("L3CacheSpeed")
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

// SetLevel sets the value of Level for the instance
func (instance *Win32_Processor) SetPropertyLevel(value uint16) (err error) {
	return instance.SetProperty("Level", (value))
}

// GetLevel gets the value of Level for the instance
func (instance *Win32_Processor) GetPropertyLevel() (value uint16, err error) {
	retValue, err := instance.GetProperty("Level")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetManufacturer sets the value of Manufacturer for the instance
func (instance *Win32_Processor) SetPropertyManufacturer(value string) (err error) {
	return instance.SetProperty("Manufacturer", (value))
}

// GetManufacturer gets the value of Manufacturer for the instance
func (instance *Win32_Processor) GetPropertyManufacturer() (value string, err error) {
	retValue, err := instance.GetProperty("Manufacturer")
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

// SetNumberOfCores sets the value of NumberOfCores for the instance
func (instance *Win32_Processor) SetPropertyNumberOfCores(value uint32) (err error) {
	return instance.SetProperty("NumberOfCores", (value))
}

// GetNumberOfCores gets the value of NumberOfCores for the instance
func (instance *Win32_Processor) GetPropertyNumberOfCores() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfCores")
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

// SetNumberOfEnabledCore sets the value of NumberOfEnabledCore for the instance
func (instance *Win32_Processor) SetPropertyNumberOfEnabledCore(value uint32) (err error) {
	return instance.SetProperty("NumberOfEnabledCore", (value))
}

// GetNumberOfEnabledCore gets the value of NumberOfEnabledCore for the instance
func (instance *Win32_Processor) GetPropertyNumberOfEnabledCore() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfEnabledCore")
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

// SetNumberOfLogicalProcessors sets the value of NumberOfLogicalProcessors for the instance
func (instance *Win32_Processor) SetPropertyNumberOfLogicalProcessors(value uint32) (err error) {
	return instance.SetProperty("NumberOfLogicalProcessors", (value))
}

// GetNumberOfLogicalProcessors gets the value of NumberOfLogicalProcessors for the instance
func (instance *Win32_Processor) GetPropertyNumberOfLogicalProcessors() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfLogicalProcessors")
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

// SetPartNumber sets the value of PartNumber for the instance
func (instance *Win32_Processor) SetPropertyPartNumber(value string) (err error) {
	return instance.SetProperty("PartNumber", (value))
}

// GetPartNumber gets the value of PartNumber for the instance
func (instance *Win32_Processor) GetPropertyPartNumber() (value string, err error) {
	retValue, err := instance.GetProperty("PartNumber")
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

// SetProcessorId sets the value of ProcessorId for the instance
func (instance *Win32_Processor) SetPropertyProcessorId(value string) (err error) {
	return instance.SetProperty("ProcessorId", (value))
}

// GetProcessorId gets the value of ProcessorId for the instance
func (instance *Win32_Processor) GetPropertyProcessorId() (value string, err error) {
	retValue, err := instance.GetProperty("ProcessorId")
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

// SetProcessorType sets the value of ProcessorType for the instance
func (instance *Win32_Processor) SetPropertyProcessorType(value uint16) (err error) {
	return instance.SetProperty("ProcessorType", (value))
}

// GetProcessorType gets the value of ProcessorType for the instance
func (instance *Win32_Processor) GetPropertyProcessorType() (value uint16, err error) {
	retValue, err := instance.GetProperty("ProcessorType")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetRevision sets the value of Revision for the instance
func (instance *Win32_Processor) SetPropertyRevision(value uint16) (err error) {
	return instance.SetProperty("Revision", (value))
}

// GetRevision gets the value of Revision for the instance
func (instance *Win32_Processor) GetPropertyRevision() (value uint16, err error) {
	retValue, err := instance.GetProperty("Revision")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetSecondLevelAddressTranslationExtensions sets the value of SecondLevelAddressTranslationExtensions for the instance
func (instance *Win32_Processor) SetPropertySecondLevelAddressTranslationExtensions(value bool) (err error) {
	return instance.SetProperty("SecondLevelAddressTranslationExtensions", (value))
}

// GetSecondLevelAddressTranslationExtensions gets the value of SecondLevelAddressTranslationExtensions for the instance
func (instance *Win32_Processor) GetPropertySecondLevelAddressTranslationExtensions() (value bool, err error) {
	retValue, err := instance.GetProperty("SecondLevelAddressTranslationExtensions")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetSerialNumber sets the value of SerialNumber for the instance
func (instance *Win32_Processor) SetPropertySerialNumber(value string) (err error) {
	return instance.SetProperty("SerialNumber", (value))
}

// GetSerialNumber gets the value of SerialNumber for the instance
func (instance *Win32_Processor) GetPropertySerialNumber() (value string, err error) {
	retValue, err := instance.GetProperty("SerialNumber")
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

// SetSocketDesignation sets the value of SocketDesignation for the instance
func (instance *Win32_Processor) SetPropertySocketDesignation(value string) (err error) {
	return instance.SetProperty("SocketDesignation", (value))
}

// GetSocketDesignation gets the value of SocketDesignation for the instance
func (instance *Win32_Processor) GetPropertySocketDesignation() (value string, err error) {
	retValue, err := instance.GetProperty("SocketDesignation")
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

// SetThreadCount sets the value of ThreadCount for the instance
func (instance *Win32_Processor) SetPropertyThreadCount(value uint32) (err error) {
	return instance.SetProperty("ThreadCount", (value))
}

// GetThreadCount gets the value of ThreadCount for the instance
func (instance *Win32_Processor) GetPropertyThreadCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("ThreadCount")
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

// SetVersion sets the value of Version for the instance
func (instance *Win32_Processor) SetPropertyVersion(value string) (err error) {
	return instance.SetProperty("Version", (value))
}

// GetVersion gets the value of Version for the instance
func (instance *Win32_Processor) GetPropertyVersion() (value string, err error) {
	retValue, err := instance.GetProperty("Version")
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

// SetVirtualizationFirmwareEnabled sets the value of VirtualizationFirmwareEnabled for the instance
func (instance *Win32_Processor) SetPropertyVirtualizationFirmwareEnabled(value bool) (err error) {
	return instance.SetProperty("VirtualizationFirmwareEnabled", (value))
}

// GetVirtualizationFirmwareEnabled gets the value of VirtualizationFirmwareEnabled for the instance
func (instance *Win32_Processor) GetPropertyVirtualizationFirmwareEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("VirtualizationFirmwareEnabled")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetVMMonitorModeExtensions sets the value of VMMonitorModeExtensions for the instance
func (instance *Win32_Processor) SetPropertyVMMonitorModeExtensions(value bool) (err error) {
	return instance.SetProperty("VMMonitorModeExtensions", (value))
}

// GetVMMonitorModeExtensions gets the value of VMMonitorModeExtensions for the instance
func (instance *Win32_Processor) GetPropertyVMMonitorModeExtensions() (value bool, err error) {
	retValue, err := instance.GetProperty("VMMonitorModeExtensions")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetVoltageCaps sets the value of VoltageCaps for the instance
func (instance *Win32_Processor) SetPropertyVoltageCaps(value uint32) (err error) {
	return instance.SetProperty("VoltageCaps", (value))
}

// GetVoltageCaps gets the value of VoltageCaps for the instance
func (instance *Win32_Processor) GetPropertyVoltageCaps() (value uint32, err error) {
	retValue, err := instance.GetProperty("VoltageCaps")
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
