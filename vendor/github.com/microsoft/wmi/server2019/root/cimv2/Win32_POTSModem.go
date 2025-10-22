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

// Win32_POTSModem struct
type Win32_POTSModem struct {
	*CIM_PotsModem

	//
	AttachedTo string

	//
	BlindOff string

	//
	BlindOn string

	//
	CompatibilityFlags string

	//
	CompressionOff string

	//
	CompressionOn string

	//
	ConfigurationDialog string

	//
	DCB []uint8

	//
	Default []uint8

	//
	DeviceLoader string

	//
	DeviceType string

	//
	DriverDate string

	//
	ErrorControlForced string

	//
	ErrorControlOff string

	//
	ErrorControlOn string

	//
	FlowControlHard string

	//
	FlowControlOff string

	//
	FlowControlSoft string

	//
	InactivityScale string

	//
	Index uint32

	//
	IndexEx string

	//
	Model string

	//
	ModemInfPath string

	//
	ModemInfSection string

	//
	ModulationBell string

	//
	ModulationCCITT string

	//
	PortSubClass string

	//
	Prefix string

	//
	Properties []uint8

	//
	ProviderName string

	//
	Pulse string

	//
	Resetstring string

	//
	ResponsesKeyName string

	//
	SpeakerModeDial string

	//
	SpeakerModeOff string

	//
	SpeakerModeOn string

	//
	SpeakerModeSetup string

	//
	SpeakerVolumeHigh string

	//
	SpeakerVolumeLow string

	//
	SpeakerVolumeMed string

	//
	StringFormat string

	//
	Terminator string

	//
	Tone string

	//
	VoiceSwitchFeature string
}

func NewWin32_POTSModemEx1(instance *cim.WmiInstance) (newInstance *Win32_POTSModem, err error) {
	tmp, err := NewCIM_PotsModemEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_POTSModem{
		CIM_PotsModem: tmp,
	}
	return
}

func NewWin32_POTSModemEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_POTSModem, err error) {
	tmp, err := NewCIM_PotsModemEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_POTSModem{
		CIM_PotsModem: tmp,
	}
	return
}

// SetAttachedTo sets the value of AttachedTo for the instance
func (instance *Win32_POTSModem) SetPropertyAttachedTo(value string) (err error) {
	return instance.SetProperty("AttachedTo", (value))
}

// GetAttachedTo gets the value of AttachedTo for the instance
func (instance *Win32_POTSModem) GetPropertyAttachedTo() (value string, err error) {
	retValue, err := instance.GetProperty("AttachedTo")
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

// SetBlindOff sets the value of BlindOff for the instance
func (instance *Win32_POTSModem) SetPropertyBlindOff(value string) (err error) {
	return instance.SetProperty("BlindOff", (value))
}

// GetBlindOff gets the value of BlindOff for the instance
func (instance *Win32_POTSModem) GetPropertyBlindOff() (value string, err error) {
	retValue, err := instance.GetProperty("BlindOff")
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

// SetBlindOn sets the value of BlindOn for the instance
func (instance *Win32_POTSModem) SetPropertyBlindOn(value string) (err error) {
	return instance.SetProperty("BlindOn", (value))
}

// GetBlindOn gets the value of BlindOn for the instance
func (instance *Win32_POTSModem) GetPropertyBlindOn() (value string, err error) {
	retValue, err := instance.GetProperty("BlindOn")
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

// SetCompatibilityFlags sets the value of CompatibilityFlags for the instance
func (instance *Win32_POTSModem) SetPropertyCompatibilityFlags(value string) (err error) {
	return instance.SetProperty("CompatibilityFlags", (value))
}

// GetCompatibilityFlags gets the value of CompatibilityFlags for the instance
func (instance *Win32_POTSModem) GetPropertyCompatibilityFlags() (value string, err error) {
	retValue, err := instance.GetProperty("CompatibilityFlags")
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

// SetCompressionOff sets the value of CompressionOff for the instance
func (instance *Win32_POTSModem) SetPropertyCompressionOff(value string) (err error) {
	return instance.SetProperty("CompressionOff", (value))
}

// GetCompressionOff gets the value of CompressionOff for the instance
func (instance *Win32_POTSModem) GetPropertyCompressionOff() (value string, err error) {
	retValue, err := instance.GetProperty("CompressionOff")
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

// SetCompressionOn sets the value of CompressionOn for the instance
func (instance *Win32_POTSModem) SetPropertyCompressionOn(value string) (err error) {
	return instance.SetProperty("CompressionOn", (value))
}

// GetCompressionOn gets the value of CompressionOn for the instance
func (instance *Win32_POTSModem) GetPropertyCompressionOn() (value string, err error) {
	retValue, err := instance.GetProperty("CompressionOn")
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

// SetConfigurationDialog sets the value of ConfigurationDialog for the instance
func (instance *Win32_POTSModem) SetPropertyConfigurationDialog(value string) (err error) {
	return instance.SetProperty("ConfigurationDialog", (value))
}

// GetConfigurationDialog gets the value of ConfigurationDialog for the instance
func (instance *Win32_POTSModem) GetPropertyConfigurationDialog() (value string, err error) {
	retValue, err := instance.GetProperty("ConfigurationDialog")
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

// SetDCB sets the value of DCB for the instance
func (instance *Win32_POTSModem) SetPropertyDCB(value []uint8) (err error) {
	return instance.SetProperty("DCB", (value))
}

// GetDCB gets the value of DCB for the instance
func (instance *Win32_POTSModem) GetPropertyDCB() (value []uint8, err error) {
	retValue, err := instance.GetProperty("DCB")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint8)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint8 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint8(valuetmp))
	}

	return
}

// SetDefault sets the value of Default for the instance
func (instance *Win32_POTSModem) SetPropertyDefault(value []uint8) (err error) {
	return instance.SetProperty("Default", (value))
}

// GetDefault gets the value of Default for the instance
func (instance *Win32_POTSModem) GetPropertyDefault() (value []uint8, err error) {
	retValue, err := instance.GetProperty("Default")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint8)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint8 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint8(valuetmp))
	}

	return
}

// SetDeviceLoader sets the value of DeviceLoader for the instance
func (instance *Win32_POTSModem) SetPropertyDeviceLoader(value string) (err error) {
	return instance.SetProperty("DeviceLoader", (value))
}

// GetDeviceLoader gets the value of DeviceLoader for the instance
func (instance *Win32_POTSModem) GetPropertyDeviceLoader() (value string, err error) {
	retValue, err := instance.GetProperty("DeviceLoader")
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

// SetDeviceType sets the value of DeviceType for the instance
func (instance *Win32_POTSModem) SetPropertyDeviceType(value string) (err error) {
	return instance.SetProperty("DeviceType", (value))
}

// GetDeviceType gets the value of DeviceType for the instance
func (instance *Win32_POTSModem) GetPropertyDeviceType() (value string, err error) {
	retValue, err := instance.GetProperty("DeviceType")
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

// SetDriverDate sets the value of DriverDate for the instance
func (instance *Win32_POTSModem) SetPropertyDriverDate(value string) (err error) {
	return instance.SetProperty("DriverDate", (value))
}

// GetDriverDate gets the value of DriverDate for the instance
func (instance *Win32_POTSModem) GetPropertyDriverDate() (value string, err error) {
	retValue, err := instance.GetProperty("DriverDate")
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

// SetErrorControlForced sets the value of ErrorControlForced for the instance
func (instance *Win32_POTSModem) SetPropertyErrorControlForced(value string) (err error) {
	return instance.SetProperty("ErrorControlForced", (value))
}

// GetErrorControlForced gets the value of ErrorControlForced for the instance
func (instance *Win32_POTSModem) GetPropertyErrorControlForced() (value string, err error) {
	retValue, err := instance.GetProperty("ErrorControlForced")
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

// SetErrorControlOff sets the value of ErrorControlOff for the instance
func (instance *Win32_POTSModem) SetPropertyErrorControlOff(value string) (err error) {
	return instance.SetProperty("ErrorControlOff", (value))
}

// GetErrorControlOff gets the value of ErrorControlOff for the instance
func (instance *Win32_POTSModem) GetPropertyErrorControlOff() (value string, err error) {
	retValue, err := instance.GetProperty("ErrorControlOff")
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

// SetErrorControlOn sets the value of ErrorControlOn for the instance
func (instance *Win32_POTSModem) SetPropertyErrorControlOn(value string) (err error) {
	return instance.SetProperty("ErrorControlOn", (value))
}

// GetErrorControlOn gets the value of ErrorControlOn for the instance
func (instance *Win32_POTSModem) GetPropertyErrorControlOn() (value string, err error) {
	retValue, err := instance.GetProperty("ErrorControlOn")
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

// SetFlowControlHard sets the value of FlowControlHard for the instance
func (instance *Win32_POTSModem) SetPropertyFlowControlHard(value string) (err error) {
	return instance.SetProperty("FlowControlHard", (value))
}

// GetFlowControlHard gets the value of FlowControlHard for the instance
func (instance *Win32_POTSModem) GetPropertyFlowControlHard() (value string, err error) {
	retValue, err := instance.GetProperty("FlowControlHard")
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

// SetFlowControlOff sets the value of FlowControlOff for the instance
func (instance *Win32_POTSModem) SetPropertyFlowControlOff(value string) (err error) {
	return instance.SetProperty("FlowControlOff", (value))
}

// GetFlowControlOff gets the value of FlowControlOff for the instance
func (instance *Win32_POTSModem) GetPropertyFlowControlOff() (value string, err error) {
	retValue, err := instance.GetProperty("FlowControlOff")
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

// SetFlowControlSoft sets the value of FlowControlSoft for the instance
func (instance *Win32_POTSModem) SetPropertyFlowControlSoft(value string) (err error) {
	return instance.SetProperty("FlowControlSoft", (value))
}

// GetFlowControlSoft gets the value of FlowControlSoft for the instance
func (instance *Win32_POTSModem) GetPropertyFlowControlSoft() (value string, err error) {
	retValue, err := instance.GetProperty("FlowControlSoft")
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

// SetInactivityScale sets the value of InactivityScale for the instance
func (instance *Win32_POTSModem) SetPropertyInactivityScale(value string) (err error) {
	return instance.SetProperty("InactivityScale", (value))
}

// GetInactivityScale gets the value of InactivityScale for the instance
func (instance *Win32_POTSModem) GetPropertyInactivityScale() (value string, err error) {
	retValue, err := instance.GetProperty("InactivityScale")
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

// SetIndex sets the value of Index for the instance
func (instance *Win32_POTSModem) SetPropertyIndex(value uint32) (err error) {
	return instance.SetProperty("Index", (value))
}

// GetIndex gets the value of Index for the instance
func (instance *Win32_POTSModem) GetPropertyIndex() (value uint32, err error) {
	retValue, err := instance.GetProperty("Index")
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

// SetIndexEx sets the value of IndexEx for the instance
func (instance *Win32_POTSModem) SetPropertyIndexEx(value string) (err error) {
	return instance.SetProperty("IndexEx", (value))
}

// GetIndexEx gets the value of IndexEx for the instance
func (instance *Win32_POTSModem) GetPropertyIndexEx() (value string, err error) {
	retValue, err := instance.GetProperty("IndexEx")
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

// SetModel sets the value of Model for the instance
func (instance *Win32_POTSModem) SetPropertyModel(value string) (err error) {
	return instance.SetProperty("Model", (value))
}

// GetModel gets the value of Model for the instance
func (instance *Win32_POTSModem) GetPropertyModel() (value string, err error) {
	retValue, err := instance.GetProperty("Model")
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

// SetModemInfPath sets the value of ModemInfPath for the instance
func (instance *Win32_POTSModem) SetPropertyModemInfPath(value string) (err error) {
	return instance.SetProperty("ModemInfPath", (value))
}

// GetModemInfPath gets the value of ModemInfPath for the instance
func (instance *Win32_POTSModem) GetPropertyModemInfPath() (value string, err error) {
	retValue, err := instance.GetProperty("ModemInfPath")
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

// SetModemInfSection sets the value of ModemInfSection for the instance
func (instance *Win32_POTSModem) SetPropertyModemInfSection(value string) (err error) {
	return instance.SetProperty("ModemInfSection", (value))
}

// GetModemInfSection gets the value of ModemInfSection for the instance
func (instance *Win32_POTSModem) GetPropertyModemInfSection() (value string, err error) {
	retValue, err := instance.GetProperty("ModemInfSection")
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

// SetModulationBell sets the value of ModulationBell for the instance
func (instance *Win32_POTSModem) SetPropertyModulationBell(value string) (err error) {
	return instance.SetProperty("ModulationBell", (value))
}

// GetModulationBell gets the value of ModulationBell for the instance
func (instance *Win32_POTSModem) GetPropertyModulationBell() (value string, err error) {
	retValue, err := instance.GetProperty("ModulationBell")
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

// SetModulationCCITT sets the value of ModulationCCITT for the instance
func (instance *Win32_POTSModem) SetPropertyModulationCCITT(value string) (err error) {
	return instance.SetProperty("ModulationCCITT", (value))
}

// GetModulationCCITT gets the value of ModulationCCITT for the instance
func (instance *Win32_POTSModem) GetPropertyModulationCCITT() (value string, err error) {
	retValue, err := instance.GetProperty("ModulationCCITT")
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

// SetPortSubClass sets the value of PortSubClass for the instance
func (instance *Win32_POTSModem) SetPropertyPortSubClass(value string) (err error) {
	return instance.SetProperty("PortSubClass", (value))
}

// GetPortSubClass gets the value of PortSubClass for the instance
func (instance *Win32_POTSModem) GetPropertyPortSubClass() (value string, err error) {
	retValue, err := instance.GetProperty("PortSubClass")
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

// SetPrefix sets the value of Prefix for the instance
func (instance *Win32_POTSModem) SetPropertyPrefix(value string) (err error) {
	return instance.SetProperty("Prefix", (value))
}

// GetPrefix gets the value of Prefix for the instance
func (instance *Win32_POTSModem) GetPropertyPrefix() (value string, err error) {
	retValue, err := instance.GetProperty("Prefix")
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

// SetProperties sets the value of Properties for the instance
func (instance *Win32_POTSModem) SetPropertyProperties(value []uint8) (err error) {
	return instance.SetProperty("Properties", (value))
}

// GetProperties gets the value of Properties for the instance
func (instance *Win32_POTSModem) GetPropertyProperties() (value []uint8, err error) {
	retValue, err := instance.GetProperty("Properties")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint8)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint8 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint8(valuetmp))
	}

	return
}

// SetProviderName sets the value of ProviderName for the instance
func (instance *Win32_POTSModem) SetPropertyProviderName(value string) (err error) {
	return instance.SetProperty("ProviderName", (value))
}

// GetProviderName gets the value of ProviderName for the instance
func (instance *Win32_POTSModem) GetPropertyProviderName() (value string, err error) {
	retValue, err := instance.GetProperty("ProviderName")
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

// SetPulse sets the value of Pulse for the instance
func (instance *Win32_POTSModem) SetPropertyPulse(value string) (err error) {
	return instance.SetProperty("Pulse", (value))
}

// GetPulse gets the value of Pulse for the instance
func (instance *Win32_POTSModem) GetPropertyPulse() (value string, err error) {
	retValue, err := instance.GetProperty("Pulse")
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

// SetReset sets the value of Reset for the instance
func (instance *Win32_POTSModem) SetPropertyReset(value string) (err error) {
	return instance.SetProperty("Reset", (value))
}

// GetReset gets the value of Reset for the instance
func (instance *Win32_POTSModem) GetPropertyReset() (value string, err error) {
	retValue, err := instance.GetProperty("Reset")
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

// SetResponsesKeyName sets the value of ResponsesKeyName for the instance
func (instance *Win32_POTSModem) SetPropertyResponsesKeyName(value string) (err error) {
	return instance.SetProperty("ResponsesKeyName", (value))
}

// GetResponsesKeyName gets the value of ResponsesKeyName for the instance
func (instance *Win32_POTSModem) GetPropertyResponsesKeyName() (value string, err error) {
	retValue, err := instance.GetProperty("ResponsesKeyName")
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

// SetSpeakerModeDial sets the value of SpeakerModeDial for the instance
func (instance *Win32_POTSModem) SetPropertySpeakerModeDial(value string) (err error) {
	return instance.SetProperty("SpeakerModeDial", (value))
}

// GetSpeakerModeDial gets the value of SpeakerModeDial for the instance
func (instance *Win32_POTSModem) GetPropertySpeakerModeDial() (value string, err error) {
	retValue, err := instance.GetProperty("SpeakerModeDial")
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

// SetSpeakerModeOff sets the value of SpeakerModeOff for the instance
func (instance *Win32_POTSModem) SetPropertySpeakerModeOff(value string) (err error) {
	return instance.SetProperty("SpeakerModeOff", (value))
}

// GetSpeakerModeOff gets the value of SpeakerModeOff for the instance
func (instance *Win32_POTSModem) GetPropertySpeakerModeOff() (value string, err error) {
	retValue, err := instance.GetProperty("SpeakerModeOff")
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

// SetSpeakerModeOn sets the value of SpeakerModeOn for the instance
func (instance *Win32_POTSModem) SetPropertySpeakerModeOn(value string) (err error) {
	return instance.SetProperty("SpeakerModeOn", (value))
}

// GetSpeakerModeOn gets the value of SpeakerModeOn for the instance
func (instance *Win32_POTSModem) GetPropertySpeakerModeOn() (value string, err error) {
	retValue, err := instance.GetProperty("SpeakerModeOn")
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

// SetSpeakerModeSetup sets the value of SpeakerModeSetup for the instance
func (instance *Win32_POTSModem) SetPropertySpeakerModeSetup(value string) (err error) {
	return instance.SetProperty("SpeakerModeSetup", (value))
}

// GetSpeakerModeSetup gets the value of SpeakerModeSetup for the instance
func (instance *Win32_POTSModem) GetPropertySpeakerModeSetup() (value string, err error) {
	retValue, err := instance.GetProperty("SpeakerModeSetup")
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

// SetSpeakerVolumeHigh sets the value of SpeakerVolumeHigh for the instance
func (instance *Win32_POTSModem) SetPropertySpeakerVolumeHigh(value string) (err error) {
	return instance.SetProperty("SpeakerVolumeHigh", (value))
}

// GetSpeakerVolumeHigh gets the value of SpeakerVolumeHigh for the instance
func (instance *Win32_POTSModem) GetPropertySpeakerVolumeHigh() (value string, err error) {
	retValue, err := instance.GetProperty("SpeakerVolumeHigh")
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

// SetSpeakerVolumeLow sets the value of SpeakerVolumeLow for the instance
func (instance *Win32_POTSModem) SetPropertySpeakerVolumeLow(value string) (err error) {
	return instance.SetProperty("SpeakerVolumeLow", (value))
}

// GetSpeakerVolumeLow gets the value of SpeakerVolumeLow for the instance
func (instance *Win32_POTSModem) GetPropertySpeakerVolumeLow() (value string, err error) {
	retValue, err := instance.GetProperty("SpeakerVolumeLow")
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

// SetSpeakerVolumeMed sets the value of SpeakerVolumeMed for the instance
func (instance *Win32_POTSModem) SetPropertySpeakerVolumeMed(value string) (err error) {
	return instance.SetProperty("SpeakerVolumeMed", (value))
}

// GetSpeakerVolumeMed gets the value of SpeakerVolumeMed for the instance
func (instance *Win32_POTSModem) GetPropertySpeakerVolumeMed() (value string, err error) {
	retValue, err := instance.GetProperty("SpeakerVolumeMed")
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

// SetStringFormat sets the value of StringFormat for the instance
func (instance *Win32_POTSModem) SetPropertyStringFormat(value string) (err error) {
	return instance.SetProperty("StringFormat", (value))
}

// GetStringFormat gets the value of StringFormat for the instance
func (instance *Win32_POTSModem) GetPropertyStringFormat() (value string, err error) {
	retValue, err := instance.GetProperty("StringFormat")
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

// SetTerminator sets the value of Terminator for the instance
func (instance *Win32_POTSModem) SetPropertyTerminator(value string) (err error) {
	return instance.SetProperty("Terminator", (value))
}

// GetTerminator gets the value of Terminator for the instance
func (instance *Win32_POTSModem) GetPropertyTerminator() (value string, err error) {
	retValue, err := instance.GetProperty("Terminator")
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

// SetTone sets the value of Tone for the instance
func (instance *Win32_POTSModem) SetPropertyTone(value string) (err error) {
	return instance.SetProperty("Tone", (value))
}

// GetTone gets the value of Tone for the instance
func (instance *Win32_POTSModem) GetPropertyTone() (value string, err error) {
	retValue, err := instance.GetProperty("Tone")
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

// SetVoiceSwitchFeature sets the value of VoiceSwitchFeature for the instance
func (instance *Win32_POTSModem) SetPropertyVoiceSwitchFeature(value string) (err error) {
	return instance.SetProperty("VoiceSwitchFeature", (value))
}

// GetVoiceSwitchFeature gets the value of VoiceSwitchFeature for the instance
func (instance *Win32_POTSModem) GetPropertyVoiceSwitchFeature() (value string, err error) {
	retValue, err := instance.GetProperty("VoiceSwitchFeature")
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
