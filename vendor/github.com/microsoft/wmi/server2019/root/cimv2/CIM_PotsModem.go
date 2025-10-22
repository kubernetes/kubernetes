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

// CIM_PotsModem struct
type CIM_PotsModem struct {
	*CIM_LogicalDevice

	//
	AnswerMode uint16

	//
	CompressionInfo uint16

	//
	CountriesSupported []string

	//
	CountrySelected string

	//
	CurrentPasswords []string

	//
	DialType uint16

	//
	ErrorControlInfo uint16

	//
	InactivityTimeout uint32

	//
	MaxBaudRateToPhone uint32

	//
	MaxBaudRateToSerialPort uint32

	//
	MaxNumberOfPasswords uint16

	//
	ModulationScheme uint16

	//
	RingsBeforeAnswer uint8

	//
	SpeakerVolumeInfo uint16

	//
	SupportsCallback bool

	//
	SupportsSynchronousConnect bool

	//
	TimeOfLastReset string
}

func NewCIM_PotsModemEx1(instance *cim.WmiInstance) (newInstance *CIM_PotsModem, err error) {
	tmp, err := NewCIM_LogicalDeviceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_PotsModem{
		CIM_LogicalDevice: tmp,
	}
	return
}

func NewCIM_PotsModemEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_PotsModem, err error) {
	tmp, err := NewCIM_LogicalDeviceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_PotsModem{
		CIM_LogicalDevice: tmp,
	}
	return
}

// SetAnswerMode sets the value of AnswerMode for the instance
func (instance *CIM_PotsModem) SetPropertyAnswerMode(value uint16) (err error) {
	return instance.SetProperty("AnswerMode", (value))
}

// GetAnswerMode gets the value of AnswerMode for the instance
func (instance *CIM_PotsModem) GetPropertyAnswerMode() (value uint16, err error) {
	retValue, err := instance.GetProperty("AnswerMode")
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

// SetCompressionInfo sets the value of CompressionInfo for the instance
func (instance *CIM_PotsModem) SetPropertyCompressionInfo(value uint16) (err error) {
	return instance.SetProperty("CompressionInfo", (value))
}

// GetCompressionInfo gets the value of CompressionInfo for the instance
func (instance *CIM_PotsModem) GetPropertyCompressionInfo() (value uint16, err error) {
	retValue, err := instance.GetProperty("CompressionInfo")
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

// SetCountriesSupported sets the value of CountriesSupported for the instance
func (instance *CIM_PotsModem) SetPropertyCountriesSupported(value []string) (err error) {
	return instance.SetProperty("CountriesSupported", (value))
}

// GetCountriesSupported gets the value of CountriesSupported for the instance
func (instance *CIM_PotsModem) GetPropertyCountriesSupported() (value []string, err error) {
	retValue, err := instance.GetProperty("CountriesSupported")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetCountrySelected sets the value of CountrySelected for the instance
func (instance *CIM_PotsModem) SetPropertyCountrySelected(value string) (err error) {
	return instance.SetProperty("CountrySelected", (value))
}

// GetCountrySelected gets the value of CountrySelected for the instance
func (instance *CIM_PotsModem) GetPropertyCountrySelected() (value string, err error) {
	retValue, err := instance.GetProperty("CountrySelected")
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

// SetCurrentPasswords sets the value of CurrentPasswords for the instance
func (instance *CIM_PotsModem) SetPropertyCurrentPasswords(value []string) (err error) {
	return instance.SetProperty("CurrentPasswords", (value))
}

// GetCurrentPasswords gets the value of CurrentPasswords for the instance
func (instance *CIM_PotsModem) GetPropertyCurrentPasswords() (value []string, err error) {
	retValue, err := instance.GetProperty("CurrentPasswords")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetDialType sets the value of DialType for the instance
func (instance *CIM_PotsModem) SetPropertyDialType(value uint16) (err error) {
	return instance.SetProperty("DialType", (value))
}

// GetDialType gets the value of DialType for the instance
func (instance *CIM_PotsModem) GetPropertyDialType() (value uint16, err error) {
	retValue, err := instance.GetProperty("DialType")
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

// SetErrorControlInfo sets the value of ErrorControlInfo for the instance
func (instance *CIM_PotsModem) SetPropertyErrorControlInfo(value uint16) (err error) {
	return instance.SetProperty("ErrorControlInfo", (value))
}

// GetErrorControlInfo gets the value of ErrorControlInfo for the instance
func (instance *CIM_PotsModem) GetPropertyErrorControlInfo() (value uint16, err error) {
	retValue, err := instance.GetProperty("ErrorControlInfo")
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

// SetInactivityTimeout sets the value of InactivityTimeout for the instance
func (instance *CIM_PotsModem) SetPropertyInactivityTimeout(value uint32) (err error) {
	return instance.SetProperty("InactivityTimeout", (value))
}

// GetInactivityTimeout gets the value of InactivityTimeout for the instance
func (instance *CIM_PotsModem) GetPropertyInactivityTimeout() (value uint32, err error) {
	retValue, err := instance.GetProperty("InactivityTimeout")
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

// SetMaxBaudRateToPhone sets the value of MaxBaudRateToPhone for the instance
func (instance *CIM_PotsModem) SetPropertyMaxBaudRateToPhone(value uint32) (err error) {
	return instance.SetProperty("MaxBaudRateToPhone", (value))
}

// GetMaxBaudRateToPhone gets the value of MaxBaudRateToPhone for the instance
func (instance *CIM_PotsModem) GetPropertyMaxBaudRateToPhone() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaxBaudRateToPhone")
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

// SetMaxBaudRateToSerialPort sets the value of MaxBaudRateToSerialPort for the instance
func (instance *CIM_PotsModem) SetPropertyMaxBaudRateToSerialPort(value uint32) (err error) {
	return instance.SetProperty("MaxBaudRateToSerialPort", (value))
}

// GetMaxBaudRateToSerialPort gets the value of MaxBaudRateToSerialPort for the instance
func (instance *CIM_PotsModem) GetPropertyMaxBaudRateToSerialPort() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaxBaudRateToSerialPort")
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

// SetMaxNumberOfPasswords sets the value of MaxNumberOfPasswords for the instance
func (instance *CIM_PotsModem) SetPropertyMaxNumberOfPasswords(value uint16) (err error) {
	return instance.SetProperty("MaxNumberOfPasswords", (value))
}

// GetMaxNumberOfPasswords gets the value of MaxNumberOfPasswords for the instance
func (instance *CIM_PotsModem) GetPropertyMaxNumberOfPasswords() (value uint16, err error) {
	retValue, err := instance.GetProperty("MaxNumberOfPasswords")
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

// SetModulationScheme sets the value of ModulationScheme for the instance
func (instance *CIM_PotsModem) SetPropertyModulationScheme(value uint16) (err error) {
	return instance.SetProperty("ModulationScheme", (value))
}

// GetModulationScheme gets the value of ModulationScheme for the instance
func (instance *CIM_PotsModem) GetPropertyModulationScheme() (value uint16, err error) {
	retValue, err := instance.GetProperty("ModulationScheme")
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

// SetRingsBeforeAnswer sets the value of RingsBeforeAnswer for the instance
func (instance *CIM_PotsModem) SetPropertyRingsBeforeAnswer(value uint8) (err error) {
	return instance.SetProperty("RingsBeforeAnswer", (value))
}

// GetRingsBeforeAnswer gets the value of RingsBeforeAnswer for the instance
func (instance *CIM_PotsModem) GetPropertyRingsBeforeAnswer() (value uint8, err error) {
	retValue, err := instance.GetProperty("RingsBeforeAnswer")
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

// SetSpeakerVolumeInfo sets the value of SpeakerVolumeInfo for the instance
func (instance *CIM_PotsModem) SetPropertySpeakerVolumeInfo(value uint16) (err error) {
	return instance.SetProperty("SpeakerVolumeInfo", (value))
}

// GetSpeakerVolumeInfo gets the value of SpeakerVolumeInfo for the instance
func (instance *CIM_PotsModem) GetPropertySpeakerVolumeInfo() (value uint16, err error) {
	retValue, err := instance.GetProperty("SpeakerVolumeInfo")
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

// SetSupportsCallback sets the value of SupportsCallback for the instance
func (instance *CIM_PotsModem) SetPropertySupportsCallback(value bool) (err error) {
	return instance.SetProperty("SupportsCallback", (value))
}

// GetSupportsCallback gets the value of SupportsCallback for the instance
func (instance *CIM_PotsModem) GetPropertySupportsCallback() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsCallback")
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

// SetSupportsSynchronousConnect sets the value of SupportsSynchronousConnect for the instance
func (instance *CIM_PotsModem) SetPropertySupportsSynchronousConnect(value bool) (err error) {
	return instance.SetProperty("SupportsSynchronousConnect", (value))
}

// GetSupportsSynchronousConnect gets the value of SupportsSynchronousConnect for the instance
func (instance *CIM_PotsModem) GetPropertySupportsSynchronousConnect() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsSynchronousConnect")
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

// SetTimeOfLastReset sets the value of TimeOfLastReset for the instance
func (instance *CIM_PotsModem) SetPropertyTimeOfLastReset(value string) (err error) {
	return instance.SetProperty("TimeOfLastReset", (value))
}

// GetTimeOfLastReset gets the value of TimeOfLastReset for the instance
func (instance *CIM_PotsModem) GetPropertyTimeOfLastReset() (value string, err error) {
	retValue, err := instance.GetProperty("TimeOfLastReset")
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
