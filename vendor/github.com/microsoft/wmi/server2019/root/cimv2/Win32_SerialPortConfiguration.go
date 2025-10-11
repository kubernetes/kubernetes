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

// Win32_SerialPortConfiguration struct
type Win32_SerialPortConfiguration struct {
	*CIM_Setting

	//
	AbortReadWriteOnError bool

	//
	BaudRate uint32

	//
	BinaryModeEnabled bool

	//
	BitsPerByte uint32

	//
	ContinueXMitOnXOff bool

	//
	CTSOutflowControl bool

	//
	DiscardNULLBytes bool

	//
	DSROutflowControl bool

	//
	DSRSensitivity bool

	//
	DTRFlowControlType string

	//
	EOFCharacter uint32

	//
	ErrorReplaceCharacter uint32

	//
	ErrorReplacementEnabled bool

	//
	EventCharacter uint32

	//
	IsBusy bool

	//
	Name string

	//
	Parity string

	//
	ParityCheckEnabled bool

	//
	RTSFlowControlType string

	//
	StopBits string

	//
	XOffCharacter uint32

	//
	XOffXMitThreshold uint32

	//
	XOnCharacter uint32

	//
	XOnXMitThreshold uint32

	//
	XOnXOffInFlowControl uint32

	//
	XOnXOffOutFlowControl uint32
}

func NewWin32_SerialPortConfigurationEx1(instance *cim.WmiInstance) (newInstance *Win32_SerialPortConfiguration, err error) {
	tmp, err := NewCIM_SettingEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_SerialPortConfiguration{
		CIM_Setting: tmp,
	}
	return
}

func NewWin32_SerialPortConfigurationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_SerialPortConfiguration, err error) {
	tmp, err := NewCIM_SettingEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_SerialPortConfiguration{
		CIM_Setting: tmp,
	}
	return
}

// SetAbortReadWriteOnError sets the value of AbortReadWriteOnError for the instance
func (instance *Win32_SerialPortConfiguration) SetPropertyAbortReadWriteOnError(value bool) (err error) {
	return instance.SetProperty("AbortReadWriteOnError", (value))
}

// GetAbortReadWriteOnError gets the value of AbortReadWriteOnError for the instance
func (instance *Win32_SerialPortConfiguration) GetPropertyAbortReadWriteOnError() (value bool, err error) {
	retValue, err := instance.GetProperty("AbortReadWriteOnError")
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

// SetBaudRate sets the value of BaudRate for the instance
func (instance *Win32_SerialPortConfiguration) SetPropertyBaudRate(value uint32) (err error) {
	return instance.SetProperty("BaudRate", (value))
}

// GetBaudRate gets the value of BaudRate for the instance
func (instance *Win32_SerialPortConfiguration) GetPropertyBaudRate() (value uint32, err error) {
	retValue, err := instance.GetProperty("BaudRate")
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

// SetBinaryModeEnabled sets the value of BinaryModeEnabled for the instance
func (instance *Win32_SerialPortConfiguration) SetPropertyBinaryModeEnabled(value bool) (err error) {
	return instance.SetProperty("BinaryModeEnabled", (value))
}

// GetBinaryModeEnabled gets the value of BinaryModeEnabled for the instance
func (instance *Win32_SerialPortConfiguration) GetPropertyBinaryModeEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("BinaryModeEnabled")
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

// SetBitsPerByte sets the value of BitsPerByte for the instance
func (instance *Win32_SerialPortConfiguration) SetPropertyBitsPerByte(value uint32) (err error) {
	return instance.SetProperty("BitsPerByte", (value))
}

// GetBitsPerByte gets the value of BitsPerByte for the instance
func (instance *Win32_SerialPortConfiguration) GetPropertyBitsPerByte() (value uint32, err error) {
	retValue, err := instance.GetProperty("BitsPerByte")
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

// SetContinueXMitOnXOff sets the value of ContinueXMitOnXOff for the instance
func (instance *Win32_SerialPortConfiguration) SetPropertyContinueXMitOnXOff(value bool) (err error) {
	return instance.SetProperty("ContinueXMitOnXOff", (value))
}

// GetContinueXMitOnXOff gets the value of ContinueXMitOnXOff for the instance
func (instance *Win32_SerialPortConfiguration) GetPropertyContinueXMitOnXOff() (value bool, err error) {
	retValue, err := instance.GetProperty("ContinueXMitOnXOff")
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

// SetCTSOutflowControl sets the value of CTSOutflowControl for the instance
func (instance *Win32_SerialPortConfiguration) SetPropertyCTSOutflowControl(value bool) (err error) {
	return instance.SetProperty("CTSOutflowControl", (value))
}

// GetCTSOutflowControl gets the value of CTSOutflowControl for the instance
func (instance *Win32_SerialPortConfiguration) GetPropertyCTSOutflowControl() (value bool, err error) {
	retValue, err := instance.GetProperty("CTSOutflowControl")
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

// SetDiscardNULLBytes sets the value of DiscardNULLBytes for the instance
func (instance *Win32_SerialPortConfiguration) SetPropertyDiscardNULLBytes(value bool) (err error) {
	return instance.SetProperty("DiscardNULLBytes", (value))
}

// GetDiscardNULLBytes gets the value of DiscardNULLBytes for the instance
func (instance *Win32_SerialPortConfiguration) GetPropertyDiscardNULLBytes() (value bool, err error) {
	retValue, err := instance.GetProperty("DiscardNULLBytes")
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

// SetDSROutflowControl sets the value of DSROutflowControl for the instance
func (instance *Win32_SerialPortConfiguration) SetPropertyDSROutflowControl(value bool) (err error) {
	return instance.SetProperty("DSROutflowControl", (value))
}

// GetDSROutflowControl gets the value of DSROutflowControl for the instance
func (instance *Win32_SerialPortConfiguration) GetPropertyDSROutflowControl() (value bool, err error) {
	retValue, err := instance.GetProperty("DSROutflowControl")
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

// SetDSRSensitivity sets the value of DSRSensitivity for the instance
func (instance *Win32_SerialPortConfiguration) SetPropertyDSRSensitivity(value bool) (err error) {
	return instance.SetProperty("DSRSensitivity", (value))
}

// GetDSRSensitivity gets the value of DSRSensitivity for the instance
func (instance *Win32_SerialPortConfiguration) GetPropertyDSRSensitivity() (value bool, err error) {
	retValue, err := instance.GetProperty("DSRSensitivity")
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

// SetDTRFlowControlType sets the value of DTRFlowControlType for the instance
func (instance *Win32_SerialPortConfiguration) SetPropertyDTRFlowControlType(value string) (err error) {
	return instance.SetProperty("DTRFlowControlType", (value))
}

// GetDTRFlowControlType gets the value of DTRFlowControlType for the instance
func (instance *Win32_SerialPortConfiguration) GetPropertyDTRFlowControlType() (value string, err error) {
	retValue, err := instance.GetProperty("DTRFlowControlType")
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

// SetEOFCharacter sets the value of EOFCharacter for the instance
func (instance *Win32_SerialPortConfiguration) SetPropertyEOFCharacter(value uint32) (err error) {
	return instance.SetProperty("EOFCharacter", (value))
}

// GetEOFCharacter gets the value of EOFCharacter for the instance
func (instance *Win32_SerialPortConfiguration) GetPropertyEOFCharacter() (value uint32, err error) {
	retValue, err := instance.GetProperty("EOFCharacter")
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

// SetErrorReplaceCharacter sets the value of ErrorReplaceCharacter for the instance
func (instance *Win32_SerialPortConfiguration) SetPropertyErrorReplaceCharacter(value uint32) (err error) {
	return instance.SetProperty("ErrorReplaceCharacter", (value))
}

// GetErrorReplaceCharacter gets the value of ErrorReplaceCharacter for the instance
func (instance *Win32_SerialPortConfiguration) GetPropertyErrorReplaceCharacter() (value uint32, err error) {
	retValue, err := instance.GetProperty("ErrorReplaceCharacter")
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

// SetErrorReplacementEnabled sets the value of ErrorReplacementEnabled for the instance
func (instance *Win32_SerialPortConfiguration) SetPropertyErrorReplacementEnabled(value bool) (err error) {
	return instance.SetProperty("ErrorReplacementEnabled", (value))
}

// GetErrorReplacementEnabled gets the value of ErrorReplacementEnabled for the instance
func (instance *Win32_SerialPortConfiguration) GetPropertyErrorReplacementEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("ErrorReplacementEnabled")
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

// SetEventCharacter sets the value of EventCharacter for the instance
func (instance *Win32_SerialPortConfiguration) SetPropertyEventCharacter(value uint32) (err error) {
	return instance.SetProperty("EventCharacter", (value))
}

// GetEventCharacter gets the value of EventCharacter for the instance
func (instance *Win32_SerialPortConfiguration) GetPropertyEventCharacter() (value uint32, err error) {
	retValue, err := instance.GetProperty("EventCharacter")
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

// SetIsBusy sets the value of IsBusy for the instance
func (instance *Win32_SerialPortConfiguration) SetPropertyIsBusy(value bool) (err error) {
	return instance.SetProperty("IsBusy", (value))
}

// GetIsBusy gets the value of IsBusy for the instance
func (instance *Win32_SerialPortConfiguration) GetPropertyIsBusy() (value bool, err error) {
	retValue, err := instance.GetProperty("IsBusy")
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

// SetName sets the value of Name for the instance
func (instance *Win32_SerialPortConfiguration) SetPropertyName(value string) (err error) {
	return instance.SetProperty("Name", (value))
}

// GetName gets the value of Name for the instance
func (instance *Win32_SerialPortConfiguration) GetPropertyName() (value string, err error) {
	retValue, err := instance.GetProperty("Name")
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

// SetParity sets the value of Parity for the instance
func (instance *Win32_SerialPortConfiguration) SetPropertyParity(value string) (err error) {
	return instance.SetProperty("Parity", (value))
}

// GetParity gets the value of Parity for the instance
func (instance *Win32_SerialPortConfiguration) GetPropertyParity() (value string, err error) {
	retValue, err := instance.GetProperty("Parity")
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

// SetParityCheckEnabled sets the value of ParityCheckEnabled for the instance
func (instance *Win32_SerialPortConfiguration) SetPropertyParityCheckEnabled(value bool) (err error) {
	return instance.SetProperty("ParityCheckEnabled", (value))
}

// GetParityCheckEnabled gets the value of ParityCheckEnabled for the instance
func (instance *Win32_SerialPortConfiguration) GetPropertyParityCheckEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("ParityCheckEnabled")
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

// SetRTSFlowControlType sets the value of RTSFlowControlType for the instance
func (instance *Win32_SerialPortConfiguration) SetPropertyRTSFlowControlType(value string) (err error) {
	return instance.SetProperty("RTSFlowControlType", (value))
}

// GetRTSFlowControlType gets the value of RTSFlowControlType for the instance
func (instance *Win32_SerialPortConfiguration) GetPropertyRTSFlowControlType() (value string, err error) {
	retValue, err := instance.GetProperty("RTSFlowControlType")
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

// SetStopBits sets the value of StopBits for the instance
func (instance *Win32_SerialPortConfiguration) SetPropertyStopBits(value string) (err error) {
	return instance.SetProperty("StopBits", (value))
}

// GetStopBits gets the value of StopBits for the instance
func (instance *Win32_SerialPortConfiguration) GetPropertyStopBits() (value string, err error) {
	retValue, err := instance.GetProperty("StopBits")
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

// SetXOffCharacter sets the value of XOffCharacter for the instance
func (instance *Win32_SerialPortConfiguration) SetPropertyXOffCharacter(value uint32) (err error) {
	return instance.SetProperty("XOffCharacter", (value))
}

// GetXOffCharacter gets the value of XOffCharacter for the instance
func (instance *Win32_SerialPortConfiguration) GetPropertyXOffCharacter() (value uint32, err error) {
	retValue, err := instance.GetProperty("XOffCharacter")
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

// SetXOffXMitThreshold sets the value of XOffXMitThreshold for the instance
func (instance *Win32_SerialPortConfiguration) SetPropertyXOffXMitThreshold(value uint32) (err error) {
	return instance.SetProperty("XOffXMitThreshold", (value))
}

// GetXOffXMitThreshold gets the value of XOffXMitThreshold for the instance
func (instance *Win32_SerialPortConfiguration) GetPropertyXOffXMitThreshold() (value uint32, err error) {
	retValue, err := instance.GetProperty("XOffXMitThreshold")
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

// SetXOnCharacter sets the value of XOnCharacter for the instance
func (instance *Win32_SerialPortConfiguration) SetPropertyXOnCharacter(value uint32) (err error) {
	return instance.SetProperty("XOnCharacter", (value))
}

// GetXOnCharacter gets the value of XOnCharacter for the instance
func (instance *Win32_SerialPortConfiguration) GetPropertyXOnCharacter() (value uint32, err error) {
	retValue, err := instance.GetProperty("XOnCharacter")
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

// SetXOnXMitThreshold sets the value of XOnXMitThreshold for the instance
func (instance *Win32_SerialPortConfiguration) SetPropertyXOnXMitThreshold(value uint32) (err error) {
	return instance.SetProperty("XOnXMitThreshold", (value))
}

// GetXOnXMitThreshold gets the value of XOnXMitThreshold for the instance
func (instance *Win32_SerialPortConfiguration) GetPropertyXOnXMitThreshold() (value uint32, err error) {
	retValue, err := instance.GetProperty("XOnXMitThreshold")
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

// SetXOnXOffInFlowControl sets the value of XOnXOffInFlowControl for the instance
func (instance *Win32_SerialPortConfiguration) SetPropertyXOnXOffInFlowControl(value uint32) (err error) {
	return instance.SetProperty("XOnXOffInFlowControl", (value))
}

// GetXOnXOffInFlowControl gets the value of XOnXOffInFlowControl for the instance
func (instance *Win32_SerialPortConfiguration) GetPropertyXOnXOffInFlowControl() (value uint32, err error) {
	retValue, err := instance.GetProperty("XOnXOffInFlowControl")
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

// SetXOnXOffOutFlowControl sets the value of XOnXOffOutFlowControl for the instance
func (instance *Win32_SerialPortConfiguration) SetPropertyXOnXOffOutFlowControl(value uint32) (err error) {
	return instance.SetProperty("XOnXOffOutFlowControl", (value))
}

// GetXOnXOffOutFlowControl gets the value of XOnXOffOutFlowControl for the instance
func (instance *Win32_SerialPortConfiguration) GetPropertyXOnXOffOutFlowControl() (value uint32, err error) {
	retValue, err := instance.GetProperty("XOnXOffOutFlowControl")
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
