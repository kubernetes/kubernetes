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

// Win32_SerialPort struct
type Win32_SerialPort struct {
	*CIM_SerialController

	//
	Binary bool

	//
	MaximumInputBufferSize uint32

	//
	MaximumOutputBufferSize uint32

	//
	OSAutoDiscovered bool

	//
	ProviderType string

	//
	SettableBaudRate bool

	//
	SettableDataBits bool

	//
	SettableFlowControl bool

	//
	SettableParity bool

	//
	SettableParityCheck bool

	//
	SettableRLSD bool

	//
	SettableStopBits bool

	//
	Supports16BitMode bool

	//
	SupportsDTRDSR bool

	//
	SupportsElapsedTimeouts bool

	//
	SupportsIntTimeouts bool

	//
	SupportsParityCheck bool

	//
	SupportsRLSD bool

	//
	SupportsRTSCTS bool

	//
	SupportsSpecialCharacters bool

	//
	SupportsXOnXOff bool

	//
	SupportsXOnXOffSet bool
}

func NewWin32_SerialPortEx1(instance *cim.WmiInstance) (newInstance *Win32_SerialPort, err error) {
	tmp, err := NewCIM_SerialControllerEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_SerialPort{
		CIM_SerialController: tmp,
	}
	return
}

func NewWin32_SerialPortEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_SerialPort, err error) {
	tmp, err := NewCIM_SerialControllerEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_SerialPort{
		CIM_SerialController: tmp,
	}
	return
}

// SetBinary sets the value of Binary for the instance
func (instance *Win32_SerialPort) SetPropertyBinary(value bool) (err error) {
	return instance.SetProperty("Binary", (value))
}

// GetBinary gets the value of Binary for the instance
func (instance *Win32_SerialPort) GetPropertyBinary() (value bool, err error) {
	retValue, err := instance.GetProperty("Binary")
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

// SetMaximumInputBufferSize sets the value of MaximumInputBufferSize for the instance
func (instance *Win32_SerialPort) SetPropertyMaximumInputBufferSize(value uint32) (err error) {
	return instance.SetProperty("MaximumInputBufferSize", (value))
}

// GetMaximumInputBufferSize gets the value of MaximumInputBufferSize for the instance
func (instance *Win32_SerialPort) GetPropertyMaximumInputBufferSize() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaximumInputBufferSize")
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

// SetMaximumOutputBufferSize sets the value of MaximumOutputBufferSize for the instance
func (instance *Win32_SerialPort) SetPropertyMaximumOutputBufferSize(value uint32) (err error) {
	return instance.SetProperty("MaximumOutputBufferSize", (value))
}

// GetMaximumOutputBufferSize gets the value of MaximumOutputBufferSize for the instance
func (instance *Win32_SerialPort) GetPropertyMaximumOutputBufferSize() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaximumOutputBufferSize")
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

// SetOSAutoDiscovered sets the value of OSAutoDiscovered for the instance
func (instance *Win32_SerialPort) SetPropertyOSAutoDiscovered(value bool) (err error) {
	return instance.SetProperty("OSAutoDiscovered", (value))
}

// GetOSAutoDiscovered gets the value of OSAutoDiscovered for the instance
func (instance *Win32_SerialPort) GetPropertyOSAutoDiscovered() (value bool, err error) {
	retValue, err := instance.GetProperty("OSAutoDiscovered")
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

// SetProviderType sets the value of ProviderType for the instance
func (instance *Win32_SerialPort) SetPropertyProviderType(value string) (err error) {
	return instance.SetProperty("ProviderType", (value))
}

// GetProviderType gets the value of ProviderType for the instance
func (instance *Win32_SerialPort) GetPropertyProviderType() (value string, err error) {
	retValue, err := instance.GetProperty("ProviderType")
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

// SetSettableBaudRate sets the value of SettableBaudRate for the instance
func (instance *Win32_SerialPort) SetPropertySettableBaudRate(value bool) (err error) {
	return instance.SetProperty("SettableBaudRate", (value))
}

// GetSettableBaudRate gets the value of SettableBaudRate for the instance
func (instance *Win32_SerialPort) GetPropertySettableBaudRate() (value bool, err error) {
	retValue, err := instance.GetProperty("SettableBaudRate")
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

// SetSettableDataBits sets the value of SettableDataBits for the instance
func (instance *Win32_SerialPort) SetPropertySettableDataBits(value bool) (err error) {
	return instance.SetProperty("SettableDataBits", (value))
}

// GetSettableDataBits gets the value of SettableDataBits for the instance
func (instance *Win32_SerialPort) GetPropertySettableDataBits() (value bool, err error) {
	retValue, err := instance.GetProperty("SettableDataBits")
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

// SetSettableFlowControl sets the value of SettableFlowControl for the instance
func (instance *Win32_SerialPort) SetPropertySettableFlowControl(value bool) (err error) {
	return instance.SetProperty("SettableFlowControl", (value))
}

// GetSettableFlowControl gets the value of SettableFlowControl for the instance
func (instance *Win32_SerialPort) GetPropertySettableFlowControl() (value bool, err error) {
	retValue, err := instance.GetProperty("SettableFlowControl")
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

// SetSettableParity sets the value of SettableParity for the instance
func (instance *Win32_SerialPort) SetPropertySettableParity(value bool) (err error) {
	return instance.SetProperty("SettableParity", (value))
}

// GetSettableParity gets the value of SettableParity for the instance
func (instance *Win32_SerialPort) GetPropertySettableParity() (value bool, err error) {
	retValue, err := instance.GetProperty("SettableParity")
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

// SetSettableParityCheck sets the value of SettableParityCheck for the instance
func (instance *Win32_SerialPort) SetPropertySettableParityCheck(value bool) (err error) {
	return instance.SetProperty("SettableParityCheck", (value))
}

// GetSettableParityCheck gets the value of SettableParityCheck for the instance
func (instance *Win32_SerialPort) GetPropertySettableParityCheck() (value bool, err error) {
	retValue, err := instance.GetProperty("SettableParityCheck")
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

// SetSettableRLSD sets the value of SettableRLSD for the instance
func (instance *Win32_SerialPort) SetPropertySettableRLSD(value bool) (err error) {
	return instance.SetProperty("SettableRLSD", (value))
}

// GetSettableRLSD gets the value of SettableRLSD for the instance
func (instance *Win32_SerialPort) GetPropertySettableRLSD() (value bool, err error) {
	retValue, err := instance.GetProperty("SettableRLSD")
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

// SetSettableStopBits sets the value of SettableStopBits for the instance
func (instance *Win32_SerialPort) SetPropertySettableStopBits(value bool) (err error) {
	return instance.SetProperty("SettableStopBits", (value))
}

// GetSettableStopBits gets the value of SettableStopBits for the instance
func (instance *Win32_SerialPort) GetPropertySettableStopBits() (value bool, err error) {
	retValue, err := instance.GetProperty("SettableStopBits")
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

// SetSupports16BitMode sets the value of Supports16BitMode for the instance
func (instance *Win32_SerialPort) SetPropertySupports16BitMode(value bool) (err error) {
	return instance.SetProperty("Supports16BitMode", (value))
}

// GetSupports16BitMode gets the value of Supports16BitMode for the instance
func (instance *Win32_SerialPort) GetPropertySupports16BitMode() (value bool, err error) {
	retValue, err := instance.GetProperty("Supports16BitMode")
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

// SetSupportsDTRDSR sets the value of SupportsDTRDSR for the instance
func (instance *Win32_SerialPort) SetPropertySupportsDTRDSR(value bool) (err error) {
	return instance.SetProperty("SupportsDTRDSR", (value))
}

// GetSupportsDTRDSR gets the value of SupportsDTRDSR for the instance
func (instance *Win32_SerialPort) GetPropertySupportsDTRDSR() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsDTRDSR")
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

// SetSupportsElapsedTimeouts sets the value of SupportsElapsedTimeouts for the instance
func (instance *Win32_SerialPort) SetPropertySupportsElapsedTimeouts(value bool) (err error) {
	return instance.SetProperty("SupportsElapsedTimeouts", (value))
}

// GetSupportsElapsedTimeouts gets the value of SupportsElapsedTimeouts for the instance
func (instance *Win32_SerialPort) GetPropertySupportsElapsedTimeouts() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsElapsedTimeouts")
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

// SetSupportsIntTimeouts sets the value of SupportsIntTimeouts for the instance
func (instance *Win32_SerialPort) SetPropertySupportsIntTimeouts(value bool) (err error) {
	return instance.SetProperty("SupportsIntTimeouts", (value))
}

// GetSupportsIntTimeouts gets the value of SupportsIntTimeouts for the instance
func (instance *Win32_SerialPort) GetPropertySupportsIntTimeouts() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsIntTimeouts")
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

// SetSupportsParityCheck sets the value of SupportsParityCheck for the instance
func (instance *Win32_SerialPort) SetPropertySupportsParityCheck(value bool) (err error) {
	return instance.SetProperty("SupportsParityCheck", (value))
}

// GetSupportsParityCheck gets the value of SupportsParityCheck for the instance
func (instance *Win32_SerialPort) GetPropertySupportsParityCheck() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsParityCheck")
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

// SetSupportsRLSD sets the value of SupportsRLSD for the instance
func (instance *Win32_SerialPort) SetPropertySupportsRLSD(value bool) (err error) {
	return instance.SetProperty("SupportsRLSD", (value))
}

// GetSupportsRLSD gets the value of SupportsRLSD for the instance
func (instance *Win32_SerialPort) GetPropertySupportsRLSD() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsRLSD")
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

// SetSupportsRTSCTS sets the value of SupportsRTSCTS for the instance
func (instance *Win32_SerialPort) SetPropertySupportsRTSCTS(value bool) (err error) {
	return instance.SetProperty("SupportsRTSCTS", (value))
}

// GetSupportsRTSCTS gets the value of SupportsRTSCTS for the instance
func (instance *Win32_SerialPort) GetPropertySupportsRTSCTS() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsRTSCTS")
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

// SetSupportsSpecialCharacters sets the value of SupportsSpecialCharacters for the instance
func (instance *Win32_SerialPort) SetPropertySupportsSpecialCharacters(value bool) (err error) {
	return instance.SetProperty("SupportsSpecialCharacters", (value))
}

// GetSupportsSpecialCharacters gets the value of SupportsSpecialCharacters for the instance
func (instance *Win32_SerialPort) GetPropertySupportsSpecialCharacters() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsSpecialCharacters")
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

// SetSupportsXOnXOff sets the value of SupportsXOnXOff for the instance
func (instance *Win32_SerialPort) SetPropertySupportsXOnXOff(value bool) (err error) {
	return instance.SetProperty("SupportsXOnXOff", (value))
}

// GetSupportsXOnXOff gets the value of SupportsXOnXOff for the instance
func (instance *Win32_SerialPort) GetPropertySupportsXOnXOff() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsXOnXOff")
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

// SetSupportsXOnXOffSet sets the value of SupportsXOnXOffSet for the instance
func (instance *Win32_SerialPort) SetPropertySupportsXOnXOffSet(value bool) (err error) {
	return instance.SetProperty("SupportsXOnXOffSet", (value))
}

// GetSupportsXOnXOffSet gets the value of SupportsXOnXOffSet for the instance
func (instance *Win32_SerialPort) GetPropertySupportsXOnXOffSet() (value bool, err error) {
	retValue, err := instance.GetProperty("SupportsXOnXOffSet")
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
