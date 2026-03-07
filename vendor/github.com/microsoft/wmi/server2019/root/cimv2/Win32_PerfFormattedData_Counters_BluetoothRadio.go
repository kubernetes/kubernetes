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

// Win32_PerfFormattedData_Counters_BluetoothRadio struct
type Win32_PerfFormattedData_Counters_BluetoothRadio struct {
	*Win32_PerfFormattedData

	//
	ACLflusheventsPersec uint32

	//
	ClassicACLbytesreadPersec uint32

	//
	ClassicACLbyteswrittenPersec uint32

	//
	ClassicACLConnections uint32

	//
	ClassicACLwritecredits uint32

	//
	InquiryScanDutyCyclePercent uint32

	//
	InquiryScanInterval uint32

	//
	InquiryScanWindow uint32

	//
	LEACLbytesreadPersec uint32

	//
	LEACLbyteswrittenPersec uint32

	//
	LEACLConnections uint32

	//
	LEACLwritecredits uint32

	//
	LEScanDutyCyclePercent uint32

	//
	LEScanInterval uint32

	//
	LEScanWindow uint32

	//
	PageScanDutyCyclePercent uint32

	//
	PageScanInterval uint32

	//
	PageScanWindow uint32

	//
	SCObytesreadPersec uint32

	//
	SCObyteswrittenPersec uint32

	//
	SCOConnections uint32

	//
	SidebandSCOConnections uint32
}

func NewWin32_PerfFormattedData_Counters_BluetoothRadioEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_BluetoothRadio, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_BluetoothRadio{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_BluetoothRadioEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_BluetoothRadio, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_BluetoothRadio{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetACLflusheventsPersec sets the value of ACLflusheventsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) SetPropertyACLflusheventsPersec(value uint32) (err error) {
	return instance.SetProperty("ACLflusheventsPersec", (value))
}

// GetACLflusheventsPersec gets the value of ACLflusheventsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) GetPropertyACLflusheventsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ACLflusheventsPersec")
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

// SetClassicACLbytesreadPersec sets the value of ClassicACLbytesreadPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) SetPropertyClassicACLbytesreadPersec(value uint32) (err error) {
	return instance.SetProperty("ClassicACLbytesreadPersec", (value))
}

// GetClassicACLbytesreadPersec gets the value of ClassicACLbytesreadPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) GetPropertyClassicACLbytesreadPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ClassicACLbytesreadPersec")
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

// SetClassicACLbyteswrittenPersec sets the value of ClassicACLbyteswrittenPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) SetPropertyClassicACLbyteswrittenPersec(value uint32) (err error) {
	return instance.SetProperty("ClassicACLbyteswrittenPersec", (value))
}

// GetClassicACLbyteswrittenPersec gets the value of ClassicACLbyteswrittenPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) GetPropertyClassicACLbyteswrittenPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ClassicACLbyteswrittenPersec")
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

// SetClassicACLConnections sets the value of ClassicACLConnections for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) SetPropertyClassicACLConnections(value uint32) (err error) {
	return instance.SetProperty("ClassicACLConnections", (value))
}

// GetClassicACLConnections gets the value of ClassicACLConnections for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) GetPropertyClassicACLConnections() (value uint32, err error) {
	retValue, err := instance.GetProperty("ClassicACLConnections")
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

// SetClassicACLwritecredits sets the value of ClassicACLwritecredits for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) SetPropertyClassicACLwritecredits(value uint32) (err error) {
	return instance.SetProperty("ClassicACLwritecredits", (value))
}

// GetClassicACLwritecredits gets the value of ClassicACLwritecredits for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) GetPropertyClassicACLwritecredits() (value uint32, err error) {
	retValue, err := instance.GetProperty("ClassicACLwritecredits")
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

// SetInquiryScanDutyCyclePercent sets the value of InquiryScanDutyCyclePercent for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) SetPropertyInquiryScanDutyCyclePercent(value uint32) (err error) {
	return instance.SetProperty("InquiryScanDutyCyclePercent", (value))
}

// GetInquiryScanDutyCyclePercent gets the value of InquiryScanDutyCyclePercent for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) GetPropertyInquiryScanDutyCyclePercent() (value uint32, err error) {
	retValue, err := instance.GetProperty("InquiryScanDutyCyclePercent")
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

// SetInquiryScanInterval sets the value of InquiryScanInterval for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) SetPropertyInquiryScanInterval(value uint32) (err error) {
	return instance.SetProperty("InquiryScanInterval", (value))
}

// GetInquiryScanInterval gets the value of InquiryScanInterval for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) GetPropertyInquiryScanInterval() (value uint32, err error) {
	retValue, err := instance.GetProperty("InquiryScanInterval")
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

// SetInquiryScanWindow sets the value of InquiryScanWindow for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) SetPropertyInquiryScanWindow(value uint32) (err error) {
	return instance.SetProperty("InquiryScanWindow", (value))
}

// GetInquiryScanWindow gets the value of InquiryScanWindow for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) GetPropertyInquiryScanWindow() (value uint32, err error) {
	retValue, err := instance.GetProperty("InquiryScanWindow")
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

// SetLEACLbytesreadPersec sets the value of LEACLbytesreadPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) SetPropertyLEACLbytesreadPersec(value uint32) (err error) {
	return instance.SetProperty("LEACLbytesreadPersec", (value))
}

// GetLEACLbytesreadPersec gets the value of LEACLbytesreadPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) GetPropertyLEACLbytesreadPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("LEACLbytesreadPersec")
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

// SetLEACLbyteswrittenPersec sets the value of LEACLbyteswrittenPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) SetPropertyLEACLbyteswrittenPersec(value uint32) (err error) {
	return instance.SetProperty("LEACLbyteswrittenPersec", (value))
}

// GetLEACLbyteswrittenPersec gets the value of LEACLbyteswrittenPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) GetPropertyLEACLbyteswrittenPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("LEACLbyteswrittenPersec")
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

// SetLEACLConnections sets the value of LEACLConnections for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) SetPropertyLEACLConnections(value uint32) (err error) {
	return instance.SetProperty("LEACLConnections", (value))
}

// GetLEACLConnections gets the value of LEACLConnections for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) GetPropertyLEACLConnections() (value uint32, err error) {
	retValue, err := instance.GetProperty("LEACLConnections")
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

// SetLEACLwritecredits sets the value of LEACLwritecredits for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) SetPropertyLEACLwritecredits(value uint32) (err error) {
	return instance.SetProperty("LEACLwritecredits", (value))
}

// GetLEACLwritecredits gets the value of LEACLwritecredits for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) GetPropertyLEACLwritecredits() (value uint32, err error) {
	retValue, err := instance.GetProperty("LEACLwritecredits")
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

// SetLEScanDutyCyclePercent sets the value of LEScanDutyCyclePercent for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) SetPropertyLEScanDutyCyclePercent(value uint32) (err error) {
	return instance.SetProperty("LEScanDutyCyclePercent", (value))
}

// GetLEScanDutyCyclePercent gets the value of LEScanDutyCyclePercent for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) GetPropertyLEScanDutyCyclePercent() (value uint32, err error) {
	retValue, err := instance.GetProperty("LEScanDutyCyclePercent")
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

// SetLEScanInterval sets the value of LEScanInterval for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) SetPropertyLEScanInterval(value uint32) (err error) {
	return instance.SetProperty("LEScanInterval", (value))
}

// GetLEScanInterval gets the value of LEScanInterval for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) GetPropertyLEScanInterval() (value uint32, err error) {
	retValue, err := instance.GetProperty("LEScanInterval")
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

// SetLEScanWindow sets the value of LEScanWindow for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) SetPropertyLEScanWindow(value uint32) (err error) {
	return instance.SetProperty("LEScanWindow", (value))
}

// GetLEScanWindow gets the value of LEScanWindow for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) GetPropertyLEScanWindow() (value uint32, err error) {
	retValue, err := instance.GetProperty("LEScanWindow")
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

// SetPageScanDutyCyclePercent sets the value of PageScanDutyCyclePercent for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) SetPropertyPageScanDutyCyclePercent(value uint32) (err error) {
	return instance.SetProperty("PageScanDutyCyclePercent", (value))
}

// GetPageScanDutyCyclePercent gets the value of PageScanDutyCyclePercent for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) GetPropertyPageScanDutyCyclePercent() (value uint32, err error) {
	retValue, err := instance.GetProperty("PageScanDutyCyclePercent")
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

// SetPageScanInterval sets the value of PageScanInterval for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) SetPropertyPageScanInterval(value uint32) (err error) {
	return instance.SetProperty("PageScanInterval", (value))
}

// GetPageScanInterval gets the value of PageScanInterval for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) GetPropertyPageScanInterval() (value uint32, err error) {
	retValue, err := instance.GetProperty("PageScanInterval")
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

// SetPageScanWindow sets the value of PageScanWindow for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) SetPropertyPageScanWindow(value uint32) (err error) {
	return instance.SetProperty("PageScanWindow", (value))
}

// GetPageScanWindow gets the value of PageScanWindow for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) GetPropertyPageScanWindow() (value uint32, err error) {
	retValue, err := instance.GetProperty("PageScanWindow")
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

// SetSCObytesreadPersec sets the value of SCObytesreadPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) SetPropertySCObytesreadPersec(value uint32) (err error) {
	return instance.SetProperty("SCObytesreadPersec", (value))
}

// GetSCObytesreadPersec gets the value of SCObytesreadPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) GetPropertySCObytesreadPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("SCObytesreadPersec")
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

// SetSCObyteswrittenPersec sets the value of SCObyteswrittenPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) SetPropertySCObyteswrittenPersec(value uint32) (err error) {
	return instance.SetProperty("SCObyteswrittenPersec", (value))
}

// GetSCObyteswrittenPersec gets the value of SCObyteswrittenPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) GetPropertySCObyteswrittenPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("SCObyteswrittenPersec")
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

// SetSCOConnections sets the value of SCOConnections for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) SetPropertySCOConnections(value uint32) (err error) {
	return instance.SetProperty("SCOConnections", (value))
}

// GetSCOConnections gets the value of SCOConnections for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) GetPropertySCOConnections() (value uint32, err error) {
	retValue, err := instance.GetProperty("SCOConnections")
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

// SetSidebandSCOConnections sets the value of SidebandSCOConnections for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) SetPropertySidebandSCOConnections(value uint32) (err error) {
	return instance.SetProperty("SidebandSCOConnections", (value))
}

// GetSidebandSCOConnections gets the value of SidebandSCOConnections for the instance
func (instance *Win32_PerfFormattedData_Counters_BluetoothRadio) GetPropertySidebandSCOConnections() (value uint32, err error) {
	retValue, err := instance.GetProperty("SidebandSCOConnections")
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
