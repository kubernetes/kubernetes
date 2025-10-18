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

// Win32_ReliabilityRecords struct
type Win32_ReliabilityRecords struct {
	*Win32_Reliability

	//
	ComputerName string

	//
	EventIdentifier uint32

	//
	InsertionStrings []string

	//
	Logfile string

	//
	Message string

	//
	ProductName string

	//
	RecordNumber uint32

	//
	SourceName string

	//
	TimeGenerated string

	//
	User string
}

func NewWin32_ReliabilityRecordsEx1(instance *cim.WmiInstance) (newInstance *Win32_ReliabilityRecords, err error) {
	tmp, err := NewWin32_ReliabilityEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_ReliabilityRecords{
		Win32_Reliability: tmp,
	}
	return
}

func NewWin32_ReliabilityRecordsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ReliabilityRecords, err error) {
	tmp, err := NewWin32_ReliabilityEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ReliabilityRecords{
		Win32_Reliability: tmp,
	}
	return
}

// SetComputerName sets the value of ComputerName for the instance
func (instance *Win32_ReliabilityRecords) SetPropertyComputerName(value string) (err error) {
	return instance.SetProperty("ComputerName", (value))
}

// GetComputerName gets the value of ComputerName for the instance
func (instance *Win32_ReliabilityRecords) GetPropertyComputerName() (value string, err error) {
	retValue, err := instance.GetProperty("ComputerName")
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

// SetEventIdentifier sets the value of EventIdentifier for the instance
func (instance *Win32_ReliabilityRecords) SetPropertyEventIdentifier(value uint32) (err error) {
	return instance.SetProperty("EventIdentifier", (value))
}

// GetEventIdentifier gets the value of EventIdentifier for the instance
func (instance *Win32_ReliabilityRecords) GetPropertyEventIdentifier() (value uint32, err error) {
	retValue, err := instance.GetProperty("EventIdentifier")
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

// SetInsertionStrings sets the value of InsertionStrings for the instance
func (instance *Win32_ReliabilityRecords) SetPropertyInsertionStrings(value []string) (err error) {
	return instance.SetProperty("InsertionStrings", (value))
}

// GetInsertionStrings gets the value of InsertionStrings for the instance
func (instance *Win32_ReliabilityRecords) GetPropertyInsertionStrings() (value []string, err error) {
	retValue, err := instance.GetProperty("InsertionStrings")
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

// SetLogfile sets the value of Logfile for the instance
func (instance *Win32_ReliabilityRecords) SetPropertyLogfile(value string) (err error) {
	return instance.SetProperty("Logfile", (value))
}

// GetLogfile gets the value of Logfile for the instance
func (instance *Win32_ReliabilityRecords) GetPropertyLogfile() (value string, err error) {
	retValue, err := instance.GetProperty("Logfile")
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

// SetMessage sets the value of Message for the instance
func (instance *Win32_ReliabilityRecords) SetPropertyMessage(value string) (err error) {
	return instance.SetProperty("Message", (value))
}

// GetMessage gets the value of Message for the instance
func (instance *Win32_ReliabilityRecords) GetPropertyMessage() (value string, err error) {
	retValue, err := instance.GetProperty("Message")
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

// SetProductName sets the value of ProductName for the instance
func (instance *Win32_ReliabilityRecords) SetPropertyProductName(value string) (err error) {
	return instance.SetProperty("ProductName", (value))
}

// GetProductName gets the value of ProductName for the instance
func (instance *Win32_ReliabilityRecords) GetPropertyProductName() (value string, err error) {
	retValue, err := instance.GetProperty("ProductName")
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

// SetRecordNumber sets the value of RecordNumber for the instance
func (instance *Win32_ReliabilityRecords) SetPropertyRecordNumber(value uint32) (err error) {
	return instance.SetProperty("RecordNumber", (value))
}

// GetRecordNumber gets the value of RecordNumber for the instance
func (instance *Win32_ReliabilityRecords) GetPropertyRecordNumber() (value uint32, err error) {
	retValue, err := instance.GetProperty("RecordNumber")
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

// SetSourceName sets the value of SourceName for the instance
func (instance *Win32_ReliabilityRecords) SetPropertySourceName(value string) (err error) {
	return instance.SetProperty("SourceName", (value))
}

// GetSourceName gets the value of SourceName for the instance
func (instance *Win32_ReliabilityRecords) GetPropertySourceName() (value string, err error) {
	retValue, err := instance.GetProperty("SourceName")
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

// SetTimeGenerated sets the value of TimeGenerated for the instance
func (instance *Win32_ReliabilityRecords) SetPropertyTimeGenerated(value string) (err error) {
	return instance.SetProperty("TimeGenerated", (value))
}

// GetTimeGenerated gets the value of TimeGenerated for the instance
func (instance *Win32_ReliabilityRecords) GetPropertyTimeGenerated() (value string, err error) {
	retValue, err := instance.GetProperty("TimeGenerated")
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

// SetUser sets the value of User for the instance
func (instance *Win32_ReliabilityRecords) SetPropertyUser(value string) (err error) {
	return instance.SetProperty("User", (value))
}

// GetUser gets the value of User for the instance
func (instance *Win32_ReliabilityRecords) GetPropertyUser() (value string, err error) {
	retValue, err := instance.GetProperty("User")
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

//

// <param name="RecordCount" type="uint32 "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_ReliabilityRecords) GetRecordCount( /* OUT */ RecordCount uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetRecordCount")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}
