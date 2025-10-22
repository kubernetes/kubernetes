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

// CIM_Memory struct
type CIM_Memory struct {
	*CIM_StorageExtent

	//
	AdditionalErrorData []uint8

	//
	CorrectableError bool

	//
	EndingAddress uint64

	//
	ErrorAccess uint16

	//
	ErrorAddress uint64

	//
	ErrorData []uint8

	//
	ErrorDataOrder uint16

	//
	ErrorInfo uint16

	//
	ErrorResolution uint64

	//
	ErrorTime string

	//
	ErrorTransferSize uint32

	//
	OtherErrorDescription string

	//
	StartingAddress uint64

	//
	SystemLevelAddress bool
}

func NewCIM_MemoryEx1(instance *cim.WmiInstance) (newInstance *CIM_Memory, err error) {
	tmp, err := NewCIM_StorageExtentEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_Memory{
		CIM_StorageExtent: tmp,
	}
	return
}

func NewCIM_MemoryEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_Memory, err error) {
	tmp, err := NewCIM_StorageExtentEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_Memory{
		CIM_StorageExtent: tmp,
	}
	return
}

// SetAdditionalErrorData sets the value of AdditionalErrorData for the instance
func (instance *CIM_Memory) SetPropertyAdditionalErrorData(value []uint8) (err error) {
	return instance.SetProperty("AdditionalErrorData", (value))
}

// GetAdditionalErrorData gets the value of AdditionalErrorData for the instance
func (instance *CIM_Memory) GetPropertyAdditionalErrorData() (value []uint8, err error) {
	retValue, err := instance.GetProperty("AdditionalErrorData")
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

// SetCorrectableError sets the value of CorrectableError for the instance
func (instance *CIM_Memory) SetPropertyCorrectableError(value bool) (err error) {
	return instance.SetProperty("CorrectableError", (value))
}

// GetCorrectableError gets the value of CorrectableError for the instance
func (instance *CIM_Memory) GetPropertyCorrectableError() (value bool, err error) {
	retValue, err := instance.GetProperty("CorrectableError")
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

// SetEndingAddress sets the value of EndingAddress for the instance
func (instance *CIM_Memory) SetPropertyEndingAddress(value uint64) (err error) {
	return instance.SetProperty("EndingAddress", (value))
}

// GetEndingAddress gets the value of EndingAddress for the instance
func (instance *CIM_Memory) GetPropertyEndingAddress() (value uint64, err error) {
	retValue, err := instance.GetProperty("EndingAddress")
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

// SetErrorAccess sets the value of ErrorAccess for the instance
func (instance *CIM_Memory) SetPropertyErrorAccess(value uint16) (err error) {
	return instance.SetProperty("ErrorAccess", (value))
}

// GetErrorAccess gets the value of ErrorAccess for the instance
func (instance *CIM_Memory) GetPropertyErrorAccess() (value uint16, err error) {
	retValue, err := instance.GetProperty("ErrorAccess")
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

// SetErrorAddress sets the value of ErrorAddress for the instance
func (instance *CIM_Memory) SetPropertyErrorAddress(value uint64) (err error) {
	return instance.SetProperty("ErrorAddress", (value))
}

// GetErrorAddress gets the value of ErrorAddress for the instance
func (instance *CIM_Memory) GetPropertyErrorAddress() (value uint64, err error) {
	retValue, err := instance.GetProperty("ErrorAddress")
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

// SetErrorData sets the value of ErrorData for the instance
func (instance *CIM_Memory) SetPropertyErrorData(value []uint8) (err error) {
	return instance.SetProperty("ErrorData", (value))
}

// GetErrorData gets the value of ErrorData for the instance
func (instance *CIM_Memory) GetPropertyErrorData() (value []uint8, err error) {
	retValue, err := instance.GetProperty("ErrorData")
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

// SetErrorDataOrder sets the value of ErrorDataOrder for the instance
func (instance *CIM_Memory) SetPropertyErrorDataOrder(value uint16) (err error) {
	return instance.SetProperty("ErrorDataOrder", (value))
}

// GetErrorDataOrder gets the value of ErrorDataOrder for the instance
func (instance *CIM_Memory) GetPropertyErrorDataOrder() (value uint16, err error) {
	retValue, err := instance.GetProperty("ErrorDataOrder")
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

// SetErrorInfo sets the value of ErrorInfo for the instance
func (instance *CIM_Memory) SetPropertyErrorInfo(value uint16) (err error) {
	return instance.SetProperty("ErrorInfo", (value))
}

// GetErrorInfo gets the value of ErrorInfo for the instance
func (instance *CIM_Memory) GetPropertyErrorInfo() (value uint16, err error) {
	retValue, err := instance.GetProperty("ErrorInfo")
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

// SetErrorResolution sets the value of ErrorResolution for the instance
func (instance *CIM_Memory) SetPropertyErrorResolution(value uint64) (err error) {
	return instance.SetProperty("ErrorResolution", (value))
}

// GetErrorResolution gets the value of ErrorResolution for the instance
func (instance *CIM_Memory) GetPropertyErrorResolution() (value uint64, err error) {
	retValue, err := instance.GetProperty("ErrorResolution")
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

// SetErrorTime sets the value of ErrorTime for the instance
func (instance *CIM_Memory) SetPropertyErrorTime(value string) (err error) {
	return instance.SetProperty("ErrorTime", (value))
}

// GetErrorTime gets the value of ErrorTime for the instance
func (instance *CIM_Memory) GetPropertyErrorTime() (value string, err error) {
	retValue, err := instance.GetProperty("ErrorTime")
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

// SetErrorTransferSize sets the value of ErrorTransferSize for the instance
func (instance *CIM_Memory) SetPropertyErrorTransferSize(value uint32) (err error) {
	return instance.SetProperty("ErrorTransferSize", (value))
}

// GetErrorTransferSize gets the value of ErrorTransferSize for the instance
func (instance *CIM_Memory) GetPropertyErrorTransferSize() (value uint32, err error) {
	retValue, err := instance.GetProperty("ErrorTransferSize")
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

// SetOtherErrorDescription sets the value of OtherErrorDescription for the instance
func (instance *CIM_Memory) SetPropertyOtherErrorDescription(value string) (err error) {
	return instance.SetProperty("OtherErrorDescription", (value))
}

// GetOtherErrorDescription gets the value of OtherErrorDescription for the instance
func (instance *CIM_Memory) GetPropertyOtherErrorDescription() (value string, err error) {
	retValue, err := instance.GetProperty("OtherErrorDescription")
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

// SetStartingAddress sets the value of StartingAddress for the instance
func (instance *CIM_Memory) SetPropertyStartingAddress(value uint64) (err error) {
	return instance.SetProperty("StartingAddress", (value))
}

// GetStartingAddress gets the value of StartingAddress for the instance
func (instance *CIM_Memory) GetPropertyStartingAddress() (value uint64, err error) {
	retValue, err := instance.GetProperty("StartingAddress")
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

// SetSystemLevelAddress sets the value of SystemLevelAddress for the instance
func (instance *CIM_Memory) SetPropertySystemLevelAddress(value bool) (err error) {
	return instance.SetProperty("SystemLevelAddress", (value))
}

// GetSystemLevelAddress gets the value of SystemLevelAddress for the instance
func (instance *CIM_Memory) GetPropertySystemLevelAddress() (value bool, err error) {
	retValue, err := instance.GetProperty("SystemLevelAddress")
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
