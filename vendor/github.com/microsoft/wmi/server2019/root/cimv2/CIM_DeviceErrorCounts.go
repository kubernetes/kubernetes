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

// CIM_DeviceErrorCounts struct
type CIM_DeviceErrorCounts struct {
	*CIM_StatisticalInformation

	//
	CriticalErrorCount uint64

	//
	DeviceCreationClassName string

	//
	DeviceID string

	//
	IndeterminateErrorCount uint64

	//
	MajorErrorCount uint64

	//
	MinorErrorCount uint64

	//
	SystemCreationClassName string

	//
	SystemName string

	//
	WarningCount uint64
}

func NewCIM_DeviceErrorCountsEx1(instance *cim.WmiInstance) (newInstance *CIM_DeviceErrorCounts, err error) {
	tmp, err := NewCIM_StatisticalInformationEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_DeviceErrorCounts{
		CIM_StatisticalInformation: tmp,
	}
	return
}

func NewCIM_DeviceErrorCountsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_DeviceErrorCounts, err error) {
	tmp, err := NewCIM_StatisticalInformationEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_DeviceErrorCounts{
		CIM_StatisticalInformation: tmp,
	}
	return
}

// SetCriticalErrorCount sets the value of CriticalErrorCount for the instance
func (instance *CIM_DeviceErrorCounts) SetPropertyCriticalErrorCount(value uint64) (err error) {
	return instance.SetProperty("CriticalErrorCount", (value))
}

// GetCriticalErrorCount gets the value of CriticalErrorCount for the instance
func (instance *CIM_DeviceErrorCounts) GetPropertyCriticalErrorCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("CriticalErrorCount")
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

// SetDeviceCreationClassName sets the value of DeviceCreationClassName for the instance
func (instance *CIM_DeviceErrorCounts) SetPropertyDeviceCreationClassName(value string) (err error) {
	return instance.SetProperty("DeviceCreationClassName", (value))
}

// GetDeviceCreationClassName gets the value of DeviceCreationClassName for the instance
func (instance *CIM_DeviceErrorCounts) GetPropertyDeviceCreationClassName() (value string, err error) {
	retValue, err := instance.GetProperty("DeviceCreationClassName")
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

// SetDeviceID sets the value of DeviceID for the instance
func (instance *CIM_DeviceErrorCounts) SetPropertyDeviceID(value string) (err error) {
	return instance.SetProperty("DeviceID", (value))
}

// GetDeviceID gets the value of DeviceID for the instance
func (instance *CIM_DeviceErrorCounts) GetPropertyDeviceID() (value string, err error) {
	retValue, err := instance.GetProperty("DeviceID")
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

// SetIndeterminateErrorCount sets the value of IndeterminateErrorCount for the instance
func (instance *CIM_DeviceErrorCounts) SetPropertyIndeterminateErrorCount(value uint64) (err error) {
	return instance.SetProperty("IndeterminateErrorCount", (value))
}

// GetIndeterminateErrorCount gets the value of IndeterminateErrorCount for the instance
func (instance *CIM_DeviceErrorCounts) GetPropertyIndeterminateErrorCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("IndeterminateErrorCount")
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

// SetMajorErrorCount sets the value of MajorErrorCount for the instance
func (instance *CIM_DeviceErrorCounts) SetPropertyMajorErrorCount(value uint64) (err error) {
	return instance.SetProperty("MajorErrorCount", (value))
}

// GetMajorErrorCount gets the value of MajorErrorCount for the instance
func (instance *CIM_DeviceErrorCounts) GetPropertyMajorErrorCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("MajorErrorCount")
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

// SetMinorErrorCount sets the value of MinorErrorCount for the instance
func (instance *CIM_DeviceErrorCounts) SetPropertyMinorErrorCount(value uint64) (err error) {
	return instance.SetProperty("MinorErrorCount", (value))
}

// GetMinorErrorCount gets the value of MinorErrorCount for the instance
func (instance *CIM_DeviceErrorCounts) GetPropertyMinorErrorCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("MinorErrorCount")
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

// SetSystemCreationClassName sets the value of SystemCreationClassName for the instance
func (instance *CIM_DeviceErrorCounts) SetPropertySystemCreationClassName(value string) (err error) {
	return instance.SetProperty("SystemCreationClassName", (value))
}

// GetSystemCreationClassName gets the value of SystemCreationClassName for the instance
func (instance *CIM_DeviceErrorCounts) GetPropertySystemCreationClassName() (value string, err error) {
	retValue, err := instance.GetProperty("SystemCreationClassName")
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

// SetSystemName sets the value of SystemName for the instance
func (instance *CIM_DeviceErrorCounts) SetPropertySystemName(value string) (err error) {
	return instance.SetProperty("SystemName", (value))
}

// GetSystemName gets the value of SystemName for the instance
func (instance *CIM_DeviceErrorCounts) GetPropertySystemName() (value string, err error) {
	retValue, err := instance.GetProperty("SystemName")
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

// SetWarningCount sets the value of WarningCount for the instance
func (instance *CIM_DeviceErrorCounts) SetPropertyWarningCount(value uint64) (err error) {
	return instance.SetProperty("WarningCount", (value))
}

// GetWarningCount gets the value of WarningCount for the instance
func (instance *CIM_DeviceErrorCounts) GetPropertyWarningCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("WarningCount")
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

// <param name="SelectedCounter" type="uint16 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *CIM_DeviceErrorCounts) ResetCounter( /* IN */ SelectedCounter uint16) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("ResetCounter", SelectedCounter)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}
