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

// Win32_ReliabilityStabilityMetrics struct
type Win32_ReliabilityStabilityMetrics struct {
	*Win32_Reliability

	//
	EndMeasurementDate string

	//
	RelID string

	//
	StartMeasurementDate string

	//
	SystemStabilityIndex float64

	//
	TimeGenerated string
}

func NewWin32_ReliabilityStabilityMetricsEx1(instance *cim.WmiInstance) (newInstance *Win32_ReliabilityStabilityMetrics, err error) {
	tmp, err := NewWin32_ReliabilityEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_ReliabilityStabilityMetrics{
		Win32_Reliability: tmp,
	}
	return
}

func NewWin32_ReliabilityStabilityMetricsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ReliabilityStabilityMetrics, err error) {
	tmp, err := NewWin32_ReliabilityEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ReliabilityStabilityMetrics{
		Win32_Reliability: tmp,
	}
	return
}

// SetEndMeasurementDate sets the value of EndMeasurementDate for the instance
func (instance *Win32_ReliabilityStabilityMetrics) SetPropertyEndMeasurementDate(value string) (err error) {
	return instance.SetProperty("EndMeasurementDate", (value))
}

// GetEndMeasurementDate gets the value of EndMeasurementDate for the instance
func (instance *Win32_ReliabilityStabilityMetrics) GetPropertyEndMeasurementDate() (value string, err error) {
	retValue, err := instance.GetProperty("EndMeasurementDate")
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

// SetRelID sets the value of RelID for the instance
func (instance *Win32_ReliabilityStabilityMetrics) SetPropertyRelID(value string) (err error) {
	return instance.SetProperty("RelID", (value))
}

// GetRelID gets the value of RelID for the instance
func (instance *Win32_ReliabilityStabilityMetrics) GetPropertyRelID() (value string, err error) {
	retValue, err := instance.GetProperty("RelID")
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

// SetStartMeasurementDate sets the value of StartMeasurementDate for the instance
func (instance *Win32_ReliabilityStabilityMetrics) SetPropertyStartMeasurementDate(value string) (err error) {
	return instance.SetProperty("StartMeasurementDate", (value))
}

// GetStartMeasurementDate gets the value of StartMeasurementDate for the instance
func (instance *Win32_ReliabilityStabilityMetrics) GetPropertyStartMeasurementDate() (value string, err error) {
	retValue, err := instance.GetProperty("StartMeasurementDate")
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

// SetSystemStabilityIndex sets the value of SystemStabilityIndex for the instance
func (instance *Win32_ReliabilityStabilityMetrics) SetPropertySystemStabilityIndex(value float64) (err error) {
	return instance.SetProperty("SystemStabilityIndex", (value))
}

// GetSystemStabilityIndex gets the value of SystemStabilityIndex for the instance
func (instance *Win32_ReliabilityStabilityMetrics) GetPropertySystemStabilityIndex() (value float64, err error) {
	retValue, err := instance.GetProperty("SystemStabilityIndex")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(float64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " float64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = float64(valuetmp)

	return
}

// SetTimeGenerated sets the value of TimeGenerated for the instance
func (instance *Win32_ReliabilityStabilityMetrics) SetPropertyTimeGenerated(value string) (err error) {
	return instance.SetProperty("TimeGenerated", (value))
}

// GetTimeGenerated gets the value of TimeGenerated for the instance
func (instance *Win32_ReliabilityStabilityMetrics) GetPropertyTimeGenerated() (value string, err error) {
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

//

// <param name="RecordCount" type="uint32 "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_ReliabilityStabilityMetrics) GetRecordCount( /* OUT */ RecordCount uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetRecordCount")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}
