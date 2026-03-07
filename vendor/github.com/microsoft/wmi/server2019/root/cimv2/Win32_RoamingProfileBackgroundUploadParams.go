// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.CIMV2
//////////////////////////////////////////////
package cimv2

import (
	"github.com/microsoft/wmi/pkg/base/instance"
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// Win32_RoamingProfileBackgroundUploadParams struct
type Win32_RoamingProfileBackgroundUploadParams struct {
	*cim.WmiInstance

	// The time interval, in hours.
	Interval uint16

	// Indicates when a background upload should be performed. One of the following values can be specified. SpecificTime - Perform the background upload at the time of day specified in the Time property. SetInterval  - Perform the background upload at the interval specified in the Interval property.
	SchedulingMethod RoamingProfileBackgroundUploadParams_SchedulingMethod

	// An integer value that represents the hour, in 24-hour time, for the time of day when they sync should occur. This must be an integer value from 0 to 23.
	Time uint16
}

func NewWin32_RoamingProfileBackgroundUploadParamsEx1(instance *cim.WmiInstance) (newInstance *Win32_RoamingProfileBackgroundUploadParams, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_RoamingProfileBackgroundUploadParams{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_RoamingProfileBackgroundUploadParamsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_RoamingProfileBackgroundUploadParams, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_RoamingProfileBackgroundUploadParams{
		WmiInstance: tmp,
	}
	return
}

// SetInterval sets the value of Interval for the instance
func (instance *Win32_RoamingProfileBackgroundUploadParams) SetPropertyInterval(value uint16) (err error) {
	return instance.SetProperty("Interval", (value))
}

// GetInterval gets the value of Interval for the instance
func (instance *Win32_RoamingProfileBackgroundUploadParams) GetPropertyInterval() (value uint16, err error) {
	retValue, err := instance.GetProperty("Interval")
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

// SetSchedulingMethod sets the value of SchedulingMethod for the instance
func (instance *Win32_RoamingProfileBackgroundUploadParams) SetPropertySchedulingMethod(value RoamingProfileBackgroundUploadParams_SchedulingMethod) (err error) {
	return instance.SetProperty("SchedulingMethod", (value))
}

// GetSchedulingMethod gets the value of SchedulingMethod for the instance
func (instance *Win32_RoamingProfileBackgroundUploadParams) GetPropertySchedulingMethod() (value RoamingProfileBackgroundUploadParams_SchedulingMethod, err error) {
	retValue, err := instance.GetProperty("SchedulingMethod")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = RoamingProfileBackgroundUploadParams_SchedulingMethod(valuetmp)

	return
}

// SetTime sets the value of Time for the instance
func (instance *Win32_RoamingProfileBackgroundUploadParams) SetPropertyTime(value uint16) (err error) {
	return instance.SetProperty("Time", (value))
}

// GetTime gets the value of Time for the instance
func (instance *Win32_RoamingProfileBackgroundUploadParams) GetPropertyTime() (value uint16, err error) {
	retValue, err := instance.GetProperty("Time")
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
