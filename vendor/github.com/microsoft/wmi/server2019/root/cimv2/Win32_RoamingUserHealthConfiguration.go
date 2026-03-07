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

// Win32_RoamingUserHealthConfiguration struct
type Win32_RoamingUserHealthConfiguration struct {
	*cim.WmiInstance

	// Configure how the Win32_UserProfile::HealthStatus property should reflect the use of temporary profiles.
	HealthStatusForTempProfiles RoamingUserHealthConfiguration_HealthStatusForTempProfiles

	// This is the time threshold, in hours, after which the profile health is reported as Caution when the profile has not been downloaded yet
	LastProfileDownloadIntervalCautionInHours uint16

	// This is the time threshold, in hours, after which the profile health is reported as Unhealthy when the profile has not been uploaded yet
	LastProfileDownloadIntervalUnhealthyInHours uint16

	// This is the time threshold, in hours, after which the profile health is reported as Caution when the profile has not been uploaded yet
	LastProfileUploadIntervalCautionInHours uint16

	// This is the time threshold, in hours, after which the profile health is reported as Unhealthy when the profile has not been downloaded yet
	LastProfileUploadIntervalUnhealthyInHours uint16
}

func NewWin32_RoamingUserHealthConfigurationEx1(instance *cim.WmiInstance) (newInstance *Win32_RoamingUserHealthConfiguration, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_RoamingUserHealthConfiguration{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_RoamingUserHealthConfigurationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_RoamingUserHealthConfiguration, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_RoamingUserHealthConfiguration{
		WmiInstance: tmp,
	}
	return
}

// SetHealthStatusForTempProfiles sets the value of HealthStatusForTempProfiles for the instance
func (instance *Win32_RoamingUserHealthConfiguration) SetPropertyHealthStatusForTempProfiles(value RoamingUserHealthConfiguration_HealthStatusForTempProfiles) (err error) {
	return instance.SetProperty("HealthStatusForTempProfiles", (value))
}

// GetHealthStatusForTempProfiles gets the value of HealthStatusForTempProfiles for the instance
func (instance *Win32_RoamingUserHealthConfiguration) GetPropertyHealthStatusForTempProfiles() (value RoamingUserHealthConfiguration_HealthStatusForTempProfiles, err error) {
	retValue, err := instance.GetProperty("HealthStatusForTempProfiles")
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

	value = RoamingUserHealthConfiguration_HealthStatusForTempProfiles(valuetmp)

	return
}

// SetLastProfileDownloadIntervalCautionInHours sets the value of LastProfileDownloadIntervalCautionInHours for the instance
func (instance *Win32_RoamingUserHealthConfiguration) SetPropertyLastProfileDownloadIntervalCautionInHours(value uint16) (err error) {
	return instance.SetProperty("LastProfileDownloadIntervalCautionInHours", (value))
}

// GetLastProfileDownloadIntervalCautionInHours gets the value of LastProfileDownloadIntervalCautionInHours for the instance
func (instance *Win32_RoamingUserHealthConfiguration) GetPropertyLastProfileDownloadIntervalCautionInHours() (value uint16, err error) {
	retValue, err := instance.GetProperty("LastProfileDownloadIntervalCautionInHours")
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

// SetLastProfileDownloadIntervalUnhealthyInHours sets the value of LastProfileDownloadIntervalUnhealthyInHours for the instance
func (instance *Win32_RoamingUserHealthConfiguration) SetPropertyLastProfileDownloadIntervalUnhealthyInHours(value uint16) (err error) {
	return instance.SetProperty("LastProfileDownloadIntervalUnhealthyInHours", (value))
}

// GetLastProfileDownloadIntervalUnhealthyInHours gets the value of LastProfileDownloadIntervalUnhealthyInHours for the instance
func (instance *Win32_RoamingUserHealthConfiguration) GetPropertyLastProfileDownloadIntervalUnhealthyInHours() (value uint16, err error) {
	retValue, err := instance.GetProperty("LastProfileDownloadIntervalUnhealthyInHours")
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

// SetLastProfileUploadIntervalCautionInHours sets the value of LastProfileUploadIntervalCautionInHours for the instance
func (instance *Win32_RoamingUserHealthConfiguration) SetPropertyLastProfileUploadIntervalCautionInHours(value uint16) (err error) {
	return instance.SetProperty("LastProfileUploadIntervalCautionInHours", (value))
}

// GetLastProfileUploadIntervalCautionInHours gets the value of LastProfileUploadIntervalCautionInHours for the instance
func (instance *Win32_RoamingUserHealthConfiguration) GetPropertyLastProfileUploadIntervalCautionInHours() (value uint16, err error) {
	retValue, err := instance.GetProperty("LastProfileUploadIntervalCautionInHours")
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

// SetLastProfileUploadIntervalUnhealthyInHours sets the value of LastProfileUploadIntervalUnhealthyInHours for the instance
func (instance *Win32_RoamingUserHealthConfiguration) SetPropertyLastProfileUploadIntervalUnhealthyInHours(value uint16) (err error) {
	return instance.SetProperty("LastProfileUploadIntervalUnhealthyInHours", (value))
}

// GetLastProfileUploadIntervalUnhealthyInHours gets the value of LastProfileUploadIntervalUnhealthyInHours for the instance
func (instance *Win32_RoamingUserHealthConfiguration) GetPropertyLastProfileUploadIntervalUnhealthyInHours() (value uint16, err error) {
	retValue, err := instance.GetProperty("LastProfileUploadIntervalUnhealthyInHours")
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
