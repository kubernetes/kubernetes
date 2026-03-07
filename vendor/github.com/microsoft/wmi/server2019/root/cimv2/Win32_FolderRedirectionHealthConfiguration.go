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

// Win32_FolderRedirectionHealthConfiguration struct
type Win32_FolderRedirectionHealthConfiguration struct {
	*cim.WmiInstance

	// Cautious threshold, in hours. If the number of hours since the last attempted synchronization is greater than or equal to this threshold, the HealthStatus property of the Win32_FolderRedirectionHealth class is set to Caution.
	LastSyncDurationCautionInHours uint32

	// Unhealthy threshold, in hours. If the number of hours since the last attempted synchronization is greater than or equal to this threshold, the HealthStatus property of the Win32_FolderRedirectionHealth class is set to Unhealthy.
	LastSyncDurationUnhealthyInHours uint32
}

func NewWin32_FolderRedirectionHealthConfigurationEx1(instance *cim.WmiInstance) (newInstance *Win32_FolderRedirectionHealthConfiguration, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_FolderRedirectionHealthConfiguration{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_FolderRedirectionHealthConfigurationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_FolderRedirectionHealthConfiguration, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_FolderRedirectionHealthConfiguration{
		WmiInstance: tmp,
	}
	return
}

// SetLastSyncDurationCautionInHours sets the value of LastSyncDurationCautionInHours for the instance
func (instance *Win32_FolderRedirectionHealthConfiguration) SetPropertyLastSyncDurationCautionInHours(value uint32) (err error) {
	return instance.SetProperty("LastSyncDurationCautionInHours", (value))
}

// GetLastSyncDurationCautionInHours gets the value of LastSyncDurationCautionInHours for the instance
func (instance *Win32_FolderRedirectionHealthConfiguration) GetPropertyLastSyncDurationCautionInHours() (value uint32, err error) {
	retValue, err := instance.GetProperty("LastSyncDurationCautionInHours")
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

// SetLastSyncDurationUnhealthyInHours sets the value of LastSyncDurationUnhealthyInHours for the instance
func (instance *Win32_FolderRedirectionHealthConfiguration) SetPropertyLastSyncDurationUnhealthyInHours(value uint32) (err error) {
	return instance.SetProperty("LastSyncDurationUnhealthyInHours", (value))
}

// GetLastSyncDurationUnhealthyInHours gets the value of LastSyncDurationUnhealthyInHours for the instance
func (instance *Win32_FolderRedirectionHealthConfiguration) GetPropertyLastSyncDurationUnhealthyInHours() (value uint32, err error) {
	retValue, err := instance.GetProperty("LastSyncDurationUnhealthyInHours")
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
