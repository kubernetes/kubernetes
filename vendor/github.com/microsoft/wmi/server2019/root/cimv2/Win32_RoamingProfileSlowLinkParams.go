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

// Win32_RoamingProfileSlowLinkParams struct
type Win32_RoamingProfileSlowLinkParams struct {
	*cim.WmiInstance

	// The connection speed, in kilobytes per second (kbps). This threshold is used to determine if the connection is a slow link. If the server's transfer rate in kbps is less than this threshold, the connection is considered to be slow. This property applies to IP networks.
	ConnectionTransferRate uint32

	// The slow-network connection timeout, in milliseconds. This threshold is used to determine if the connection is a slow link. If the delay in milliseconds is greater than this threshold, the connection is considered to be slow. This property applies to non-IP networks.
	TimeOut uint16
}

func NewWin32_RoamingProfileSlowLinkParamsEx1(instance *cim.WmiInstance) (newInstance *Win32_RoamingProfileSlowLinkParams, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_RoamingProfileSlowLinkParams{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_RoamingProfileSlowLinkParamsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_RoamingProfileSlowLinkParams, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_RoamingProfileSlowLinkParams{
		WmiInstance: tmp,
	}
	return
}

// SetConnectionTransferRate sets the value of ConnectionTransferRate for the instance
func (instance *Win32_RoamingProfileSlowLinkParams) SetPropertyConnectionTransferRate(value uint32) (err error) {
	return instance.SetProperty("ConnectionTransferRate", (value))
}

// GetConnectionTransferRate gets the value of ConnectionTransferRate for the instance
func (instance *Win32_RoamingProfileSlowLinkParams) GetPropertyConnectionTransferRate() (value uint32, err error) {
	retValue, err := instance.GetProperty("ConnectionTransferRate")
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

// SetTimeOut sets the value of TimeOut for the instance
func (instance *Win32_RoamingProfileSlowLinkParams) SetPropertyTimeOut(value uint16) (err error) {
	return instance.SetProperty("TimeOut", (value))
}

// GetTimeOut gets the value of TimeOut for the instance
func (instance *Win32_RoamingProfileSlowLinkParams) GetPropertyTimeOut() (value uint16, err error) {
	retValue, err := instance.GetProperty("TimeOut")
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
