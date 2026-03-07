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

// Win32_OfflineFilesDiskSpaceLimit struct
type Win32_OfflineFilesDiskSpaceLimit struct {
	*cim.WmiInstance

	//
	AutoCacheSizeInMB uint32

	//
	TotalSizeInMB uint32
}

func NewWin32_OfflineFilesDiskSpaceLimitEx1(instance *cim.WmiInstance) (newInstance *Win32_OfflineFilesDiskSpaceLimit, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_OfflineFilesDiskSpaceLimit{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_OfflineFilesDiskSpaceLimitEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_OfflineFilesDiskSpaceLimit, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_OfflineFilesDiskSpaceLimit{
		WmiInstance: tmp,
	}
	return
}

// SetAutoCacheSizeInMB sets the value of AutoCacheSizeInMB for the instance
func (instance *Win32_OfflineFilesDiskSpaceLimit) SetPropertyAutoCacheSizeInMB(value uint32) (err error) {
	return instance.SetProperty("AutoCacheSizeInMB", (value))
}

// GetAutoCacheSizeInMB gets the value of AutoCacheSizeInMB for the instance
func (instance *Win32_OfflineFilesDiskSpaceLimit) GetPropertyAutoCacheSizeInMB() (value uint32, err error) {
	retValue, err := instance.GetProperty("AutoCacheSizeInMB")
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

// SetTotalSizeInMB sets the value of TotalSizeInMB for the instance
func (instance *Win32_OfflineFilesDiskSpaceLimit) SetPropertyTotalSizeInMB(value uint32) (err error) {
	return instance.SetProperty("TotalSizeInMB", (value))
}

// GetTotalSizeInMB gets the value of TotalSizeInMB for the instance
func (instance *Win32_OfflineFilesDiskSpaceLimit) GetPropertyTotalSizeInMB() (value uint32, err error) {
	retValue, err := instance.GetProperty("TotalSizeInMB")
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
