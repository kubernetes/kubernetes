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

// Win32_DiskQuota struct
type Win32_DiskQuota struct {
	*cim.WmiInstance

	//
	DiskSpaceUsed uint64

	//
	Limit uint64

	//
	QuotaVolume Win32_LogicalDisk

	//
	Status uint32

	//
	User Win32_Account

	//
	WarningLimit uint64
}

func NewWin32_DiskQuotaEx1(instance *cim.WmiInstance) (newInstance *Win32_DiskQuota, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_DiskQuota{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_DiskQuotaEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_DiskQuota, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_DiskQuota{
		WmiInstance: tmp,
	}
	return
}

// SetDiskSpaceUsed sets the value of DiskSpaceUsed for the instance
func (instance *Win32_DiskQuota) SetPropertyDiskSpaceUsed(value uint64) (err error) {
	return instance.SetProperty("DiskSpaceUsed", (value))
}

// GetDiskSpaceUsed gets the value of DiskSpaceUsed for the instance
func (instance *Win32_DiskQuota) GetPropertyDiskSpaceUsed() (value uint64, err error) {
	retValue, err := instance.GetProperty("DiskSpaceUsed")
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

// SetLimit sets the value of Limit for the instance
func (instance *Win32_DiskQuota) SetPropertyLimit(value uint64) (err error) {
	return instance.SetProperty("Limit", (value))
}

// GetLimit gets the value of Limit for the instance
func (instance *Win32_DiskQuota) GetPropertyLimit() (value uint64, err error) {
	retValue, err := instance.GetProperty("Limit")
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

// SetQuotaVolume sets the value of QuotaVolume for the instance
func (instance *Win32_DiskQuota) SetPropertyQuotaVolume(value Win32_LogicalDisk) (err error) {
	return instance.SetProperty("QuotaVolume", (value))
}

// GetQuotaVolume gets the value of QuotaVolume for the instance
func (instance *Win32_DiskQuota) GetPropertyQuotaVolume() (value Win32_LogicalDisk, err error) {
	retValue, err := instance.GetProperty("QuotaVolume")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_LogicalDisk)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_LogicalDisk is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_LogicalDisk(valuetmp)

	return
}

// SetStatus sets the value of Status for the instance
func (instance *Win32_DiskQuota) SetPropertyStatus(value uint32) (err error) {
	return instance.SetProperty("Status", (value))
}

// GetStatus gets the value of Status for the instance
func (instance *Win32_DiskQuota) GetPropertyStatus() (value uint32, err error) {
	retValue, err := instance.GetProperty("Status")
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

// SetUser sets the value of User for the instance
func (instance *Win32_DiskQuota) SetPropertyUser(value Win32_Account) (err error) {
	return instance.SetProperty("User", (value))
}

// GetUser gets the value of User for the instance
func (instance *Win32_DiskQuota) GetPropertyUser() (value Win32_Account, err error) {
	retValue, err := instance.GetProperty("User")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_Account)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_Account is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_Account(valuetmp)

	return
}

// SetWarningLimit sets the value of WarningLimit for the instance
func (instance *Win32_DiskQuota) SetPropertyWarningLimit(value uint64) (err error) {
	return instance.SetProperty("WarningLimit", (value))
}

// GetWarningLimit gets the value of WarningLimit for the instance
func (instance *Win32_DiskQuota) GetPropertyWarningLimit() (value uint64, err error) {
	retValue, err := instance.GetProperty("WarningLimit")
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
