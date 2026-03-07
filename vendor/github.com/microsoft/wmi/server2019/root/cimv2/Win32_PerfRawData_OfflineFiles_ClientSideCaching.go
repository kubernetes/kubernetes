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

// Win32_PerfRawData_OfflineFiles_ClientSideCaching struct
type Win32_PerfRawData_OfflineFiles_ClientSideCaching struct {
	*Win32_PerfRawData

	//
	ApplicationBytesReadFromCache uint64

	//
	ApplicationBytesReadFromServer uint64

	//
	ApplicationBytesReadFromServerNotCached uint64

	//
	PrefetchBytesReadFromCache uint64

	//
	PrefetchBytesReadFromServer uint64

	//
	PrefetchOperationsQueued uint32

	//
	SMBBranchCacheBytesPublished uint64

	//
	SMBBranchCacheBytesReceived uint64

	//
	SMBBranchCacheBytesRequested uint64

	//
	SMBBranchCacheBytesRequestedFromServer uint64

	//
	SMBBranchCacheHashBytesReceived uint64

	//
	SMBBranchCacheHashesReceived uint32

	//
	SMBBranchCacheHashesRequested uint32
}

func NewWin32_PerfRawData_OfflineFiles_ClientSideCachingEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_OfflineFiles_ClientSideCaching, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_OfflineFiles_ClientSideCaching{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_OfflineFiles_ClientSideCachingEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_OfflineFiles_ClientSideCaching, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_OfflineFiles_ClientSideCaching{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetApplicationBytesReadFromCache sets the value of ApplicationBytesReadFromCache for the instance
func (instance *Win32_PerfRawData_OfflineFiles_ClientSideCaching) SetPropertyApplicationBytesReadFromCache(value uint64) (err error) {
	return instance.SetProperty("ApplicationBytesReadFromCache", (value))
}

// GetApplicationBytesReadFromCache gets the value of ApplicationBytesReadFromCache for the instance
func (instance *Win32_PerfRawData_OfflineFiles_ClientSideCaching) GetPropertyApplicationBytesReadFromCache() (value uint64, err error) {
	retValue, err := instance.GetProperty("ApplicationBytesReadFromCache")
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

// SetApplicationBytesReadFromServer sets the value of ApplicationBytesReadFromServer for the instance
func (instance *Win32_PerfRawData_OfflineFiles_ClientSideCaching) SetPropertyApplicationBytesReadFromServer(value uint64) (err error) {
	return instance.SetProperty("ApplicationBytesReadFromServer", (value))
}

// GetApplicationBytesReadFromServer gets the value of ApplicationBytesReadFromServer for the instance
func (instance *Win32_PerfRawData_OfflineFiles_ClientSideCaching) GetPropertyApplicationBytesReadFromServer() (value uint64, err error) {
	retValue, err := instance.GetProperty("ApplicationBytesReadFromServer")
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

// SetApplicationBytesReadFromServerNotCached sets the value of ApplicationBytesReadFromServerNotCached for the instance
func (instance *Win32_PerfRawData_OfflineFiles_ClientSideCaching) SetPropertyApplicationBytesReadFromServerNotCached(value uint64) (err error) {
	return instance.SetProperty("ApplicationBytesReadFromServerNotCached", (value))
}

// GetApplicationBytesReadFromServerNotCached gets the value of ApplicationBytesReadFromServerNotCached for the instance
func (instance *Win32_PerfRawData_OfflineFiles_ClientSideCaching) GetPropertyApplicationBytesReadFromServerNotCached() (value uint64, err error) {
	retValue, err := instance.GetProperty("ApplicationBytesReadFromServerNotCached")
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

// SetPrefetchBytesReadFromCache sets the value of PrefetchBytesReadFromCache for the instance
func (instance *Win32_PerfRawData_OfflineFiles_ClientSideCaching) SetPropertyPrefetchBytesReadFromCache(value uint64) (err error) {
	return instance.SetProperty("PrefetchBytesReadFromCache", (value))
}

// GetPrefetchBytesReadFromCache gets the value of PrefetchBytesReadFromCache for the instance
func (instance *Win32_PerfRawData_OfflineFiles_ClientSideCaching) GetPropertyPrefetchBytesReadFromCache() (value uint64, err error) {
	retValue, err := instance.GetProperty("PrefetchBytesReadFromCache")
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

// SetPrefetchBytesReadFromServer sets the value of PrefetchBytesReadFromServer for the instance
func (instance *Win32_PerfRawData_OfflineFiles_ClientSideCaching) SetPropertyPrefetchBytesReadFromServer(value uint64) (err error) {
	return instance.SetProperty("PrefetchBytesReadFromServer", (value))
}

// GetPrefetchBytesReadFromServer gets the value of PrefetchBytesReadFromServer for the instance
func (instance *Win32_PerfRawData_OfflineFiles_ClientSideCaching) GetPropertyPrefetchBytesReadFromServer() (value uint64, err error) {
	retValue, err := instance.GetProperty("PrefetchBytesReadFromServer")
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

// SetPrefetchOperationsQueued sets the value of PrefetchOperationsQueued for the instance
func (instance *Win32_PerfRawData_OfflineFiles_ClientSideCaching) SetPropertyPrefetchOperationsQueued(value uint32) (err error) {
	return instance.SetProperty("PrefetchOperationsQueued", (value))
}

// GetPrefetchOperationsQueued gets the value of PrefetchOperationsQueued for the instance
func (instance *Win32_PerfRawData_OfflineFiles_ClientSideCaching) GetPropertyPrefetchOperationsQueued() (value uint32, err error) {
	retValue, err := instance.GetProperty("PrefetchOperationsQueued")
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

// SetSMBBranchCacheBytesPublished sets the value of SMBBranchCacheBytesPublished for the instance
func (instance *Win32_PerfRawData_OfflineFiles_ClientSideCaching) SetPropertySMBBranchCacheBytesPublished(value uint64) (err error) {
	return instance.SetProperty("SMBBranchCacheBytesPublished", (value))
}

// GetSMBBranchCacheBytesPublished gets the value of SMBBranchCacheBytesPublished for the instance
func (instance *Win32_PerfRawData_OfflineFiles_ClientSideCaching) GetPropertySMBBranchCacheBytesPublished() (value uint64, err error) {
	retValue, err := instance.GetProperty("SMBBranchCacheBytesPublished")
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

// SetSMBBranchCacheBytesReceived sets the value of SMBBranchCacheBytesReceived for the instance
func (instance *Win32_PerfRawData_OfflineFiles_ClientSideCaching) SetPropertySMBBranchCacheBytesReceived(value uint64) (err error) {
	return instance.SetProperty("SMBBranchCacheBytesReceived", (value))
}

// GetSMBBranchCacheBytesReceived gets the value of SMBBranchCacheBytesReceived for the instance
func (instance *Win32_PerfRawData_OfflineFiles_ClientSideCaching) GetPropertySMBBranchCacheBytesReceived() (value uint64, err error) {
	retValue, err := instance.GetProperty("SMBBranchCacheBytesReceived")
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

// SetSMBBranchCacheBytesRequested sets the value of SMBBranchCacheBytesRequested for the instance
func (instance *Win32_PerfRawData_OfflineFiles_ClientSideCaching) SetPropertySMBBranchCacheBytesRequested(value uint64) (err error) {
	return instance.SetProperty("SMBBranchCacheBytesRequested", (value))
}

// GetSMBBranchCacheBytesRequested gets the value of SMBBranchCacheBytesRequested for the instance
func (instance *Win32_PerfRawData_OfflineFiles_ClientSideCaching) GetPropertySMBBranchCacheBytesRequested() (value uint64, err error) {
	retValue, err := instance.GetProperty("SMBBranchCacheBytesRequested")
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

// SetSMBBranchCacheBytesRequestedFromServer sets the value of SMBBranchCacheBytesRequestedFromServer for the instance
func (instance *Win32_PerfRawData_OfflineFiles_ClientSideCaching) SetPropertySMBBranchCacheBytesRequestedFromServer(value uint64) (err error) {
	return instance.SetProperty("SMBBranchCacheBytesRequestedFromServer", (value))
}

// GetSMBBranchCacheBytesRequestedFromServer gets the value of SMBBranchCacheBytesRequestedFromServer for the instance
func (instance *Win32_PerfRawData_OfflineFiles_ClientSideCaching) GetPropertySMBBranchCacheBytesRequestedFromServer() (value uint64, err error) {
	retValue, err := instance.GetProperty("SMBBranchCacheBytesRequestedFromServer")
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

// SetSMBBranchCacheHashBytesReceived sets the value of SMBBranchCacheHashBytesReceived for the instance
func (instance *Win32_PerfRawData_OfflineFiles_ClientSideCaching) SetPropertySMBBranchCacheHashBytesReceived(value uint64) (err error) {
	return instance.SetProperty("SMBBranchCacheHashBytesReceived", (value))
}

// GetSMBBranchCacheHashBytesReceived gets the value of SMBBranchCacheHashBytesReceived for the instance
func (instance *Win32_PerfRawData_OfflineFiles_ClientSideCaching) GetPropertySMBBranchCacheHashBytesReceived() (value uint64, err error) {
	retValue, err := instance.GetProperty("SMBBranchCacheHashBytesReceived")
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

// SetSMBBranchCacheHashesReceived sets the value of SMBBranchCacheHashesReceived for the instance
func (instance *Win32_PerfRawData_OfflineFiles_ClientSideCaching) SetPropertySMBBranchCacheHashesReceived(value uint32) (err error) {
	return instance.SetProperty("SMBBranchCacheHashesReceived", (value))
}

// GetSMBBranchCacheHashesReceived gets the value of SMBBranchCacheHashesReceived for the instance
func (instance *Win32_PerfRawData_OfflineFiles_ClientSideCaching) GetPropertySMBBranchCacheHashesReceived() (value uint32, err error) {
	retValue, err := instance.GetProperty("SMBBranchCacheHashesReceived")
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

// SetSMBBranchCacheHashesRequested sets the value of SMBBranchCacheHashesRequested for the instance
func (instance *Win32_PerfRawData_OfflineFiles_ClientSideCaching) SetPropertySMBBranchCacheHashesRequested(value uint32) (err error) {
	return instance.SetProperty("SMBBranchCacheHashesRequested", (value))
}

// GetSMBBranchCacheHashesRequested gets the value of SMBBranchCacheHashesRequested for the instance
func (instance *Win32_PerfRawData_OfflineFiles_ClientSideCaching) GetPropertySMBBranchCacheHashesRequested() (value uint32, err error) {
	retValue, err := instance.GetProperty("SMBBranchCacheHashesRequested")
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
