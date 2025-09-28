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

// Win32_PerfRawData_NETMemoryCache40_NETMemoryCache40 struct
type Win32_PerfRawData_NETMemoryCache40_NETMemoryCache40 struct {
	*Win32_PerfRawData

	//
	CacheEntries uint32

	//
	CacheHitRatio uint32

	//
	CacheHitRatio_Base uint32

	//
	CacheHits uint32

	//
	CacheMisses uint32

	//
	CacheTrims uint32

	//
	CacheTurnoverRate uint32
}

func NewWin32_PerfRawData_NETMemoryCache40_NETMemoryCache40Ex1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_NETMemoryCache40_NETMemoryCache40, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_NETMemoryCache40_NETMemoryCache40{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_NETMemoryCache40_NETMemoryCache40Ex6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_NETMemoryCache40_NETMemoryCache40, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_NETMemoryCache40_NETMemoryCache40{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetCacheEntries sets the value of CacheEntries for the instance
func (instance *Win32_PerfRawData_NETMemoryCache40_NETMemoryCache40) SetPropertyCacheEntries(value uint32) (err error) {
	return instance.SetProperty("CacheEntries", (value))
}

// GetCacheEntries gets the value of CacheEntries for the instance
func (instance *Win32_PerfRawData_NETMemoryCache40_NETMemoryCache40) GetPropertyCacheEntries() (value uint32, err error) {
	retValue, err := instance.GetProperty("CacheEntries")
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

// SetCacheHitRatio sets the value of CacheHitRatio for the instance
func (instance *Win32_PerfRawData_NETMemoryCache40_NETMemoryCache40) SetPropertyCacheHitRatio(value uint32) (err error) {
	return instance.SetProperty("CacheHitRatio", (value))
}

// GetCacheHitRatio gets the value of CacheHitRatio for the instance
func (instance *Win32_PerfRawData_NETMemoryCache40_NETMemoryCache40) GetPropertyCacheHitRatio() (value uint32, err error) {
	retValue, err := instance.GetProperty("CacheHitRatio")
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

// SetCacheHitRatio_Base sets the value of CacheHitRatio_Base for the instance
func (instance *Win32_PerfRawData_NETMemoryCache40_NETMemoryCache40) SetPropertyCacheHitRatio_Base(value uint32) (err error) {
	return instance.SetProperty("CacheHitRatio_Base", (value))
}

// GetCacheHitRatio_Base gets the value of CacheHitRatio_Base for the instance
func (instance *Win32_PerfRawData_NETMemoryCache40_NETMemoryCache40) GetPropertyCacheHitRatio_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("CacheHitRatio_Base")
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

// SetCacheHits sets the value of CacheHits for the instance
func (instance *Win32_PerfRawData_NETMemoryCache40_NETMemoryCache40) SetPropertyCacheHits(value uint32) (err error) {
	return instance.SetProperty("CacheHits", (value))
}

// GetCacheHits gets the value of CacheHits for the instance
func (instance *Win32_PerfRawData_NETMemoryCache40_NETMemoryCache40) GetPropertyCacheHits() (value uint32, err error) {
	retValue, err := instance.GetProperty("CacheHits")
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

// SetCacheMisses sets the value of CacheMisses for the instance
func (instance *Win32_PerfRawData_NETMemoryCache40_NETMemoryCache40) SetPropertyCacheMisses(value uint32) (err error) {
	return instance.SetProperty("CacheMisses", (value))
}

// GetCacheMisses gets the value of CacheMisses for the instance
func (instance *Win32_PerfRawData_NETMemoryCache40_NETMemoryCache40) GetPropertyCacheMisses() (value uint32, err error) {
	retValue, err := instance.GetProperty("CacheMisses")
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

// SetCacheTrims sets the value of CacheTrims for the instance
func (instance *Win32_PerfRawData_NETMemoryCache40_NETMemoryCache40) SetPropertyCacheTrims(value uint32) (err error) {
	return instance.SetProperty("CacheTrims", (value))
}

// GetCacheTrims gets the value of CacheTrims for the instance
func (instance *Win32_PerfRawData_NETMemoryCache40_NETMemoryCache40) GetPropertyCacheTrims() (value uint32, err error) {
	retValue, err := instance.GetProperty("CacheTrims")
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

// SetCacheTurnoverRate sets the value of CacheTurnoverRate for the instance
func (instance *Win32_PerfRawData_NETMemoryCache40_NETMemoryCache40) SetPropertyCacheTurnoverRate(value uint32) (err error) {
	return instance.SetProperty("CacheTurnoverRate", (value))
}

// GetCacheTurnoverRate gets the value of CacheTurnoverRate for the instance
func (instance *Win32_PerfRawData_NETMemoryCache40_NETMemoryCache40) GetPropertyCacheTurnoverRate() (value uint32, err error) {
	retValue, err := instance.GetProperty("CacheTurnoverRate")
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
