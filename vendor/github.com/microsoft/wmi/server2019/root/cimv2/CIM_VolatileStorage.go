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

// CIM_VolatileStorage struct
type CIM_VolatileStorage struct {
	*CIM_Memory

	//
	Cacheable bool

	//
	CacheType uint16
}

func NewCIM_VolatileStorageEx1(instance *cim.WmiInstance) (newInstance *CIM_VolatileStorage, err error) {
	tmp, err := NewCIM_MemoryEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_VolatileStorage{
		CIM_Memory: tmp,
	}
	return
}

func NewCIM_VolatileStorageEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_VolatileStorage, err error) {
	tmp, err := NewCIM_MemoryEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_VolatileStorage{
		CIM_Memory: tmp,
	}
	return
}

// SetCacheable sets the value of Cacheable for the instance
func (instance *CIM_VolatileStorage) SetPropertyCacheable(value bool) (err error) {
	return instance.SetProperty("Cacheable", (value))
}

// GetCacheable gets the value of Cacheable for the instance
func (instance *CIM_VolatileStorage) GetPropertyCacheable() (value bool, err error) {
	retValue, err := instance.GetProperty("Cacheable")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetCacheType sets the value of CacheType for the instance
func (instance *CIM_VolatileStorage) SetPropertyCacheType(value uint16) (err error) {
	return instance.SetProperty("CacheType", (value))
}

// GetCacheType gets the value of CacheType for the instance
func (instance *CIM_VolatileStorage) GetPropertyCacheType() (value uint16, err error) {
	retValue, err := instance.GetProperty("CacheType")
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
