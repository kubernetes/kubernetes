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

// CIM_NonVolatileStorage struct
type CIM_NonVolatileStorage struct {
	*CIM_Memory

	//
	IsWriteable bool
}

func NewCIM_NonVolatileStorageEx1(instance *cim.WmiInstance) (newInstance *CIM_NonVolatileStorage, err error) {
	tmp, err := NewCIM_MemoryEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_NonVolatileStorage{
		CIM_Memory: tmp,
	}
	return
}

func NewCIM_NonVolatileStorageEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_NonVolatileStorage, err error) {
	tmp, err := NewCIM_MemoryEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_NonVolatileStorage{
		CIM_Memory: tmp,
	}
	return
}

// SetIsWriteable sets the value of IsWriteable for the instance
func (instance *CIM_NonVolatileStorage) SetPropertyIsWriteable(value bool) (err error) {
	return instance.SetProperty("IsWriteable", (value))
}

// GetIsWriteable gets the value of IsWriteable for the instance
func (instance *CIM_NonVolatileStorage) GetPropertyIsWriteable() (value bool, err error) {
	retValue, err := instance.GetProperty("IsWriteable")
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
