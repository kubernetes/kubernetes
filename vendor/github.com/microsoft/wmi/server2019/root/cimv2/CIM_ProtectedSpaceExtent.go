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

// CIM_ProtectedSpaceExtent struct
type CIM_ProtectedSpaceExtent struct {
	*CIM_StorageExtent

	//
	UserDataStripeDepth uint64
}

func NewCIM_ProtectedSpaceExtentEx1(instance *cim.WmiInstance) (newInstance *CIM_ProtectedSpaceExtent, err error) {
	tmp, err := NewCIM_StorageExtentEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_ProtectedSpaceExtent{
		CIM_StorageExtent: tmp,
	}
	return
}

func NewCIM_ProtectedSpaceExtentEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_ProtectedSpaceExtent, err error) {
	tmp, err := NewCIM_StorageExtentEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_ProtectedSpaceExtent{
		CIM_StorageExtent: tmp,
	}
	return
}

// SetUserDataStripeDepth sets the value of UserDataStripeDepth for the instance
func (instance *CIM_ProtectedSpaceExtent) SetPropertyUserDataStripeDepth(value uint64) (err error) {
	return instance.SetProperty("UserDataStripeDepth", (value))
}

// GetUserDataStripeDepth gets the value of UserDataStripeDepth for the instance
func (instance *CIM_ProtectedSpaceExtent) GetPropertyUserDataStripeDepth() (value uint64, err error) {
	retValue, err := instance.GetProperty("UserDataStripeDepth")
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
