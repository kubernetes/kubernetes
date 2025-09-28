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

// CIM_VolumeSet struct
type CIM_VolumeSet struct {
	*CIM_StorageExtent

	//
	PSExtentInterleaveDepth uint64

	//
	PSExtentStripeLength uint64
}

func NewCIM_VolumeSetEx1(instance *cim.WmiInstance) (newInstance *CIM_VolumeSet, err error) {
	tmp, err := NewCIM_StorageExtentEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_VolumeSet{
		CIM_StorageExtent: tmp,
	}
	return
}

func NewCIM_VolumeSetEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_VolumeSet, err error) {
	tmp, err := NewCIM_StorageExtentEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_VolumeSet{
		CIM_StorageExtent: tmp,
	}
	return
}

// SetPSExtentInterleaveDepth sets the value of PSExtentInterleaveDepth for the instance
func (instance *CIM_VolumeSet) SetPropertyPSExtentInterleaveDepth(value uint64) (err error) {
	return instance.SetProperty("PSExtentInterleaveDepth", (value))
}

// GetPSExtentInterleaveDepth gets the value of PSExtentInterleaveDepth for the instance
func (instance *CIM_VolumeSet) GetPropertyPSExtentInterleaveDepth() (value uint64, err error) {
	retValue, err := instance.GetProperty("PSExtentInterleaveDepth")
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

// SetPSExtentStripeLength sets the value of PSExtentStripeLength for the instance
func (instance *CIM_VolumeSet) SetPropertyPSExtentStripeLength(value uint64) (err error) {
	return instance.SetProperty("PSExtentStripeLength", (value))
}

// GetPSExtentStripeLength gets the value of PSExtentStripeLength for the instance
func (instance *CIM_VolumeSet) GetPropertyPSExtentStripeLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("PSExtentStripeLength")
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
