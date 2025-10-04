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

// CIM_DiskPartition struct
type CIM_DiskPartition struct {
	*CIM_StorageExtent

	//
	Bootable bool

	//
	PrimaryPartition bool
}

func NewCIM_DiskPartitionEx1(instance *cim.WmiInstance) (newInstance *CIM_DiskPartition, err error) {
	tmp, err := NewCIM_StorageExtentEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_DiskPartition{
		CIM_StorageExtent: tmp,
	}
	return
}

func NewCIM_DiskPartitionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_DiskPartition, err error) {
	tmp, err := NewCIM_StorageExtentEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_DiskPartition{
		CIM_StorageExtent: tmp,
	}
	return
}

// SetBootable sets the value of Bootable for the instance
func (instance *CIM_DiskPartition) SetPropertyBootable(value bool) (err error) {
	return instance.SetProperty("Bootable", (value))
}

// GetBootable gets the value of Bootable for the instance
func (instance *CIM_DiskPartition) GetPropertyBootable() (value bool, err error) {
	retValue, err := instance.GetProperty("Bootable")
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

// SetPrimaryPartition sets the value of PrimaryPartition for the instance
func (instance *CIM_DiskPartition) SetPropertyPrimaryPartition(value bool) (err error) {
	return instance.SetProperty("PrimaryPartition", (value))
}

// GetPrimaryPartition gets the value of PrimaryPartition for the instance
func (instance *CIM_DiskPartition) GetPropertyPrimaryPartition() (value bool, err error) {
	retValue, err := instance.GetProperty("PrimaryPartition")
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
