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

// Win32_DiskPartition struct
type Win32_DiskPartition struct {
	*CIM_DiskPartition

	//
	BootPartition bool

	//
	DiskIndex uint32

	//
	HiddenSectors uint32

	//
	Index uint32

	//
	RewritePartition bool

	//
	Size uint64

	//
	StartingOffset uint64

	//
	Type string
}

func NewWin32_DiskPartitionEx1(instance *cim.WmiInstance) (newInstance *Win32_DiskPartition, err error) {
	tmp, err := NewCIM_DiskPartitionEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_DiskPartition{
		CIM_DiskPartition: tmp,
	}
	return
}

func NewWin32_DiskPartitionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_DiskPartition, err error) {
	tmp, err := NewCIM_DiskPartitionEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_DiskPartition{
		CIM_DiskPartition: tmp,
	}
	return
}

// SetBootPartition sets the value of BootPartition for the instance
func (instance *Win32_DiskPartition) SetPropertyBootPartition(value bool) (err error) {
	return instance.SetProperty("BootPartition", (value))
}

// GetBootPartition gets the value of BootPartition for the instance
func (instance *Win32_DiskPartition) GetPropertyBootPartition() (value bool, err error) {
	retValue, err := instance.GetProperty("BootPartition")
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

// SetDiskIndex sets the value of DiskIndex for the instance
func (instance *Win32_DiskPartition) SetPropertyDiskIndex(value uint32) (err error) {
	return instance.SetProperty("DiskIndex", (value))
}

// GetDiskIndex gets the value of DiskIndex for the instance
func (instance *Win32_DiskPartition) GetPropertyDiskIndex() (value uint32, err error) {
	retValue, err := instance.GetProperty("DiskIndex")
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

// SetHiddenSectors sets the value of HiddenSectors for the instance
func (instance *Win32_DiskPartition) SetPropertyHiddenSectors(value uint32) (err error) {
	return instance.SetProperty("HiddenSectors", (value))
}

// GetHiddenSectors gets the value of HiddenSectors for the instance
func (instance *Win32_DiskPartition) GetPropertyHiddenSectors() (value uint32, err error) {
	retValue, err := instance.GetProperty("HiddenSectors")
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

// SetIndex sets the value of Index for the instance
func (instance *Win32_DiskPartition) SetPropertyIndex(value uint32) (err error) {
	return instance.SetProperty("Index", (value))
}

// GetIndex gets the value of Index for the instance
func (instance *Win32_DiskPartition) GetPropertyIndex() (value uint32, err error) {
	retValue, err := instance.GetProperty("Index")
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

// SetRewritePartition sets the value of RewritePartition for the instance
func (instance *Win32_DiskPartition) SetPropertyRewritePartition(value bool) (err error) {
	return instance.SetProperty("RewritePartition", (value))
}

// GetRewritePartition gets the value of RewritePartition for the instance
func (instance *Win32_DiskPartition) GetPropertyRewritePartition() (value bool, err error) {
	retValue, err := instance.GetProperty("RewritePartition")
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

// SetSize sets the value of Size for the instance
func (instance *Win32_DiskPartition) SetPropertySize(value uint64) (err error) {
	return instance.SetProperty("Size", (value))
}

// GetSize gets the value of Size for the instance
func (instance *Win32_DiskPartition) GetPropertySize() (value uint64, err error) {
	retValue, err := instance.GetProperty("Size")
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

// SetStartingOffset sets the value of StartingOffset for the instance
func (instance *Win32_DiskPartition) SetPropertyStartingOffset(value uint64) (err error) {
	return instance.SetProperty("StartingOffset", (value))
}

// GetStartingOffset gets the value of StartingOffset for the instance
func (instance *Win32_DiskPartition) GetPropertyStartingOffset() (value uint64, err error) {
	retValue, err := instance.GetProperty("StartingOffset")
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

// SetType sets the value of Type for the instance
func (instance *Win32_DiskPartition) SetPropertyType(value string) (err error) {
	return instance.SetProperty("Type", (value))
}

// GetType gets the value of Type for the instance
func (instance *Win32_DiskPartition) GetPropertyType() (value string, err error) {
	retValue, err := instance.GetProperty("Type")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}
