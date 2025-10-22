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

// Win32_Registry struct
type Win32_Registry struct {
	*CIM_LogicalElement

	//
	CurrentSize uint32

	//
	MaximumSize uint32

	//
	ProposedSize uint32
}

func NewWin32_RegistryEx1(instance *cim.WmiInstance) (newInstance *Win32_Registry, err error) {
	tmp, err := NewCIM_LogicalElementEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_Registry{
		CIM_LogicalElement: tmp,
	}
	return
}

func NewWin32_RegistryEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_Registry, err error) {
	tmp, err := NewCIM_LogicalElementEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_Registry{
		CIM_LogicalElement: tmp,
	}
	return
}

// SetCurrentSize sets the value of CurrentSize for the instance
func (instance *Win32_Registry) SetPropertyCurrentSize(value uint32) (err error) {
	return instance.SetProperty("CurrentSize", (value))
}

// GetCurrentSize gets the value of CurrentSize for the instance
func (instance *Win32_Registry) GetPropertyCurrentSize() (value uint32, err error) {
	retValue, err := instance.GetProperty("CurrentSize")
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

// SetMaximumSize sets the value of MaximumSize for the instance
func (instance *Win32_Registry) SetPropertyMaximumSize(value uint32) (err error) {
	return instance.SetProperty("MaximumSize", (value))
}

// GetMaximumSize gets the value of MaximumSize for the instance
func (instance *Win32_Registry) GetPropertyMaximumSize() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaximumSize")
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

// SetProposedSize sets the value of ProposedSize for the instance
func (instance *Win32_Registry) SetPropertyProposedSize(value uint32) (err error) {
	return instance.SetProperty("ProposedSize", (value))
}

// GetProposedSize gets the value of ProposedSize for the instance
func (instance *Win32_Registry) GetPropertyProposedSize() (value uint32, err error) {
	retValue, err := instance.GetProperty("ProposedSize")
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
