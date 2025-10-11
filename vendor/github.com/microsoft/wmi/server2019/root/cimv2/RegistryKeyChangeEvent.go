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

// RegistryKeyChangeEvent struct
type RegistryKeyChangeEvent struct {
	*RegistryEvent

	//
	Hive string

	//
	KeyPath string
}

func NewRegistryKeyChangeEventEx1(instance *cim.WmiInstance) (newInstance *RegistryKeyChangeEvent, err error) {
	tmp, err := NewRegistryEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &RegistryKeyChangeEvent{
		RegistryEvent: tmp,
	}
	return
}

func NewRegistryKeyChangeEventEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *RegistryKeyChangeEvent, err error) {
	tmp, err := NewRegistryEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &RegistryKeyChangeEvent{
		RegistryEvent: tmp,
	}
	return
}

// SetHive sets the value of Hive for the instance
func (instance *RegistryKeyChangeEvent) SetPropertyHive(value string) (err error) {
	return instance.SetProperty("Hive", (value))
}

// GetHive gets the value of Hive for the instance
func (instance *RegistryKeyChangeEvent) GetPropertyHive() (value string, err error) {
	retValue, err := instance.GetProperty("Hive")
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

// SetKeyPath sets the value of KeyPath for the instance
func (instance *RegistryKeyChangeEvent) SetPropertyKeyPath(value string) (err error) {
	return instance.SetProperty("KeyPath", (value))
}

// GetKeyPath gets the value of KeyPath for the instance
func (instance *RegistryKeyChangeEvent) GetPropertyKeyPath() (value string, err error) {
	retValue, err := instance.GetProperty("KeyPath")
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
