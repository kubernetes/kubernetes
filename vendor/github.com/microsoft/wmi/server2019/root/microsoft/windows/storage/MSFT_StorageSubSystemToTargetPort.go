// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.Microsoft.Windows.Storage
//////////////////////////////////////////////
package storage

import (
	"github.com/microsoft/wmi/pkg/base/instance"
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// MSFT_StorageSubSystemToTargetPort struct
type MSFT_StorageSubSystemToTargetPort struct {
	*cim.WmiInstance

	//
	StorageSubSystem MSFT_StorageSubSystem

	//
	TargetPort MSFT_TargetPort
}

func NewMSFT_StorageSubSystemToTargetPortEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageSubSystemToTargetPort, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageSubSystemToTargetPort{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StorageSubSystemToTargetPortEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageSubSystemToTargetPort, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageSubSystemToTargetPort{
		WmiInstance: tmp,
	}
	return
}

// SetStorageSubSystem sets the value of StorageSubSystem for the instance
func (instance *MSFT_StorageSubSystemToTargetPort) SetPropertyStorageSubSystem(value MSFT_StorageSubSystem) (err error) {
	return instance.SetProperty("StorageSubSystem", (value))
}

// GetStorageSubSystem gets the value of StorageSubSystem for the instance
func (instance *MSFT_StorageSubSystemToTargetPort) GetPropertyStorageSubSystem() (value MSFT_StorageSubSystem, err error) {
	retValue, err := instance.GetProperty("StorageSubSystem")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_StorageSubSystem)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_StorageSubSystem is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_StorageSubSystem(valuetmp)

	return
}

// SetTargetPort sets the value of TargetPort for the instance
func (instance *MSFT_StorageSubSystemToTargetPort) SetPropertyTargetPort(value MSFT_TargetPort) (err error) {
	return instance.SetProperty("TargetPort", (value))
}

// GetTargetPort gets the value of TargetPort for the instance
func (instance *MSFT_StorageSubSystemToTargetPort) GetPropertyTargetPort() (value MSFT_TargetPort, err error) {
	retValue, err := instance.GetProperty("TargetPort")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_TargetPort)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_TargetPort is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_TargetPort(valuetmp)

	return
}
