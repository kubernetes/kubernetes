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

// MSFT_StorageSubSystemToMaskingSet struct
type MSFT_StorageSubSystemToMaskingSet struct {
	*cim.WmiInstance

	//
	MaskingSet MSFT_MaskingSet

	//
	StorageSubSystem MSFT_StorageSubSystem
}

func NewMSFT_StorageSubSystemToMaskingSetEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageSubSystemToMaskingSet, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageSubSystemToMaskingSet{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StorageSubSystemToMaskingSetEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageSubSystemToMaskingSet, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageSubSystemToMaskingSet{
		WmiInstance: tmp,
	}
	return
}

// SetMaskingSet sets the value of MaskingSet for the instance
func (instance *MSFT_StorageSubSystemToMaskingSet) SetPropertyMaskingSet(value MSFT_MaskingSet) (err error) {
	return instance.SetProperty("MaskingSet", (value))
}

// GetMaskingSet gets the value of MaskingSet for the instance
func (instance *MSFT_StorageSubSystemToMaskingSet) GetPropertyMaskingSet() (value MSFT_MaskingSet, err error) {
	retValue, err := instance.GetProperty("MaskingSet")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_MaskingSet)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_MaskingSet is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_MaskingSet(valuetmp)

	return
}

// SetStorageSubSystem sets the value of StorageSubSystem for the instance
func (instance *MSFT_StorageSubSystemToMaskingSet) SetPropertyStorageSubSystem(value MSFT_StorageSubSystem) (err error) {
	return instance.SetProperty("StorageSubSystem", (value))
}

// GetStorageSubSystem gets the value of StorageSubSystem for the instance
func (instance *MSFT_StorageSubSystemToMaskingSet) GetPropertyStorageSubSystem() (value MSFT_StorageSubSystem, err error) {
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
