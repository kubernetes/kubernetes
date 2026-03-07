// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.Microsoft.Windows.Storage
//////////////////////////////////////////////
package storage

import (
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// MSFT_VirtualDiskToVirtualDisk struct
type MSFT_VirtualDiskToVirtualDisk struct {
	*MSFT_Synchronized

	//
	SourceVirtualDisk MSFT_VirtualDisk

	//
	TargetVirtualDisk MSFT_VirtualDisk
}

func NewMSFT_VirtualDiskToVirtualDiskEx1(instance *cim.WmiInstance) (newInstance *MSFT_VirtualDiskToVirtualDisk, err error) {
	tmp, err := NewMSFT_SynchronizedEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_VirtualDiskToVirtualDisk{
		MSFT_Synchronized: tmp,
	}
	return
}

func NewMSFT_VirtualDiskToVirtualDiskEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_VirtualDiskToVirtualDisk, err error) {
	tmp, err := NewMSFT_SynchronizedEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_VirtualDiskToVirtualDisk{
		MSFT_Synchronized: tmp,
	}
	return
}

// SetSourceVirtualDisk sets the value of SourceVirtualDisk for the instance
func (instance *MSFT_VirtualDiskToVirtualDisk) SetPropertySourceVirtualDisk(value MSFT_VirtualDisk) (err error) {
	return instance.SetProperty("SourceVirtualDisk", (value))
}

// GetSourceVirtualDisk gets the value of SourceVirtualDisk for the instance
func (instance *MSFT_VirtualDiskToVirtualDisk) GetPropertySourceVirtualDisk() (value MSFT_VirtualDisk, err error) {
	retValue, err := instance.GetProperty("SourceVirtualDisk")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_VirtualDisk)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_VirtualDisk is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_VirtualDisk(valuetmp)

	return
}

// SetTargetVirtualDisk sets the value of TargetVirtualDisk for the instance
func (instance *MSFT_VirtualDiskToVirtualDisk) SetPropertyTargetVirtualDisk(value MSFT_VirtualDisk) (err error) {
	return instance.SetProperty("TargetVirtualDisk", (value))
}

// GetTargetVirtualDisk gets the value of TargetVirtualDisk for the instance
func (instance *MSFT_VirtualDiskToVirtualDisk) GetPropertyTargetVirtualDisk() (value MSFT_VirtualDisk, err error) {
	retValue, err := instance.GetProperty("TargetVirtualDisk")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_VirtualDisk)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_VirtualDisk is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_VirtualDisk(valuetmp)

	return
}
