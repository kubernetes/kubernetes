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

// MSFT_StorageSetting struct
type MSFT_StorageSetting struct {
	*cim.WmiInstance

	//
	NewDiskPolicy uint16

	//
	ScrubPolicy uint32
}

func NewMSFT_StorageSettingEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageSetting, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageSetting{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StorageSettingEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageSetting, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageSetting{
		WmiInstance: tmp,
	}
	return
}

// SetNewDiskPolicy sets the value of NewDiskPolicy for the instance
func (instance *MSFT_StorageSetting) SetPropertyNewDiskPolicy(value uint16) (err error) {
	return instance.SetProperty("NewDiskPolicy", (value))
}

// GetNewDiskPolicy gets the value of NewDiskPolicy for the instance
func (instance *MSFT_StorageSetting) GetPropertyNewDiskPolicy() (value uint16, err error) {
	retValue, err := instance.GetProperty("NewDiskPolicy")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetScrubPolicy sets the value of ScrubPolicy for the instance
func (instance *MSFT_StorageSetting) SetPropertyScrubPolicy(value uint32) (err error) {
	return instance.SetProperty("ScrubPolicy", (value))
}

// GetScrubPolicy gets the value of ScrubPolicy for the instance
func (instance *MSFT_StorageSetting) GetPropertyScrubPolicy() (value uint32, err error) {
	retValue, err := instance.GetProperty("ScrubPolicy")
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

//

// <param name="ReturnValue" type="uint32 "></param>
// <param name="StorageSetting" type="MSFT_StorageSetting "></param>
func (instance *MSFT_StorageSetting) Get( /* OUT */ StorageSetting MSFT_StorageSetting) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("Get")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="NewDiskPolicy" type="uint16 "></param>
// <param name="ScrubPolicy" type="uint32 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageSetting) Set( /* IN */ NewDiskPolicy uint16,
	/* IN */ ScrubPolicy uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Set", NewDiskPolicy, ScrubPolicy)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageSetting) UpdateHostStorageCache() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("UpdateHostStorageCache")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}
