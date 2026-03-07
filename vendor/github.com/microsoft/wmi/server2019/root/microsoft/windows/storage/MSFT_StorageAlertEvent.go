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

// MSFT_StorageAlertEvent struct
type MSFT_StorageAlertEvent struct {
	*MSFT_StorageEvent

	//
	AlertType uint16
}

func NewMSFT_StorageAlertEventEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageAlertEvent, err error) {
	tmp, err := NewMSFT_StorageEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageAlertEvent{
		MSFT_StorageEvent: tmp,
	}
	return
}

func NewMSFT_StorageAlertEventEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageAlertEvent, err error) {
	tmp, err := NewMSFT_StorageEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageAlertEvent{
		MSFT_StorageEvent: tmp,
	}
	return
}

// SetAlertType sets the value of AlertType for the instance
func (instance *MSFT_StorageAlertEvent) SetPropertyAlertType(value uint16) (err error) {
	return instance.SetProperty("AlertType", (value))
}

// GetAlertType gets the value of AlertType for the instance
func (instance *MSFT_StorageAlertEvent) GetPropertyAlertType() (value uint16, err error) {
	retValue, err := instance.GetProperty("AlertType")
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
