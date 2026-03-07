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

// MSFT_NCProvClientConnected struct
type MSFT_NCProvClientConnected struct {
	*MSFT_NCProvEvent

	//
	Inproc bool
}

func NewMSFT_NCProvClientConnectedEx1(instance *cim.WmiInstance) (newInstance *MSFT_NCProvClientConnected, err error) {
	tmp, err := NewMSFT_NCProvEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_NCProvClientConnected{
		MSFT_NCProvEvent: tmp,
	}
	return
}

func NewMSFT_NCProvClientConnectedEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_NCProvClientConnected, err error) {
	tmp, err := NewMSFT_NCProvEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_NCProvClientConnected{
		MSFT_NCProvEvent: tmp,
	}
	return
}

// SetInproc sets the value of Inproc for the instance
func (instance *MSFT_NCProvClientConnected) SetPropertyInproc(value bool) (err error) {
	return instance.SetProperty("Inproc", (value))
}

// GetInproc gets the value of Inproc for the instance
func (instance *MSFT_NCProvClientConnected) GetPropertyInproc() (value bool, err error) {
	retValue, err := instance.GetProperty("Inproc")
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
