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

// MSFT_NCProvCancelQuery struct
type MSFT_NCProvCancelQuery struct {
	*MSFT_NCProvEvent

	//
	ID uint32
}

func NewMSFT_NCProvCancelQueryEx1(instance *cim.WmiInstance) (newInstance *MSFT_NCProvCancelQuery, err error) {
	tmp, err := NewMSFT_NCProvEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_NCProvCancelQuery{
		MSFT_NCProvEvent: tmp,
	}
	return
}

func NewMSFT_NCProvCancelQueryEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_NCProvCancelQuery, err error) {
	tmp, err := NewMSFT_NCProvEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_NCProvCancelQuery{
		MSFT_NCProvEvent: tmp,
	}
	return
}

// SetID sets the value of ID for the instance
func (instance *MSFT_NCProvCancelQuery) SetPropertyID(value uint32) (err error) {
	return instance.SetProperty("ID", (value))
}

// GetID gets the value of ID for the instance
func (instance *MSFT_NCProvCancelQuery) GetPropertyID() (value uint32, err error) {
	retValue, err := instance.GetProperty("ID")
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
