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

// MSFT_MaskingSetToInitiatorId struct
type MSFT_MaskingSetToInitiatorId struct {
	*cim.WmiInstance

	//
	InitiatorId MSFT_InitiatorId

	//
	MaskingSet MSFT_MaskingSet
}

func NewMSFT_MaskingSetToInitiatorIdEx1(instance *cim.WmiInstance) (newInstance *MSFT_MaskingSetToInitiatorId, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_MaskingSetToInitiatorId{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_MaskingSetToInitiatorIdEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_MaskingSetToInitiatorId, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_MaskingSetToInitiatorId{
		WmiInstance: tmp,
	}
	return
}

// SetInitiatorId sets the value of InitiatorId for the instance
func (instance *MSFT_MaskingSetToInitiatorId) SetPropertyInitiatorId(value MSFT_InitiatorId) (err error) {
	return instance.SetProperty("InitiatorId", (value))
}

// GetInitiatorId gets the value of InitiatorId for the instance
func (instance *MSFT_MaskingSetToInitiatorId) GetPropertyInitiatorId() (value MSFT_InitiatorId, err error) {
	retValue, err := instance.GetProperty("InitiatorId")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_InitiatorId)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_InitiatorId is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_InitiatorId(valuetmp)

	return
}

// SetMaskingSet sets the value of MaskingSet for the instance
func (instance *MSFT_MaskingSetToInitiatorId) SetPropertyMaskingSet(value MSFT_MaskingSet) (err error) {
	return instance.SetProperty("MaskingSet", (value))
}

// GetMaskingSet gets the value of MaskingSet for the instance
func (instance *MSFT_MaskingSetToInitiatorId) GetPropertyMaskingSet() (value MSFT_MaskingSet, err error) {
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
