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

// MSFT_NetTakeOwnership struct
type MSFT_NetTakeOwnership struct {
	*MSFT_SCMEventLogEvent

	//
	RegistryKey string
}

func NewMSFT_NetTakeOwnershipEx1(instance *cim.WmiInstance) (newInstance *MSFT_NetTakeOwnership, err error) {
	tmp, err := NewMSFT_SCMEventLogEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_NetTakeOwnership{
		MSFT_SCMEventLogEvent: tmp,
	}
	return
}

func NewMSFT_NetTakeOwnershipEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_NetTakeOwnership, err error) {
	tmp, err := NewMSFT_SCMEventLogEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_NetTakeOwnership{
		MSFT_SCMEventLogEvent: tmp,
	}
	return
}

// SetRegistryKey sets the value of RegistryKey for the instance
func (instance *MSFT_NetTakeOwnership) SetPropertyRegistryKey(value string) (err error) {
	return instance.SetProperty("RegistryKey", (value))
}

// GetRegistryKey gets the value of RegistryKey for the instance
func (instance *MSFT_NetTakeOwnership) GetPropertyRegistryKey() (value string, err error) {
	retValue, err := instance.GetProperty("RegistryKey")
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
