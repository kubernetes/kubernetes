// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.CIMV2
//////////////////////////////////////////////
package cimv2

import (
	"github.com/microsoft/wmi/pkg/base/instance"
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// NTEventlogProviderConfig struct
type NTEventlogProviderConfig struct {
	*cim.WmiInstance

	//
	LastBootUpTime string
}

func NewNTEventlogProviderConfigEx1(instance *cim.WmiInstance) (newInstance *NTEventlogProviderConfig, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &NTEventlogProviderConfig{
		WmiInstance: tmp,
	}
	return
}

func NewNTEventlogProviderConfigEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *NTEventlogProviderConfig, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &NTEventlogProviderConfig{
		WmiInstance: tmp,
	}
	return
}

// SetLastBootUpTime sets the value of LastBootUpTime for the instance
func (instance *NTEventlogProviderConfig) SetPropertyLastBootUpTime(value string) (err error) {
	return instance.SetProperty("LastBootUpTime", (value))
}

// GetLastBootUpTime gets the value of LastBootUpTime for the instance
func (instance *NTEventlogProviderConfig) GetPropertyLastBootUpTime() (value string, err error) {
	retValue, err := instance.GetProperty("LastBootUpTime")
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
