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

// __NotifyStatus struct
type __NotifyStatus struct {
	*cim.WmiInstance

	//
	StatusCode uint32
}

func New__NotifyStatusEx1(instance *cim.WmiInstance) (newInstance *__NotifyStatus, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &__NotifyStatus{
		WmiInstance: tmp,
	}
	return
}

func New__NotifyStatusEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *__NotifyStatus, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &__NotifyStatus{
		WmiInstance: tmp,
	}
	return
}

// SetStatusCode sets the value of StatusCode for the instance
func (instance *__NotifyStatus) SetPropertyStatusCode(value uint32) (err error) {
	return instance.SetProperty("StatusCode", (value))
}

// GetStatusCode gets the value of StatusCode for the instance
func (instance *__NotifyStatus) GetPropertyStatusCode() (value uint32, err error) {
	retValue, err := instance.GetProperty("StatusCode")
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
