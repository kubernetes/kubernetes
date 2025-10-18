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

// MSFT_HealthRecordReal64 struct
type MSFT_HealthRecordReal64 struct {
	*MSFT_HealthRecord

	//
	Value float64
}

func NewMSFT_HealthRecordReal64Ex1(instance *cim.WmiInstance) (newInstance *MSFT_HealthRecordReal64, err error) {
	tmp, err := NewMSFT_HealthRecordEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_HealthRecordReal64{
		MSFT_HealthRecord: tmp,
	}
	return
}

func NewMSFT_HealthRecordReal64Ex6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_HealthRecordReal64, err error) {
	tmp, err := NewMSFT_HealthRecordEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_HealthRecordReal64{
		MSFT_HealthRecord: tmp,
	}
	return
}

// SetValue sets the value of Value for the instance
func (instance *MSFT_HealthRecordReal64) SetPropertyValue(value float64) (err error) {
	return instance.SetProperty("Value", (value))
}

// GetValue gets the value of Value for the instance
func (instance *MSFT_HealthRecordReal64) GetPropertyValue() (value float64, err error) {
	retValue, err := instance.GetProperty("Value")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(float64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " float64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = float64(valuetmp)

	return
}
