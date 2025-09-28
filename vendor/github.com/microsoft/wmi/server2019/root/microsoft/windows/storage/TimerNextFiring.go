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

// __TimerNextFiring struct
type __TimerNextFiring struct {
	*__IndicationRelated

	//
	NextEvent64BitTime int64

	//
	TimerId string
}

func New__TimerNextFiringEx1(instance *cim.WmiInstance) (newInstance *__TimerNextFiring, err error) {
	tmp, err := New__IndicationRelatedEx1(instance)

	if err != nil {
		return
	}
	newInstance = &__TimerNextFiring{
		__IndicationRelated: tmp,
	}
	return
}

func New__TimerNextFiringEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *__TimerNextFiring, err error) {
	tmp, err := New__IndicationRelatedEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &__TimerNextFiring{
		__IndicationRelated: tmp,
	}
	return
}

// SetNextEvent64BitTime sets the value of NextEvent64BitTime for the instance
func (instance *__TimerNextFiring) SetPropertyNextEvent64BitTime(value int64) (err error) {
	return instance.SetProperty("NextEvent64BitTime", (value))
}

// GetNextEvent64BitTime gets the value of NextEvent64BitTime for the instance
func (instance *__TimerNextFiring) GetPropertyNextEvent64BitTime() (value int64, err error) {
	retValue, err := instance.GetProperty("NextEvent64BitTime")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = int64(valuetmp)

	return
}

// SetTimerId sets the value of TimerId for the instance
func (instance *__TimerNextFiring) SetPropertyTimerId(value string) (err error) {
	return instance.SetProperty("TimerId", (value))
}

// GetTimerId gets the value of TimerId for the instance
func (instance *__TimerNextFiring) GetPropertyTimerId() (value string, err error) {
	retValue, err := instance.GetProperty("TimerId")
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
