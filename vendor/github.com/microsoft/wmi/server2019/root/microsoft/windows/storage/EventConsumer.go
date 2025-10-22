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

// __EventConsumer struct
type __EventConsumer struct {
	*__IndicationRelated

	//
	CreatorSID []uint8

	//
	MachineName string

	//
	MaximumQueueSize uint32
}

func New__EventConsumerEx1(instance *cim.WmiInstance) (newInstance *__EventConsumer, err error) {
	tmp, err := New__IndicationRelatedEx1(instance)

	if err != nil {
		return
	}
	newInstance = &__EventConsumer{
		__IndicationRelated: tmp,
	}
	return
}

func New__EventConsumerEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *__EventConsumer, err error) {
	tmp, err := New__IndicationRelatedEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &__EventConsumer{
		__IndicationRelated: tmp,
	}
	return
}

// SetCreatorSID sets the value of CreatorSID for the instance
func (instance *__EventConsumer) SetPropertyCreatorSID(value []uint8) (err error) {
	return instance.SetProperty("CreatorSID", (value))
}

// GetCreatorSID gets the value of CreatorSID for the instance
func (instance *__EventConsumer) GetPropertyCreatorSID() (value []uint8, err error) {
	retValue, err := instance.GetProperty("CreatorSID")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint8)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint8 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint8(valuetmp))
	}

	return
}

// SetMachineName sets the value of MachineName for the instance
func (instance *__EventConsumer) SetPropertyMachineName(value string) (err error) {
	return instance.SetProperty("MachineName", (value))
}

// GetMachineName gets the value of MachineName for the instance
func (instance *__EventConsumer) GetPropertyMachineName() (value string, err error) {
	retValue, err := instance.GetProperty("MachineName")
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

// SetMaximumQueueSize sets the value of MaximumQueueSize for the instance
func (instance *__EventConsumer) SetPropertyMaximumQueueSize(value uint32) (err error) {
	return instance.SetProperty("MaximumQueueSize", (value))
}

// GetMaximumQueueSize gets the value of MaximumQueueSize for the instance
func (instance *__EventConsumer) GetPropertyMaximumQueueSize() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaximumQueueSize")
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
