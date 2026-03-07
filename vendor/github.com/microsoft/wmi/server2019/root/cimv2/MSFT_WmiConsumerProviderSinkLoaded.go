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

// MSFT_WmiConsumerProviderSinkLoaded struct
type MSFT_WmiConsumerProviderSinkLoaded struct {
	*MSFT_WmiConsumerProviderEvent

	//
	Consumer __EventConsumer
}

func NewMSFT_WmiConsumerProviderSinkLoadedEx1(instance *cim.WmiInstance) (newInstance *MSFT_WmiConsumerProviderSinkLoaded, err error) {
	tmp, err := NewMSFT_WmiConsumerProviderEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_WmiConsumerProviderSinkLoaded{
		MSFT_WmiConsumerProviderEvent: tmp,
	}
	return
}

func NewMSFT_WmiConsumerProviderSinkLoadedEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_WmiConsumerProviderSinkLoaded, err error) {
	tmp, err := NewMSFT_WmiConsumerProviderEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_WmiConsumerProviderSinkLoaded{
		MSFT_WmiConsumerProviderEvent: tmp,
	}
	return
}

// SetConsumer sets the value of Consumer for the instance
func (instance *MSFT_WmiConsumerProviderSinkLoaded) SetPropertyConsumer(value __EventConsumer) (err error) {
	return instance.SetProperty("Consumer", (value))
}

// GetConsumer gets the value of Consumer for the instance
func (instance *MSFT_WmiConsumerProviderSinkLoaded) GetPropertyConsumer() (value __EventConsumer, err error) {
	retValue, err := instance.GetProperty("Consumer")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(__EventConsumer)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " __EventConsumer is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = __EventConsumer(valuetmp)

	return
}
