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
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
)

// MSFT_WmiConsumerProviderUnloaded struct
type MSFT_WmiConsumerProviderUnloaded struct {
	*MSFT_WmiConsumerProviderEvent
}

func NewMSFT_WmiConsumerProviderUnloadedEx1(instance *cim.WmiInstance) (newInstance *MSFT_WmiConsumerProviderUnloaded, err error) {
	tmp, err := NewMSFT_WmiConsumerProviderEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_WmiConsumerProviderUnloaded{
		MSFT_WmiConsumerProviderEvent: tmp,
	}
	return
}

func NewMSFT_WmiConsumerProviderUnloadedEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_WmiConsumerProviderUnloaded, err error) {
	tmp, err := NewMSFT_WmiConsumerProviderEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_WmiConsumerProviderUnloaded{
		MSFT_WmiConsumerProviderEvent: tmp,
	}
	return
}
