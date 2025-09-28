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

// MSFT_WmiThreadPoolThreadCreated struct
type MSFT_WmiThreadPoolThreadCreated struct {
	*MSFT_WmiThreadPoolEvent
}

func NewMSFT_WmiThreadPoolThreadCreatedEx1(instance *cim.WmiInstance) (newInstance *MSFT_WmiThreadPoolThreadCreated, err error) {
	tmp, err := NewMSFT_WmiThreadPoolEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_WmiThreadPoolThreadCreated{
		MSFT_WmiThreadPoolEvent: tmp,
	}
	return
}

func NewMSFT_WmiThreadPoolThreadCreatedEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_WmiThreadPoolThreadCreated, err error) {
	tmp, err := NewMSFT_WmiThreadPoolEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_WmiThreadPoolThreadCreated{
		MSFT_WmiThreadPoolEvent: tmp,
	}
	return
}
