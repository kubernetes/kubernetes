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

// MSFT_WmiThreadPoolThreadDeleted struct
type MSFT_WmiThreadPoolThreadDeleted struct {
	*MSFT_WmiThreadPoolEvent
}

func NewMSFT_WmiThreadPoolThreadDeletedEx1(instance *cim.WmiInstance) (newInstance *MSFT_WmiThreadPoolThreadDeleted, err error) {
	tmp, err := NewMSFT_WmiThreadPoolEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_WmiThreadPoolThreadDeleted{
		MSFT_WmiThreadPoolEvent: tmp,
	}
	return
}

func NewMSFT_WmiThreadPoolThreadDeletedEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_WmiThreadPoolThreadDeleted, err error) {
	tmp, err := NewMSFT_WmiThreadPoolEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_WmiThreadPoolThreadDeleted{
		MSFT_WmiThreadPoolEvent: tmp,
	}
	return
}
