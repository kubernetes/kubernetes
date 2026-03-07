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
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
)

// MSFT_StorageArrivalEvent struct
type MSFT_StorageArrivalEvent struct {
	*MSFT_StorageEvent
}

func NewMSFT_StorageArrivalEventEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageArrivalEvent, err error) {
	tmp, err := NewMSFT_StorageEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageArrivalEvent{
		MSFT_StorageEvent: tmp,
	}
	return
}

func NewMSFT_StorageArrivalEventEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageArrivalEvent, err error) {
	tmp, err := NewMSFT_StorageEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageArrivalEvent{
		MSFT_StorageEvent: tmp,
	}
	return
}
