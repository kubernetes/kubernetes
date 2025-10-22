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

// MSFT_StorageModificationEvent struct
type MSFT_StorageModificationEvent struct {
	*MSFT_StorageEvent
}

func NewMSFT_StorageModificationEventEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageModificationEvent, err error) {
	tmp, err := NewMSFT_StorageEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageModificationEvent{
		MSFT_StorageEvent: tmp,
	}
	return
}

func NewMSFT_StorageModificationEventEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageModificationEvent, err error) {
	tmp, err := NewMSFT_StorageEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageModificationEvent{
		MSFT_StorageEvent: tmp,
	}
	return
}
