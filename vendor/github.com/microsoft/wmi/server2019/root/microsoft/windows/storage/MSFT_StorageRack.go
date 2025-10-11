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

// MSFT_StorageRack struct
type MSFT_StorageRack struct {
	*MSFT_StorageFaultDomain
}

func NewMSFT_StorageRackEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageRack, err error) {
	tmp, err := NewMSFT_StorageFaultDomainEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageRack{
		MSFT_StorageFaultDomain: tmp,
	}
	return
}

func NewMSFT_StorageRackEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageRack, err error) {
	tmp, err := NewMSFT_StorageFaultDomainEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageRack{
		MSFT_StorageFaultDomain: tmp,
	}
	return
}
