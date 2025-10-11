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

// MSFT_StorageChassis struct
type MSFT_StorageChassis struct {
	*MSFT_StorageFaultDomain
}

func NewMSFT_StorageChassisEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageChassis, err error) {
	tmp, err := NewMSFT_StorageFaultDomainEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageChassis{
		MSFT_StorageFaultDomain: tmp,
	}
	return
}

func NewMSFT_StorageChassisEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageChassis, err error) {
	tmp, err := NewMSFT_StorageFaultDomainEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageChassis{
		MSFT_StorageFaultDomain: tmp,
	}
	return
}
