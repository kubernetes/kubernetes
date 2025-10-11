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

// MSFT_StorageExtendedStatus struct
type MSFT_StorageExtendedStatus struct {
	*CIM_Error
}

func NewMSFT_StorageExtendedStatusEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageExtendedStatus, err error) {
	tmp, err := NewCIM_ErrorEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageExtendedStatus{
		CIM_Error: tmp,
	}
	return
}

func NewMSFT_StorageExtendedStatusEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageExtendedStatus, err error) {
	tmp, err := NewCIM_ErrorEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageExtendedStatus{
		CIM_Error: tmp,
	}
	return
}
