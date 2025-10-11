// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.Microsoft.Windows.Storage
//////////////////////////////////////////////
package storage

import (
	"github.com/microsoft/wmi/pkg/base/instance"
	"github.com/microsoft/wmi/pkg/base/query"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
)

// MSFT_StorageJobOutParams struct
type MSFT_StorageJobOutParams struct {
	*cim.WmiInstance
}

func NewMSFT_StorageJobOutParamsEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageJobOutParams, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageJobOutParams{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StorageJobOutParamsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageJobOutParams, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageJobOutParams{
		WmiInstance: tmp,
	}
	return
}
