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

// MSFT_WmiFilterDeactivated struct
type MSFT_WmiFilterDeactivated struct {
	*MSFT_WmiFilterEvent
}

func NewMSFT_WmiFilterDeactivatedEx1(instance *cim.WmiInstance) (newInstance *MSFT_WmiFilterDeactivated, err error) {
	tmp, err := NewMSFT_WmiFilterEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_WmiFilterDeactivated{
		MSFT_WmiFilterEvent: tmp,
	}
	return
}

func NewMSFT_WmiFilterDeactivatedEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_WmiFilterDeactivated, err error) {
	tmp, err := NewMSFT_WmiFilterEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_WmiFilterDeactivated{
		MSFT_WmiFilterEvent: tmp,
	}
	return
}
