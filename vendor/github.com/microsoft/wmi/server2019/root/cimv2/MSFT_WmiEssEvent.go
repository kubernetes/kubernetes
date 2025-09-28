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

// MSFT_WmiEssEvent struct
type MSFT_WmiEssEvent struct {
	*MSFT_WmiSelfEvent
}

func NewMSFT_WmiEssEventEx1(instance *cim.WmiInstance) (newInstance *MSFT_WmiEssEvent, err error) {
	tmp, err := NewMSFT_WmiSelfEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_WmiEssEvent{
		MSFT_WmiSelfEvent: tmp,
	}
	return
}

func NewMSFT_WmiEssEventEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_WmiEssEvent, err error) {
	tmp, err := NewMSFT_WmiSelfEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_WmiEssEvent{
		MSFT_WmiSelfEvent: tmp,
	}
	return
}
