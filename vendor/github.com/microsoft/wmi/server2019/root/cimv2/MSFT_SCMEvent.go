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

// MSFT_SCMEvent struct
type MSFT_SCMEvent struct {
	*__ExtrinsicEvent
}

func NewMSFT_SCMEventEx1(instance *cim.WmiInstance) (newInstance *MSFT_SCMEvent, err error) {
	tmp, err := New__ExtrinsicEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_SCMEvent{
		__ExtrinsicEvent: tmp,
	}
	return
}

func NewMSFT_SCMEventEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_SCMEvent, err error) {
	tmp, err := New__ExtrinsicEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_SCMEvent{
		__ExtrinsicEvent: tmp,
	}
	return
}
