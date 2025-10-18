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

// MSFT_SCMEventLogEvent struct
type MSFT_SCMEventLogEvent struct {
	*MSFT_SCMEvent
}

func NewMSFT_SCMEventLogEventEx1(instance *cim.WmiInstance) (newInstance *MSFT_SCMEventLogEvent, err error) {
	tmp, err := NewMSFT_SCMEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_SCMEventLogEvent{
		MSFT_SCMEvent: tmp,
	}
	return
}

func NewMSFT_SCMEventLogEventEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_SCMEventLogEvent, err error) {
	tmp, err := NewMSFT_SCMEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_SCMEventLogEvent{
		MSFT_SCMEvent: tmp,
	}
	return
}
