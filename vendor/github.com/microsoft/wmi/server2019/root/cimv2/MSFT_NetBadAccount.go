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

// MSFT_NetBadAccount struct
type MSFT_NetBadAccount struct {
	*MSFT_SCMEventLogEvent
}

func NewMSFT_NetBadAccountEx1(instance *cim.WmiInstance) (newInstance *MSFT_NetBadAccount, err error) {
	tmp, err := NewMSFT_SCMEventLogEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_NetBadAccount{
		MSFT_SCMEventLogEvent: tmp,
	}
	return
}

func NewMSFT_NetBadAccountEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_NetBadAccount, err error) {
	tmp, err := NewMSFT_SCMEventLogEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_NetBadAccount{
		MSFT_SCMEventLogEvent: tmp,
	}
	return
}
