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

// MSFT_NetCircularDependencyAuto struct
type MSFT_NetCircularDependencyAuto struct {
	*MSFT_SCMEventLogEvent
}

func NewMSFT_NetCircularDependencyAutoEx1(instance *cim.WmiInstance) (newInstance *MSFT_NetCircularDependencyAuto, err error) {
	tmp, err := NewMSFT_SCMEventLogEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_NetCircularDependencyAuto{
		MSFT_SCMEventLogEvent: tmp,
	}
	return
}

func NewMSFT_NetCircularDependencyAutoEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_NetCircularDependencyAuto, err error) {
	tmp, err := NewMSFT_SCMEventLogEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_NetCircularDependencyAuto{
		MSFT_SCMEventLogEvent: tmp,
	}
	return
}
