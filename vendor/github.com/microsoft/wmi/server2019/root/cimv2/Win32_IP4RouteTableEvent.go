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

// Win32_IP4RouteTableEvent struct
type Win32_IP4RouteTableEvent struct {
	*__ExtrinsicEvent
}

func NewWin32_IP4RouteTableEventEx1(instance *cim.WmiInstance) (newInstance *Win32_IP4RouteTableEvent, err error) {
	tmp, err := New__ExtrinsicEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_IP4RouteTableEvent{
		__ExtrinsicEvent: tmp,
	}
	return
}

func NewWin32_IP4RouteTableEventEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_IP4RouteTableEvent, err error) {
	tmp, err := New__ExtrinsicEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_IP4RouteTableEvent{
		__ExtrinsicEvent: tmp,
	}
	return
}
