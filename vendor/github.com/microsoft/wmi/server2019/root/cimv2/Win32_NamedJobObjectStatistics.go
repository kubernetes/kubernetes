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

// Win32_NamedJobObjectStatistics struct
type Win32_NamedJobObjectStatistics struct {
	*Win32_CollectionStatistics
}

func NewWin32_NamedJobObjectStatisticsEx1(instance *cim.WmiInstance) (newInstance *Win32_NamedJobObjectStatistics, err error) {
	tmp, err := NewWin32_CollectionStatisticsEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_NamedJobObjectStatistics{
		Win32_CollectionStatistics: tmp,
	}
	return
}

func NewWin32_NamedJobObjectStatisticsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_NamedJobObjectStatistics, err error) {
	tmp, err := NewWin32_CollectionStatisticsEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_NamedJobObjectStatistics{
		Win32_CollectionStatistics: tmp,
	}
	return
}
