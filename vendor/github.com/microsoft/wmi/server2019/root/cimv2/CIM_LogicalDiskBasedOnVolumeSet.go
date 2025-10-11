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

// CIM_LogicalDiskBasedOnVolumeSet struct
type CIM_LogicalDiskBasedOnVolumeSet struct {
	*CIM_BasedOn
}

func NewCIM_LogicalDiskBasedOnVolumeSetEx1(instance *cim.WmiInstance) (newInstance *CIM_LogicalDiskBasedOnVolumeSet, err error) {
	tmp, err := NewCIM_BasedOnEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_LogicalDiskBasedOnVolumeSet{
		CIM_BasedOn: tmp,
	}
	return
}

func NewCIM_LogicalDiskBasedOnVolumeSetEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_LogicalDiskBasedOnVolumeSet, err error) {
	tmp, err := NewCIM_BasedOnEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_LogicalDiskBasedOnVolumeSet{
		CIM_BasedOn: tmp,
	}
	return
}
