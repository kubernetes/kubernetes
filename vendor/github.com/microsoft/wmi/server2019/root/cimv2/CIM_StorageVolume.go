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

// CIM_StorageVolume struct
type CIM_StorageVolume struct {
	*CIM_StorageExtent
}

func NewCIM_StorageVolumeEx1(instance *cim.WmiInstance) (newInstance *CIM_StorageVolume, err error) {
	tmp, err := NewCIM_StorageExtentEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_StorageVolume{
		CIM_StorageExtent: tmp,
	}
	return
}

func NewCIM_StorageVolumeEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_StorageVolume, err error) {
	tmp, err := NewCIM_StorageExtentEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_StorageVolume{
		CIM_StorageExtent: tmp,
	}
	return
}
