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

// Win32_DiskDriveToDiskPartition struct
type Win32_DiskDriveToDiskPartition struct {
	*CIM_MediaPresent
}

func NewWin32_DiskDriveToDiskPartitionEx1(instance *cim.WmiInstance) (newInstance *Win32_DiskDriveToDiskPartition, err error) {
	tmp, err := NewCIM_MediaPresentEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_DiskDriveToDiskPartition{
		CIM_MediaPresent: tmp,
	}
	return
}

func NewWin32_DiskDriveToDiskPartitionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_DiskDriveToDiskPartition, err error) {
	tmp, err := NewCIM_MediaPresentEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_DiskDriveToDiskPartition{
		CIM_MediaPresent: tmp,
	}
	return
}
