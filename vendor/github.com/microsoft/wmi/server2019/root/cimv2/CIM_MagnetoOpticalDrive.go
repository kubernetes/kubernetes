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

// CIM_MagnetoOpticalDrive struct
type CIM_MagnetoOpticalDrive struct {
	*CIM_MediaAccessDevice
}

func NewCIM_MagnetoOpticalDriveEx1(instance *cim.WmiInstance) (newInstance *CIM_MagnetoOpticalDrive, err error) {
	tmp, err := NewCIM_MediaAccessDeviceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_MagnetoOpticalDrive{
		CIM_MediaAccessDevice: tmp,
	}
	return
}

func NewCIM_MagnetoOpticalDriveEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_MagnetoOpticalDrive, err error) {
	tmp, err := NewCIM_MediaAccessDeviceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_MagnetoOpticalDrive{
		CIM_MediaAccessDevice: tmp,
	}
	return
}
