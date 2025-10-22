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

// Win32_SoftwareFeatureSoftwareElements struct
type Win32_SoftwareFeatureSoftwareElements struct {
	*CIM_SoftwareFeatureSoftwareElements
}

func NewWin32_SoftwareFeatureSoftwareElementsEx1(instance *cim.WmiInstance) (newInstance *Win32_SoftwareFeatureSoftwareElements, err error) {
	tmp, err := NewCIM_SoftwareFeatureSoftwareElementsEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_SoftwareFeatureSoftwareElements{
		CIM_SoftwareFeatureSoftwareElements: tmp,
	}
	return
}

func NewWin32_SoftwareFeatureSoftwareElementsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_SoftwareFeatureSoftwareElements, err error) {
	tmp, err := NewCIM_SoftwareFeatureSoftwareElementsEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_SoftwareFeatureSoftwareElements{
		CIM_SoftwareFeatureSoftwareElements: tmp,
	}
	return
}
