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

// CIM_VideoBIOSFeatureVideoBIOSElements struct
type CIM_VideoBIOSFeatureVideoBIOSElements struct {
	*CIM_SoftwareFeatureSoftwareElements
}

func NewCIM_VideoBIOSFeatureVideoBIOSElementsEx1(instance *cim.WmiInstance) (newInstance *CIM_VideoBIOSFeatureVideoBIOSElements, err error) {
	tmp, err := NewCIM_SoftwareFeatureSoftwareElementsEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_VideoBIOSFeatureVideoBIOSElements{
		CIM_SoftwareFeatureSoftwareElements: tmp,
	}
	return
}

func NewCIM_VideoBIOSFeatureVideoBIOSElementsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_VideoBIOSFeatureVideoBIOSElements, err error) {
	tmp, err := NewCIM_SoftwareFeatureSoftwareElementsEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_VideoBIOSFeatureVideoBIOSElements{
		CIM_SoftwareFeatureSoftwareElements: tmp,
	}
	return
}
