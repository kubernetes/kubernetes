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
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// CIM_StorageRedundancyGroup struct
type CIM_StorageRedundancyGroup struct {
	*CIM_RedundancyGroup

	//
	TypeOfAlgorithm uint16
}

func NewCIM_StorageRedundancyGroupEx1(instance *cim.WmiInstance) (newInstance *CIM_StorageRedundancyGroup, err error) {
	tmp, err := NewCIM_RedundancyGroupEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_StorageRedundancyGroup{
		CIM_RedundancyGroup: tmp,
	}
	return
}

func NewCIM_StorageRedundancyGroupEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_StorageRedundancyGroup, err error) {
	tmp, err := NewCIM_RedundancyGroupEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_StorageRedundancyGroup{
		CIM_RedundancyGroup: tmp,
	}
	return
}

// SetTypeOfAlgorithm sets the value of TypeOfAlgorithm for the instance
func (instance *CIM_StorageRedundancyGroup) SetPropertyTypeOfAlgorithm(value uint16) (err error) {
	return instance.SetProperty("TypeOfAlgorithm", (value))
}

// GetTypeOfAlgorithm gets the value of TypeOfAlgorithm for the instance
func (instance *CIM_StorageRedundancyGroup) GetPropertyTypeOfAlgorithm() (value uint16, err error) {
	retValue, err := instance.GetProperty("TypeOfAlgorithm")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}
