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

// CIM_ExtraCapacityGroup struct
type CIM_ExtraCapacityGroup struct {
	*CIM_RedundancyGroup

	//
	MinNumberNeeded uint32
}

func NewCIM_ExtraCapacityGroupEx1(instance *cim.WmiInstance) (newInstance *CIM_ExtraCapacityGroup, err error) {
	tmp, err := NewCIM_RedundancyGroupEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_ExtraCapacityGroup{
		CIM_RedundancyGroup: tmp,
	}
	return
}

func NewCIM_ExtraCapacityGroupEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_ExtraCapacityGroup, err error) {
	tmp, err := NewCIM_RedundancyGroupEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_ExtraCapacityGroup{
		CIM_RedundancyGroup: tmp,
	}
	return
}

// SetMinNumberNeeded sets the value of MinNumberNeeded for the instance
func (instance *CIM_ExtraCapacityGroup) SetPropertyMinNumberNeeded(value uint32) (err error) {
	return instance.SetProperty("MinNumberNeeded", (value))
}

// GetMinNumberNeeded gets the value of MinNumberNeeded for the instance
func (instance *CIM_ExtraCapacityGroup) GetPropertyMinNumberNeeded() (value uint32, err error) {
	retValue, err := instance.GetProperty("MinNumberNeeded")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}
