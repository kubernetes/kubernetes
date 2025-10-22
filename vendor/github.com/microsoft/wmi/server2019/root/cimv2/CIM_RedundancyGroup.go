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

// CIM_RedundancyGroup struct
type CIM_RedundancyGroup struct {
	*CIM_LogicalElement

	//
	CreationClassName string

	//
	RedundancyStatus uint16
}

func NewCIM_RedundancyGroupEx1(instance *cim.WmiInstance) (newInstance *CIM_RedundancyGroup, err error) {
	tmp, err := NewCIM_LogicalElementEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_RedundancyGroup{
		CIM_LogicalElement: tmp,
	}
	return
}

func NewCIM_RedundancyGroupEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_RedundancyGroup, err error) {
	tmp, err := NewCIM_LogicalElementEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_RedundancyGroup{
		CIM_LogicalElement: tmp,
	}
	return
}

// SetCreationClassName sets the value of CreationClassName for the instance
func (instance *CIM_RedundancyGroup) SetPropertyCreationClassName(value string) (err error) {
	return instance.SetProperty("CreationClassName", (value))
}

// GetCreationClassName gets the value of CreationClassName for the instance
func (instance *CIM_RedundancyGroup) GetPropertyCreationClassName() (value string, err error) {
	retValue, err := instance.GetProperty("CreationClassName")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetRedundancyStatus sets the value of RedundancyStatus for the instance
func (instance *CIM_RedundancyGroup) SetPropertyRedundancyStatus(value uint16) (err error) {
	return instance.SetProperty("RedundancyStatus", (value))
}

// GetRedundancyStatus gets the value of RedundancyStatus for the instance
func (instance *CIM_RedundancyGroup) GetPropertyRedundancyStatus() (value uint16, err error) {
	retValue, err := instance.GetProperty("RedundancyStatus")
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
