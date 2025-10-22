// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.CIMV2
//////////////////////////////////////////////
package cimv2

import (
	"github.com/microsoft/wmi/pkg/base/instance"
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// CIM_ParticipatesInSet struct
type CIM_ParticipatesInSet struct {
	*cim.WmiInstance

	//
	Element CIM_PhysicalElement

	//
	Set CIM_ReplacementSet
}

func NewCIM_ParticipatesInSetEx1(instance *cim.WmiInstance) (newInstance *CIM_ParticipatesInSet, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &CIM_ParticipatesInSet{
		WmiInstance: tmp,
	}
	return
}

func NewCIM_ParticipatesInSetEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_ParticipatesInSet, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_ParticipatesInSet{
		WmiInstance: tmp,
	}
	return
}

// SetElement sets the value of Element for the instance
func (instance *CIM_ParticipatesInSet) SetPropertyElement(value CIM_PhysicalElement) (err error) {
	return instance.SetProperty("Element", (value))
}

// GetElement gets the value of Element for the instance
func (instance *CIM_ParticipatesInSet) GetPropertyElement() (value CIM_PhysicalElement, err error) {
	retValue, err := instance.GetProperty("Element")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_PhysicalElement)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_PhysicalElement is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_PhysicalElement(valuetmp)

	return
}

// SetSet sets the value of Set for the instance
func (instance *CIM_ParticipatesInSet) SetPropertySet(value CIM_ReplacementSet) (err error) {
	return instance.SetProperty("Set", (value))
}

// GetSet gets the value of Set for the instance
func (instance *CIM_ParticipatesInSet) GetPropertySet() (value CIM_ReplacementSet, err error) {
	retValue, err := instance.GetProperty("Set")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_ReplacementSet)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_ReplacementSet is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_ReplacementSet(valuetmp)

	return
}
