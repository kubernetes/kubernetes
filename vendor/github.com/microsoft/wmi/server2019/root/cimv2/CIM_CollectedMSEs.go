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

// CIM_CollectedMSEs struct
type CIM_CollectedMSEs struct {
	*cim.WmiInstance

	//
	Collection CIM_CollectionOfMSEs

	//
	Member CIM_ManagedSystemElement
}

func NewCIM_CollectedMSEsEx1(instance *cim.WmiInstance) (newInstance *CIM_CollectedMSEs, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &CIM_CollectedMSEs{
		WmiInstance: tmp,
	}
	return
}

func NewCIM_CollectedMSEsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_CollectedMSEs, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_CollectedMSEs{
		WmiInstance: tmp,
	}
	return
}

// SetCollection sets the value of Collection for the instance
func (instance *CIM_CollectedMSEs) SetPropertyCollection(value CIM_CollectionOfMSEs) (err error) {
	return instance.SetProperty("Collection", (value))
}

// GetCollection gets the value of Collection for the instance
func (instance *CIM_CollectedMSEs) GetPropertyCollection() (value CIM_CollectionOfMSEs, err error) {
	retValue, err := instance.GetProperty("Collection")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_CollectionOfMSEs)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_CollectionOfMSEs is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_CollectionOfMSEs(valuetmp)

	return
}

// SetMember sets the value of Member for the instance
func (instance *CIM_CollectedMSEs) SetPropertyMember(value CIM_ManagedSystemElement) (err error) {
	return instance.SetProperty("Member", (value))
}

// GetMember gets the value of Member for the instance
func (instance *CIM_CollectedMSEs) GetPropertyMember() (value CIM_ManagedSystemElement, err error) {
	retValue, err := instance.GetProperty("Member")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_ManagedSystemElement)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_ManagedSystemElement is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_ManagedSystemElement(valuetmp)

	return
}
