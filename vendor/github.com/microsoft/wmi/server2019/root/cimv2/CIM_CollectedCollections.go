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

// CIM_CollectedCollections struct
type CIM_CollectedCollections struct {
	*cim.WmiInstance

	//
	Collection CIM_CollectionOfMSEs

	//
	CollectionInCollection CIM_CollectionOfMSEs
}

func NewCIM_CollectedCollectionsEx1(instance *cim.WmiInstance) (newInstance *CIM_CollectedCollections, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &CIM_CollectedCollections{
		WmiInstance: tmp,
	}
	return
}

func NewCIM_CollectedCollectionsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_CollectedCollections, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_CollectedCollections{
		WmiInstance: tmp,
	}
	return
}

// SetCollection sets the value of Collection for the instance
func (instance *CIM_CollectedCollections) SetPropertyCollection(value CIM_CollectionOfMSEs) (err error) {
	return instance.SetProperty("Collection", (value))
}

// GetCollection gets the value of Collection for the instance
func (instance *CIM_CollectedCollections) GetPropertyCollection() (value CIM_CollectionOfMSEs, err error) {
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

// SetCollectionInCollection sets the value of CollectionInCollection for the instance
func (instance *CIM_CollectedCollections) SetPropertyCollectionInCollection(value CIM_CollectionOfMSEs) (err error) {
	return instance.SetProperty("CollectionInCollection", (value))
}

// GetCollectionInCollection gets the value of CollectionInCollection for the instance
func (instance *CIM_CollectedCollections) GetPropertyCollectionInCollection() (value CIM_CollectionOfMSEs, err error) {
	retValue, err := instance.GetProperty("CollectionInCollection")
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
