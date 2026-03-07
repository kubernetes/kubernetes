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

// CIM_CollectionSetting struct
type CIM_CollectionSetting struct {
	*cim.WmiInstance

	//
	Collection CIM_CollectionOfMSEs

	//
	Setting CIM_Setting
}

func NewCIM_CollectionSettingEx1(instance *cim.WmiInstance) (newInstance *CIM_CollectionSetting, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &CIM_CollectionSetting{
		WmiInstance: tmp,
	}
	return
}

func NewCIM_CollectionSettingEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_CollectionSetting, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_CollectionSetting{
		WmiInstance: tmp,
	}
	return
}

// SetCollection sets the value of Collection for the instance
func (instance *CIM_CollectionSetting) SetPropertyCollection(value CIM_CollectionOfMSEs) (err error) {
	return instance.SetProperty("Collection", (value))
}

// GetCollection gets the value of Collection for the instance
func (instance *CIM_CollectionSetting) GetPropertyCollection() (value CIM_CollectionOfMSEs, err error) {
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

// SetSetting sets the value of Setting for the instance
func (instance *CIM_CollectionSetting) SetPropertySetting(value CIM_Setting) (err error) {
	return instance.SetProperty("Setting", (value))
}

// GetSetting gets the value of Setting for the instance
func (instance *CIM_CollectionSetting) GetPropertySetting() (value CIM_Setting, err error) {
	retValue, err := instance.GetProperty("Setting")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_Setting)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_Setting is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_Setting(valuetmp)

	return
}
