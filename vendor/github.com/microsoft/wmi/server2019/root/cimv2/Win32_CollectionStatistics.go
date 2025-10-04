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

// Win32_CollectionStatistics struct
type Win32_CollectionStatistics struct {
	*cim.WmiInstance

	//
	Collection CIM_CollectionOfMSEs

	//
	Stats CIM_StatisticalInformation
}

func NewWin32_CollectionStatisticsEx1(instance *cim.WmiInstance) (newInstance *Win32_CollectionStatistics, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_CollectionStatistics{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_CollectionStatisticsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_CollectionStatistics, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_CollectionStatistics{
		WmiInstance: tmp,
	}
	return
}

// SetCollection sets the value of Collection for the instance
func (instance *Win32_CollectionStatistics) SetPropertyCollection(value CIM_CollectionOfMSEs) (err error) {
	return instance.SetProperty("Collection", (value))
}

// GetCollection gets the value of Collection for the instance
func (instance *Win32_CollectionStatistics) GetPropertyCollection() (value CIM_CollectionOfMSEs, err error) {
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

// SetStats sets the value of Stats for the instance
func (instance *Win32_CollectionStatistics) SetPropertyStats(value CIM_StatisticalInformation) (err error) {
	return instance.SetProperty("Stats", (value))
}

// GetStats gets the value of Stats for the instance
func (instance *Win32_CollectionStatistics) GetPropertyStats() (value CIM_StatisticalInformation, err error) {
	retValue, err := instance.GetProperty("Stats")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_StatisticalInformation)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_StatisticalInformation is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_StatisticalInformation(valuetmp)

	return
}
