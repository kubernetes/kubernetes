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

// CIM_RelatedStatistics struct
type CIM_RelatedStatistics struct {
	*cim.WmiInstance

	//
	RelatedStats CIM_StatisticalInformation

	//
	Stats CIM_StatisticalInformation
}

func NewCIM_RelatedStatisticsEx1(instance *cim.WmiInstance) (newInstance *CIM_RelatedStatistics, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &CIM_RelatedStatistics{
		WmiInstance: tmp,
	}
	return
}

func NewCIM_RelatedStatisticsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_RelatedStatistics, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_RelatedStatistics{
		WmiInstance: tmp,
	}
	return
}

// SetRelatedStats sets the value of RelatedStats for the instance
func (instance *CIM_RelatedStatistics) SetPropertyRelatedStats(value CIM_StatisticalInformation) (err error) {
	return instance.SetProperty("RelatedStats", (value))
}

// GetRelatedStats gets the value of RelatedStats for the instance
func (instance *CIM_RelatedStatistics) GetPropertyRelatedStats() (value CIM_StatisticalInformation, err error) {
	retValue, err := instance.GetProperty("RelatedStats")
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

// SetStats sets the value of Stats for the instance
func (instance *CIM_RelatedStatistics) SetPropertyStats(value CIM_StatisticalInformation) (err error) {
	return instance.SetProperty("Stats", (value))
}

// GetStats gets the value of Stats for the instance
func (instance *CIM_RelatedStatistics) GetPropertyStats() (value CIM_StatisticalInformation, err error) {
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
