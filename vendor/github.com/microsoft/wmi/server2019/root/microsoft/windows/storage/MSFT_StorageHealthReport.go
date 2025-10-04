// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.Microsoft.Windows.Storage
//////////////////////////////////////////////
package storage

import (
	"github.com/microsoft/wmi/pkg/base/instance"
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// MSFT_StorageHealthReport struct
type MSFT_StorageHealthReport struct {
	*cim.WmiInstance

	//
	Records []MSFT_HealthRecord

	//
	ReportedObjectUniqueId string

	//
	StorageSubsystemUniqueId string
}

func NewMSFT_StorageHealthReportEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageHealthReport, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageHealthReport{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StorageHealthReportEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageHealthReport, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageHealthReport{
		WmiInstance: tmp,
	}
	return
}

// SetRecords sets the value of Records for the instance
func (instance *MSFT_StorageHealthReport) SetPropertyRecords(value []MSFT_HealthRecord) (err error) {
	return instance.SetProperty("Records", (value))
}

// GetRecords gets the value of Records for the instance
func (instance *MSFT_StorageHealthReport) GetPropertyRecords() (value []MSFT_HealthRecord, err error) {
	retValue, err := instance.GetProperty("Records")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(MSFT_HealthRecord)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " MSFT_HealthRecord is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, MSFT_HealthRecord(valuetmp))
	}

	return
}

// SetReportedObjectUniqueId sets the value of ReportedObjectUniqueId for the instance
func (instance *MSFT_StorageHealthReport) SetPropertyReportedObjectUniqueId(value string) (err error) {
	return instance.SetProperty("ReportedObjectUniqueId", (value))
}

// GetReportedObjectUniqueId gets the value of ReportedObjectUniqueId for the instance
func (instance *MSFT_StorageHealthReport) GetPropertyReportedObjectUniqueId() (value string, err error) {
	retValue, err := instance.GetProperty("ReportedObjectUniqueId")
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

// SetStorageSubsystemUniqueId sets the value of StorageSubsystemUniqueId for the instance
func (instance *MSFT_StorageHealthReport) SetPropertyStorageSubsystemUniqueId(value string) (err error) {
	return instance.SetProperty("StorageSubsystemUniqueId", (value))
}

// GetStorageSubsystemUniqueId gets the value of StorageSubsystemUniqueId for the instance
func (instance *MSFT_StorageHealthReport) GetPropertyStorageSubsystemUniqueId() (value string, err error) {
	retValue, err := instance.GetProperty("StorageSubsystemUniqueId")
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
