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

// MSFT_StorageJobToAffectedStorageObject struct
type MSFT_StorageJobToAffectedStorageObject struct {
	*cim.WmiInstance

	//
	AffectedStorageObject MSFT_StorageObject

	//
	StorageJob MSFT_StorageJob
}

func NewMSFT_StorageJobToAffectedStorageObjectEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageJobToAffectedStorageObject, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageJobToAffectedStorageObject{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StorageJobToAffectedStorageObjectEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageJobToAffectedStorageObject, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageJobToAffectedStorageObject{
		WmiInstance: tmp,
	}
	return
}

// SetAffectedStorageObject sets the value of AffectedStorageObject for the instance
func (instance *MSFT_StorageJobToAffectedStorageObject) SetPropertyAffectedStorageObject(value MSFT_StorageObject) (err error) {
	return instance.SetProperty("AffectedStorageObject", (value))
}

// GetAffectedStorageObject gets the value of AffectedStorageObject for the instance
func (instance *MSFT_StorageJobToAffectedStorageObject) GetPropertyAffectedStorageObject() (value MSFT_StorageObject, err error) {
	retValue, err := instance.GetProperty("AffectedStorageObject")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_StorageObject)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_StorageObject is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_StorageObject(valuetmp)

	return
}

// SetStorageJob sets the value of StorageJob for the instance
func (instance *MSFT_StorageJobToAffectedStorageObject) SetPropertyStorageJob(value MSFT_StorageJob) (err error) {
	return instance.SetProperty("StorageJob", (value))
}

// GetStorageJob gets the value of StorageJob for the instance
func (instance *MSFT_StorageJobToAffectedStorageObject) GetPropertyStorageJob() (value MSFT_StorageJob, err error) {
	retValue, err := instance.GetProperty("StorageJob")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_StorageJob)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_StorageJob is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_StorageJob(valuetmp)

	return
}
