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

// MSFT_StorageQoSPolicyStore struct
type MSFT_StorageQoSPolicyStore struct {
	*cim.WmiInstance

	//
	Id string

	//
	IOPSNormalizationSize uint32
}

func NewMSFT_StorageQoSPolicyStoreEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageQoSPolicyStore, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageQoSPolicyStore{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StorageQoSPolicyStoreEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageQoSPolicyStore, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageQoSPolicyStore{
		WmiInstance: tmp,
	}
	return
}

// SetId sets the value of Id for the instance
func (instance *MSFT_StorageQoSPolicyStore) SetPropertyId(value string) (err error) {
	return instance.SetProperty("Id", (value))
}

// GetId gets the value of Id for the instance
func (instance *MSFT_StorageQoSPolicyStore) GetPropertyId() (value string, err error) {
	retValue, err := instance.GetProperty("Id")
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

// SetIOPSNormalizationSize sets the value of IOPSNormalizationSize for the instance
func (instance *MSFT_StorageQoSPolicyStore) SetPropertyIOPSNormalizationSize(value uint32) (err error) {
	return instance.SetProperty("IOPSNormalizationSize", (value))
}

// GetIOPSNormalizationSize gets the value of IOPSNormalizationSize for the instance
func (instance *MSFT_StorageQoSPolicyStore) GetPropertyIOPSNormalizationSize() (value uint32, err error) {
	retValue, err := instance.GetProperty("IOPSNormalizationSize")
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

//

// <param name="Policy" type="MSFT_StorageQoSPolicy "></param>

// <param name="Policy" type="MSFT_StorageQoSPolicy "></param>
// <param name="ReturnValue" type="int32 "></param>
func (instance *MSFT_StorageQoSPolicyStore) CreatePolicy( /* IN/OUT */ Policy MSFT_StorageQoSPolicy) (result int32, err error) {
	retVal, err := instance.InvokeMethod("CreatePolicy")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = int32(retValue)
	return

}

//

// <param name="IOPSNormalizationSize" type="uint32 "></param>

// <param name="ReturnValue" type="int32 "></param>
func (instance *MSFT_StorageQoSPolicyStore) SetAttributes( /* IN */ IOPSNormalizationSize uint32) (result int32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetAttributes", IOPSNormalizationSize)
	if err != nil {
		return
	}
	result = int32(retVal)
	return

}
