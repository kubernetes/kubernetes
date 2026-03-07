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

// MSFT_StorageQoSPolicyToChildPolicy struct
type MSFT_StorageQoSPolicyToChildPolicy struct {
	*cim.WmiInstance

	//
	ChildPolicy MSFT_StorageQoSPolicy

	//
	ParentPolicy MSFT_StorageQoSPolicy
}

func NewMSFT_StorageQoSPolicyToChildPolicyEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageQoSPolicyToChildPolicy, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageQoSPolicyToChildPolicy{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StorageQoSPolicyToChildPolicyEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageQoSPolicyToChildPolicy, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageQoSPolicyToChildPolicy{
		WmiInstance: tmp,
	}
	return
}

// SetChildPolicy sets the value of ChildPolicy for the instance
func (instance *MSFT_StorageQoSPolicyToChildPolicy) SetPropertyChildPolicy(value MSFT_StorageQoSPolicy) (err error) {
	return instance.SetProperty("ChildPolicy", (value))
}

// GetChildPolicy gets the value of ChildPolicy for the instance
func (instance *MSFT_StorageQoSPolicyToChildPolicy) GetPropertyChildPolicy() (value MSFT_StorageQoSPolicy, err error) {
	retValue, err := instance.GetProperty("ChildPolicy")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_StorageQoSPolicy)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_StorageQoSPolicy is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_StorageQoSPolicy(valuetmp)

	return
}

// SetParentPolicy sets the value of ParentPolicy for the instance
func (instance *MSFT_StorageQoSPolicyToChildPolicy) SetPropertyParentPolicy(value MSFT_StorageQoSPolicy) (err error) {
	return instance.SetProperty("ParentPolicy", (value))
}

// GetParentPolicy gets the value of ParentPolicy for the instance
func (instance *MSFT_StorageQoSPolicyToChildPolicy) GetPropertyParentPolicy() (value MSFT_StorageQoSPolicy, err error) {
	retValue, err := instance.GetProperty("ParentPolicy")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_StorageQoSPolicy)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_StorageQoSPolicy is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_StorageQoSPolicy(valuetmp)

	return
}
