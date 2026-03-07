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

// MSFT_StorageQoSPolicyToFlow struct
type MSFT_StorageQoSPolicyToFlow struct {
	*cim.WmiInstance

	//
	Flow MSFT_StorageQoSFlow

	//
	Policy MSFT_StorageQoSPolicy
}

func NewMSFT_StorageQoSPolicyToFlowEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageQoSPolicyToFlow, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageQoSPolicyToFlow{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StorageQoSPolicyToFlowEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageQoSPolicyToFlow, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageQoSPolicyToFlow{
		WmiInstance: tmp,
	}
	return
}

// SetFlow sets the value of Flow for the instance
func (instance *MSFT_StorageQoSPolicyToFlow) SetPropertyFlow(value MSFT_StorageQoSFlow) (err error) {
	return instance.SetProperty("Flow", (value))
}

// GetFlow gets the value of Flow for the instance
func (instance *MSFT_StorageQoSPolicyToFlow) GetPropertyFlow() (value MSFT_StorageQoSFlow, err error) {
	retValue, err := instance.GetProperty("Flow")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_StorageQoSFlow)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_StorageQoSFlow is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_StorageQoSFlow(valuetmp)

	return
}

// SetPolicy sets the value of Policy for the instance
func (instance *MSFT_StorageQoSPolicyToFlow) SetPropertyPolicy(value MSFT_StorageQoSPolicy) (err error) {
	return instance.SetProperty("Policy", (value))
}

// GetPolicy gets the value of Policy for the instance
func (instance *MSFT_StorageQoSPolicyToFlow) GetPropertyPolicy() (value MSFT_StorageQoSPolicy, err error) {
	retValue, err := instance.GetProperty("Policy")
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
