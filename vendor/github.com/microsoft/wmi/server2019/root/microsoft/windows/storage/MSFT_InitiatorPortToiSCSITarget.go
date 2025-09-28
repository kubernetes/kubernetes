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

// MSFT_InitiatorPortToiSCSITarget struct
type MSFT_InitiatorPortToiSCSITarget struct {
	*cim.WmiInstance

	//
	InitiatorPort MSFT_InitiatorPort

	//
	iSCSITarget MSFT_iSCSITarget
}

func NewMSFT_InitiatorPortToiSCSITargetEx1(instance *cim.WmiInstance) (newInstance *MSFT_InitiatorPortToiSCSITarget, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_InitiatorPortToiSCSITarget{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_InitiatorPortToiSCSITargetEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_InitiatorPortToiSCSITarget, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_InitiatorPortToiSCSITarget{
		WmiInstance: tmp,
	}
	return
}

// SetInitiatorPort sets the value of InitiatorPort for the instance
func (instance *MSFT_InitiatorPortToiSCSITarget) SetPropertyInitiatorPort(value MSFT_InitiatorPort) (err error) {
	return instance.SetProperty("InitiatorPort", (value))
}

// GetInitiatorPort gets the value of InitiatorPort for the instance
func (instance *MSFT_InitiatorPortToiSCSITarget) GetPropertyInitiatorPort() (value MSFT_InitiatorPort, err error) {
	retValue, err := instance.GetProperty("InitiatorPort")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_InitiatorPort)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_InitiatorPort is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_InitiatorPort(valuetmp)

	return
}

// SetiSCSITarget sets the value of iSCSITarget for the instance
func (instance *MSFT_InitiatorPortToiSCSITarget) SetPropertyiSCSITarget(value MSFT_iSCSITarget) (err error) {
	return instance.SetProperty("iSCSITarget", (value))
}

// GetiSCSITarget gets the value of iSCSITarget for the instance
func (instance *MSFT_InitiatorPortToiSCSITarget) GetPropertyiSCSITarget() (value MSFT_iSCSITarget, err error) {
	retValue, err := instance.GetProperty("iSCSITarget")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_iSCSITarget)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_iSCSITarget is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_iSCSITarget(valuetmp)

	return
}
