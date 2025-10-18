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

// MSFT_iSCSITargetToiSCSIConnection struct
type MSFT_iSCSITargetToiSCSIConnection struct {
	*cim.WmiInstance

	//
	iSCSIConnection MSFT_iSCSIConnection

	//
	iSCSITarget MSFT_iSCSITarget
}

func NewMSFT_iSCSITargetToiSCSIConnectionEx1(instance *cim.WmiInstance) (newInstance *MSFT_iSCSITargetToiSCSIConnection, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_iSCSITargetToiSCSIConnection{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_iSCSITargetToiSCSIConnectionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_iSCSITargetToiSCSIConnection, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_iSCSITargetToiSCSIConnection{
		WmiInstance: tmp,
	}
	return
}

// SetiSCSIConnection sets the value of iSCSIConnection for the instance
func (instance *MSFT_iSCSITargetToiSCSIConnection) SetPropertyiSCSIConnection(value MSFT_iSCSIConnection) (err error) {
	return instance.SetProperty("iSCSIConnection", (value))
}

// GetiSCSIConnection gets the value of iSCSIConnection for the instance
func (instance *MSFT_iSCSITargetToiSCSIConnection) GetPropertyiSCSIConnection() (value MSFT_iSCSIConnection, err error) {
	retValue, err := instance.GetProperty("iSCSIConnection")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_iSCSIConnection)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_iSCSIConnection is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_iSCSIConnection(valuetmp)

	return
}

// SetiSCSITarget sets the value of iSCSITarget for the instance
func (instance *MSFT_iSCSITargetToiSCSIConnection) SetPropertyiSCSITarget(value MSFT_iSCSITarget) (err error) {
	return instance.SetProperty("iSCSITarget", (value))
}

// GetiSCSITarget gets the value of iSCSITarget for the instance
func (instance *MSFT_iSCSITargetToiSCSIConnection) GetPropertyiSCSITarget() (value MSFT_iSCSITarget, err error) {
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
