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

// MSFT_iSCSIConnectionToiSCSITargetPortal struct
type MSFT_iSCSIConnectionToiSCSITargetPortal struct {
	*cim.WmiInstance

	//
	iSCSIConnection MSFT_iSCSIConnection

	//
	iSCSITargetPortal MSFT_iSCSITargetPortal
}

func NewMSFT_iSCSIConnectionToiSCSITargetPortalEx1(instance *cim.WmiInstance) (newInstance *MSFT_iSCSIConnectionToiSCSITargetPortal, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_iSCSIConnectionToiSCSITargetPortal{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_iSCSIConnectionToiSCSITargetPortalEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_iSCSIConnectionToiSCSITargetPortal, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_iSCSIConnectionToiSCSITargetPortal{
		WmiInstance: tmp,
	}
	return
}

// SetiSCSIConnection sets the value of iSCSIConnection for the instance
func (instance *MSFT_iSCSIConnectionToiSCSITargetPortal) SetPropertyiSCSIConnection(value MSFT_iSCSIConnection) (err error) {
	return instance.SetProperty("iSCSIConnection", (value))
}

// GetiSCSIConnection gets the value of iSCSIConnection for the instance
func (instance *MSFT_iSCSIConnectionToiSCSITargetPortal) GetPropertyiSCSIConnection() (value MSFT_iSCSIConnection, err error) {
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

// SetiSCSITargetPortal sets the value of iSCSITargetPortal for the instance
func (instance *MSFT_iSCSIConnectionToiSCSITargetPortal) SetPropertyiSCSITargetPortal(value MSFT_iSCSITargetPortal) (err error) {
	return instance.SetProperty("iSCSITargetPortal", (value))
}

// GetiSCSITargetPortal gets the value of iSCSITargetPortal for the instance
func (instance *MSFT_iSCSIConnectionToiSCSITargetPortal) GetPropertyiSCSITargetPortal() (value MSFT_iSCSITargetPortal, err error) {
	retValue, err := instance.GetProperty("iSCSITargetPortal")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_iSCSITargetPortal)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_iSCSITargetPortal is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_iSCSITargetPortal(valuetmp)

	return
}
