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

// MSFT_iSCSISessionToiSCSITargetPortal struct
type MSFT_iSCSISessionToiSCSITargetPortal struct {
	*cim.WmiInstance

	//
	iSCSISession MSFT_iSCSISession

	//
	iSCSITargetPortal MSFT_iSCSITargetPortal
}

func NewMSFT_iSCSISessionToiSCSITargetPortalEx1(instance *cim.WmiInstance) (newInstance *MSFT_iSCSISessionToiSCSITargetPortal, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_iSCSISessionToiSCSITargetPortal{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_iSCSISessionToiSCSITargetPortalEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_iSCSISessionToiSCSITargetPortal, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_iSCSISessionToiSCSITargetPortal{
		WmiInstance: tmp,
	}
	return
}

// SetiSCSISession sets the value of iSCSISession for the instance
func (instance *MSFT_iSCSISessionToiSCSITargetPortal) SetPropertyiSCSISession(value MSFT_iSCSISession) (err error) {
	return instance.SetProperty("iSCSISession", (value))
}

// GetiSCSISession gets the value of iSCSISession for the instance
func (instance *MSFT_iSCSISessionToiSCSITargetPortal) GetPropertyiSCSISession() (value MSFT_iSCSISession, err error) {
	retValue, err := instance.GetProperty("iSCSISession")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_iSCSISession)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_iSCSISession is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_iSCSISession(valuetmp)

	return
}

// SetiSCSITargetPortal sets the value of iSCSITargetPortal for the instance
func (instance *MSFT_iSCSISessionToiSCSITargetPortal) SetPropertyiSCSITargetPortal(value MSFT_iSCSITargetPortal) (err error) {
	return instance.SetProperty("iSCSITargetPortal", (value))
}

// GetiSCSITargetPortal gets the value of iSCSITargetPortal for the instance
func (instance *MSFT_iSCSISessionToiSCSITargetPortal) GetPropertyiSCSITargetPortal() (value MSFT_iSCSITargetPortal, err error) {
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
