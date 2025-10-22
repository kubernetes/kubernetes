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

// MSFT_iSCSITargetToiSCSISession struct
type MSFT_iSCSITargetToiSCSISession struct {
	*cim.WmiInstance

	//
	iSCSISession MSFT_iSCSISession

	//
	iSCSITarget MSFT_iSCSITarget
}

func NewMSFT_iSCSITargetToiSCSISessionEx1(instance *cim.WmiInstance) (newInstance *MSFT_iSCSITargetToiSCSISession, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_iSCSITargetToiSCSISession{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_iSCSITargetToiSCSISessionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_iSCSITargetToiSCSISession, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_iSCSITargetToiSCSISession{
		WmiInstance: tmp,
	}
	return
}

// SetiSCSISession sets the value of iSCSISession for the instance
func (instance *MSFT_iSCSITargetToiSCSISession) SetPropertyiSCSISession(value MSFT_iSCSISession) (err error) {
	return instance.SetProperty("iSCSISession", (value))
}

// GetiSCSISession gets the value of iSCSISession for the instance
func (instance *MSFT_iSCSITargetToiSCSISession) GetPropertyiSCSISession() (value MSFT_iSCSISession, err error) {
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

// SetiSCSITarget sets the value of iSCSITarget for the instance
func (instance *MSFT_iSCSITargetToiSCSISession) SetPropertyiSCSITarget(value MSFT_iSCSITarget) (err error) {
	return instance.SetProperty("iSCSITarget", (value))
}

// GetiSCSITarget gets the value of iSCSITarget for the instance
func (instance *MSFT_iSCSITargetToiSCSISession) GetPropertyiSCSITarget() (value MSFT_iSCSITarget, err error) {
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
