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

// MSFT_iSCSISessionToiSCSIConnection struct
type MSFT_iSCSISessionToiSCSIConnection struct {
	*cim.WmiInstance

	//
	iSCSIConnection MSFT_iSCSIConnection

	//
	iSCSISession MSFT_iSCSISession
}

func NewMSFT_iSCSISessionToiSCSIConnectionEx1(instance *cim.WmiInstance) (newInstance *MSFT_iSCSISessionToiSCSIConnection, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_iSCSISessionToiSCSIConnection{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_iSCSISessionToiSCSIConnectionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_iSCSISessionToiSCSIConnection, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_iSCSISessionToiSCSIConnection{
		WmiInstance: tmp,
	}
	return
}

// SetiSCSIConnection sets the value of iSCSIConnection for the instance
func (instance *MSFT_iSCSISessionToiSCSIConnection) SetPropertyiSCSIConnection(value MSFT_iSCSIConnection) (err error) {
	return instance.SetProperty("iSCSIConnection", (value))
}

// GetiSCSIConnection gets the value of iSCSIConnection for the instance
func (instance *MSFT_iSCSISessionToiSCSIConnection) GetPropertyiSCSIConnection() (value MSFT_iSCSIConnection, err error) {
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

// SetiSCSISession sets the value of iSCSISession for the instance
func (instance *MSFT_iSCSISessionToiSCSIConnection) SetPropertyiSCSISession(value MSFT_iSCSISession) (err error) {
	return instance.SetProperty("iSCSISession", (value))
}

// GetiSCSISession gets the value of iSCSISession for the instance
func (instance *MSFT_iSCSISessionToiSCSIConnection) GetPropertyiSCSISession() (value MSFT_iSCSISession, err error) {
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
