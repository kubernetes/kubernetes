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

// MSFT_InitiatorPortToiSCSIConnection struct
type MSFT_InitiatorPortToiSCSIConnection struct {
	*cim.WmiInstance

	//
	InitiatorPort MSFT_InitiatorPort

	//
	iSCSIConnection MSFT_iSCSIConnection
}

func NewMSFT_InitiatorPortToiSCSIConnectionEx1(instance *cim.WmiInstance) (newInstance *MSFT_InitiatorPortToiSCSIConnection, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_InitiatorPortToiSCSIConnection{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_InitiatorPortToiSCSIConnectionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_InitiatorPortToiSCSIConnection, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_InitiatorPortToiSCSIConnection{
		WmiInstance: tmp,
	}
	return
}

// SetInitiatorPort sets the value of InitiatorPort for the instance
func (instance *MSFT_InitiatorPortToiSCSIConnection) SetPropertyInitiatorPort(value MSFT_InitiatorPort) (err error) {
	return instance.SetProperty("InitiatorPort", (value))
}

// GetInitiatorPort gets the value of InitiatorPort for the instance
func (instance *MSFT_InitiatorPortToiSCSIConnection) GetPropertyInitiatorPort() (value MSFT_InitiatorPort, err error) {
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

// SetiSCSIConnection sets the value of iSCSIConnection for the instance
func (instance *MSFT_InitiatorPortToiSCSIConnection) SetPropertyiSCSIConnection(value MSFT_iSCSIConnection) (err error) {
	return instance.SetProperty("iSCSIConnection", (value))
}

// GetiSCSIConnection gets the value of iSCSIConnection for the instance
func (instance *MSFT_InitiatorPortToiSCSIConnection) GetPropertyiSCSIConnection() (value MSFT_iSCSIConnection, err error) {
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
