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

// MSFT_iSCSISessionToDisk struct
type MSFT_iSCSISessionToDisk struct {
	*cim.WmiInstance

	//
	Disk MSFT_Disk

	//
	iSCSISession MSFT_iSCSISession
}

func NewMSFT_iSCSISessionToDiskEx1(instance *cim.WmiInstance) (newInstance *MSFT_iSCSISessionToDisk, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_iSCSISessionToDisk{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_iSCSISessionToDiskEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_iSCSISessionToDisk, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_iSCSISessionToDisk{
		WmiInstance: tmp,
	}
	return
}

// SetDisk sets the value of Disk for the instance
func (instance *MSFT_iSCSISessionToDisk) SetPropertyDisk(value MSFT_Disk) (err error) {
	return instance.SetProperty("Disk", (value))
}

// GetDisk gets the value of Disk for the instance
func (instance *MSFT_iSCSISessionToDisk) GetPropertyDisk() (value MSFT_Disk, err error) {
	retValue, err := instance.GetProperty("Disk")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_Disk)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_Disk is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_Disk(valuetmp)

	return
}

// SetiSCSISession sets the value of iSCSISession for the instance
func (instance *MSFT_iSCSISessionToDisk) SetPropertyiSCSISession(value MSFT_iSCSISession) (err error) {
	return instance.SetProperty("iSCSISession", (value))
}

// GetiSCSISession gets the value of iSCSISession for the instance
func (instance *MSFT_iSCSISessionToDisk) GetPropertyiSCSISession() (value MSFT_iSCSISession, err error) {
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
