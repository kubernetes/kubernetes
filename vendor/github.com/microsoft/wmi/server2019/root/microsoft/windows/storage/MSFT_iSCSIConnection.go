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

// MSFT_iSCSIConnection struct
type MSFT_iSCSIConnection struct {
	*cim.WmiInstance

	//
	ConnectionIdentifier string

	//
	InitiatorAddress string

	//
	InitiatorPortNumber uint32

	//
	TargetAddress string

	//
	TargetPortNumber uint32
}

func NewMSFT_iSCSIConnectionEx1(instance *cim.WmiInstance) (newInstance *MSFT_iSCSIConnection, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_iSCSIConnection{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_iSCSIConnectionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_iSCSIConnection, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_iSCSIConnection{
		WmiInstance: tmp,
	}
	return
}

// SetConnectionIdentifier sets the value of ConnectionIdentifier for the instance
func (instance *MSFT_iSCSIConnection) SetPropertyConnectionIdentifier(value string) (err error) {
	return instance.SetProperty("ConnectionIdentifier", (value))
}

// GetConnectionIdentifier gets the value of ConnectionIdentifier for the instance
func (instance *MSFT_iSCSIConnection) GetPropertyConnectionIdentifier() (value string, err error) {
	retValue, err := instance.GetProperty("ConnectionIdentifier")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetInitiatorAddress sets the value of InitiatorAddress for the instance
func (instance *MSFT_iSCSIConnection) SetPropertyInitiatorAddress(value string) (err error) {
	return instance.SetProperty("InitiatorAddress", (value))
}

// GetInitiatorAddress gets the value of InitiatorAddress for the instance
func (instance *MSFT_iSCSIConnection) GetPropertyInitiatorAddress() (value string, err error) {
	retValue, err := instance.GetProperty("InitiatorAddress")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetInitiatorPortNumber sets the value of InitiatorPortNumber for the instance
func (instance *MSFT_iSCSIConnection) SetPropertyInitiatorPortNumber(value uint32) (err error) {
	return instance.SetProperty("InitiatorPortNumber", (value))
}

// GetInitiatorPortNumber gets the value of InitiatorPortNumber for the instance
func (instance *MSFT_iSCSIConnection) GetPropertyInitiatorPortNumber() (value uint32, err error) {
	retValue, err := instance.GetProperty("InitiatorPortNumber")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}

// SetTargetAddress sets the value of TargetAddress for the instance
func (instance *MSFT_iSCSIConnection) SetPropertyTargetAddress(value string) (err error) {
	return instance.SetProperty("TargetAddress", (value))
}

// GetTargetAddress gets the value of TargetAddress for the instance
func (instance *MSFT_iSCSIConnection) GetPropertyTargetAddress() (value string, err error) {
	retValue, err := instance.GetProperty("TargetAddress")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetTargetPortNumber sets the value of TargetPortNumber for the instance
func (instance *MSFT_iSCSIConnection) SetPropertyTargetPortNumber(value uint32) (err error) {
	return instance.SetProperty("TargetPortNumber", (value))
}

// GetTargetPortNumber gets the value of TargetPortNumber for the instance
func (instance *MSFT_iSCSIConnection) GetPropertyTargetPortNumber() (value uint32, err error) {
	retValue, err := instance.GetProperty("TargetPortNumber")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}
