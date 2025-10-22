// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.Microsoft.Windows.Storage
//////////////////////////////////////////////
package storage

import (
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// MSFT_InitiatorId struct
type MSFT_InitiatorId struct {
	*MSFT_StorageObject

	//
	HostType []uint16

	//
	InitiatorAddress string

	//
	OtherHostTypeDescription []string

	//
	Type uint16
}

func NewMSFT_InitiatorIdEx1(instance *cim.WmiInstance) (newInstance *MSFT_InitiatorId, err error) {
	tmp, err := NewMSFT_StorageObjectEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_InitiatorId{
		MSFT_StorageObject: tmp,
	}
	return
}

func NewMSFT_InitiatorIdEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_InitiatorId, err error) {
	tmp, err := NewMSFT_StorageObjectEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_InitiatorId{
		MSFT_StorageObject: tmp,
	}
	return
}

// SetHostType sets the value of HostType for the instance
func (instance *MSFT_InitiatorId) SetPropertyHostType(value []uint16) (err error) {
	return instance.SetProperty("HostType", (value))
}

// GetHostType gets the value of HostType for the instance
func (instance *MSFT_InitiatorId) GetPropertyHostType() (value []uint16, err error) {
	retValue, err := instance.GetProperty("HostType")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint16)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint16(valuetmp))
	}

	return
}

// SetInitiatorAddress sets the value of InitiatorAddress for the instance
func (instance *MSFT_InitiatorId) SetPropertyInitiatorAddress(value string) (err error) {
	return instance.SetProperty("InitiatorAddress", (value))
}

// GetInitiatorAddress gets the value of InitiatorAddress for the instance
func (instance *MSFT_InitiatorId) GetPropertyInitiatorAddress() (value string, err error) {
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

// SetOtherHostTypeDescription sets the value of OtherHostTypeDescription for the instance
func (instance *MSFT_InitiatorId) SetPropertyOtherHostTypeDescription(value []string) (err error) {
	return instance.SetProperty("OtherHostTypeDescription", (value))
}

// GetOtherHostTypeDescription gets the value of OtherHostTypeDescription for the instance
func (instance *MSFT_InitiatorId) GetPropertyOtherHostTypeDescription() (value []string, err error) {
	retValue, err := instance.GetProperty("OtherHostTypeDescription")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetType sets the value of Type for the instance
func (instance *MSFT_InitiatorId) SetPropertyType(value uint16) (err error) {
	return instance.SetProperty("Type", (value))
}

// GetType gets the value of Type for the instance
func (instance *MSFT_InitiatorId) GetPropertyType() (value uint16, err error) {
	retValue, err := instance.GetProperty("Type")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

//

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_InitiatorId) DeleteObject( /* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("DeleteObject")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}
