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

// MSFT_StorageHealthStatusChangeEvent struct
type MSFT_StorageHealthStatusChangeEvent struct {
	*MSFT_StorageEvent

	//
	CurrentHealthStatus uint16

	//
	PreviousHealthStatus uint16

	//
	SourceUniqueId string

	//
	StorageSubsystemUniqueId string
}

func NewMSFT_StorageHealthStatusChangeEventEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageHealthStatusChangeEvent, err error) {
	tmp, err := NewMSFT_StorageEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageHealthStatusChangeEvent{
		MSFT_StorageEvent: tmp,
	}
	return
}

func NewMSFT_StorageHealthStatusChangeEventEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageHealthStatusChangeEvent, err error) {
	tmp, err := NewMSFT_StorageEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageHealthStatusChangeEvent{
		MSFT_StorageEvent: tmp,
	}
	return
}

// SetCurrentHealthStatus sets the value of CurrentHealthStatus for the instance
func (instance *MSFT_StorageHealthStatusChangeEvent) SetPropertyCurrentHealthStatus(value uint16) (err error) {
	return instance.SetProperty("CurrentHealthStatus", (value))
}

// GetCurrentHealthStatus gets the value of CurrentHealthStatus for the instance
func (instance *MSFT_StorageHealthStatusChangeEvent) GetPropertyCurrentHealthStatus() (value uint16, err error) {
	retValue, err := instance.GetProperty("CurrentHealthStatus")
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

// SetPreviousHealthStatus sets the value of PreviousHealthStatus for the instance
func (instance *MSFT_StorageHealthStatusChangeEvent) SetPropertyPreviousHealthStatus(value uint16) (err error) {
	return instance.SetProperty("PreviousHealthStatus", (value))
}

// GetPreviousHealthStatus gets the value of PreviousHealthStatus for the instance
func (instance *MSFT_StorageHealthStatusChangeEvent) GetPropertyPreviousHealthStatus() (value uint16, err error) {
	retValue, err := instance.GetProperty("PreviousHealthStatus")
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

// SetSourceUniqueId sets the value of SourceUniqueId for the instance
func (instance *MSFT_StorageHealthStatusChangeEvent) SetPropertySourceUniqueId(value string) (err error) {
	return instance.SetProperty("SourceUniqueId", (value))
}

// GetSourceUniqueId gets the value of SourceUniqueId for the instance
func (instance *MSFT_StorageHealthStatusChangeEvent) GetPropertySourceUniqueId() (value string, err error) {
	retValue, err := instance.GetProperty("SourceUniqueId")
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

// SetStorageSubsystemUniqueId sets the value of StorageSubsystemUniqueId for the instance
func (instance *MSFT_StorageHealthStatusChangeEvent) SetPropertyStorageSubsystemUniqueId(value string) (err error) {
	return instance.SetProperty("StorageSubsystemUniqueId", (value))
}

// GetStorageSubsystemUniqueId gets the value of StorageSubsystemUniqueId for the instance
func (instance *MSFT_StorageHealthStatusChangeEvent) GetPropertyStorageSubsystemUniqueId() (value string, err error) {
	retValue, err := instance.GetProperty("StorageSubsystemUniqueId")
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
