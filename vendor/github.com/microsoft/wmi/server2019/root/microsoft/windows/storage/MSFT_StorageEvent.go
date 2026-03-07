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

// MSFT_StorageEvent struct
type MSFT_StorageEvent struct {
	*cim.WmiInstance

	//
	Description string

	//
	EventTime string

	//
	PerceivedSeverity uint16

	//
	SourceClassName string

	//
	SourceInstance MSFT_StorageObject

	//
	SourceNamespace string

	//
	SourceObjectId string

	//
	SourceServer string

	//
	StorageSubsystemObjectId string
}

func NewMSFT_StorageEventEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageEvent, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageEvent{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StorageEventEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageEvent, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageEvent{
		WmiInstance: tmp,
	}
	return
}

// SetDescription sets the value of Description for the instance
func (instance *MSFT_StorageEvent) SetPropertyDescription(value string) (err error) {
	return instance.SetProperty("Description", (value))
}

// GetDescription gets the value of Description for the instance
func (instance *MSFT_StorageEvent) GetPropertyDescription() (value string, err error) {
	retValue, err := instance.GetProperty("Description")
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

// SetEventTime sets the value of EventTime for the instance
func (instance *MSFT_StorageEvent) SetPropertyEventTime(value string) (err error) {
	return instance.SetProperty("EventTime", (value))
}

// GetEventTime gets the value of EventTime for the instance
func (instance *MSFT_StorageEvent) GetPropertyEventTime() (value string, err error) {
	retValue, err := instance.GetProperty("EventTime")
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

// SetPerceivedSeverity sets the value of PerceivedSeverity for the instance
func (instance *MSFT_StorageEvent) SetPropertyPerceivedSeverity(value uint16) (err error) {
	return instance.SetProperty("PerceivedSeverity", (value))
}

// GetPerceivedSeverity gets the value of PerceivedSeverity for the instance
func (instance *MSFT_StorageEvent) GetPropertyPerceivedSeverity() (value uint16, err error) {
	retValue, err := instance.GetProperty("PerceivedSeverity")
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

// SetSourceClassName sets the value of SourceClassName for the instance
func (instance *MSFT_StorageEvent) SetPropertySourceClassName(value string) (err error) {
	return instance.SetProperty("SourceClassName", (value))
}

// GetSourceClassName gets the value of SourceClassName for the instance
func (instance *MSFT_StorageEvent) GetPropertySourceClassName() (value string, err error) {
	retValue, err := instance.GetProperty("SourceClassName")
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

// SetSourceInstance sets the value of SourceInstance for the instance
func (instance *MSFT_StorageEvent) SetPropertySourceInstance(value MSFT_StorageObject) (err error) {
	return instance.SetProperty("SourceInstance", (value))
}

// GetSourceInstance gets the value of SourceInstance for the instance
func (instance *MSFT_StorageEvent) GetPropertySourceInstance() (value MSFT_StorageObject, err error) {
	retValue, err := instance.GetProperty("SourceInstance")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(MSFT_StorageObject)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " MSFT_StorageObject is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = MSFT_StorageObject(valuetmp)

	return
}

// SetSourceNamespace sets the value of SourceNamespace for the instance
func (instance *MSFT_StorageEvent) SetPropertySourceNamespace(value string) (err error) {
	return instance.SetProperty("SourceNamespace", (value))
}

// GetSourceNamespace gets the value of SourceNamespace for the instance
func (instance *MSFT_StorageEvent) GetPropertySourceNamespace() (value string, err error) {
	retValue, err := instance.GetProperty("SourceNamespace")
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

// SetSourceObjectId sets the value of SourceObjectId for the instance
func (instance *MSFT_StorageEvent) SetPropertySourceObjectId(value string) (err error) {
	return instance.SetProperty("SourceObjectId", (value))
}

// GetSourceObjectId gets the value of SourceObjectId for the instance
func (instance *MSFT_StorageEvent) GetPropertySourceObjectId() (value string, err error) {
	retValue, err := instance.GetProperty("SourceObjectId")
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

// SetSourceServer sets the value of SourceServer for the instance
func (instance *MSFT_StorageEvent) SetPropertySourceServer(value string) (err error) {
	return instance.SetProperty("SourceServer", (value))
}

// GetSourceServer gets the value of SourceServer for the instance
func (instance *MSFT_StorageEvent) GetPropertySourceServer() (value string, err error) {
	retValue, err := instance.GetProperty("SourceServer")
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

// SetStorageSubsystemObjectId sets the value of StorageSubsystemObjectId for the instance
func (instance *MSFT_StorageEvent) SetPropertyStorageSubsystemObjectId(value string) (err error) {
	return instance.SetProperty("StorageSubsystemObjectId", (value))
}

// GetStorageSubsystemObjectId gets the value of StorageSubsystemObjectId for the instance
func (instance *MSFT_StorageEvent) GetPropertyStorageSubsystemObjectId() (value string, err error) {
	retValue, err := instance.GetProperty("StorageSubsystemObjectId")
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
