// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.CIMV2
//////////////////////////////////////////////
package cimv2

import (
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// Msft_WmiProvider_LoadOperationFailureEvent struct
type Msft_WmiProvider_LoadOperationFailureEvent struct {
	*Msft_WmiProvider_OperationEvent

	//
	Clsid string

	//
	InProcServer bool

	//
	InProcServerPath string

	//
	LocalServer bool

	//
	LocalServerPath string

	//
	ResultCode uint32

	//
	ServerName string

	//
	Synchronisation uint32

	//
	ThreadingModel uint32
}

func NewMsft_WmiProvider_LoadOperationFailureEventEx1(instance *cim.WmiInstance) (newInstance *Msft_WmiProvider_LoadOperationFailureEvent, err error) {
	tmp, err := NewMsft_WmiProvider_OperationEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Msft_WmiProvider_LoadOperationFailureEvent{
		Msft_WmiProvider_OperationEvent: tmp,
	}
	return
}

func NewMsft_WmiProvider_LoadOperationFailureEventEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Msft_WmiProvider_LoadOperationFailureEvent, err error) {
	tmp, err := NewMsft_WmiProvider_OperationEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Msft_WmiProvider_LoadOperationFailureEvent{
		Msft_WmiProvider_OperationEvent: tmp,
	}
	return
}

// SetClsid sets the value of Clsid for the instance
func (instance *Msft_WmiProvider_LoadOperationFailureEvent) SetPropertyClsid(value string) (err error) {
	return instance.SetProperty("Clsid", (value))
}

// GetClsid gets the value of Clsid for the instance
func (instance *Msft_WmiProvider_LoadOperationFailureEvent) GetPropertyClsid() (value string, err error) {
	retValue, err := instance.GetProperty("Clsid")
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

// SetInProcServer sets the value of InProcServer for the instance
func (instance *Msft_WmiProvider_LoadOperationFailureEvent) SetPropertyInProcServer(value bool) (err error) {
	return instance.SetProperty("InProcServer", (value))
}

// GetInProcServer gets the value of InProcServer for the instance
func (instance *Msft_WmiProvider_LoadOperationFailureEvent) GetPropertyInProcServer() (value bool, err error) {
	retValue, err := instance.GetProperty("InProcServer")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetInProcServerPath sets the value of InProcServerPath for the instance
func (instance *Msft_WmiProvider_LoadOperationFailureEvent) SetPropertyInProcServerPath(value string) (err error) {
	return instance.SetProperty("InProcServerPath", (value))
}

// GetInProcServerPath gets the value of InProcServerPath for the instance
func (instance *Msft_WmiProvider_LoadOperationFailureEvent) GetPropertyInProcServerPath() (value string, err error) {
	retValue, err := instance.GetProperty("InProcServerPath")
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

// SetLocalServer sets the value of LocalServer for the instance
func (instance *Msft_WmiProvider_LoadOperationFailureEvent) SetPropertyLocalServer(value bool) (err error) {
	return instance.SetProperty("LocalServer", (value))
}

// GetLocalServer gets the value of LocalServer for the instance
func (instance *Msft_WmiProvider_LoadOperationFailureEvent) GetPropertyLocalServer() (value bool, err error) {
	retValue, err := instance.GetProperty("LocalServer")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetLocalServerPath sets the value of LocalServerPath for the instance
func (instance *Msft_WmiProvider_LoadOperationFailureEvent) SetPropertyLocalServerPath(value string) (err error) {
	return instance.SetProperty("LocalServerPath", (value))
}

// GetLocalServerPath gets the value of LocalServerPath for the instance
func (instance *Msft_WmiProvider_LoadOperationFailureEvent) GetPropertyLocalServerPath() (value string, err error) {
	retValue, err := instance.GetProperty("LocalServerPath")
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

// SetResultCode sets the value of ResultCode for the instance
func (instance *Msft_WmiProvider_LoadOperationFailureEvent) SetPropertyResultCode(value uint32) (err error) {
	return instance.SetProperty("ResultCode", (value))
}

// GetResultCode gets the value of ResultCode for the instance
func (instance *Msft_WmiProvider_LoadOperationFailureEvent) GetPropertyResultCode() (value uint32, err error) {
	retValue, err := instance.GetProperty("ResultCode")
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

// SetServerName sets the value of ServerName for the instance
func (instance *Msft_WmiProvider_LoadOperationFailureEvent) SetPropertyServerName(value string) (err error) {
	return instance.SetProperty("ServerName", (value))
}

// GetServerName gets the value of ServerName for the instance
func (instance *Msft_WmiProvider_LoadOperationFailureEvent) GetPropertyServerName() (value string, err error) {
	retValue, err := instance.GetProperty("ServerName")
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

// SetSynchronisation sets the value of Synchronisation for the instance
func (instance *Msft_WmiProvider_LoadOperationFailureEvent) SetPropertySynchronisation(value uint32) (err error) {
	return instance.SetProperty("Synchronisation", (value))
}

// GetSynchronisation gets the value of Synchronisation for the instance
func (instance *Msft_WmiProvider_LoadOperationFailureEvent) GetPropertySynchronisation() (value uint32, err error) {
	retValue, err := instance.GetProperty("Synchronisation")
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

// SetThreadingModel sets the value of ThreadingModel for the instance
func (instance *Msft_WmiProvider_LoadOperationFailureEvent) SetPropertyThreadingModel(value uint32) (err error) {
	return instance.SetProperty("ThreadingModel", (value))
}

// GetThreadingModel gets the value of ThreadingModel for the instance
func (instance *Msft_WmiProvider_LoadOperationFailureEvent) GetPropertyThreadingModel() (value uint32, err error) {
	retValue, err := instance.GetProperty("ThreadingModel")
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
