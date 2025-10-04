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

// Msft_WmiProvider_ComServerLoadOperationFailureEvent struct
type Msft_WmiProvider_ComServerLoadOperationFailureEvent struct {
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
}

func NewMsft_WmiProvider_ComServerLoadOperationFailureEventEx1(instance *cim.WmiInstance) (newInstance *Msft_WmiProvider_ComServerLoadOperationFailureEvent, err error) {
	tmp, err := NewMsft_WmiProvider_OperationEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Msft_WmiProvider_ComServerLoadOperationFailureEvent{
		Msft_WmiProvider_OperationEvent: tmp,
	}
	return
}

func NewMsft_WmiProvider_ComServerLoadOperationFailureEventEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Msft_WmiProvider_ComServerLoadOperationFailureEvent, err error) {
	tmp, err := NewMsft_WmiProvider_OperationEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Msft_WmiProvider_ComServerLoadOperationFailureEvent{
		Msft_WmiProvider_OperationEvent: tmp,
	}
	return
}

// SetClsid sets the value of Clsid for the instance
func (instance *Msft_WmiProvider_ComServerLoadOperationFailureEvent) SetPropertyClsid(value string) (err error) {
	return instance.SetProperty("Clsid", (value))
}

// GetClsid gets the value of Clsid for the instance
func (instance *Msft_WmiProvider_ComServerLoadOperationFailureEvent) GetPropertyClsid() (value string, err error) {
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
func (instance *Msft_WmiProvider_ComServerLoadOperationFailureEvent) SetPropertyInProcServer(value bool) (err error) {
	return instance.SetProperty("InProcServer", (value))
}

// GetInProcServer gets the value of InProcServer for the instance
func (instance *Msft_WmiProvider_ComServerLoadOperationFailureEvent) GetPropertyInProcServer() (value bool, err error) {
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
func (instance *Msft_WmiProvider_ComServerLoadOperationFailureEvent) SetPropertyInProcServerPath(value string) (err error) {
	return instance.SetProperty("InProcServerPath", (value))
}

// GetInProcServerPath gets the value of InProcServerPath for the instance
func (instance *Msft_WmiProvider_ComServerLoadOperationFailureEvent) GetPropertyInProcServerPath() (value string, err error) {
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
func (instance *Msft_WmiProvider_ComServerLoadOperationFailureEvent) SetPropertyLocalServer(value bool) (err error) {
	return instance.SetProperty("LocalServer", (value))
}

// GetLocalServer gets the value of LocalServer for the instance
func (instance *Msft_WmiProvider_ComServerLoadOperationFailureEvent) GetPropertyLocalServer() (value bool, err error) {
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
func (instance *Msft_WmiProvider_ComServerLoadOperationFailureEvent) SetPropertyLocalServerPath(value string) (err error) {
	return instance.SetProperty("LocalServerPath", (value))
}

// GetLocalServerPath gets the value of LocalServerPath for the instance
func (instance *Msft_WmiProvider_ComServerLoadOperationFailureEvent) GetPropertyLocalServerPath() (value string, err error) {
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
func (instance *Msft_WmiProvider_ComServerLoadOperationFailureEvent) SetPropertyResultCode(value uint32) (err error) {
	return instance.SetProperty("ResultCode", (value))
}

// GetResultCode gets the value of ResultCode for the instance
func (instance *Msft_WmiProvider_ComServerLoadOperationFailureEvent) GetPropertyResultCode() (value uint32, err error) {
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
func (instance *Msft_WmiProvider_ComServerLoadOperationFailureEvent) SetPropertyServerName(value string) (err error) {
	return instance.SetProperty("ServerName", (value))
}

// GetServerName gets the value of ServerName for the instance
func (instance *Msft_WmiProvider_ComServerLoadOperationFailureEvent) GetPropertyServerName() (value string, err error) {
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
