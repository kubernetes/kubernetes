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

// Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000 struct
type Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000 struct {
	*Win32_PerfRawData

	//
	ConnectionsAcceptedovernetpipe uint32

	//
	ConnectionsAcceptedovernettcp uint32

	//
	ConnectionsDispatchedovernetpipe uint32

	//
	ConnectionsDispatchedovernettcp uint32

	//
	DispatchFailuresovernetpipe uint32

	//
	DispatchFailuresovernettcp uint32

	//
	ProtocolFailuresovernetpipe uint32

	//
	ProtocolFailuresovernettcp uint32

	//
	RegistrationsActivefornetpipe uint32

	//
	RegistrationsActivefornettcp uint32

	//
	UrisRegisteredfornetpipe uint32

	//
	UrisRegisteredfornettcp uint32

	//
	UrisUnregisteredfornetpipe uint32

	//
	UrisUnregisteredfornettcp uint32
}

func NewWin32_PerfRawData_SMSvcHost4000_SMSvcHost4000Ex1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_SMSvcHost4000_SMSvcHost4000Ex6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetConnectionsAcceptedovernetpipe sets the value of ConnectionsAcceptedovernetpipe for the instance
func (instance *Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000) SetPropertyConnectionsAcceptedovernetpipe(value uint32) (err error) {
	return instance.SetProperty("ConnectionsAcceptedovernetpipe", (value))
}

// GetConnectionsAcceptedovernetpipe gets the value of ConnectionsAcceptedovernetpipe for the instance
func (instance *Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000) GetPropertyConnectionsAcceptedovernetpipe() (value uint32, err error) {
	retValue, err := instance.GetProperty("ConnectionsAcceptedovernetpipe")
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

// SetConnectionsAcceptedovernettcp sets the value of ConnectionsAcceptedovernettcp for the instance
func (instance *Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000) SetPropertyConnectionsAcceptedovernettcp(value uint32) (err error) {
	return instance.SetProperty("ConnectionsAcceptedovernettcp", (value))
}

// GetConnectionsAcceptedovernettcp gets the value of ConnectionsAcceptedovernettcp for the instance
func (instance *Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000) GetPropertyConnectionsAcceptedovernettcp() (value uint32, err error) {
	retValue, err := instance.GetProperty("ConnectionsAcceptedovernettcp")
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

// SetConnectionsDispatchedovernetpipe sets the value of ConnectionsDispatchedovernetpipe for the instance
func (instance *Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000) SetPropertyConnectionsDispatchedovernetpipe(value uint32) (err error) {
	return instance.SetProperty("ConnectionsDispatchedovernetpipe", (value))
}

// GetConnectionsDispatchedovernetpipe gets the value of ConnectionsDispatchedovernetpipe for the instance
func (instance *Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000) GetPropertyConnectionsDispatchedovernetpipe() (value uint32, err error) {
	retValue, err := instance.GetProperty("ConnectionsDispatchedovernetpipe")
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

// SetConnectionsDispatchedovernettcp sets the value of ConnectionsDispatchedovernettcp for the instance
func (instance *Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000) SetPropertyConnectionsDispatchedovernettcp(value uint32) (err error) {
	return instance.SetProperty("ConnectionsDispatchedovernettcp", (value))
}

// GetConnectionsDispatchedovernettcp gets the value of ConnectionsDispatchedovernettcp for the instance
func (instance *Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000) GetPropertyConnectionsDispatchedovernettcp() (value uint32, err error) {
	retValue, err := instance.GetProperty("ConnectionsDispatchedovernettcp")
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

// SetDispatchFailuresovernetpipe sets the value of DispatchFailuresovernetpipe for the instance
func (instance *Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000) SetPropertyDispatchFailuresovernetpipe(value uint32) (err error) {
	return instance.SetProperty("DispatchFailuresovernetpipe", (value))
}

// GetDispatchFailuresovernetpipe gets the value of DispatchFailuresovernetpipe for the instance
func (instance *Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000) GetPropertyDispatchFailuresovernetpipe() (value uint32, err error) {
	retValue, err := instance.GetProperty("DispatchFailuresovernetpipe")
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

// SetDispatchFailuresovernettcp sets the value of DispatchFailuresovernettcp for the instance
func (instance *Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000) SetPropertyDispatchFailuresovernettcp(value uint32) (err error) {
	return instance.SetProperty("DispatchFailuresovernettcp", (value))
}

// GetDispatchFailuresovernettcp gets the value of DispatchFailuresovernettcp for the instance
func (instance *Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000) GetPropertyDispatchFailuresovernettcp() (value uint32, err error) {
	retValue, err := instance.GetProperty("DispatchFailuresovernettcp")
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

// SetProtocolFailuresovernetpipe sets the value of ProtocolFailuresovernetpipe for the instance
func (instance *Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000) SetPropertyProtocolFailuresovernetpipe(value uint32) (err error) {
	return instance.SetProperty("ProtocolFailuresovernetpipe", (value))
}

// GetProtocolFailuresovernetpipe gets the value of ProtocolFailuresovernetpipe for the instance
func (instance *Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000) GetPropertyProtocolFailuresovernetpipe() (value uint32, err error) {
	retValue, err := instance.GetProperty("ProtocolFailuresovernetpipe")
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

// SetProtocolFailuresovernettcp sets the value of ProtocolFailuresovernettcp for the instance
func (instance *Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000) SetPropertyProtocolFailuresovernettcp(value uint32) (err error) {
	return instance.SetProperty("ProtocolFailuresovernettcp", (value))
}

// GetProtocolFailuresovernettcp gets the value of ProtocolFailuresovernettcp for the instance
func (instance *Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000) GetPropertyProtocolFailuresovernettcp() (value uint32, err error) {
	retValue, err := instance.GetProperty("ProtocolFailuresovernettcp")
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

// SetRegistrationsActivefornetpipe sets the value of RegistrationsActivefornetpipe for the instance
func (instance *Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000) SetPropertyRegistrationsActivefornetpipe(value uint32) (err error) {
	return instance.SetProperty("RegistrationsActivefornetpipe", (value))
}

// GetRegistrationsActivefornetpipe gets the value of RegistrationsActivefornetpipe for the instance
func (instance *Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000) GetPropertyRegistrationsActivefornetpipe() (value uint32, err error) {
	retValue, err := instance.GetProperty("RegistrationsActivefornetpipe")
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

// SetRegistrationsActivefornettcp sets the value of RegistrationsActivefornettcp for the instance
func (instance *Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000) SetPropertyRegistrationsActivefornettcp(value uint32) (err error) {
	return instance.SetProperty("RegistrationsActivefornettcp", (value))
}

// GetRegistrationsActivefornettcp gets the value of RegistrationsActivefornettcp for the instance
func (instance *Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000) GetPropertyRegistrationsActivefornettcp() (value uint32, err error) {
	retValue, err := instance.GetProperty("RegistrationsActivefornettcp")
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

// SetUrisRegisteredfornetpipe sets the value of UrisRegisteredfornetpipe for the instance
func (instance *Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000) SetPropertyUrisRegisteredfornetpipe(value uint32) (err error) {
	return instance.SetProperty("UrisRegisteredfornetpipe", (value))
}

// GetUrisRegisteredfornetpipe gets the value of UrisRegisteredfornetpipe for the instance
func (instance *Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000) GetPropertyUrisRegisteredfornetpipe() (value uint32, err error) {
	retValue, err := instance.GetProperty("UrisRegisteredfornetpipe")
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

// SetUrisRegisteredfornettcp sets the value of UrisRegisteredfornettcp for the instance
func (instance *Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000) SetPropertyUrisRegisteredfornettcp(value uint32) (err error) {
	return instance.SetProperty("UrisRegisteredfornettcp", (value))
}

// GetUrisRegisteredfornettcp gets the value of UrisRegisteredfornettcp for the instance
func (instance *Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000) GetPropertyUrisRegisteredfornettcp() (value uint32, err error) {
	retValue, err := instance.GetProperty("UrisRegisteredfornettcp")
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

// SetUrisUnregisteredfornetpipe sets the value of UrisUnregisteredfornetpipe for the instance
func (instance *Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000) SetPropertyUrisUnregisteredfornetpipe(value uint32) (err error) {
	return instance.SetProperty("UrisUnregisteredfornetpipe", (value))
}

// GetUrisUnregisteredfornetpipe gets the value of UrisUnregisteredfornetpipe for the instance
func (instance *Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000) GetPropertyUrisUnregisteredfornetpipe() (value uint32, err error) {
	retValue, err := instance.GetProperty("UrisUnregisteredfornetpipe")
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

// SetUrisUnregisteredfornettcp sets the value of UrisUnregisteredfornettcp for the instance
func (instance *Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000) SetPropertyUrisUnregisteredfornettcp(value uint32) (err error) {
	return instance.SetProperty("UrisUnregisteredfornettcp", (value))
}

// GetUrisUnregisteredfornettcp gets the value of UrisUnregisteredfornettcp for the instance
func (instance *Win32_PerfRawData_SMSvcHost4000_SMSvcHost4000) GetPropertyUrisUnregisteredfornettcp() (value uint32, err error) {
	retValue, err := instance.GetProperty("UrisUnregisteredfornettcp")
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
