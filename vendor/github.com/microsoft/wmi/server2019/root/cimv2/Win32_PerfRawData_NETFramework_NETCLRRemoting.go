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

// Win32_PerfRawData_NETFramework_NETCLRRemoting struct
type Win32_PerfRawData_NETFramework_NETCLRRemoting struct {
	*Win32_PerfRawData

	//
	Channels uint32

	//
	ContextBoundClassesLoaded uint32

	//
	ContextBoundObjectsAllocPersec uint32

	//
	ContextProxies uint32

	//
	Contexts uint32

	//
	RemoteCallsPersec uint32

	//
	TotalRemoteCalls uint32
}

func NewWin32_PerfRawData_NETFramework_NETCLRRemotingEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_NETFramework_NETCLRRemoting, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_NETFramework_NETCLRRemoting{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_NETFramework_NETCLRRemotingEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_NETFramework_NETCLRRemoting, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_NETFramework_NETCLRRemoting{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetChannels sets the value of Channels for the instance
func (instance *Win32_PerfRawData_NETFramework_NETCLRRemoting) SetPropertyChannels(value uint32) (err error) {
	return instance.SetProperty("Channels", (value))
}

// GetChannels gets the value of Channels for the instance
func (instance *Win32_PerfRawData_NETFramework_NETCLRRemoting) GetPropertyChannels() (value uint32, err error) {
	retValue, err := instance.GetProperty("Channels")
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

// SetContextBoundClassesLoaded sets the value of ContextBoundClassesLoaded for the instance
func (instance *Win32_PerfRawData_NETFramework_NETCLRRemoting) SetPropertyContextBoundClassesLoaded(value uint32) (err error) {
	return instance.SetProperty("ContextBoundClassesLoaded", (value))
}

// GetContextBoundClassesLoaded gets the value of ContextBoundClassesLoaded for the instance
func (instance *Win32_PerfRawData_NETFramework_NETCLRRemoting) GetPropertyContextBoundClassesLoaded() (value uint32, err error) {
	retValue, err := instance.GetProperty("ContextBoundClassesLoaded")
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

// SetContextBoundObjectsAllocPersec sets the value of ContextBoundObjectsAllocPersec for the instance
func (instance *Win32_PerfRawData_NETFramework_NETCLRRemoting) SetPropertyContextBoundObjectsAllocPersec(value uint32) (err error) {
	return instance.SetProperty("ContextBoundObjectsAllocPersec", (value))
}

// GetContextBoundObjectsAllocPersec gets the value of ContextBoundObjectsAllocPersec for the instance
func (instance *Win32_PerfRawData_NETFramework_NETCLRRemoting) GetPropertyContextBoundObjectsAllocPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ContextBoundObjectsAllocPersec")
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

// SetContextProxies sets the value of ContextProxies for the instance
func (instance *Win32_PerfRawData_NETFramework_NETCLRRemoting) SetPropertyContextProxies(value uint32) (err error) {
	return instance.SetProperty("ContextProxies", (value))
}

// GetContextProxies gets the value of ContextProxies for the instance
func (instance *Win32_PerfRawData_NETFramework_NETCLRRemoting) GetPropertyContextProxies() (value uint32, err error) {
	retValue, err := instance.GetProperty("ContextProxies")
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

// SetContexts sets the value of Contexts for the instance
func (instance *Win32_PerfRawData_NETFramework_NETCLRRemoting) SetPropertyContexts(value uint32) (err error) {
	return instance.SetProperty("Contexts", (value))
}

// GetContexts gets the value of Contexts for the instance
func (instance *Win32_PerfRawData_NETFramework_NETCLRRemoting) GetPropertyContexts() (value uint32, err error) {
	retValue, err := instance.GetProperty("Contexts")
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

// SetRemoteCallsPersec sets the value of RemoteCallsPersec for the instance
func (instance *Win32_PerfRawData_NETFramework_NETCLRRemoting) SetPropertyRemoteCallsPersec(value uint32) (err error) {
	return instance.SetProperty("RemoteCallsPersec", (value))
}

// GetRemoteCallsPersec gets the value of RemoteCallsPersec for the instance
func (instance *Win32_PerfRawData_NETFramework_NETCLRRemoting) GetPropertyRemoteCallsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("RemoteCallsPersec")
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

// SetTotalRemoteCalls sets the value of TotalRemoteCalls for the instance
func (instance *Win32_PerfRawData_NETFramework_NETCLRRemoting) SetPropertyTotalRemoteCalls(value uint32) (err error) {
	return instance.SetProperty("TotalRemoteCalls", (value))
}

// GetTotalRemoteCalls gets the value of TotalRemoteCalls for the instance
func (instance *Win32_PerfRawData_NETFramework_NETCLRRemoting) GetPropertyTotalRemoteCalls() (value uint32, err error) {
	retValue, err := instance.GetProperty("TotalRemoteCalls")
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
