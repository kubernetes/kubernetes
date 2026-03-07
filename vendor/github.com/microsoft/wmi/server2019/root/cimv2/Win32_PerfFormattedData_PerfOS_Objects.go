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

// Win32_PerfFormattedData_PerfOS_Objects struct
type Win32_PerfFormattedData_PerfOS_Objects struct {
	*Win32_PerfFormattedData

	//
	Events uint32

	//
	Mutexes uint32

	//
	Processes uint32

	//
	Sections uint32

	//
	Semaphores uint32

	//
	Threads uint32
}

func NewWin32_PerfFormattedData_PerfOS_ObjectsEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_PerfOS_Objects, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_PerfOS_Objects{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_PerfOS_ObjectsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_PerfOS_Objects, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_PerfOS_Objects{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetEvents sets the value of Events for the instance
func (instance *Win32_PerfFormattedData_PerfOS_Objects) SetPropertyEvents(value uint32) (err error) {
	return instance.SetProperty("Events", (value))
}

// GetEvents gets the value of Events for the instance
func (instance *Win32_PerfFormattedData_PerfOS_Objects) GetPropertyEvents() (value uint32, err error) {
	retValue, err := instance.GetProperty("Events")
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

// SetMutexes sets the value of Mutexes for the instance
func (instance *Win32_PerfFormattedData_PerfOS_Objects) SetPropertyMutexes(value uint32) (err error) {
	return instance.SetProperty("Mutexes", (value))
}

// GetMutexes gets the value of Mutexes for the instance
func (instance *Win32_PerfFormattedData_PerfOS_Objects) GetPropertyMutexes() (value uint32, err error) {
	retValue, err := instance.GetProperty("Mutexes")
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

// SetProcesses sets the value of Processes for the instance
func (instance *Win32_PerfFormattedData_PerfOS_Objects) SetPropertyProcesses(value uint32) (err error) {
	return instance.SetProperty("Processes", (value))
}

// GetProcesses gets the value of Processes for the instance
func (instance *Win32_PerfFormattedData_PerfOS_Objects) GetPropertyProcesses() (value uint32, err error) {
	retValue, err := instance.GetProperty("Processes")
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

// SetSections sets the value of Sections for the instance
func (instance *Win32_PerfFormattedData_PerfOS_Objects) SetPropertySections(value uint32) (err error) {
	return instance.SetProperty("Sections", (value))
}

// GetSections gets the value of Sections for the instance
func (instance *Win32_PerfFormattedData_PerfOS_Objects) GetPropertySections() (value uint32, err error) {
	retValue, err := instance.GetProperty("Sections")
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

// SetSemaphores sets the value of Semaphores for the instance
func (instance *Win32_PerfFormattedData_PerfOS_Objects) SetPropertySemaphores(value uint32) (err error) {
	return instance.SetProperty("Semaphores", (value))
}

// GetSemaphores gets the value of Semaphores for the instance
func (instance *Win32_PerfFormattedData_PerfOS_Objects) GetPropertySemaphores() (value uint32, err error) {
	retValue, err := instance.GetProperty("Semaphores")
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

// SetThreads sets the value of Threads for the instance
func (instance *Win32_PerfFormattedData_PerfOS_Objects) SetPropertyThreads(value uint32) (err error) {
	return instance.SetProperty("Threads", (value))
}

// GetThreads gets the value of Threads for the instance
func (instance *Win32_PerfFormattedData_PerfOS_Objects) GetPropertyThreads() (value uint32, err error) {
	retValue, err := instance.GetProperty("Threads")
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
