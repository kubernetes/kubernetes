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

// Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000 struct
type Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000 struct {
	*Win32_PerfFormattedData

	//
	AverageWorkflowLoadTime uint32

	//
	AverageWorkflowPersistTime uint32

	//
	WorkflowsAborted uint32

	//
	WorkflowsAbortedPerSecond uint32

	//
	WorkflowsCompleted uint32

	//
	WorkflowsCompletedPerSecond uint32

	//
	WorkflowsCreated uint32

	//
	WorkflowsCreatedPerSecond uint32

	//
	WorkflowsExecuting uint32

	//
	WorkflowsIdlePerSecond uint32

	//
	WorkflowsInMemory uint32

	//
	WorkflowsLoaded uint32

	//
	WorkflowsLoadedPerSecond uint32

	//
	WorkflowsPersisted uint32

	//
	WorkflowsPersistedPerSecond uint32

	//
	WorkflowsSuspended uint32

	//
	WorkflowsSuspendedPerSecond uint32

	//
	WorkflowsTerminated uint32

	//
	WorkflowsTerminatedPerSecond uint32

	//
	WorkflowsUnloaded uint32

	//
	WorkflowsUnloadedPerSecond uint32
}

func NewWin32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000Ex1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000Ex6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetAverageWorkflowLoadTime sets the value of AverageWorkflowLoadTime for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) SetPropertyAverageWorkflowLoadTime(value uint32) (err error) {
	return instance.SetProperty("AverageWorkflowLoadTime", (value))
}

// GetAverageWorkflowLoadTime gets the value of AverageWorkflowLoadTime for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) GetPropertyAverageWorkflowLoadTime() (value uint32, err error) {
	retValue, err := instance.GetProperty("AverageWorkflowLoadTime")
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

// SetAverageWorkflowPersistTime sets the value of AverageWorkflowPersistTime for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) SetPropertyAverageWorkflowPersistTime(value uint32) (err error) {
	return instance.SetProperty("AverageWorkflowPersistTime", (value))
}

// GetAverageWorkflowPersistTime gets the value of AverageWorkflowPersistTime for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) GetPropertyAverageWorkflowPersistTime() (value uint32, err error) {
	retValue, err := instance.GetProperty("AverageWorkflowPersistTime")
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

// SetWorkflowsAborted sets the value of WorkflowsAborted for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) SetPropertyWorkflowsAborted(value uint32) (err error) {
	return instance.SetProperty("WorkflowsAborted", (value))
}

// GetWorkflowsAborted gets the value of WorkflowsAborted for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) GetPropertyWorkflowsAborted() (value uint32, err error) {
	retValue, err := instance.GetProperty("WorkflowsAborted")
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

// SetWorkflowsAbortedPerSecond sets the value of WorkflowsAbortedPerSecond for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) SetPropertyWorkflowsAbortedPerSecond(value uint32) (err error) {
	return instance.SetProperty("WorkflowsAbortedPerSecond", (value))
}

// GetWorkflowsAbortedPerSecond gets the value of WorkflowsAbortedPerSecond for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) GetPropertyWorkflowsAbortedPerSecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("WorkflowsAbortedPerSecond")
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

// SetWorkflowsCompleted sets the value of WorkflowsCompleted for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) SetPropertyWorkflowsCompleted(value uint32) (err error) {
	return instance.SetProperty("WorkflowsCompleted", (value))
}

// GetWorkflowsCompleted gets the value of WorkflowsCompleted for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) GetPropertyWorkflowsCompleted() (value uint32, err error) {
	retValue, err := instance.GetProperty("WorkflowsCompleted")
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

// SetWorkflowsCompletedPerSecond sets the value of WorkflowsCompletedPerSecond for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) SetPropertyWorkflowsCompletedPerSecond(value uint32) (err error) {
	return instance.SetProperty("WorkflowsCompletedPerSecond", (value))
}

// GetWorkflowsCompletedPerSecond gets the value of WorkflowsCompletedPerSecond for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) GetPropertyWorkflowsCompletedPerSecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("WorkflowsCompletedPerSecond")
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

// SetWorkflowsCreated sets the value of WorkflowsCreated for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) SetPropertyWorkflowsCreated(value uint32) (err error) {
	return instance.SetProperty("WorkflowsCreated", (value))
}

// GetWorkflowsCreated gets the value of WorkflowsCreated for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) GetPropertyWorkflowsCreated() (value uint32, err error) {
	retValue, err := instance.GetProperty("WorkflowsCreated")
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

// SetWorkflowsCreatedPerSecond sets the value of WorkflowsCreatedPerSecond for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) SetPropertyWorkflowsCreatedPerSecond(value uint32) (err error) {
	return instance.SetProperty("WorkflowsCreatedPerSecond", (value))
}

// GetWorkflowsCreatedPerSecond gets the value of WorkflowsCreatedPerSecond for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) GetPropertyWorkflowsCreatedPerSecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("WorkflowsCreatedPerSecond")
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

// SetWorkflowsExecuting sets the value of WorkflowsExecuting for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) SetPropertyWorkflowsExecuting(value uint32) (err error) {
	return instance.SetProperty("WorkflowsExecuting", (value))
}

// GetWorkflowsExecuting gets the value of WorkflowsExecuting for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) GetPropertyWorkflowsExecuting() (value uint32, err error) {
	retValue, err := instance.GetProperty("WorkflowsExecuting")
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

// SetWorkflowsIdlePerSecond sets the value of WorkflowsIdlePerSecond for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) SetPropertyWorkflowsIdlePerSecond(value uint32) (err error) {
	return instance.SetProperty("WorkflowsIdlePerSecond", (value))
}

// GetWorkflowsIdlePerSecond gets the value of WorkflowsIdlePerSecond for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) GetPropertyWorkflowsIdlePerSecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("WorkflowsIdlePerSecond")
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

// SetWorkflowsInMemory sets the value of WorkflowsInMemory for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) SetPropertyWorkflowsInMemory(value uint32) (err error) {
	return instance.SetProperty("WorkflowsInMemory", (value))
}

// GetWorkflowsInMemory gets the value of WorkflowsInMemory for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) GetPropertyWorkflowsInMemory() (value uint32, err error) {
	retValue, err := instance.GetProperty("WorkflowsInMemory")
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

// SetWorkflowsLoaded sets the value of WorkflowsLoaded for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) SetPropertyWorkflowsLoaded(value uint32) (err error) {
	return instance.SetProperty("WorkflowsLoaded", (value))
}

// GetWorkflowsLoaded gets the value of WorkflowsLoaded for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) GetPropertyWorkflowsLoaded() (value uint32, err error) {
	retValue, err := instance.GetProperty("WorkflowsLoaded")
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

// SetWorkflowsLoadedPerSecond sets the value of WorkflowsLoadedPerSecond for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) SetPropertyWorkflowsLoadedPerSecond(value uint32) (err error) {
	return instance.SetProperty("WorkflowsLoadedPerSecond", (value))
}

// GetWorkflowsLoadedPerSecond gets the value of WorkflowsLoadedPerSecond for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) GetPropertyWorkflowsLoadedPerSecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("WorkflowsLoadedPerSecond")
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

// SetWorkflowsPersisted sets the value of WorkflowsPersisted for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) SetPropertyWorkflowsPersisted(value uint32) (err error) {
	return instance.SetProperty("WorkflowsPersisted", (value))
}

// GetWorkflowsPersisted gets the value of WorkflowsPersisted for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) GetPropertyWorkflowsPersisted() (value uint32, err error) {
	retValue, err := instance.GetProperty("WorkflowsPersisted")
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

// SetWorkflowsPersistedPerSecond sets the value of WorkflowsPersistedPerSecond for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) SetPropertyWorkflowsPersistedPerSecond(value uint32) (err error) {
	return instance.SetProperty("WorkflowsPersistedPerSecond", (value))
}

// GetWorkflowsPersistedPerSecond gets the value of WorkflowsPersistedPerSecond for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) GetPropertyWorkflowsPersistedPerSecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("WorkflowsPersistedPerSecond")
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

// SetWorkflowsSuspended sets the value of WorkflowsSuspended for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) SetPropertyWorkflowsSuspended(value uint32) (err error) {
	return instance.SetProperty("WorkflowsSuspended", (value))
}

// GetWorkflowsSuspended gets the value of WorkflowsSuspended for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) GetPropertyWorkflowsSuspended() (value uint32, err error) {
	retValue, err := instance.GetProperty("WorkflowsSuspended")
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

// SetWorkflowsSuspendedPerSecond sets the value of WorkflowsSuspendedPerSecond for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) SetPropertyWorkflowsSuspendedPerSecond(value uint32) (err error) {
	return instance.SetProperty("WorkflowsSuspendedPerSecond", (value))
}

// GetWorkflowsSuspendedPerSecond gets the value of WorkflowsSuspendedPerSecond for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) GetPropertyWorkflowsSuspendedPerSecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("WorkflowsSuspendedPerSecond")
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

// SetWorkflowsTerminated sets the value of WorkflowsTerminated for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) SetPropertyWorkflowsTerminated(value uint32) (err error) {
	return instance.SetProperty("WorkflowsTerminated", (value))
}

// GetWorkflowsTerminated gets the value of WorkflowsTerminated for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) GetPropertyWorkflowsTerminated() (value uint32, err error) {
	retValue, err := instance.GetProperty("WorkflowsTerminated")
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

// SetWorkflowsTerminatedPerSecond sets the value of WorkflowsTerminatedPerSecond for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) SetPropertyWorkflowsTerminatedPerSecond(value uint32) (err error) {
	return instance.SetProperty("WorkflowsTerminatedPerSecond", (value))
}

// GetWorkflowsTerminatedPerSecond gets the value of WorkflowsTerminatedPerSecond for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) GetPropertyWorkflowsTerminatedPerSecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("WorkflowsTerminatedPerSecond")
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

// SetWorkflowsUnloaded sets the value of WorkflowsUnloaded for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) SetPropertyWorkflowsUnloaded(value uint32) (err error) {
	return instance.SetProperty("WorkflowsUnloaded", (value))
}

// GetWorkflowsUnloaded gets the value of WorkflowsUnloaded for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) GetPropertyWorkflowsUnloaded() (value uint32, err error) {
	retValue, err := instance.GetProperty("WorkflowsUnloaded")
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

// SetWorkflowsUnloadedPerSecond sets the value of WorkflowsUnloadedPerSecond for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) SetPropertyWorkflowsUnloadedPerSecond(value uint32) (err error) {
	return instance.SetProperty("WorkflowsUnloadedPerSecond", (value))
}

// GetWorkflowsUnloadedPerSecond gets the value of WorkflowsUnloadedPerSecond for the instance
func (instance *Win32_PerfFormattedData_WorkflowServiceHost4000_WorkflowServiceHost4000) GetPropertyWorkflowsUnloadedPerSecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("WorkflowsUnloadedPerSecond")
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
