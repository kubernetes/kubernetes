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

// CIM_Process struct
type CIM_Process struct {
	*CIM_LogicalElement

	//
	CreationClassName string

	//
	CreationDate string

	//
	CSCreationClassName string

	//
	CSName string

	//
	ExecutionState uint16

	//
	Handle string

	//
	KernelModeTime uint64

	//
	OSCreationClassName string

	//
	OSName string

	//
	Priority uint32

	//
	TerminationDate string

	//
	UserModeTime uint64

	//
	WorkingSetSize uint64
}

func NewCIM_ProcessEx1(instance *cim.WmiInstance) (newInstance *CIM_Process, err error) {
	tmp, err := NewCIM_LogicalElementEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_Process{
		CIM_LogicalElement: tmp,
	}
	return
}

func NewCIM_ProcessEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_Process, err error) {
	tmp, err := NewCIM_LogicalElementEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_Process{
		CIM_LogicalElement: tmp,
	}
	return
}

// SetCreationClassName sets the value of CreationClassName for the instance
func (instance *CIM_Process) SetPropertyCreationClassName(value string) (err error) {
	return instance.SetProperty("CreationClassName", (value))
}

// GetCreationClassName gets the value of CreationClassName for the instance
func (instance *CIM_Process) GetPropertyCreationClassName() (value string, err error) {
	retValue, err := instance.GetProperty("CreationClassName")
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

// SetCreationDate sets the value of CreationDate for the instance
func (instance *CIM_Process) SetPropertyCreationDate(value string) (err error) {
	return instance.SetProperty("CreationDate", (value))
}

// GetCreationDate gets the value of CreationDate for the instance
func (instance *CIM_Process) GetPropertyCreationDate() (value string, err error) {
	retValue, err := instance.GetProperty("CreationDate")
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

// SetCSCreationClassName sets the value of CSCreationClassName for the instance
func (instance *CIM_Process) SetPropertyCSCreationClassName(value string) (err error) {
	return instance.SetProperty("CSCreationClassName", (value))
}

// GetCSCreationClassName gets the value of CSCreationClassName for the instance
func (instance *CIM_Process) GetPropertyCSCreationClassName() (value string, err error) {
	retValue, err := instance.GetProperty("CSCreationClassName")
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

// SetCSName sets the value of CSName for the instance
func (instance *CIM_Process) SetPropertyCSName(value string) (err error) {
	return instance.SetProperty("CSName", (value))
}

// GetCSName gets the value of CSName for the instance
func (instance *CIM_Process) GetPropertyCSName() (value string, err error) {
	retValue, err := instance.GetProperty("CSName")
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

// SetExecutionState sets the value of ExecutionState for the instance
func (instance *CIM_Process) SetPropertyExecutionState(value uint16) (err error) {
	return instance.SetProperty("ExecutionState", (value))
}

// GetExecutionState gets the value of ExecutionState for the instance
func (instance *CIM_Process) GetPropertyExecutionState() (value uint16, err error) {
	retValue, err := instance.GetProperty("ExecutionState")
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

// SetHandle sets the value of Handle for the instance
func (instance *CIM_Process) SetPropertyHandle(value string) (err error) {
	return instance.SetProperty("Handle", (value))
}

// GetHandle gets the value of Handle for the instance
func (instance *CIM_Process) GetPropertyHandle() (value string, err error) {
	retValue, err := instance.GetProperty("Handle")
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

// SetKernelModeTime sets the value of KernelModeTime for the instance
func (instance *CIM_Process) SetPropertyKernelModeTime(value uint64) (err error) {
	return instance.SetProperty("KernelModeTime", (value))
}

// GetKernelModeTime gets the value of KernelModeTime for the instance
func (instance *CIM_Process) GetPropertyKernelModeTime() (value uint64, err error) {
	retValue, err := instance.GetProperty("KernelModeTime")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetOSCreationClassName sets the value of OSCreationClassName for the instance
func (instance *CIM_Process) SetPropertyOSCreationClassName(value string) (err error) {
	return instance.SetProperty("OSCreationClassName", (value))
}

// GetOSCreationClassName gets the value of OSCreationClassName for the instance
func (instance *CIM_Process) GetPropertyOSCreationClassName() (value string, err error) {
	retValue, err := instance.GetProperty("OSCreationClassName")
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

// SetOSName sets the value of OSName for the instance
func (instance *CIM_Process) SetPropertyOSName(value string) (err error) {
	return instance.SetProperty("OSName", (value))
}

// GetOSName gets the value of OSName for the instance
func (instance *CIM_Process) GetPropertyOSName() (value string, err error) {
	retValue, err := instance.GetProperty("OSName")
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

// SetPriority sets the value of Priority for the instance
func (instance *CIM_Process) SetPropertyPriority(value uint32) (err error) {
	return instance.SetProperty("Priority", (value))
}

// GetPriority gets the value of Priority for the instance
func (instance *CIM_Process) GetPropertyPriority() (value uint32, err error) {
	retValue, err := instance.GetProperty("Priority")
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

// SetTerminationDate sets the value of TerminationDate for the instance
func (instance *CIM_Process) SetPropertyTerminationDate(value string) (err error) {
	return instance.SetProperty("TerminationDate", (value))
}

// GetTerminationDate gets the value of TerminationDate for the instance
func (instance *CIM_Process) GetPropertyTerminationDate() (value string, err error) {
	retValue, err := instance.GetProperty("TerminationDate")
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

// SetUserModeTime sets the value of UserModeTime for the instance
func (instance *CIM_Process) SetPropertyUserModeTime(value uint64) (err error) {
	return instance.SetProperty("UserModeTime", (value))
}

// GetUserModeTime gets the value of UserModeTime for the instance
func (instance *CIM_Process) GetPropertyUserModeTime() (value uint64, err error) {
	retValue, err := instance.GetProperty("UserModeTime")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetWorkingSetSize sets the value of WorkingSetSize for the instance
func (instance *CIM_Process) SetPropertyWorkingSetSize(value uint64) (err error) {
	return instance.SetProperty("WorkingSetSize", (value))
}

// GetWorkingSetSize gets the value of WorkingSetSize for the instance
func (instance *CIM_Process) GetPropertyWorkingSetSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("WorkingSetSize")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}
