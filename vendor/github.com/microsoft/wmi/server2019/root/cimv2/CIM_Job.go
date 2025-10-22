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

// CIM_Job struct
type CIM_Job struct {
	*CIM_LogicalElement

	//
	ElapsedTime string

	//
	JobStatus string

	//
	Notify string

	//
	Owner string

	//
	Priority uint32

	//
	StartTime string

	//
	TimeSubmitted string

	//
	UntilTime string
}

func NewCIM_JobEx1(instance *cim.WmiInstance) (newInstance *CIM_Job, err error) {
	tmp, err := NewCIM_LogicalElementEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_Job{
		CIM_LogicalElement: tmp,
	}
	return
}

func NewCIM_JobEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_Job, err error) {
	tmp, err := NewCIM_LogicalElementEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_Job{
		CIM_LogicalElement: tmp,
	}
	return
}

// SetElapsedTime sets the value of ElapsedTime for the instance
func (instance *CIM_Job) SetPropertyElapsedTime(value string) (err error) {
	return instance.SetProperty("ElapsedTime", (value))
}

// GetElapsedTime gets the value of ElapsedTime for the instance
func (instance *CIM_Job) GetPropertyElapsedTime() (value string, err error) {
	retValue, err := instance.GetProperty("ElapsedTime")
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

// SetJobStatus sets the value of JobStatus for the instance
func (instance *CIM_Job) SetPropertyJobStatus(value string) (err error) {
	return instance.SetProperty("JobStatus", (value))
}

// GetJobStatus gets the value of JobStatus for the instance
func (instance *CIM_Job) GetPropertyJobStatus() (value string, err error) {
	retValue, err := instance.GetProperty("JobStatus")
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

// SetNotify sets the value of Notify for the instance
func (instance *CIM_Job) SetPropertyNotify(value string) (err error) {
	return instance.SetProperty("Notify", (value))
}

// GetNotify gets the value of Notify for the instance
func (instance *CIM_Job) GetPropertyNotify() (value string, err error) {
	retValue, err := instance.GetProperty("Notify")
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

// SetOwner sets the value of Owner for the instance
func (instance *CIM_Job) SetPropertyOwner(value string) (err error) {
	return instance.SetProperty("Owner", (value))
}

// GetOwner gets the value of Owner for the instance
func (instance *CIM_Job) GetPropertyOwner() (value string, err error) {
	retValue, err := instance.GetProperty("Owner")
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
func (instance *CIM_Job) SetPropertyPriority(value uint32) (err error) {
	return instance.SetProperty("Priority", (value))
}

// GetPriority gets the value of Priority for the instance
func (instance *CIM_Job) GetPropertyPriority() (value uint32, err error) {
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

// SetStartTime sets the value of StartTime for the instance
func (instance *CIM_Job) SetPropertyStartTime(value string) (err error) {
	return instance.SetProperty("StartTime", (value))
}

// GetStartTime gets the value of StartTime for the instance
func (instance *CIM_Job) GetPropertyStartTime() (value string, err error) {
	retValue, err := instance.GetProperty("StartTime")
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

// SetTimeSubmitted sets the value of TimeSubmitted for the instance
func (instance *CIM_Job) SetPropertyTimeSubmitted(value string) (err error) {
	return instance.SetProperty("TimeSubmitted", (value))
}

// GetTimeSubmitted gets the value of TimeSubmitted for the instance
func (instance *CIM_Job) GetPropertyTimeSubmitted() (value string, err error) {
	retValue, err := instance.GetProperty("TimeSubmitted")
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

// SetUntilTime sets the value of UntilTime for the instance
func (instance *CIM_Job) SetPropertyUntilTime(value string) (err error) {
	return instance.SetProperty("UntilTime", (value))
}

// GetUntilTime gets the value of UntilTime for the instance
func (instance *CIM_Job) GetPropertyUntilTime() (value string, err error) {
	retValue, err := instance.GetProperty("UntilTime")
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
