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

// Win32_Thread struct
type Win32_Thread struct {
	*CIM_Thread

	//
	ElapsedTime uint64

	//
	PriorityBase uint32

	//
	StartAddress uint32

	//
	ThreadState uint32

	//
	ThreadWaitReason uint32
}

func NewWin32_ThreadEx1(instance *cim.WmiInstance) (newInstance *Win32_Thread, err error) {
	tmp, err := NewCIM_ThreadEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_Thread{
		CIM_Thread: tmp,
	}
	return
}

func NewWin32_ThreadEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_Thread, err error) {
	tmp, err := NewCIM_ThreadEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_Thread{
		CIM_Thread: tmp,
	}
	return
}

// SetElapsedTime sets the value of ElapsedTime for the instance
func (instance *Win32_Thread) SetPropertyElapsedTime(value uint64) (err error) {
	return instance.SetProperty("ElapsedTime", (value))
}

// GetElapsedTime gets the value of ElapsedTime for the instance
func (instance *Win32_Thread) GetPropertyElapsedTime() (value uint64, err error) {
	retValue, err := instance.GetProperty("ElapsedTime")
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

// SetPriorityBase sets the value of PriorityBase for the instance
func (instance *Win32_Thread) SetPropertyPriorityBase(value uint32) (err error) {
	return instance.SetProperty("PriorityBase", (value))
}

// GetPriorityBase gets the value of PriorityBase for the instance
func (instance *Win32_Thread) GetPropertyPriorityBase() (value uint32, err error) {
	retValue, err := instance.GetProperty("PriorityBase")
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

// SetStartAddress sets the value of StartAddress for the instance
func (instance *Win32_Thread) SetPropertyStartAddress(value uint32) (err error) {
	return instance.SetProperty("StartAddress", (value))
}

// GetStartAddress gets the value of StartAddress for the instance
func (instance *Win32_Thread) GetPropertyStartAddress() (value uint32, err error) {
	retValue, err := instance.GetProperty("StartAddress")
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

// SetThreadState sets the value of ThreadState for the instance
func (instance *Win32_Thread) SetPropertyThreadState(value uint32) (err error) {
	return instance.SetProperty("ThreadState", (value))
}

// GetThreadState gets the value of ThreadState for the instance
func (instance *Win32_Thread) GetPropertyThreadState() (value uint32, err error) {
	retValue, err := instance.GetProperty("ThreadState")
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

// SetThreadWaitReason sets the value of ThreadWaitReason for the instance
func (instance *Win32_Thread) SetPropertyThreadWaitReason(value uint32) (err error) {
	return instance.SetProperty("ThreadWaitReason", (value))
}

// GetThreadWaitReason gets the value of ThreadWaitReason for the instance
func (instance *Win32_Thread) GetPropertyThreadWaitReason() (value uint32, err error) {
	retValue, err := instance.GetProperty("ThreadWaitReason")
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
