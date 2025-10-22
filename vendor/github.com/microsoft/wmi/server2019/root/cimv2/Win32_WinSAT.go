// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.CIMV2
//////////////////////////////////////////////
package cimv2

import (
	"github.com/microsoft/wmi/pkg/base/instance"
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// Win32_WinSAT struct
type Win32_WinSAT struct {
	*cim.WmiInstance

	//
	CPUScore float32

	//
	D3DScore float32

	//
	DiskScore float32

	//
	GraphicsScore float32

	//
	MemoryScore float32

	//
	TimeTaken string

	//
	WinSATAssessmentState WinSAT_WinSATAssessmentState

	//
	WinSPRLevel float32
}

func NewWin32_WinSATEx1(instance *cim.WmiInstance) (newInstance *Win32_WinSAT, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_WinSAT{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_WinSATEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_WinSAT, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_WinSAT{
		WmiInstance: tmp,
	}
	return
}

// SetCPUScore sets the value of CPUScore for the instance
func (instance *Win32_WinSAT) SetPropertyCPUScore(value float32) (err error) {
	return instance.SetProperty("CPUScore", (value))
}

// GetCPUScore gets the value of CPUScore for the instance
func (instance *Win32_WinSAT) GetPropertyCPUScore() (value float32, err error) {
	retValue, err := instance.GetProperty("CPUScore")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(float32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " float32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = float32(valuetmp)

	return
}

// SetD3DScore sets the value of D3DScore for the instance
func (instance *Win32_WinSAT) SetPropertyD3DScore(value float32) (err error) {
	return instance.SetProperty("D3DScore", (value))
}

// GetD3DScore gets the value of D3DScore for the instance
func (instance *Win32_WinSAT) GetPropertyD3DScore() (value float32, err error) {
	retValue, err := instance.GetProperty("D3DScore")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(float32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " float32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = float32(valuetmp)

	return
}

// SetDiskScore sets the value of DiskScore for the instance
func (instance *Win32_WinSAT) SetPropertyDiskScore(value float32) (err error) {
	return instance.SetProperty("DiskScore", (value))
}

// GetDiskScore gets the value of DiskScore for the instance
func (instance *Win32_WinSAT) GetPropertyDiskScore() (value float32, err error) {
	retValue, err := instance.GetProperty("DiskScore")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(float32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " float32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = float32(valuetmp)

	return
}

// SetGraphicsScore sets the value of GraphicsScore for the instance
func (instance *Win32_WinSAT) SetPropertyGraphicsScore(value float32) (err error) {
	return instance.SetProperty("GraphicsScore", (value))
}

// GetGraphicsScore gets the value of GraphicsScore for the instance
func (instance *Win32_WinSAT) GetPropertyGraphicsScore() (value float32, err error) {
	retValue, err := instance.GetProperty("GraphicsScore")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(float32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " float32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = float32(valuetmp)

	return
}

// SetMemoryScore sets the value of MemoryScore for the instance
func (instance *Win32_WinSAT) SetPropertyMemoryScore(value float32) (err error) {
	return instance.SetProperty("MemoryScore", (value))
}

// GetMemoryScore gets the value of MemoryScore for the instance
func (instance *Win32_WinSAT) GetPropertyMemoryScore() (value float32, err error) {
	retValue, err := instance.GetProperty("MemoryScore")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(float32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " float32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = float32(valuetmp)

	return
}

// SetTimeTaken sets the value of TimeTaken for the instance
func (instance *Win32_WinSAT) SetPropertyTimeTaken(value string) (err error) {
	return instance.SetProperty("TimeTaken", (value))
}

// GetTimeTaken gets the value of TimeTaken for the instance
func (instance *Win32_WinSAT) GetPropertyTimeTaken() (value string, err error) {
	retValue, err := instance.GetProperty("TimeTaken")
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

// SetWinSATAssessmentState sets the value of WinSATAssessmentState for the instance
func (instance *Win32_WinSAT) SetPropertyWinSATAssessmentState(value WinSAT_WinSATAssessmentState) (err error) {
	return instance.SetProperty("WinSATAssessmentState", (value))
}

// GetWinSATAssessmentState gets the value of WinSATAssessmentState for the instance
func (instance *Win32_WinSAT) GetPropertyWinSATAssessmentState() (value WinSAT_WinSATAssessmentState, err error) {
	retValue, err := instance.GetProperty("WinSATAssessmentState")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = WinSAT_WinSATAssessmentState(valuetmp)

	return
}

// SetWinSPRLevel sets the value of WinSPRLevel for the instance
func (instance *Win32_WinSAT) SetPropertyWinSPRLevel(value float32) (err error) {
	return instance.SetProperty("WinSPRLevel", (value))
}

// GetWinSPRLevel gets the value of WinSPRLevel for the instance
func (instance *Win32_WinSAT) GetPropertyWinSPRLevel() (value float32, err error) {
	retValue, err := instance.GetProperty("WinSPRLevel")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(float32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " float32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = float32(valuetmp)

	return
}
