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

// Win32_NamedJobObjectLimitSetting struct
type Win32_NamedJobObjectLimitSetting struct {
	*CIM_Setting

	//
	ActiveProcessLimit uint32

	//
	Affinity uint32

	//
	JobMemoryLimit uint32

	//
	LimitFlags uint32

	//
	MaximumWorkingSetSize uint32

	//
	MinimumWorkingSetSize uint32

	//
	PerJobUserTimeLimit uint64

	//
	PerProcessUserTimeLimit uint64

	//
	PriorityClass uint32

	//
	ProcessMemoryLimit uint32

	//
	SchedulingClass uint32
}

func NewWin32_NamedJobObjectLimitSettingEx1(instance *cim.WmiInstance) (newInstance *Win32_NamedJobObjectLimitSetting, err error) {
	tmp, err := NewCIM_SettingEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_NamedJobObjectLimitSetting{
		CIM_Setting: tmp,
	}
	return
}

func NewWin32_NamedJobObjectLimitSettingEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_NamedJobObjectLimitSetting, err error) {
	tmp, err := NewCIM_SettingEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_NamedJobObjectLimitSetting{
		CIM_Setting: tmp,
	}
	return
}

// SetActiveProcessLimit sets the value of ActiveProcessLimit for the instance
func (instance *Win32_NamedJobObjectLimitSetting) SetPropertyActiveProcessLimit(value uint32) (err error) {
	return instance.SetProperty("ActiveProcessLimit", (value))
}

// GetActiveProcessLimit gets the value of ActiveProcessLimit for the instance
func (instance *Win32_NamedJobObjectLimitSetting) GetPropertyActiveProcessLimit() (value uint32, err error) {
	retValue, err := instance.GetProperty("ActiveProcessLimit")
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

// SetAffinity sets the value of Affinity for the instance
func (instance *Win32_NamedJobObjectLimitSetting) SetPropertyAffinity(value uint32) (err error) {
	return instance.SetProperty("Affinity", (value))
}

// GetAffinity gets the value of Affinity for the instance
func (instance *Win32_NamedJobObjectLimitSetting) GetPropertyAffinity() (value uint32, err error) {
	retValue, err := instance.GetProperty("Affinity")
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

// SetJobMemoryLimit sets the value of JobMemoryLimit for the instance
func (instance *Win32_NamedJobObjectLimitSetting) SetPropertyJobMemoryLimit(value uint32) (err error) {
	return instance.SetProperty("JobMemoryLimit", (value))
}

// GetJobMemoryLimit gets the value of JobMemoryLimit for the instance
func (instance *Win32_NamedJobObjectLimitSetting) GetPropertyJobMemoryLimit() (value uint32, err error) {
	retValue, err := instance.GetProperty("JobMemoryLimit")
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

// SetLimitFlags sets the value of LimitFlags for the instance
func (instance *Win32_NamedJobObjectLimitSetting) SetPropertyLimitFlags(value uint32) (err error) {
	return instance.SetProperty("LimitFlags", (value))
}

// GetLimitFlags gets the value of LimitFlags for the instance
func (instance *Win32_NamedJobObjectLimitSetting) GetPropertyLimitFlags() (value uint32, err error) {
	retValue, err := instance.GetProperty("LimitFlags")
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

// SetMaximumWorkingSetSize sets the value of MaximumWorkingSetSize for the instance
func (instance *Win32_NamedJobObjectLimitSetting) SetPropertyMaximumWorkingSetSize(value uint32) (err error) {
	return instance.SetProperty("MaximumWorkingSetSize", (value))
}

// GetMaximumWorkingSetSize gets the value of MaximumWorkingSetSize for the instance
func (instance *Win32_NamedJobObjectLimitSetting) GetPropertyMaximumWorkingSetSize() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaximumWorkingSetSize")
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

// SetMinimumWorkingSetSize sets the value of MinimumWorkingSetSize for the instance
func (instance *Win32_NamedJobObjectLimitSetting) SetPropertyMinimumWorkingSetSize(value uint32) (err error) {
	return instance.SetProperty("MinimumWorkingSetSize", (value))
}

// GetMinimumWorkingSetSize gets the value of MinimumWorkingSetSize for the instance
func (instance *Win32_NamedJobObjectLimitSetting) GetPropertyMinimumWorkingSetSize() (value uint32, err error) {
	retValue, err := instance.GetProperty("MinimumWorkingSetSize")
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

// SetPerJobUserTimeLimit sets the value of PerJobUserTimeLimit for the instance
func (instance *Win32_NamedJobObjectLimitSetting) SetPropertyPerJobUserTimeLimit(value uint64) (err error) {
	return instance.SetProperty("PerJobUserTimeLimit", (value))
}

// GetPerJobUserTimeLimit gets the value of PerJobUserTimeLimit for the instance
func (instance *Win32_NamedJobObjectLimitSetting) GetPropertyPerJobUserTimeLimit() (value uint64, err error) {
	retValue, err := instance.GetProperty("PerJobUserTimeLimit")
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

// SetPerProcessUserTimeLimit sets the value of PerProcessUserTimeLimit for the instance
func (instance *Win32_NamedJobObjectLimitSetting) SetPropertyPerProcessUserTimeLimit(value uint64) (err error) {
	return instance.SetProperty("PerProcessUserTimeLimit", (value))
}

// GetPerProcessUserTimeLimit gets the value of PerProcessUserTimeLimit for the instance
func (instance *Win32_NamedJobObjectLimitSetting) GetPropertyPerProcessUserTimeLimit() (value uint64, err error) {
	retValue, err := instance.GetProperty("PerProcessUserTimeLimit")
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

// SetPriorityClass sets the value of PriorityClass for the instance
func (instance *Win32_NamedJobObjectLimitSetting) SetPropertyPriorityClass(value uint32) (err error) {
	return instance.SetProperty("PriorityClass", (value))
}

// GetPriorityClass gets the value of PriorityClass for the instance
func (instance *Win32_NamedJobObjectLimitSetting) GetPropertyPriorityClass() (value uint32, err error) {
	retValue, err := instance.GetProperty("PriorityClass")
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

// SetProcessMemoryLimit sets the value of ProcessMemoryLimit for the instance
func (instance *Win32_NamedJobObjectLimitSetting) SetPropertyProcessMemoryLimit(value uint32) (err error) {
	return instance.SetProperty("ProcessMemoryLimit", (value))
}

// GetProcessMemoryLimit gets the value of ProcessMemoryLimit for the instance
func (instance *Win32_NamedJobObjectLimitSetting) GetPropertyProcessMemoryLimit() (value uint32, err error) {
	retValue, err := instance.GetProperty("ProcessMemoryLimit")
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

// SetSchedulingClass sets the value of SchedulingClass for the instance
func (instance *Win32_NamedJobObjectLimitSetting) SetPropertySchedulingClass(value uint32) (err error) {
	return instance.SetProperty("SchedulingClass", (value))
}

// GetSchedulingClass gets the value of SchedulingClass for the instance
func (instance *Win32_NamedJobObjectLimitSetting) GetPropertySchedulingClass() (value uint32, err error) {
	retValue, err := instance.GetProperty("SchedulingClass")
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
