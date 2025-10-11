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

// Win32_DesktopMonitor struct
type Win32_DesktopMonitor struct {
	*CIM_DesktopMonitor

	//
	MonitorManufacturer string

	//
	MonitorType string

	//
	PixelsPerXLogicalInch uint32

	//
	PixelsPerYLogicalInch uint32
}

func NewWin32_DesktopMonitorEx1(instance *cim.WmiInstance) (newInstance *Win32_DesktopMonitor, err error) {
	tmp, err := NewCIM_DesktopMonitorEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_DesktopMonitor{
		CIM_DesktopMonitor: tmp,
	}
	return
}

func NewWin32_DesktopMonitorEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_DesktopMonitor, err error) {
	tmp, err := NewCIM_DesktopMonitorEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_DesktopMonitor{
		CIM_DesktopMonitor: tmp,
	}
	return
}

// SetMonitorManufacturer sets the value of MonitorManufacturer for the instance
func (instance *Win32_DesktopMonitor) SetPropertyMonitorManufacturer(value string) (err error) {
	return instance.SetProperty("MonitorManufacturer", (value))
}

// GetMonitorManufacturer gets the value of MonitorManufacturer for the instance
func (instance *Win32_DesktopMonitor) GetPropertyMonitorManufacturer() (value string, err error) {
	retValue, err := instance.GetProperty("MonitorManufacturer")
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

// SetMonitorType sets the value of MonitorType for the instance
func (instance *Win32_DesktopMonitor) SetPropertyMonitorType(value string) (err error) {
	return instance.SetProperty("MonitorType", (value))
}

// GetMonitorType gets the value of MonitorType for the instance
func (instance *Win32_DesktopMonitor) GetPropertyMonitorType() (value string, err error) {
	retValue, err := instance.GetProperty("MonitorType")
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

// SetPixelsPerXLogicalInch sets the value of PixelsPerXLogicalInch for the instance
func (instance *Win32_DesktopMonitor) SetPropertyPixelsPerXLogicalInch(value uint32) (err error) {
	return instance.SetProperty("PixelsPerXLogicalInch", (value))
}

// GetPixelsPerXLogicalInch gets the value of PixelsPerXLogicalInch for the instance
func (instance *Win32_DesktopMonitor) GetPropertyPixelsPerXLogicalInch() (value uint32, err error) {
	retValue, err := instance.GetProperty("PixelsPerXLogicalInch")
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

// SetPixelsPerYLogicalInch sets the value of PixelsPerYLogicalInch for the instance
func (instance *Win32_DesktopMonitor) SetPropertyPixelsPerYLogicalInch(value uint32) (err error) {
	return instance.SetProperty("PixelsPerYLogicalInch", (value))
}

// GetPixelsPerYLogicalInch gets the value of PixelsPerYLogicalInch for the instance
func (instance *Win32_DesktopMonitor) GetPropertyPixelsPerYLogicalInch() (value uint32, err error) {
	retValue, err := instance.GetProperty("PixelsPerYLogicalInch")
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
