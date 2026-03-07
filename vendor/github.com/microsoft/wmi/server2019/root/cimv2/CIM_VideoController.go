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

// CIM_VideoController struct
type CIM_VideoController struct {
	*CIM_Controller

	//
	AcceleratorCapabilities []uint16

	//
	CapabilityDescriptions []string

	//
	CurrentBitsPerPixel uint32

	//
	CurrentHorizontalResolution uint32

	//
	CurrentNumberOfColors uint64

	//
	CurrentNumberOfColumns uint32

	//
	CurrentNumberOfRows uint32

	//
	CurrentRefreshRate uint32

	//
	CurrentScanMode uint16

	//
	CurrentVerticalResolution uint32

	//
	MaxMemorySupported uint32

	//
	MaxRefreshRate uint32

	//
	MinRefreshRate uint32

	//
	NumberOfVideoPages uint32

	//
	VideoMemoryType uint16

	//
	VideoProcessor string
}

func NewCIM_VideoControllerEx1(instance *cim.WmiInstance) (newInstance *CIM_VideoController, err error) {
	tmp, err := NewCIM_ControllerEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_VideoController{
		CIM_Controller: tmp,
	}
	return
}

func NewCIM_VideoControllerEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_VideoController, err error) {
	tmp, err := NewCIM_ControllerEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_VideoController{
		CIM_Controller: tmp,
	}
	return
}

// SetAcceleratorCapabilities sets the value of AcceleratorCapabilities for the instance
func (instance *CIM_VideoController) SetPropertyAcceleratorCapabilities(value []uint16) (err error) {
	return instance.SetProperty("AcceleratorCapabilities", (value))
}

// GetAcceleratorCapabilities gets the value of AcceleratorCapabilities for the instance
func (instance *CIM_VideoController) GetPropertyAcceleratorCapabilities() (value []uint16, err error) {
	retValue, err := instance.GetProperty("AcceleratorCapabilities")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint16)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint16(valuetmp))
	}

	return
}

// SetCapabilityDescriptions sets the value of CapabilityDescriptions for the instance
func (instance *CIM_VideoController) SetPropertyCapabilityDescriptions(value []string) (err error) {
	return instance.SetProperty("CapabilityDescriptions", (value))
}

// GetCapabilityDescriptions gets the value of CapabilityDescriptions for the instance
func (instance *CIM_VideoController) GetPropertyCapabilityDescriptions() (value []string, err error) {
	retValue, err := instance.GetProperty("CapabilityDescriptions")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetCurrentBitsPerPixel sets the value of CurrentBitsPerPixel for the instance
func (instance *CIM_VideoController) SetPropertyCurrentBitsPerPixel(value uint32) (err error) {
	return instance.SetProperty("CurrentBitsPerPixel", (value))
}

// GetCurrentBitsPerPixel gets the value of CurrentBitsPerPixel for the instance
func (instance *CIM_VideoController) GetPropertyCurrentBitsPerPixel() (value uint32, err error) {
	retValue, err := instance.GetProperty("CurrentBitsPerPixel")
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

// SetCurrentHorizontalResolution sets the value of CurrentHorizontalResolution for the instance
func (instance *CIM_VideoController) SetPropertyCurrentHorizontalResolution(value uint32) (err error) {
	return instance.SetProperty("CurrentHorizontalResolution", (value))
}

// GetCurrentHorizontalResolution gets the value of CurrentHorizontalResolution for the instance
func (instance *CIM_VideoController) GetPropertyCurrentHorizontalResolution() (value uint32, err error) {
	retValue, err := instance.GetProperty("CurrentHorizontalResolution")
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

// SetCurrentNumberOfColors sets the value of CurrentNumberOfColors for the instance
func (instance *CIM_VideoController) SetPropertyCurrentNumberOfColors(value uint64) (err error) {
	return instance.SetProperty("CurrentNumberOfColors", (value))
}

// GetCurrentNumberOfColors gets the value of CurrentNumberOfColors for the instance
func (instance *CIM_VideoController) GetPropertyCurrentNumberOfColors() (value uint64, err error) {
	retValue, err := instance.GetProperty("CurrentNumberOfColors")
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

// SetCurrentNumberOfColumns sets the value of CurrentNumberOfColumns for the instance
func (instance *CIM_VideoController) SetPropertyCurrentNumberOfColumns(value uint32) (err error) {
	return instance.SetProperty("CurrentNumberOfColumns", (value))
}

// GetCurrentNumberOfColumns gets the value of CurrentNumberOfColumns for the instance
func (instance *CIM_VideoController) GetPropertyCurrentNumberOfColumns() (value uint32, err error) {
	retValue, err := instance.GetProperty("CurrentNumberOfColumns")
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

// SetCurrentNumberOfRows sets the value of CurrentNumberOfRows for the instance
func (instance *CIM_VideoController) SetPropertyCurrentNumberOfRows(value uint32) (err error) {
	return instance.SetProperty("CurrentNumberOfRows", (value))
}

// GetCurrentNumberOfRows gets the value of CurrentNumberOfRows for the instance
func (instance *CIM_VideoController) GetPropertyCurrentNumberOfRows() (value uint32, err error) {
	retValue, err := instance.GetProperty("CurrentNumberOfRows")
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

// SetCurrentRefreshRate sets the value of CurrentRefreshRate for the instance
func (instance *CIM_VideoController) SetPropertyCurrentRefreshRate(value uint32) (err error) {
	return instance.SetProperty("CurrentRefreshRate", (value))
}

// GetCurrentRefreshRate gets the value of CurrentRefreshRate for the instance
func (instance *CIM_VideoController) GetPropertyCurrentRefreshRate() (value uint32, err error) {
	retValue, err := instance.GetProperty("CurrentRefreshRate")
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

// SetCurrentScanMode sets the value of CurrentScanMode for the instance
func (instance *CIM_VideoController) SetPropertyCurrentScanMode(value uint16) (err error) {
	return instance.SetProperty("CurrentScanMode", (value))
}

// GetCurrentScanMode gets the value of CurrentScanMode for the instance
func (instance *CIM_VideoController) GetPropertyCurrentScanMode() (value uint16, err error) {
	retValue, err := instance.GetProperty("CurrentScanMode")
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

// SetCurrentVerticalResolution sets the value of CurrentVerticalResolution for the instance
func (instance *CIM_VideoController) SetPropertyCurrentVerticalResolution(value uint32) (err error) {
	return instance.SetProperty("CurrentVerticalResolution", (value))
}

// GetCurrentVerticalResolution gets the value of CurrentVerticalResolution for the instance
func (instance *CIM_VideoController) GetPropertyCurrentVerticalResolution() (value uint32, err error) {
	retValue, err := instance.GetProperty("CurrentVerticalResolution")
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

// SetMaxMemorySupported sets the value of MaxMemorySupported for the instance
func (instance *CIM_VideoController) SetPropertyMaxMemorySupported(value uint32) (err error) {
	return instance.SetProperty("MaxMemorySupported", (value))
}

// GetMaxMemorySupported gets the value of MaxMemorySupported for the instance
func (instance *CIM_VideoController) GetPropertyMaxMemorySupported() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaxMemorySupported")
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

// SetMaxRefreshRate sets the value of MaxRefreshRate for the instance
func (instance *CIM_VideoController) SetPropertyMaxRefreshRate(value uint32) (err error) {
	return instance.SetProperty("MaxRefreshRate", (value))
}

// GetMaxRefreshRate gets the value of MaxRefreshRate for the instance
func (instance *CIM_VideoController) GetPropertyMaxRefreshRate() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaxRefreshRate")
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

// SetMinRefreshRate sets the value of MinRefreshRate for the instance
func (instance *CIM_VideoController) SetPropertyMinRefreshRate(value uint32) (err error) {
	return instance.SetProperty("MinRefreshRate", (value))
}

// GetMinRefreshRate gets the value of MinRefreshRate for the instance
func (instance *CIM_VideoController) GetPropertyMinRefreshRate() (value uint32, err error) {
	retValue, err := instance.GetProperty("MinRefreshRate")
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

// SetNumberOfVideoPages sets the value of NumberOfVideoPages for the instance
func (instance *CIM_VideoController) SetPropertyNumberOfVideoPages(value uint32) (err error) {
	return instance.SetProperty("NumberOfVideoPages", (value))
}

// GetNumberOfVideoPages gets the value of NumberOfVideoPages for the instance
func (instance *CIM_VideoController) GetPropertyNumberOfVideoPages() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfVideoPages")
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

// SetVideoMemoryType sets the value of VideoMemoryType for the instance
func (instance *CIM_VideoController) SetPropertyVideoMemoryType(value uint16) (err error) {
	return instance.SetProperty("VideoMemoryType", (value))
}

// GetVideoMemoryType gets the value of VideoMemoryType for the instance
func (instance *CIM_VideoController) GetPropertyVideoMemoryType() (value uint16, err error) {
	retValue, err := instance.GetProperty("VideoMemoryType")
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

// SetVideoProcessor sets the value of VideoProcessor for the instance
func (instance *CIM_VideoController) SetPropertyVideoProcessor(value string) (err error) {
	return instance.SetProperty("VideoProcessor", (value))
}

// GetVideoProcessor gets the value of VideoProcessor for the instance
func (instance *CIM_VideoController) GetPropertyVideoProcessor() (value string, err error) {
	retValue, err := instance.GetProperty("VideoProcessor")
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
