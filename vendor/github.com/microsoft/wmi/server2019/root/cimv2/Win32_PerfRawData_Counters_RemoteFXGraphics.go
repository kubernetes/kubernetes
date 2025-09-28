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

// Win32_PerfRawData_Counters_RemoteFXGraphics struct
type Win32_PerfRawData_Counters_RemoteFXGraphics struct {
	*Win32_PerfRawData

	//
	AverageEncodingTime uint32

	//
	FrameQuality uint32

	//
	FramesSkippedPerSecondInsufficientClientResources uint32

	//
	FramesSkippedPerSecondInsufficientNetworkResources uint32

	//
	FramesSkippedPerSecondInsufficientServerResources uint32

	//
	GraphicsCompressionratio uint32

	//
	InputFramesPerSecond uint32

	//
	OutputFramesPerSecond uint32

	//
	SourceFramesPerSecond uint32
}

func NewWin32_PerfRawData_Counters_RemoteFXGraphicsEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_Counters_RemoteFXGraphics, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_RemoteFXGraphics{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_Counters_RemoteFXGraphicsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_Counters_RemoteFXGraphics, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_RemoteFXGraphics{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetAverageEncodingTime sets the value of AverageEncodingTime for the instance
func (instance *Win32_PerfRawData_Counters_RemoteFXGraphics) SetPropertyAverageEncodingTime(value uint32) (err error) {
	return instance.SetProperty("AverageEncodingTime", (value))
}

// GetAverageEncodingTime gets the value of AverageEncodingTime for the instance
func (instance *Win32_PerfRawData_Counters_RemoteFXGraphics) GetPropertyAverageEncodingTime() (value uint32, err error) {
	retValue, err := instance.GetProperty("AverageEncodingTime")
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

// SetFrameQuality sets the value of FrameQuality for the instance
func (instance *Win32_PerfRawData_Counters_RemoteFXGraphics) SetPropertyFrameQuality(value uint32) (err error) {
	return instance.SetProperty("FrameQuality", (value))
}

// GetFrameQuality gets the value of FrameQuality for the instance
func (instance *Win32_PerfRawData_Counters_RemoteFXGraphics) GetPropertyFrameQuality() (value uint32, err error) {
	retValue, err := instance.GetProperty("FrameQuality")
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

// SetFramesSkippedPerSecondInsufficientClientResources sets the value of FramesSkippedPerSecondInsufficientClientResources for the instance
func (instance *Win32_PerfRawData_Counters_RemoteFXGraphics) SetPropertyFramesSkippedPerSecondInsufficientClientResources(value uint32) (err error) {
	return instance.SetProperty("FramesSkippedPerSecondInsufficientClientResources", (value))
}

// GetFramesSkippedPerSecondInsufficientClientResources gets the value of FramesSkippedPerSecondInsufficientClientResources for the instance
func (instance *Win32_PerfRawData_Counters_RemoteFXGraphics) GetPropertyFramesSkippedPerSecondInsufficientClientResources() (value uint32, err error) {
	retValue, err := instance.GetProperty("FramesSkippedPerSecondInsufficientClientResources")
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

// SetFramesSkippedPerSecondInsufficientNetworkResources sets the value of FramesSkippedPerSecondInsufficientNetworkResources for the instance
func (instance *Win32_PerfRawData_Counters_RemoteFXGraphics) SetPropertyFramesSkippedPerSecondInsufficientNetworkResources(value uint32) (err error) {
	return instance.SetProperty("FramesSkippedPerSecondInsufficientNetworkResources", (value))
}

// GetFramesSkippedPerSecondInsufficientNetworkResources gets the value of FramesSkippedPerSecondInsufficientNetworkResources for the instance
func (instance *Win32_PerfRawData_Counters_RemoteFXGraphics) GetPropertyFramesSkippedPerSecondInsufficientNetworkResources() (value uint32, err error) {
	retValue, err := instance.GetProperty("FramesSkippedPerSecondInsufficientNetworkResources")
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

// SetFramesSkippedPerSecondInsufficientServerResources sets the value of FramesSkippedPerSecondInsufficientServerResources for the instance
func (instance *Win32_PerfRawData_Counters_RemoteFXGraphics) SetPropertyFramesSkippedPerSecondInsufficientServerResources(value uint32) (err error) {
	return instance.SetProperty("FramesSkippedPerSecondInsufficientServerResources", (value))
}

// GetFramesSkippedPerSecondInsufficientServerResources gets the value of FramesSkippedPerSecondInsufficientServerResources for the instance
func (instance *Win32_PerfRawData_Counters_RemoteFXGraphics) GetPropertyFramesSkippedPerSecondInsufficientServerResources() (value uint32, err error) {
	retValue, err := instance.GetProperty("FramesSkippedPerSecondInsufficientServerResources")
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

// SetGraphicsCompressionratio sets the value of GraphicsCompressionratio for the instance
func (instance *Win32_PerfRawData_Counters_RemoteFXGraphics) SetPropertyGraphicsCompressionratio(value uint32) (err error) {
	return instance.SetProperty("GraphicsCompressionratio", (value))
}

// GetGraphicsCompressionratio gets the value of GraphicsCompressionratio for the instance
func (instance *Win32_PerfRawData_Counters_RemoteFXGraphics) GetPropertyGraphicsCompressionratio() (value uint32, err error) {
	retValue, err := instance.GetProperty("GraphicsCompressionratio")
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

// SetInputFramesPerSecond sets the value of InputFramesPerSecond for the instance
func (instance *Win32_PerfRawData_Counters_RemoteFXGraphics) SetPropertyInputFramesPerSecond(value uint32) (err error) {
	return instance.SetProperty("InputFramesPerSecond", (value))
}

// GetInputFramesPerSecond gets the value of InputFramesPerSecond for the instance
func (instance *Win32_PerfRawData_Counters_RemoteFXGraphics) GetPropertyInputFramesPerSecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("InputFramesPerSecond")
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

// SetOutputFramesPerSecond sets the value of OutputFramesPerSecond for the instance
func (instance *Win32_PerfRawData_Counters_RemoteFXGraphics) SetPropertyOutputFramesPerSecond(value uint32) (err error) {
	return instance.SetProperty("OutputFramesPerSecond", (value))
}

// GetOutputFramesPerSecond gets the value of OutputFramesPerSecond for the instance
func (instance *Win32_PerfRawData_Counters_RemoteFXGraphics) GetPropertyOutputFramesPerSecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("OutputFramesPerSecond")
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

// SetSourceFramesPerSecond sets the value of SourceFramesPerSecond for the instance
func (instance *Win32_PerfRawData_Counters_RemoteFXGraphics) SetPropertySourceFramesPerSecond(value uint32) (err error) {
	return instance.SetProperty("SourceFramesPerSecond", (value))
}

// GetSourceFramesPerSecond gets the value of SourceFramesPerSecond for the instance
func (instance *Win32_PerfRawData_Counters_RemoteFXGraphics) GetPropertySourceFramesPerSecond() (value uint32, err error) {
	retValue, err := instance.GetProperty("SourceFramesPerSecond")
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
