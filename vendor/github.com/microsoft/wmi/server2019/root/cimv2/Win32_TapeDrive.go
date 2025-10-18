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

// Win32_TapeDrive struct
type Win32_TapeDrive struct {
	*CIM_TapeDrive

	//
	Compression uint32

	//
	ECC uint32

	//
	FeaturesHigh uint32

	//
	FeaturesLow uint32

	//
	Id string

	//
	Manufacturer string

	//
	MediaType string

	//
	ReportSetMarks uint32
}

func NewWin32_TapeDriveEx1(instance *cim.WmiInstance) (newInstance *Win32_TapeDrive, err error) {
	tmp, err := NewCIM_TapeDriveEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_TapeDrive{
		CIM_TapeDrive: tmp,
	}
	return
}

func NewWin32_TapeDriveEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_TapeDrive, err error) {
	tmp, err := NewCIM_TapeDriveEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_TapeDrive{
		CIM_TapeDrive: tmp,
	}
	return
}

// SetCompression sets the value of Compression for the instance
func (instance *Win32_TapeDrive) SetPropertyCompression(value uint32) (err error) {
	return instance.SetProperty("Compression", (value))
}

// GetCompression gets the value of Compression for the instance
func (instance *Win32_TapeDrive) GetPropertyCompression() (value uint32, err error) {
	retValue, err := instance.GetProperty("Compression")
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

// SetECC sets the value of ECC for the instance
func (instance *Win32_TapeDrive) SetPropertyECC(value uint32) (err error) {
	return instance.SetProperty("ECC", (value))
}

// GetECC gets the value of ECC for the instance
func (instance *Win32_TapeDrive) GetPropertyECC() (value uint32, err error) {
	retValue, err := instance.GetProperty("ECC")
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

// SetFeaturesHigh sets the value of FeaturesHigh for the instance
func (instance *Win32_TapeDrive) SetPropertyFeaturesHigh(value uint32) (err error) {
	return instance.SetProperty("FeaturesHigh", (value))
}

// GetFeaturesHigh gets the value of FeaturesHigh for the instance
func (instance *Win32_TapeDrive) GetPropertyFeaturesHigh() (value uint32, err error) {
	retValue, err := instance.GetProperty("FeaturesHigh")
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

// SetFeaturesLow sets the value of FeaturesLow for the instance
func (instance *Win32_TapeDrive) SetPropertyFeaturesLow(value uint32) (err error) {
	return instance.SetProperty("FeaturesLow", (value))
}

// GetFeaturesLow gets the value of FeaturesLow for the instance
func (instance *Win32_TapeDrive) GetPropertyFeaturesLow() (value uint32, err error) {
	retValue, err := instance.GetProperty("FeaturesLow")
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

// SetId sets the value of Id for the instance
func (instance *Win32_TapeDrive) SetPropertyId(value string) (err error) {
	return instance.SetProperty("Id", (value))
}

// GetId gets the value of Id for the instance
func (instance *Win32_TapeDrive) GetPropertyId() (value string, err error) {
	retValue, err := instance.GetProperty("Id")
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

// SetManufacturer sets the value of Manufacturer for the instance
func (instance *Win32_TapeDrive) SetPropertyManufacturer(value string) (err error) {
	return instance.SetProperty("Manufacturer", (value))
}

// GetManufacturer gets the value of Manufacturer for the instance
func (instance *Win32_TapeDrive) GetPropertyManufacturer() (value string, err error) {
	retValue, err := instance.GetProperty("Manufacturer")
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

// SetMediaType sets the value of MediaType for the instance
func (instance *Win32_TapeDrive) SetPropertyMediaType(value string) (err error) {
	return instance.SetProperty("MediaType", (value))
}

// GetMediaType gets the value of MediaType for the instance
func (instance *Win32_TapeDrive) GetPropertyMediaType() (value string, err error) {
	retValue, err := instance.GetProperty("MediaType")
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

// SetReportSetMarks sets the value of ReportSetMarks for the instance
func (instance *Win32_TapeDrive) SetPropertyReportSetMarks(value uint32) (err error) {
	return instance.SetProperty("ReportSetMarks", (value))
}

// GetReportSetMarks gets the value of ReportSetMarks for the instance
func (instance *Win32_TapeDrive) GetPropertyReportSetMarks() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReportSetMarks")
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
