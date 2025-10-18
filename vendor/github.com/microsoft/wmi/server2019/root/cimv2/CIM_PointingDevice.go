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

// CIM_PointingDevice struct
type CIM_PointingDevice struct {
	*CIM_UserDevice

	//
	Handedness uint16

	//
	NumberOfButtons uint8

	//
	PointingType uint16

	//
	Resolution uint32
}

func NewCIM_PointingDeviceEx1(instance *cim.WmiInstance) (newInstance *CIM_PointingDevice, err error) {
	tmp, err := NewCIM_UserDeviceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_PointingDevice{
		CIM_UserDevice: tmp,
	}
	return
}

func NewCIM_PointingDeviceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_PointingDevice, err error) {
	tmp, err := NewCIM_UserDeviceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_PointingDevice{
		CIM_UserDevice: tmp,
	}
	return
}

// SetHandedness sets the value of Handedness for the instance
func (instance *CIM_PointingDevice) SetPropertyHandedness(value uint16) (err error) {
	return instance.SetProperty("Handedness", (value))
}

// GetHandedness gets the value of Handedness for the instance
func (instance *CIM_PointingDevice) GetPropertyHandedness() (value uint16, err error) {
	retValue, err := instance.GetProperty("Handedness")
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

// SetNumberOfButtons sets the value of NumberOfButtons for the instance
func (instance *CIM_PointingDevice) SetPropertyNumberOfButtons(value uint8) (err error) {
	return instance.SetProperty("NumberOfButtons", (value))
}

// GetNumberOfButtons gets the value of NumberOfButtons for the instance
func (instance *CIM_PointingDevice) GetPropertyNumberOfButtons() (value uint8, err error) {
	retValue, err := instance.GetProperty("NumberOfButtons")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint8)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint8 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint8(valuetmp)

	return
}

// SetPointingType sets the value of PointingType for the instance
func (instance *CIM_PointingDevice) SetPropertyPointingType(value uint16) (err error) {
	return instance.SetProperty("PointingType", (value))
}

// GetPointingType gets the value of PointingType for the instance
func (instance *CIM_PointingDevice) GetPropertyPointingType() (value uint16, err error) {
	retValue, err := instance.GetProperty("PointingType")
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

// SetResolution sets the value of Resolution for the instance
func (instance *CIM_PointingDevice) SetPropertyResolution(value uint32) (err error) {
	return instance.SetProperty("Resolution", (value))
}

// GetResolution gets the value of Resolution for the instance
func (instance *CIM_PointingDevice) GetPropertyResolution() (value uint32, err error) {
	retValue, err := instance.GetProperty("Resolution")
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
