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

// CIM_SoftwareElementVersionCheck struct
type CIM_SoftwareElementVersionCheck struct {
	*CIM_Check

	//
	LowerSoftwareElementVersion string

	//
	SoftwareElementName string

	//
	SoftwareElementStateDesired uint16

	//
	TargetOperatingSystemDesired uint16

	//
	UpperSoftwareElementVersion string
}

func NewCIM_SoftwareElementVersionCheckEx1(instance *cim.WmiInstance) (newInstance *CIM_SoftwareElementVersionCheck, err error) {
	tmp, err := NewCIM_CheckEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_SoftwareElementVersionCheck{
		CIM_Check: tmp,
	}
	return
}

func NewCIM_SoftwareElementVersionCheckEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_SoftwareElementVersionCheck, err error) {
	tmp, err := NewCIM_CheckEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_SoftwareElementVersionCheck{
		CIM_Check: tmp,
	}
	return
}

// SetLowerSoftwareElementVersion sets the value of LowerSoftwareElementVersion for the instance
func (instance *CIM_SoftwareElementVersionCheck) SetPropertyLowerSoftwareElementVersion(value string) (err error) {
	return instance.SetProperty("LowerSoftwareElementVersion", (value))
}

// GetLowerSoftwareElementVersion gets the value of LowerSoftwareElementVersion for the instance
func (instance *CIM_SoftwareElementVersionCheck) GetPropertyLowerSoftwareElementVersion() (value string, err error) {
	retValue, err := instance.GetProperty("LowerSoftwareElementVersion")
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

// SetSoftwareElementName sets the value of SoftwareElementName for the instance
func (instance *CIM_SoftwareElementVersionCheck) SetPropertySoftwareElementName(value string) (err error) {
	return instance.SetProperty("SoftwareElementName", (value))
}

// GetSoftwareElementName gets the value of SoftwareElementName for the instance
func (instance *CIM_SoftwareElementVersionCheck) GetPropertySoftwareElementName() (value string, err error) {
	retValue, err := instance.GetProperty("SoftwareElementName")
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

// SetSoftwareElementStateDesired sets the value of SoftwareElementStateDesired for the instance
func (instance *CIM_SoftwareElementVersionCheck) SetPropertySoftwareElementStateDesired(value uint16) (err error) {
	return instance.SetProperty("SoftwareElementStateDesired", (value))
}

// GetSoftwareElementStateDesired gets the value of SoftwareElementStateDesired for the instance
func (instance *CIM_SoftwareElementVersionCheck) GetPropertySoftwareElementStateDesired() (value uint16, err error) {
	retValue, err := instance.GetProperty("SoftwareElementStateDesired")
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

// SetTargetOperatingSystemDesired sets the value of TargetOperatingSystemDesired for the instance
func (instance *CIM_SoftwareElementVersionCheck) SetPropertyTargetOperatingSystemDesired(value uint16) (err error) {
	return instance.SetProperty("TargetOperatingSystemDesired", (value))
}

// GetTargetOperatingSystemDesired gets the value of TargetOperatingSystemDesired for the instance
func (instance *CIM_SoftwareElementVersionCheck) GetPropertyTargetOperatingSystemDesired() (value uint16, err error) {
	retValue, err := instance.GetProperty("TargetOperatingSystemDesired")
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

// SetUpperSoftwareElementVersion sets the value of UpperSoftwareElementVersion for the instance
func (instance *CIM_SoftwareElementVersionCheck) SetPropertyUpperSoftwareElementVersion(value string) (err error) {
	return instance.SetProperty("UpperSoftwareElementVersion", (value))
}

// GetUpperSoftwareElementVersion gets the value of UpperSoftwareElementVersion for the instance
func (instance *CIM_SoftwareElementVersionCheck) GetPropertyUpperSoftwareElementVersion() (value string, err error) {
	retValue, err := instance.GetProperty("UpperSoftwareElementVersion")
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
