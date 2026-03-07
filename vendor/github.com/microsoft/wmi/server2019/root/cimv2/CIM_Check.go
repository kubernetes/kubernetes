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

// CIM_Check struct
type CIM_Check struct {
	*cim.WmiInstance

	//
	Caption string

	//
	CheckID string

	//
	CheckMode bool

	//
	Description string

	//
	Name string

	//
	SoftwareElementID string

	//
	SoftwareElementState uint16

	//
	TargetOperatingSystem uint16

	//
	Version string
}

func NewCIM_CheckEx1(instance *cim.WmiInstance) (newInstance *CIM_Check, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &CIM_Check{
		WmiInstance: tmp,
	}
	return
}

func NewCIM_CheckEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_Check, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_Check{
		WmiInstance: tmp,
	}
	return
}

// SetCaption sets the value of Caption for the instance
func (instance *CIM_Check) SetPropertyCaption(value string) (err error) {
	return instance.SetProperty("Caption", (value))
}

// GetCaption gets the value of Caption for the instance
func (instance *CIM_Check) GetPropertyCaption() (value string, err error) {
	retValue, err := instance.GetProperty("Caption")
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

// SetCheckID sets the value of CheckID for the instance
func (instance *CIM_Check) SetPropertyCheckID(value string) (err error) {
	return instance.SetProperty("CheckID", (value))
}

// GetCheckID gets the value of CheckID for the instance
func (instance *CIM_Check) GetPropertyCheckID() (value string, err error) {
	retValue, err := instance.GetProperty("CheckID")
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

// SetCheckMode sets the value of CheckMode for the instance
func (instance *CIM_Check) SetPropertyCheckMode(value bool) (err error) {
	return instance.SetProperty("CheckMode", (value))
}

// GetCheckMode gets the value of CheckMode for the instance
func (instance *CIM_Check) GetPropertyCheckMode() (value bool, err error) {
	retValue, err := instance.GetProperty("CheckMode")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetDescription sets the value of Description for the instance
func (instance *CIM_Check) SetPropertyDescription(value string) (err error) {
	return instance.SetProperty("Description", (value))
}

// GetDescription gets the value of Description for the instance
func (instance *CIM_Check) GetPropertyDescription() (value string, err error) {
	retValue, err := instance.GetProperty("Description")
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

// SetName sets the value of Name for the instance
func (instance *CIM_Check) SetPropertyName(value string) (err error) {
	return instance.SetProperty("Name", (value))
}

// GetName gets the value of Name for the instance
func (instance *CIM_Check) GetPropertyName() (value string, err error) {
	retValue, err := instance.GetProperty("Name")
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

// SetSoftwareElementID sets the value of SoftwareElementID for the instance
func (instance *CIM_Check) SetPropertySoftwareElementID(value string) (err error) {
	return instance.SetProperty("SoftwareElementID", (value))
}

// GetSoftwareElementID gets the value of SoftwareElementID for the instance
func (instance *CIM_Check) GetPropertySoftwareElementID() (value string, err error) {
	retValue, err := instance.GetProperty("SoftwareElementID")
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

// SetSoftwareElementState sets the value of SoftwareElementState for the instance
func (instance *CIM_Check) SetPropertySoftwareElementState(value uint16) (err error) {
	return instance.SetProperty("SoftwareElementState", (value))
}

// GetSoftwareElementState gets the value of SoftwareElementState for the instance
func (instance *CIM_Check) GetPropertySoftwareElementState() (value uint16, err error) {
	retValue, err := instance.GetProperty("SoftwareElementState")
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

// SetTargetOperatingSystem sets the value of TargetOperatingSystem for the instance
func (instance *CIM_Check) SetPropertyTargetOperatingSystem(value uint16) (err error) {
	return instance.SetProperty("TargetOperatingSystem", (value))
}

// GetTargetOperatingSystem gets the value of TargetOperatingSystem for the instance
func (instance *CIM_Check) GetPropertyTargetOperatingSystem() (value uint16, err error) {
	retValue, err := instance.GetProperty("TargetOperatingSystem")
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

// SetVersion sets the value of Version for the instance
func (instance *CIM_Check) SetPropertyVersion(value string) (err error) {
	return instance.SetProperty("Version", (value))
}

// GetVersion gets the value of Version for the instance
func (instance *CIM_Check) GetPropertyVersion() (value string, err error) {
	retValue, err := instance.GetProperty("Version")
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

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *CIM_Check) Invoke() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Invoke")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}
