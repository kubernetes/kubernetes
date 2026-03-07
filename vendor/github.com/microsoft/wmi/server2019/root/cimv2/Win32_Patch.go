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

// Win32_Patch struct
type Win32_Patch struct {
	*Win32_MSIResource

	//
	Attributes uint16

	//
	File string

	//
	PatchSize uint32

	//
	ProductCode string

	//
	Sequence int16
}

func NewWin32_PatchEx1(instance *cim.WmiInstance) (newInstance *Win32_Patch, err error) {
	tmp, err := NewWin32_MSIResourceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_Patch{
		Win32_MSIResource: tmp,
	}
	return
}

func NewWin32_PatchEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_Patch, err error) {
	tmp, err := NewWin32_MSIResourceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_Patch{
		Win32_MSIResource: tmp,
	}
	return
}

// SetAttributes sets the value of Attributes for the instance
func (instance *Win32_Patch) SetPropertyAttributes(value uint16) (err error) {
	return instance.SetProperty("Attributes", (value))
}

// GetAttributes gets the value of Attributes for the instance
func (instance *Win32_Patch) GetPropertyAttributes() (value uint16, err error) {
	retValue, err := instance.GetProperty("Attributes")
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

// SetFile sets the value of File for the instance
func (instance *Win32_Patch) SetPropertyFile(value string) (err error) {
	return instance.SetProperty("File", (value))
}

// GetFile gets the value of File for the instance
func (instance *Win32_Patch) GetPropertyFile() (value string, err error) {
	retValue, err := instance.GetProperty("File")
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

// SetPatchSize sets the value of PatchSize for the instance
func (instance *Win32_Patch) SetPropertyPatchSize(value uint32) (err error) {
	return instance.SetProperty("PatchSize", (value))
}

// GetPatchSize gets the value of PatchSize for the instance
func (instance *Win32_Patch) GetPropertyPatchSize() (value uint32, err error) {
	retValue, err := instance.GetProperty("PatchSize")
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

// SetProductCode sets the value of ProductCode for the instance
func (instance *Win32_Patch) SetPropertyProductCode(value string) (err error) {
	return instance.SetProperty("ProductCode", (value))
}

// GetProductCode gets the value of ProductCode for the instance
func (instance *Win32_Patch) GetPropertyProductCode() (value string, err error) {
	retValue, err := instance.GetProperty("ProductCode")
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

// SetSequence sets the value of Sequence for the instance
func (instance *Win32_Patch) SetPropertySequence(value int16) (err error) {
	return instance.SetProperty("Sequence", (value))
}

// GetSequence gets the value of Sequence for the instance
func (instance *Win32_Patch) GetPropertySequence() (value int16, err error) {
	retValue, err := instance.GetProperty("Sequence")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = int16(valuetmp)

	return
}
