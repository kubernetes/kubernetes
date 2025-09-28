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

// Win32_FileSpecification struct
type Win32_FileSpecification struct {
	*CIM_FileSpecification

	//
	Attributes uint16

	//
	FileID string

	//
	Language string

	//
	Sequence uint16
}

func NewWin32_FileSpecificationEx1(instance *cim.WmiInstance) (newInstance *Win32_FileSpecification, err error) {
	tmp, err := NewCIM_FileSpecificationEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_FileSpecification{
		CIM_FileSpecification: tmp,
	}
	return
}

func NewWin32_FileSpecificationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_FileSpecification, err error) {
	tmp, err := NewCIM_FileSpecificationEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_FileSpecification{
		CIM_FileSpecification: tmp,
	}
	return
}

// SetAttributes sets the value of Attributes for the instance
func (instance *Win32_FileSpecification) SetPropertyAttributes(value uint16) (err error) {
	return instance.SetProperty("Attributes", (value))
}

// GetAttributes gets the value of Attributes for the instance
func (instance *Win32_FileSpecification) GetPropertyAttributes() (value uint16, err error) {
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

// SetFileID sets the value of FileID for the instance
func (instance *Win32_FileSpecification) SetPropertyFileID(value string) (err error) {
	return instance.SetProperty("FileID", (value))
}

// GetFileID gets the value of FileID for the instance
func (instance *Win32_FileSpecification) GetPropertyFileID() (value string, err error) {
	retValue, err := instance.GetProperty("FileID")
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

// SetLanguage sets the value of Language for the instance
func (instance *Win32_FileSpecification) SetPropertyLanguage(value string) (err error) {
	return instance.SetProperty("Language", (value))
}

// GetLanguage gets the value of Language for the instance
func (instance *Win32_FileSpecification) GetPropertyLanguage() (value string, err error) {
	retValue, err := instance.GetProperty("Language")
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
func (instance *Win32_FileSpecification) SetPropertySequence(value uint16) (err error) {
	return instance.SetProperty("Sequence", (value))
}

// GetSequence gets the value of Sequence for the instance
func (instance *Win32_FileSpecification) GetPropertySequence() (value uint16, err error) {
	retValue, err := instance.GetProperty("Sequence")
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
