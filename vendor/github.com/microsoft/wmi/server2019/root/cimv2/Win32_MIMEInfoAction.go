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

// Win32_MIMEInfoAction struct
type Win32_MIMEInfoAction struct {
	*CIM_Action

	//
	CLSID string

	//
	ContentType string

	//
	Extension string
}

func NewWin32_MIMEInfoActionEx1(instance *cim.WmiInstance) (newInstance *Win32_MIMEInfoAction, err error) {
	tmp, err := NewCIM_ActionEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_MIMEInfoAction{
		CIM_Action: tmp,
	}
	return
}

func NewWin32_MIMEInfoActionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_MIMEInfoAction, err error) {
	tmp, err := NewCIM_ActionEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_MIMEInfoAction{
		CIM_Action: tmp,
	}
	return
}

// SetCLSID sets the value of CLSID for the instance
func (instance *Win32_MIMEInfoAction) SetPropertyCLSID(value string) (err error) {
	return instance.SetProperty("CLSID", (value))
}

// GetCLSID gets the value of CLSID for the instance
func (instance *Win32_MIMEInfoAction) GetPropertyCLSID() (value string, err error) {
	retValue, err := instance.GetProperty("CLSID")
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

// SetContentType sets the value of ContentType for the instance
func (instance *Win32_MIMEInfoAction) SetPropertyContentType(value string) (err error) {
	return instance.SetProperty("ContentType", (value))
}

// GetContentType gets the value of ContentType for the instance
func (instance *Win32_MIMEInfoAction) GetPropertyContentType() (value string, err error) {
	retValue, err := instance.GetProperty("ContentType")
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

// SetExtension sets the value of Extension for the instance
func (instance *Win32_MIMEInfoAction) SetPropertyExtension(value string) (err error) {
	return instance.SetProperty("Extension", (value))
}

// GetExtension gets the value of Extension for the instance
func (instance *Win32_MIMEInfoAction) GetPropertyExtension() (value string, err error) {
	retValue, err := instance.GetProperty("Extension")
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
