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

// Win32_OptionalFeature struct
type Win32_OptionalFeature struct {
	*CIM_LogicalElement

	//
	InstallState uint32
}

func NewWin32_OptionalFeatureEx1(instance *cim.WmiInstance) (newInstance *Win32_OptionalFeature, err error) {
	tmp, err := NewCIM_LogicalElementEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_OptionalFeature{
		CIM_LogicalElement: tmp,
	}
	return
}

func NewWin32_OptionalFeatureEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_OptionalFeature, err error) {
	tmp, err := NewCIM_LogicalElementEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_OptionalFeature{
		CIM_LogicalElement: tmp,
	}
	return
}

// SetInstallState sets the value of InstallState for the instance
func (instance *Win32_OptionalFeature) SetPropertyInstallState(value uint32) (err error) {
	return instance.SetProperty("InstallState", (value))
}

// GetInstallState gets the value of InstallState for the instance
func (instance *Win32_OptionalFeature) GetPropertyInstallState() (value uint32, err error) {
	retValue, err := instance.GetProperty("InstallState")
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
