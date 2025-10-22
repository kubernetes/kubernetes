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

// Win32_ShareToDirectory struct
type Win32_ShareToDirectory struct {
	*cim.WmiInstance

	//
	Share Win32_Share

	//
	SharedElement CIM_Directory
}

func NewWin32_ShareToDirectoryEx1(instance *cim.WmiInstance) (newInstance *Win32_ShareToDirectory, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_ShareToDirectory{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_ShareToDirectoryEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ShareToDirectory, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ShareToDirectory{
		WmiInstance: tmp,
	}
	return
}

// SetShare sets the value of Share for the instance
func (instance *Win32_ShareToDirectory) SetPropertyShare(value Win32_Share) (err error) {
	return instance.SetProperty("Share", (value))
}

// GetShare gets the value of Share for the instance
func (instance *Win32_ShareToDirectory) GetPropertyShare() (value Win32_Share, err error) {
	retValue, err := instance.GetProperty("Share")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_Share)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_Share is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_Share(valuetmp)

	return
}

// SetSharedElement sets the value of SharedElement for the instance
func (instance *Win32_ShareToDirectory) SetPropertySharedElement(value CIM_Directory) (err error) {
	return instance.SetProperty("SharedElement", (value))
}

// GetSharedElement gets the value of SharedElement for the instance
func (instance *Win32_ShareToDirectory) GetPropertySharedElement() (value CIM_Directory, err error) {
	retValue, err := instance.GetProperty("SharedElement")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_Directory)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_Directory is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_Directory(valuetmp)

	return
}
