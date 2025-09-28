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

// Win32_SystemEnclosure struct
type Win32_SystemEnclosure struct {
	*CIM_Chassis

	//
	SecurityStatus uint16

	//
	SMBIOSAssetTag string
}

func NewWin32_SystemEnclosureEx1(instance *cim.WmiInstance) (newInstance *Win32_SystemEnclosure, err error) {
	tmp, err := NewCIM_ChassisEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_SystemEnclosure{
		CIM_Chassis: tmp,
	}
	return
}

func NewWin32_SystemEnclosureEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_SystemEnclosure, err error) {
	tmp, err := NewCIM_ChassisEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_SystemEnclosure{
		CIM_Chassis: tmp,
	}
	return
}

// SetSecurityStatus sets the value of SecurityStatus for the instance
func (instance *Win32_SystemEnclosure) SetPropertySecurityStatus(value uint16) (err error) {
	return instance.SetProperty("SecurityStatus", (value))
}

// GetSecurityStatus gets the value of SecurityStatus for the instance
func (instance *Win32_SystemEnclosure) GetPropertySecurityStatus() (value uint16, err error) {
	retValue, err := instance.GetProperty("SecurityStatus")
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

// SetSMBIOSAssetTag sets the value of SMBIOSAssetTag for the instance
func (instance *Win32_SystemEnclosure) SetPropertySMBIOSAssetTag(value string) (err error) {
	return instance.SetProperty("SMBIOSAssetTag", (value))
}

// GetSMBIOSAssetTag gets the value of SMBIOSAssetTag for the instance
func (instance *Win32_SystemEnclosure) GetPropertySMBIOSAssetTag() (value string, err error) {
	retValue, err := instance.GetProperty("SMBIOSAssetTag")
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
