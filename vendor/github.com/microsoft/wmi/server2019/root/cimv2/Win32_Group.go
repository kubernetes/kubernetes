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
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
)

// Win32_Group struct
type Win32_Group struct {
	*Win32_Account
}

func NewWin32_GroupEx1(instance *cim.WmiInstance) (newInstance *Win32_Group, err error) {
	tmp, err := NewWin32_AccountEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_Group{
		Win32_Account: tmp,
	}
	return
}

func NewWin32_GroupEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_Group, err error) {
	tmp, err := NewWin32_AccountEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_Group{
		Win32_Account: tmp,
	}
	return
}

//

// <param name="Name" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_Group) Rename( /* IN */ Name string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Rename", Name)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}
