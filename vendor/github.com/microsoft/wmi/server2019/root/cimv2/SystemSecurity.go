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
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
)

// __SystemSecurity struct
type __SystemSecurity struct {
	*cim.WmiInstance
}

func New__SystemSecurityEx1(instance *cim.WmiInstance) (newInstance *__SystemSecurity, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &__SystemSecurity{
		WmiInstance: tmp,
	}
	return
}

func New__SystemSecurityEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *__SystemSecurity, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &__SystemSecurity{
		WmiInstance: tmp,
	}
	return
}

//

// <param name="ReturnValue" type="uint32 "></param>
// <param name="SD" type="uint8 []"></param>
func (instance *__SystemSecurity) GetSD( /* OUT */ SD []uint8) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetSD")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="Descriptor" type="__SecurityDescriptor "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *__SystemSecurity) GetSecurityDescriptor( /* OUT */ Descriptor __SecurityDescriptor) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetSecurityDescriptor")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
// <param name="ul" type="__NTLMUser9X []"></param>
func (instance *__SystemSecurity) Get9XUserList( /* OUT */ ul []__NTLMUser9X) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("Get9XUserList")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="SD" type="uint8 []"></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *__SystemSecurity) SetSD( /* IN */ SD []uint8) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetSD", SD)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="Descriptor" type="__SecurityDescriptor "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *__SystemSecurity) SetSecurityDescriptor( /* IN */ Descriptor __SecurityDescriptor) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetSecurityDescriptor", Descriptor)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ul" type="__NTLMUser9X []"></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *__SystemSecurity) Set9XUserList( /* IN */ ul []__NTLMUser9X) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Set9XUserList", ul)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
// <param name="rights" type="int32 "></param>
func (instance *__SystemSecurity) GetCallerAccessRights( /* OUT */ rights int32) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetCallerAccessRights")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}
