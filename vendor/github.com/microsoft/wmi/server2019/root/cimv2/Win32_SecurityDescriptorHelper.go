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

// Win32_SecurityDescriptorHelper struct
type Win32_SecurityDescriptorHelper struct {
	*cim.WmiInstance
}

func NewWin32_SecurityDescriptorHelperEx1(instance *cim.WmiInstance) (newInstance *Win32_SecurityDescriptorHelper, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_SecurityDescriptorHelper{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_SecurityDescriptorHelperEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_SecurityDescriptorHelper, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_SecurityDescriptorHelper{
		WmiInstance: tmp,
	}
	return
}

//

// <param name="Descriptor" type="__SecurityDescriptor "></param>

// <param name="ReturnValue" type="uint32 "></param>
// <param name="SDDL" type="string "></param>
func (instance *Win32_SecurityDescriptorHelper) Win32SDToSDDL( /* IN */ Descriptor __SecurityDescriptor,
	/* OUT */ SDDL string) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("Win32SDToSDDL", Descriptor)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="Descriptor" type="__SecurityDescriptor "></param>

// <param name="BinarySD" type="uint8 []"></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_SecurityDescriptorHelper) Win32SDToBinarySD( /* IN */ Descriptor __SecurityDescriptor,
	/* OUT */ BinarySD []uint8) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("Win32SDToBinarySD", Descriptor)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="SDDL" type="string "></param>

// <param name="Descriptor" type="__SecurityDescriptor "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_SecurityDescriptorHelper) SDDLToWin32SD( /* IN */ SDDL string,
	/* OUT */ Descriptor __SecurityDescriptor) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("SDDLToWin32SD", SDDL)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="SDDL" type="string "></param>

// <param name="BinarySD" type="uint8 []"></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_SecurityDescriptorHelper) SDDLToBinarySD( /* IN */ SDDL string,
	/* OUT */ BinarySD []uint8) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("SDDLToBinarySD", SDDL)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="BinarySD" type="uint8 []"></param>

// <param name="Descriptor" type="__SecurityDescriptor "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_SecurityDescriptorHelper) BinarySDToWin32SD( /* IN */ BinarySD []uint8,
	/* OUT */ Descriptor __SecurityDescriptor) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("BinarySDToWin32SD", BinarySD)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="BinarySD" type="uint8 []"></param>

// <param name="ReturnValue" type="uint32 "></param>
// <param name="SDDL" type="string "></param>
func (instance *Win32_SecurityDescriptorHelper) BinarySDToSDDL( /* IN */ BinarySD []uint8,
	/* OUT */ SDDL string) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("BinarySDToSDDL", BinarySD)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}
