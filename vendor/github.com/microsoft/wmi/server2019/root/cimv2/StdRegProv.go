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

// StdRegProv struct
type StdRegProv struct {
	*cim.WmiInstance
}

func NewStdRegProvEx1(instance *cim.WmiInstance) (newInstance *StdRegProv, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &StdRegProv{
		WmiInstance: tmp,
	}
	return
}

func NewStdRegProvEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *StdRegProv, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &StdRegProv{
		WmiInstance: tmp,
	}
	return
}

//

// <param name="hDefKey" type="uint32 "></param>
// <param name="sSubKeyName" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *StdRegProv) CreateKey( /* IN */ hDefKey uint32,
	/* IN */ sSubKeyName string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("CreateKey", hDefKey, sSubKeyName)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="hDefKey" type="uint32 "></param>
// <param name="sSubKeyName" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *StdRegProv) DeleteKey( /* IN */ hDefKey uint32,
	/* IN */ sSubKeyName string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("DeleteKey", hDefKey, sSubKeyName)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="hDefKey" type="uint32 "></param>
// <param name="sSubKeyName" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
// <param name="sNames" type="string []"></param>
func (instance *StdRegProv) EnumKey( /* IN */ hDefKey uint32,
	/* IN */ sSubKeyName string,
	/* OUT */ sNames []string) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("EnumKey", hDefKey, sSubKeyName)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="hDefKey" type="uint32 "></param>
// <param name="sSubKeyName" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
// <param name="sNames" type="string []"></param>
// <param name="Types" type="int32 []"></param>
func (instance *StdRegProv) EnumValues( /* IN */ hDefKey uint32,
	/* IN */ sSubKeyName string,
	/* OUT */ sNames []string,
	/* OUT */ Types []int32) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("EnumValues", hDefKey, sSubKeyName)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="hDefKey" type="uint32 "></param>
// <param name="sSubKeyName" type="string "></param>
// <param name="sValueName" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *StdRegProv) DeleteValue( /* IN */ hDefKey uint32,
	/* IN */ sSubKeyName string,
	/* IN */ sValueName string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("DeleteValue", hDefKey, sSubKeyName, sValueName)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="hDefKey" type="uint32 "></param>
// <param name="sSubKeyName" type="string "></param>
// <param name="sValueName" type="string "></param>
// <param name="uValue" type="uint32 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *StdRegProv) SetDWORDValue( /* IN */ hDefKey uint32,
	/* IN */ sSubKeyName string,
	/* IN */ sValueName string,
	/* IN */ uValue uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetDWORDValue", hDefKey, sSubKeyName, sValueName, uValue)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="hDefKey" type="uint32 "></param>
// <param name="sSubKeyName" type="string "></param>
// <param name="sValueName" type="string "></param>
// <param name="uValue" type="uint64 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *StdRegProv) SetQWORDValue( /* IN */ hDefKey uint32,
	/* IN */ sSubKeyName string,
	/* IN */ sValueName string,
	/* IN */ uValue uint64) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetQWORDValue", hDefKey, sSubKeyName, sValueName, uValue)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="hDefKey" type="uint32 "></param>
// <param name="sSubKeyName" type="string "></param>
// <param name="sValueName" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
// <param name="uValue" type="uint32 "></param>
func (instance *StdRegProv) GetDWORDValue( /* IN */ hDefKey uint32,
	/* IN */ sSubKeyName string,
	/* IN */ sValueName string,
	/* OUT */ uValue uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetDWORDValue", hDefKey, sSubKeyName, sValueName)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="hDefKey" type="uint32 "></param>
// <param name="sSubKeyName" type="string "></param>
// <param name="sValueName" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
// <param name="uValue" type="uint64 "></param>
func (instance *StdRegProv) GetQWORDValue( /* IN */ hDefKey uint32,
	/* IN */ sSubKeyName string,
	/* IN */ sValueName string,
	/* OUT */ uValue uint64) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetQWORDValue", hDefKey, sSubKeyName, sValueName)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="hDefKey" type="uint32 "></param>
// <param name="sSubKeyName" type="string "></param>
// <param name="sValue" type="string "></param>
// <param name="sValueName" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *StdRegProv) SetStringValue( /* IN */ hDefKey uint32,
	/* IN */ sSubKeyName string,
	/* IN */ sValueName string,
	/* IN */ sValue string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetStringValue", hDefKey, sSubKeyName, sValueName, sValue)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="hDefKey" type="uint32 "></param>
// <param name="sSubKeyName" type="string "></param>
// <param name="sValueName" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
// <param name="sValue" type="string "></param>
func (instance *StdRegProv) GetStringValue( /* IN */ hDefKey uint32,
	/* IN */ sSubKeyName string,
	/* IN */ sValueName string,
	/* OUT */ sValue string) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetStringValue", hDefKey, sSubKeyName, sValueName)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="hDefKey" type="uint32 "></param>
// <param name="sSubKeyName" type="string "></param>
// <param name="sValue" type="string []"></param>
// <param name="sValueName" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *StdRegProv) SetMultiStringValue( /* IN */ hDefKey uint32,
	/* IN */ sSubKeyName string,
	/* IN */ sValueName string,
	/* IN */ sValue []string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetMultiStringValue", hDefKey, sSubKeyName, sValueName, sValue)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="hDefKey" type="uint32 "></param>
// <param name="sSubKeyName" type="string "></param>
// <param name="sValueName" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
// <param name="sValue" type="string []"></param>
func (instance *StdRegProv) GetMultiStringValue( /* IN */ hDefKey uint32,
	/* IN */ sSubKeyName string,
	/* IN */ sValueName string,
	/* OUT */ sValue []string) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetMultiStringValue", hDefKey, sSubKeyName, sValueName)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="hDefKey" type="uint32 "></param>
// <param name="sSubKeyName" type="string "></param>
// <param name="sValue" type="string "></param>
// <param name="sValueName" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *StdRegProv) SetExpandedStringValue( /* IN */ hDefKey uint32,
	/* IN */ sSubKeyName string,
	/* IN */ sValueName string,
	/* IN */ sValue string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetExpandedStringValue", hDefKey, sSubKeyName, sValueName, sValue)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="hDefKey" type="uint32 "></param>
// <param name="sSubKeyName" type="string "></param>
// <param name="sValueName" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
// <param name="sValue" type="string "></param>
func (instance *StdRegProv) GetExpandedStringValue( /* IN */ hDefKey uint32,
	/* IN */ sSubKeyName string,
	/* IN */ sValueName string,
	/* OUT */ sValue string) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetExpandedStringValue", hDefKey, sSubKeyName, sValueName)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="hDefKey" type="uint32 "></param>
// <param name="sSubKeyName" type="string "></param>
// <param name="sValueName" type="string "></param>
// <param name="uValue" type="uint8 []"></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *StdRegProv) SetBinaryValue( /* IN */ hDefKey uint32,
	/* IN */ sSubKeyName string,
	/* IN */ sValueName string,
	/* IN */ uValue []uint8) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetBinaryValue", hDefKey, sSubKeyName, sValueName, uValue)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="hDefKey" type="uint32 "></param>
// <param name="sSubKeyName" type="string "></param>
// <param name="sValueName" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
// <param name="uValue" type="uint8 []"></param>
func (instance *StdRegProv) GetBinaryValue( /* IN */ hDefKey uint32,
	/* IN */ sSubKeyName string,
	/* IN */ sValueName string,
	/* OUT */ uValue []uint8) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetBinaryValue", hDefKey, sSubKeyName, sValueName)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="hDefKey" type="uint32 "></param>
// <param name="sSubKeyName" type="string "></param>
// <param name="uRequired" type="uint32 "></param>

// <param name="bGranted" type="bool "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *StdRegProv) CheckAccess( /* IN */ hDefKey uint32,
	/* IN */ sSubKeyName string,
	/* IN */ uRequired uint32,
	/* OUT */ bGranted bool) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("CheckAccess", hDefKey, sSubKeyName, uRequired)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="Descriptor" type="__SecurityDescriptor "></param>
// <param name="hDefKey" type="uint32 "></param>
// <param name="sSubKeyName" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *StdRegProv) SetSecurityDescriptor( /* IN */ hDefKey uint32,
	/* IN */ sSubKeyName string,
	/* IN */ Descriptor __SecurityDescriptor) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetSecurityDescriptor", hDefKey, sSubKeyName, Descriptor)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="hDefKey" type="uint32 "></param>
// <param name="sSubKeyName" type="string "></param>

// <param name="Descriptor" type="__SecurityDescriptor "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *StdRegProv) GetSecurityDescriptor( /* IN */ hDefKey uint32,
	/* IN */ sSubKeyName string,
	/* OUT */ Descriptor __SecurityDescriptor) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetSecurityDescriptor", hDefKey, sSubKeyName)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}
