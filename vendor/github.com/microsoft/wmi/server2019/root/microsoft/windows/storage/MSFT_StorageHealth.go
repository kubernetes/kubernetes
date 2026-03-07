// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.Microsoft.Windows.Storage
//////////////////////////////////////////////
package storage

import (
	"github.com/microsoft/wmi/pkg/base/query"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
)

// MSFT_StorageHealth struct
type MSFT_StorageHealth struct {
	*MSFT_StorageObject
}

func NewMSFT_StorageHealthEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageHealth, err error) {
	tmp, err := NewMSFT_StorageObjectEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageHealth{
		MSFT_StorageObject: tmp,
	}
	return
}

func NewMSFT_StorageHealthEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageHealth, err error) {
	tmp, err := NewMSFT_StorageObjectEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageHealth{
		MSFT_StorageObject: tmp,
	}
	return
}

//

// <param name="Name" type="string "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
// <param name="StorageHealthSetting" type="MSFT_StorageHealthSetting []"></param>
func (instance *MSFT_StorageHealth) GetSetting( /* IN */ Name string,
	/* OUT */ StorageHealthSetting []MSFT_StorageHealthSetting,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetSetting", Name)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="Name" type="string "></param>
// <param name="Value" type="string "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageHealth) SetSetting( /* IN */ Name string,
	/* IN */ Value string,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("SetSetting", Name, Value)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="Name" type="string "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageHealth) RemoveSetting( /* IN */ Name string,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("RemoveSetting", Name)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="Count" type="uint32 "></param>
// <param name="TargetObject" type="MSFT_StorageObject "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="Reports" type="MSFT_StorageHealthReport []"></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageHealth) GetReport( /* IN */ TargetObject MSFT_StorageObject,
	/* IN */ Count uint32,
	/* OUT */ Reports []MSFT_StorageHealthReport,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetReport", TargetObject, Count)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="EnableMaintenanceMode" type="bool "></param>
// <param name="IgnoreDetachedVirtualDisks" type="bool "></param>
// <param name="Manufacturer" type="string "></param>
// <param name="Model" type="string "></param>
// <param name="TargetObject" type="MSFT_StorageFaultDomain "></param>
// <param name="Timeout" type="uint32 "></param>
// <param name="ValidationFlags" type="uint16 "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageHealth) Maintenance( /* IN */ TargetObject MSFT_StorageFaultDomain,
	/* IN */ EnableMaintenanceMode bool,
	/* IN */ IgnoreDetachedVirtualDisks bool,
	/* IN */ Timeout uint32,
	/* IN */ Model string,
	/* IN */ Manufacturer string,
	/* IN */ ValidationFlags uint16,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("Maintenance", TargetObject, EnableMaintenanceMode, IgnoreDetachedVirtualDisks, Timeout, Model, Manufacturer, ValidationFlags)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="TargetObject" type="MSFT_StorageObject "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageHealth) RemoveIntent( /* IN */ TargetObject MSFT_StorageObject,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("RemoveIntent", TargetObject)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}
