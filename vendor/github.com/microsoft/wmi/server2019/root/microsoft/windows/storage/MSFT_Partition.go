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
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// MSFT_Partition struct
type MSFT_Partition struct {
	*MSFT_StorageObject

	//
	AccessPaths []string

	//
	DiskId string

	//
	DiskNumber uint32

	//
	DriveLetter byte

	//
	GptType string

	//
	Guid string

	//
	IsActive bool

	//
	IsBoot bool

	//
	IsDAX bool

	//
	IsHidden bool

	//
	IsOffline bool

	//
	IsReadOnly bool

	//
	IsShadowCopy bool

	//
	IsSystem bool

	//
	MbrType uint16

	//
	NoDefaultDriveLetter bool

	//
	Offset uint64

	//
	OperationalStatus uint16

	//
	PartitionNumber uint32

	//
	Size uint64

	//
	TransitionState uint16
}

func NewMSFT_PartitionEx1(instance *cim.WmiInstance) (newInstance *MSFT_Partition, err error) {
	tmp, err := NewMSFT_StorageObjectEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_Partition{
		MSFT_StorageObject: tmp,
	}
	return
}

func NewMSFT_PartitionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_Partition, err error) {
	tmp, err := NewMSFT_StorageObjectEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_Partition{
		MSFT_StorageObject: tmp,
	}
	return
}

// SetAccessPaths sets the value of AccessPaths for the instance
func (instance *MSFT_Partition) SetPropertyAccessPaths(value []string) (err error) {
	return instance.SetProperty("AccessPaths", (value))
}

// GetAccessPaths gets the value of AccessPaths for the instance
func (instance *MSFT_Partition) GetPropertyAccessPaths() (value []string, err error) {
	retValue, err := instance.GetProperty("AccessPaths")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetDiskId sets the value of DiskId for the instance
func (instance *MSFT_Partition) SetPropertyDiskId(value string) (err error) {
	return instance.SetProperty("DiskId", (value))
}

// GetDiskId gets the value of DiskId for the instance
func (instance *MSFT_Partition) GetPropertyDiskId() (value string, err error) {
	retValue, err := instance.GetProperty("DiskId")
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

// SetDiskNumber sets the value of DiskNumber for the instance
func (instance *MSFT_Partition) SetPropertyDiskNumber(value uint32) (err error) {
	return instance.SetProperty("DiskNumber", (value))
}

// GetDiskNumber gets the value of DiskNumber for the instance
func (instance *MSFT_Partition) GetPropertyDiskNumber() (value uint32, err error) {
	retValue, err := instance.GetProperty("DiskNumber")
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

// SetDriveLetter sets the value of DriveLetter for the instance
func (instance *MSFT_Partition) SetPropertyDriveLetter(value byte) (err error) {
	return instance.SetProperty("DriveLetter", (value))
}

// GetDriveLetter gets the value of DriveLetter for the instance
func (instance *MSFT_Partition) GetPropertyDriveLetter() (value byte, err error) {
	retValue, err := instance.GetProperty("DriveLetter")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(byte)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " byte is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = byte(valuetmp)

	return
}

// SetGptType sets the value of GptType for the instance
func (instance *MSFT_Partition) SetPropertyGptType(value string) (err error) {
	return instance.SetProperty("GptType", (value))
}

// GetGptType gets the value of GptType for the instance
func (instance *MSFT_Partition) GetPropertyGptType() (value string, err error) {
	retValue, err := instance.GetProperty("GptType")
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

// SetGuid sets the value of Guid for the instance
func (instance *MSFT_Partition) SetPropertyGuid(value string) (err error) {
	return instance.SetProperty("Guid", (value))
}

// GetGuid gets the value of Guid for the instance
func (instance *MSFT_Partition) GetPropertyGuid() (value string, err error) {
	retValue, err := instance.GetProperty("Guid")
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

// SetIsActive sets the value of IsActive for the instance
func (instance *MSFT_Partition) SetPropertyIsActive(value bool) (err error) {
	return instance.SetProperty("IsActive", (value))
}

// GetIsActive gets the value of IsActive for the instance
func (instance *MSFT_Partition) GetPropertyIsActive() (value bool, err error) {
	retValue, err := instance.GetProperty("IsActive")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetIsBoot sets the value of IsBoot for the instance
func (instance *MSFT_Partition) SetPropertyIsBoot(value bool) (err error) {
	return instance.SetProperty("IsBoot", (value))
}

// GetIsBoot gets the value of IsBoot for the instance
func (instance *MSFT_Partition) GetPropertyIsBoot() (value bool, err error) {
	retValue, err := instance.GetProperty("IsBoot")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetIsDAX sets the value of IsDAX for the instance
func (instance *MSFT_Partition) SetPropertyIsDAX(value bool) (err error) {
	return instance.SetProperty("IsDAX", (value))
}

// GetIsDAX gets the value of IsDAX for the instance
func (instance *MSFT_Partition) GetPropertyIsDAX() (value bool, err error) {
	retValue, err := instance.GetProperty("IsDAX")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetIsHidden sets the value of IsHidden for the instance
func (instance *MSFT_Partition) SetPropertyIsHidden(value bool) (err error) {
	return instance.SetProperty("IsHidden", (value))
}

// GetIsHidden gets the value of IsHidden for the instance
func (instance *MSFT_Partition) GetPropertyIsHidden() (value bool, err error) {
	retValue, err := instance.GetProperty("IsHidden")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetIsOffline sets the value of IsOffline for the instance
func (instance *MSFT_Partition) SetPropertyIsOffline(value bool) (err error) {
	return instance.SetProperty("IsOffline", (value))
}

// GetIsOffline gets the value of IsOffline for the instance
func (instance *MSFT_Partition) GetPropertyIsOffline() (value bool, err error) {
	retValue, err := instance.GetProperty("IsOffline")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetIsReadOnly sets the value of IsReadOnly for the instance
func (instance *MSFT_Partition) SetPropertyIsReadOnly(value bool) (err error) {
	return instance.SetProperty("IsReadOnly", (value))
}

// GetIsReadOnly gets the value of IsReadOnly for the instance
func (instance *MSFT_Partition) GetPropertyIsReadOnly() (value bool, err error) {
	retValue, err := instance.GetProperty("IsReadOnly")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetIsShadowCopy sets the value of IsShadowCopy for the instance
func (instance *MSFT_Partition) SetPropertyIsShadowCopy(value bool) (err error) {
	return instance.SetProperty("IsShadowCopy", (value))
}

// GetIsShadowCopy gets the value of IsShadowCopy for the instance
func (instance *MSFT_Partition) GetPropertyIsShadowCopy() (value bool, err error) {
	retValue, err := instance.GetProperty("IsShadowCopy")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetIsSystem sets the value of IsSystem for the instance
func (instance *MSFT_Partition) SetPropertyIsSystem(value bool) (err error) {
	return instance.SetProperty("IsSystem", (value))
}

// GetIsSystem gets the value of IsSystem for the instance
func (instance *MSFT_Partition) GetPropertyIsSystem() (value bool, err error) {
	retValue, err := instance.GetProperty("IsSystem")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetMbrType sets the value of MbrType for the instance
func (instance *MSFT_Partition) SetPropertyMbrType(value uint16) (err error) {
	return instance.SetProperty("MbrType", (value))
}

// GetMbrType gets the value of MbrType for the instance
func (instance *MSFT_Partition) GetPropertyMbrType() (value uint16, err error) {
	retValue, err := instance.GetProperty("MbrType")
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

// SetNoDefaultDriveLetter sets the value of NoDefaultDriveLetter for the instance
func (instance *MSFT_Partition) SetPropertyNoDefaultDriveLetter(value bool) (err error) {
	return instance.SetProperty("NoDefaultDriveLetter", (value))
}

// GetNoDefaultDriveLetter gets the value of NoDefaultDriveLetter for the instance
func (instance *MSFT_Partition) GetPropertyNoDefaultDriveLetter() (value bool, err error) {
	retValue, err := instance.GetProperty("NoDefaultDriveLetter")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetOffset sets the value of Offset for the instance
func (instance *MSFT_Partition) SetPropertyOffset(value uint64) (err error) {
	return instance.SetProperty("Offset", (value))
}

// GetOffset gets the value of Offset for the instance
func (instance *MSFT_Partition) GetPropertyOffset() (value uint64, err error) {
	retValue, err := instance.GetProperty("Offset")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetOperationalStatus sets the value of OperationalStatus for the instance
func (instance *MSFT_Partition) SetPropertyOperationalStatus(value uint16) (err error) {
	return instance.SetProperty("OperationalStatus", (value))
}

// GetOperationalStatus gets the value of OperationalStatus for the instance
func (instance *MSFT_Partition) GetPropertyOperationalStatus() (value uint16, err error) {
	retValue, err := instance.GetProperty("OperationalStatus")
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

// SetPartitionNumber sets the value of PartitionNumber for the instance
func (instance *MSFT_Partition) SetPropertyPartitionNumber(value uint32) (err error) {
	return instance.SetProperty("PartitionNumber", (value))
}

// GetPartitionNumber gets the value of PartitionNumber for the instance
func (instance *MSFT_Partition) GetPropertyPartitionNumber() (value uint32, err error) {
	retValue, err := instance.GetProperty("PartitionNumber")
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

// SetSize sets the value of Size for the instance
func (instance *MSFT_Partition) SetPropertySize(value uint64) (err error) {
	return instance.SetProperty("Size", (value))
}

// GetSize gets the value of Size for the instance
func (instance *MSFT_Partition) GetPropertySize() (value uint64, err error) {
	retValue, err := instance.GetProperty("Size")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetTransitionState sets the value of TransitionState for the instance
func (instance *MSFT_Partition) SetPropertyTransitionState(value uint16) (err error) {
	return instance.SetProperty("TransitionState", (value))
}

// GetTransitionState gets the value of TransitionState for the instance
func (instance *MSFT_Partition) GetPropertyTransitionState() (value uint16, err error) {
	retValue, err := instance.GetProperty("TransitionState")
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

//

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_Partition) DeleteObject( /* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("DeleteObject")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="AccessPaths" type="string []"></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_Partition) GetAccessPaths( /* OUT */ AccessPaths []string,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetAccessPaths")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="AccessPath" type="string "></param>
// <param name="AssignDriveLetter" type="bool "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_Partition) AddAccessPath( /* IN */ AccessPath string,
	/* IN */ AssignDriveLetter bool,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("AddAccessPath", AccessPath, AssignDriveLetter)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="AccessPath" type="string "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_Partition) RemoveAccessPath( /* IN */ AccessPath string,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("RemoveAccessPath", AccessPath)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="Size" type="uint64 "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_Partition) Resize( /* IN */ Size uint64,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("Resize", Size)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
// <param name="SizeMax" type="uint64 "></param>
// <param name="SizeMin" type="uint64 "></param>
func (instance *MSFT_Partition) GetSupportedSize( /* OUT */ SizeMin uint64,
	/* OUT */ SizeMax uint64,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetSupportedSize")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_Partition) Online( /* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("Online")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_Partition) Offline( /* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("Offline")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="GptType" type="string "></param>
// <param name="IsActive" type="bool "></param>
// <param name="IsDAX" type="bool "></param>
// <param name="IsHidden" type="bool "></param>
// <param name="IsReadOnly" type="bool "></param>
// <param name="IsShadowCopy" type="bool "></param>
// <param name="MbrType" type="uint16 "></param>
// <param name="NoDefaultDriveLetter" type="bool "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_Partition) SetAttributes( /* IN */ IsReadOnly bool,
	/* IN */ NoDefaultDriveLetter bool,
	/* IN */ IsActive bool,
	/* IN */ IsHidden bool,
	/* IN */ IsShadowCopy bool,
	/* IN */ IsDAX bool,
	/* IN */ MbrType uint16,
	/* IN */ GptType string,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("SetAttributes", IsReadOnly, NoDefaultDriveLetter, IsActive, IsHidden, IsShadowCopy, IsDAX, MbrType, GptType)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}
