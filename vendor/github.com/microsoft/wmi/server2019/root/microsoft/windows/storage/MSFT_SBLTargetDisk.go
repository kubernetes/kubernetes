// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.Microsoft.Windows.Storage
//////////////////////////////////////////////
package storage

import (
	"github.com/microsoft/wmi/pkg/base/instance"
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// MSFT_SBLTargetDisk struct
type MSFT_SBLTargetDisk struct {
	*cim.WmiInstance

	//
	CacheMode uint32

	//
	CurrentUsage uint32

	//
	DesiredUsage uint32

	//
	DeviceNumber uint32

	//
	Identifier string

	//
	IsFlash bool

	//
	IsSblCacheDevice bool

	//
	LastStateChangeTime string

	//
	ReadMediaErrorCount uint64

	//
	ReadTotalErrorCount uint64

	//
	SblAttributes uint32

	//
	State uint32

	//
	WriteMediaErrorCount uint64

	//
	WriteTotalErrorCount uint64
}

func NewMSFT_SBLTargetDiskEx1(instance *cim.WmiInstance) (newInstance *MSFT_SBLTargetDisk, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_SBLTargetDisk{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_SBLTargetDiskEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_SBLTargetDisk, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_SBLTargetDisk{
		WmiInstance: tmp,
	}
	return
}

// SetCacheMode sets the value of CacheMode for the instance
func (instance *MSFT_SBLTargetDisk) SetPropertyCacheMode(value uint32) (err error) {
	return instance.SetProperty("CacheMode", (value))
}

// GetCacheMode gets the value of CacheMode for the instance
func (instance *MSFT_SBLTargetDisk) GetPropertyCacheMode() (value uint32, err error) {
	retValue, err := instance.GetProperty("CacheMode")
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

// SetCurrentUsage sets the value of CurrentUsage for the instance
func (instance *MSFT_SBLTargetDisk) SetPropertyCurrentUsage(value uint32) (err error) {
	return instance.SetProperty("CurrentUsage", (value))
}

// GetCurrentUsage gets the value of CurrentUsage for the instance
func (instance *MSFT_SBLTargetDisk) GetPropertyCurrentUsage() (value uint32, err error) {
	retValue, err := instance.GetProperty("CurrentUsage")
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

// SetDesiredUsage sets the value of DesiredUsage for the instance
func (instance *MSFT_SBLTargetDisk) SetPropertyDesiredUsage(value uint32) (err error) {
	return instance.SetProperty("DesiredUsage", (value))
}

// GetDesiredUsage gets the value of DesiredUsage for the instance
func (instance *MSFT_SBLTargetDisk) GetPropertyDesiredUsage() (value uint32, err error) {
	retValue, err := instance.GetProperty("DesiredUsage")
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

// SetDeviceNumber sets the value of DeviceNumber for the instance
func (instance *MSFT_SBLTargetDisk) SetPropertyDeviceNumber(value uint32) (err error) {
	return instance.SetProperty("DeviceNumber", (value))
}

// GetDeviceNumber gets the value of DeviceNumber for the instance
func (instance *MSFT_SBLTargetDisk) GetPropertyDeviceNumber() (value uint32, err error) {
	retValue, err := instance.GetProperty("DeviceNumber")
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

// SetIdentifier sets the value of Identifier for the instance
func (instance *MSFT_SBLTargetDisk) SetPropertyIdentifier(value string) (err error) {
	return instance.SetProperty("Identifier", (value))
}

// GetIdentifier gets the value of Identifier for the instance
func (instance *MSFT_SBLTargetDisk) GetPropertyIdentifier() (value string, err error) {
	retValue, err := instance.GetProperty("Identifier")
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

// SetIsFlash sets the value of IsFlash for the instance
func (instance *MSFT_SBLTargetDisk) SetPropertyIsFlash(value bool) (err error) {
	return instance.SetProperty("IsFlash", (value))
}

// GetIsFlash gets the value of IsFlash for the instance
func (instance *MSFT_SBLTargetDisk) GetPropertyIsFlash() (value bool, err error) {
	retValue, err := instance.GetProperty("IsFlash")
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

// SetIsSblCacheDevice sets the value of IsSblCacheDevice for the instance
func (instance *MSFT_SBLTargetDisk) SetPropertyIsSblCacheDevice(value bool) (err error) {
	return instance.SetProperty("IsSblCacheDevice", (value))
}

// GetIsSblCacheDevice gets the value of IsSblCacheDevice for the instance
func (instance *MSFT_SBLTargetDisk) GetPropertyIsSblCacheDevice() (value bool, err error) {
	retValue, err := instance.GetProperty("IsSblCacheDevice")
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

// SetLastStateChangeTime sets the value of LastStateChangeTime for the instance
func (instance *MSFT_SBLTargetDisk) SetPropertyLastStateChangeTime(value string) (err error) {
	return instance.SetProperty("LastStateChangeTime", (value))
}

// GetLastStateChangeTime gets the value of LastStateChangeTime for the instance
func (instance *MSFT_SBLTargetDisk) GetPropertyLastStateChangeTime() (value string, err error) {
	retValue, err := instance.GetProperty("LastStateChangeTime")
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

// SetReadMediaErrorCount sets the value of ReadMediaErrorCount for the instance
func (instance *MSFT_SBLTargetDisk) SetPropertyReadMediaErrorCount(value uint64) (err error) {
	return instance.SetProperty("ReadMediaErrorCount", (value))
}

// GetReadMediaErrorCount gets the value of ReadMediaErrorCount for the instance
func (instance *MSFT_SBLTargetDisk) GetPropertyReadMediaErrorCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadMediaErrorCount")
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

// SetReadTotalErrorCount sets the value of ReadTotalErrorCount for the instance
func (instance *MSFT_SBLTargetDisk) SetPropertyReadTotalErrorCount(value uint64) (err error) {
	return instance.SetProperty("ReadTotalErrorCount", (value))
}

// GetReadTotalErrorCount gets the value of ReadTotalErrorCount for the instance
func (instance *MSFT_SBLTargetDisk) GetPropertyReadTotalErrorCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadTotalErrorCount")
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

// SetSblAttributes sets the value of SblAttributes for the instance
func (instance *MSFT_SBLTargetDisk) SetPropertySblAttributes(value uint32) (err error) {
	return instance.SetProperty("SblAttributes", (value))
}

// GetSblAttributes gets the value of SblAttributes for the instance
func (instance *MSFT_SBLTargetDisk) GetPropertySblAttributes() (value uint32, err error) {
	retValue, err := instance.GetProperty("SblAttributes")
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

// SetState sets the value of State for the instance
func (instance *MSFT_SBLTargetDisk) SetPropertyState(value uint32) (err error) {
	return instance.SetProperty("State", (value))
}

// GetState gets the value of State for the instance
func (instance *MSFT_SBLTargetDisk) GetPropertyState() (value uint32, err error) {
	retValue, err := instance.GetProperty("State")
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

// SetWriteMediaErrorCount sets the value of WriteMediaErrorCount for the instance
func (instance *MSFT_SBLTargetDisk) SetPropertyWriteMediaErrorCount(value uint64) (err error) {
	return instance.SetProperty("WriteMediaErrorCount", (value))
}

// GetWriteMediaErrorCount gets the value of WriteMediaErrorCount for the instance
func (instance *MSFT_SBLTargetDisk) GetPropertyWriteMediaErrorCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteMediaErrorCount")
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

// SetWriteTotalErrorCount sets the value of WriteTotalErrorCount for the instance
func (instance *MSFT_SBLTargetDisk) SetPropertyWriteTotalErrorCount(value uint64) (err error) {
	return instance.SetProperty("WriteTotalErrorCount", (value))
}

// GetWriteTotalErrorCount gets the value of WriteTotalErrorCount for the instance
func (instance *MSFT_SBLTargetDisk) GetPropertyWriteTotalErrorCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteTotalErrorCount")
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
