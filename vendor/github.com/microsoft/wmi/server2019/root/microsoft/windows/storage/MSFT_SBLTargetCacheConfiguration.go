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

// MSFT_SBLTargetCacheConfiguration struct
type MSFT_SBLTargetCacheConfiguration struct {
	*cim.WmiInstance

	//
	CacheBehavior uint64

	//
	CachePageSizeinKB uint32

	//
	CurrentCacheModeHDD uint32

	//
	CurrentCacheModeSSD uint32

	//
	CurrentState uint32

	//
	CurrentStateProgress uint64

	//
	CurrentStateProgressMax uint64

	//
	DesiredCacheModeHDD uint32

	//
	DesiredCacheModeSSD uint32

	//
	DesiredState uint32

	//
	FlashMetadataReserveBytes uint64

	//
	FlashReservePercent uint32

	//
	Identifier string

	//
	ProvisioningStage uint64

	//
	ProvisioningStageMax uint64

	//
	SpacesDirectEnabled bool
}

func NewMSFT_SBLTargetCacheConfigurationEx1(instance *cim.WmiInstance) (newInstance *MSFT_SBLTargetCacheConfiguration, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_SBLTargetCacheConfiguration{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_SBLTargetCacheConfigurationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_SBLTargetCacheConfiguration, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_SBLTargetCacheConfiguration{
		WmiInstance: tmp,
	}
	return
}

// SetCacheBehavior sets the value of CacheBehavior for the instance
func (instance *MSFT_SBLTargetCacheConfiguration) SetPropertyCacheBehavior(value uint64) (err error) {
	return instance.SetProperty("CacheBehavior", (value))
}

// GetCacheBehavior gets the value of CacheBehavior for the instance
func (instance *MSFT_SBLTargetCacheConfiguration) GetPropertyCacheBehavior() (value uint64, err error) {
	retValue, err := instance.GetProperty("CacheBehavior")
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

// SetCachePageSizeinKB sets the value of CachePageSizeinKB for the instance
func (instance *MSFT_SBLTargetCacheConfiguration) SetPropertyCachePageSizeinKB(value uint32) (err error) {
	return instance.SetProperty("CachePageSizeinKB", (value))
}

// GetCachePageSizeinKB gets the value of CachePageSizeinKB for the instance
func (instance *MSFT_SBLTargetCacheConfiguration) GetPropertyCachePageSizeinKB() (value uint32, err error) {
	retValue, err := instance.GetProperty("CachePageSizeinKB")
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

// SetCurrentCacheModeHDD sets the value of CurrentCacheModeHDD for the instance
func (instance *MSFT_SBLTargetCacheConfiguration) SetPropertyCurrentCacheModeHDD(value uint32) (err error) {
	return instance.SetProperty("CurrentCacheModeHDD", (value))
}

// GetCurrentCacheModeHDD gets the value of CurrentCacheModeHDD for the instance
func (instance *MSFT_SBLTargetCacheConfiguration) GetPropertyCurrentCacheModeHDD() (value uint32, err error) {
	retValue, err := instance.GetProperty("CurrentCacheModeHDD")
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

// SetCurrentCacheModeSSD sets the value of CurrentCacheModeSSD for the instance
func (instance *MSFT_SBLTargetCacheConfiguration) SetPropertyCurrentCacheModeSSD(value uint32) (err error) {
	return instance.SetProperty("CurrentCacheModeSSD", (value))
}

// GetCurrentCacheModeSSD gets the value of CurrentCacheModeSSD for the instance
func (instance *MSFT_SBLTargetCacheConfiguration) GetPropertyCurrentCacheModeSSD() (value uint32, err error) {
	retValue, err := instance.GetProperty("CurrentCacheModeSSD")
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

// SetCurrentState sets the value of CurrentState for the instance
func (instance *MSFT_SBLTargetCacheConfiguration) SetPropertyCurrentState(value uint32) (err error) {
	return instance.SetProperty("CurrentState", (value))
}

// GetCurrentState gets the value of CurrentState for the instance
func (instance *MSFT_SBLTargetCacheConfiguration) GetPropertyCurrentState() (value uint32, err error) {
	retValue, err := instance.GetProperty("CurrentState")
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

// SetCurrentStateProgress sets the value of CurrentStateProgress for the instance
func (instance *MSFT_SBLTargetCacheConfiguration) SetPropertyCurrentStateProgress(value uint64) (err error) {
	return instance.SetProperty("CurrentStateProgress", (value))
}

// GetCurrentStateProgress gets the value of CurrentStateProgress for the instance
func (instance *MSFT_SBLTargetCacheConfiguration) GetPropertyCurrentStateProgress() (value uint64, err error) {
	retValue, err := instance.GetProperty("CurrentStateProgress")
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

// SetCurrentStateProgressMax sets the value of CurrentStateProgressMax for the instance
func (instance *MSFT_SBLTargetCacheConfiguration) SetPropertyCurrentStateProgressMax(value uint64) (err error) {
	return instance.SetProperty("CurrentStateProgressMax", (value))
}

// GetCurrentStateProgressMax gets the value of CurrentStateProgressMax for the instance
func (instance *MSFT_SBLTargetCacheConfiguration) GetPropertyCurrentStateProgressMax() (value uint64, err error) {
	retValue, err := instance.GetProperty("CurrentStateProgressMax")
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

// SetDesiredCacheModeHDD sets the value of DesiredCacheModeHDD for the instance
func (instance *MSFT_SBLTargetCacheConfiguration) SetPropertyDesiredCacheModeHDD(value uint32) (err error) {
	return instance.SetProperty("DesiredCacheModeHDD", (value))
}

// GetDesiredCacheModeHDD gets the value of DesiredCacheModeHDD for the instance
func (instance *MSFT_SBLTargetCacheConfiguration) GetPropertyDesiredCacheModeHDD() (value uint32, err error) {
	retValue, err := instance.GetProperty("DesiredCacheModeHDD")
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

// SetDesiredCacheModeSSD sets the value of DesiredCacheModeSSD for the instance
func (instance *MSFT_SBLTargetCacheConfiguration) SetPropertyDesiredCacheModeSSD(value uint32) (err error) {
	return instance.SetProperty("DesiredCacheModeSSD", (value))
}

// GetDesiredCacheModeSSD gets the value of DesiredCacheModeSSD for the instance
func (instance *MSFT_SBLTargetCacheConfiguration) GetPropertyDesiredCacheModeSSD() (value uint32, err error) {
	retValue, err := instance.GetProperty("DesiredCacheModeSSD")
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

// SetDesiredState sets the value of DesiredState for the instance
func (instance *MSFT_SBLTargetCacheConfiguration) SetPropertyDesiredState(value uint32) (err error) {
	return instance.SetProperty("DesiredState", (value))
}

// GetDesiredState gets the value of DesiredState for the instance
func (instance *MSFT_SBLTargetCacheConfiguration) GetPropertyDesiredState() (value uint32, err error) {
	retValue, err := instance.GetProperty("DesiredState")
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

// SetFlashMetadataReserveBytes sets the value of FlashMetadataReserveBytes for the instance
func (instance *MSFT_SBLTargetCacheConfiguration) SetPropertyFlashMetadataReserveBytes(value uint64) (err error) {
	return instance.SetProperty("FlashMetadataReserveBytes", (value))
}

// GetFlashMetadataReserveBytes gets the value of FlashMetadataReserveBytes for the instance
func (instance *MSFT_SBLTargetCacheConfiguration) GetPropertyFlashMetadataReserveBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("FlashMetadataReserveBytes")
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

// SetFlashReservePercent sets the value of FlashReservePercent for the instance
func (instance *MSFT_SBLTargetCacheConfiguration) SetPropertyFlashReservePercent(value uint32) (err error) {
	return instance.SetProperty("FlashReservePercent", (value))
}

// GetFlashReservePercent gets the value of FlashReservePercent for the instance
func (instance *MSFT_SBLTargetCacheConfiguration) GetPropertyFlashReservePercent() (value uint32, err error) {
	retValue, err := instance.GetProperty("FlashReservePercent")
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
func (instance *MSFT_SBLTargetCacheConfiguration) SetPropertyIdentifier(value string) (err error) {
	return instance.SetProperty("Identifier", (value))
}

// GetIdentifier gets the value of Identifier for the instance
func (instance *MSFT_SBLTargetCacheConfiguration) GetPropertyIdentifier() (value string, err error) {
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

// SetProvisioningStage sets the value of ProvisioningStage for the instance
func (instance *MSFT_SBLTargetCacheConfiguration) SetPropertyProvisioningStage(value uint64) (err error) {
	return instance.SetProperty("ProvisioningStage", (value))
}

// GetProvisioningStage gets the value of ProvisioningStage for the instance
func (instance *MSFT_SBLTargetCacheConfiguration) GetPropertyProvisioningStage() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProvisioningStage")
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

// SetProvisioningStageMax sets the value of ProvisioningStageMax for the instance
func (instance *MSFT_SBLTargetCacheConfiguration) SetPropertyProvisioningStageMax(value uint64) (err error) {
	return instance.SetProperty("ProvisioningStageMax", (value))
}

// GetProvisioningStageMax gets the value of ProvisioningStageMax for the instance
func (instance *MSFT_SBLTargetCacheConfiguration) GetPropertyProvisioningStageMax() (value uint64, err error) {
	retValue, err := instance.GetProperty("ProvisioningStageMax")
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

// SetSpacesDirectEnabled sets the value of SpacesDirectEnabled for the instance
func (instance *MSFT_SBLTargetCacheConfiguration) SetPropertySpacesDirectEnabled(value bool) (err error) {
	return instance.SetProperty("SpacesDirectEnabled", (value))
}

// GetSpacesDirectEnabled gets the value of SpacesDirectEnabled for the instance
func (instance *MSFT_SBLTargetCacheConfiguration) GetPropertySpacesDirectEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("SpacesDirectEnabled")
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

//

// <param name="Description" type="string "></param>
// <param name="DiskGuid" type="string "></param>
// <param name="EnclosureId" type="string "></param>
// <param name="Manufacturer" type="string "></param>
// <param name="Name" type="string "></param>
// <param name="PoolId" type="string "></param>
// <param name="ProductId" type="string "></param>
// <param name="Serial" type="string "></param>
// <param name="SlotNumber" type="uint32 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_SBLTargetCacheConfiguration) NotifyDisk( /* IN */ DiskGuid string,
	/* IN */ PoolId string,
	/* IN */ Name string,
	/* IN */ Description string,
	/* IN */ Manufacturer string,
	/* IN */ ProductId string,
	/* IN */ Serial string,
	/* IN */ SlotNumber uint32,
	/* IN */ EnclosureId string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("NotifyDisk", DiskGuid, PoolId, Name, Description, Manufacturer, ProductId, Serial, SlotNumber, EnclosureId)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="Description" type="string "></param>
// <param name="EnclosureGuid" type="string "></param>
// <param name="Manufacturer" type="string "></param>
// <param name="Name" type="string "></param>
// <param name="ProductId" type="string "></param>
// <param name="Serial" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_SBLTargetCacheConfiguration) NotifyEnclosure( /* IN */ EnclosureGuid string,
	/* IN */ Name string,
	/* IN */ Description string,
	/* IN */ Manufacturer string,
	/* IN */ ProductId string,
	/* IN */ Serial string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("NotifyEnclosure", EnclosureGuid, Name, Description, Manufacturer, ProductId, Serial)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="DiskGuid" type="string "></param>
// <param name="StateChange" type="uint32 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_SBLTargetCacheConfiguration) NotifyDiskStateChange( /* IN */ DiskGuid string,
	/* IN */ StateChange uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("NotifyDiskStateChange", DiskGuid, StateChange)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="DiskGuid" type="string "></param>
// <param name="UseForStorageSpacesDirect" type="uint32 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_SBLTargetCacheConfiguration) SetDiskUsage( /* IN */ DiskGuid string,
	/* IN */ UseForStorageSpacesDirect uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetDiskUsage", DiskGuid, UseForStorageSpacesDirect)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="Flags" type="uint32 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_SBLTargetCacheConfiguration) StartOptimize( /* IN */ Flags uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("StartOptimize", Flags)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="CacheState" type="uint32 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_SBLTargetCacheConfiguration) CheckSystemSupportsCacheState( /* IN */ CacheState uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("CheckSystemSupportsCacheState", CacheState)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="CacheState" type="uint32 "></param>
// <param name="DiskGuid" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_SBLTargetCacheConfiguration) CheckDiskSupportsCacheState( /* IN */ DiskGuid string,
	/* IN */ CacheState uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("CheckDiskSupportsCacheState", DiskGuid, CacheState)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="CacheState" type="uint32 "></param>

// <param name="DiskGuids" type="string []"></param>
// <param name="DiskNumbers" type="uint32 []"></param>
// <param name="ReturnValue" type="uint32 "></param>
// <param name="SupportStatuses" type="uint32 []"></param>
func (instance *MSFT_SBLTargetCacheConfiguration) CheckAllDisksSupportCache( /* IN */ CacheState uint32,
	/* OUT */ DiskGuids []string,
	/* OUT */ DiskNumbers []uint32,
	/* OUT */ SupportStatuses []uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("CheckAllDisksSupportCache", CacheState)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="DiskGuid" type="string "></param>

// <param name="BoundDiskGuids" type="string []"></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_SBLTargetCacheConfiguration) QueryBoundDevices( /* IN */ DiskGuid string,
	/* OUT */ BoundDiskGuids []string) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("QueryBoundDevices", DiskGuid)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="CacheMode" type="uint32 "></param>
// <param name="DiskGuid" type="string "></param>
// <param name="Flags" type="uint32 "></param>
// <param name="Force" type="bool "></param>
// <param name="Originator" type="uint32 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_SBLTargetCacheConfiguration) SetDiskCacheMode( /* IN */ DiskGuid string,
	/* IN */ CacheMode uint32,
	/* IN */ Flags uint32,
	/* IN */ Originator uint32,
	/* IN */ Force bool) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetDiskCacheMode", DiskGuid, CacheMode, Flags, Originator, Force)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="CacheHint" type="uint32 "></param>
// <param name="DiskGuid" type="string "></param>
// <param name="Flags" type="uint32 "></param>
// <param name="Originator" type="uint32 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_SBLTargetCacheConfiguration) SetDiskCacheHint( /* IN */ DiskGuid string,
	/* IN */ CacheHint uint32,
	/* IN */ Flags uint32,
	/* IN */ Originator uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetDiskCacheHint", DiskGuid, CacheHint, Flags, Originator)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}
