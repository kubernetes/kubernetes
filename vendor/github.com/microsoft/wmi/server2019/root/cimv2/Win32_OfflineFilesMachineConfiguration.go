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

// Win32_OfflineFilesMachineConfiguration struct
type Win32_OfflineFilesMachineConfiguration struct {
	*cim.WmiInstance

	//
	BackgroundSyncEnabled bool

	//
	BackgroundSyncParams Win32_OfflineFilesBackgroundSync

	//
	DiskSpaceLimitEnabled bool

	//
	DiskSpaceLimitParams Win32_OfflineFilesDiskSpaceLimit

	//
	EconomicalAdminPinningEnabled bool

	//
	Enabled bool

	//
	ExcludedFileTypes []string

	//
	IsConfiguredByWMI bool

	//
	MakeAvailableOfflineButtonRemoved bool

	//
	OfflineFilesCacheEncrypted bool

	//
	SlowLinkEnabled bool

	//
	SlowLinkParams []string

	//
	SyncOnCostedNetworkEnabled bool

	//
	TransparentCachingLatencyThreshold uint32

	//
	WorkOfflineButtonRemoved bool
}

func NewWin32_OfflineFilesMachineConfigurationEx1(instance *cim.WmiInstance) (newInstance *Win32_OfflineFilesMachineConfiguration, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_OfflineFilesMachineConfiguration{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_OfflineFilesMachineConfigurationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_OfflineFilesMachineConfiguration, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_OfflineFilesMachineConfiguration{
		WmiInstance: tmp,
	}
	return
}

// SetBackgroundSyncEnabled sets the value of BackgroundSyncEnabled for the instance
func (instance *Win32_OfflineFilesMachineConfiguration) SetPropertyBackgroundSyncEnabled(value bool) (err error) {
	return instance.SetProperty("BackgroundSyncEnabled", (value))
}

// GetBackgroundSyncEnabled gets the value of BackgroundSyncEnabled for the instance
func (instance *Win32_OfflineFilesMachineConfiguration) GetPropertyBackgroundSyncEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("BackgroundSyncEnabled")
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

// SetBackgroundSyncParams sets the value of BackgroundSyncParams for the instance
func (instance *Win32_OfflineFilesMachineConfiguration) SetPropertyBackgroundSyncParams(value Win32_OfflineFilesBackgroundSync) (err error) {
	return instance.SetProperty("BackgroundSyncParams", (value))
}

// GetBackgroundSyncParams gets the value of BackgroundSyncParams for the instance
func (instance *Win32_OfflineFilesMachineConfiguration) GetPropertyBackgroundSyncParams() (value Win32_OfflineFilesBackgroundSync, err error) {
	retValue, err := instance.GetProperty("BackgroundSyncParams")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_OfflineFilesBackgroundSync)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_OfflineFilesBackgroundSync is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_OfflineFilesBackgroundSync(valuetmp)

	return
}

// SetDiskSpaceLimitEnabled sets the value of DiskSpaceLimitEnabled for the instance
func (instance *Win32_OfflineFilesMachineConfiguration) SetPropertyDiskSpaceLimitEnabled(value bool) (err error) {
	return instance.SetProperty("DiskSpaceLimitEnabled", (value))
}

// GetDiskSpaceLimitEnabled gets the value of DiskSpaceLimitEnabled for the instance
func (instance *Win32_OfflineFilesMachineConfiguration) GetPropertyDiskSpaceLimitEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("DiskSpaceLimitEnabled")
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

// SetDiskSpaceLimitParams sets the value of DiskSpaceLimitParams for the instance
func (instance *Win32_OfflineFilesMachineConfiguration) SetPropertyDiskSpaceLimitParams(value Win32_OfflineFilesDiskSpaceLimit) (err error) {
	return instance.SetProperty("DiskSpaceLimitParams", (value))
}

// GetDiskSpaceLimitParams gets the value of DiskSpaceLimitParams for the instance
func (instance *Win32_OfflineFilesMachineConfiguration) GetPropertyDiskSpaceLimitParams() (value Win32_OfflineFilesDiskSpaceLimit, err error) {
	retValue, err := instance.GetProperty("DiskSpaceLimitParams")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_OfflineFilesDiskSpaceLimit)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_OfflineFilesDiskSpaceLimit is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_OfflineFilesDiskSpaceLimit(valuetmp)

	return
}

// SetEconomicalAdminPinningEnabled sets the value of EconomicalAdminPinningEnabled for the instance
func (instance *Win32_OfflineFilesMachineConfiguration) SetPropertyEconomicalAdminPinningEnabled(value bool) (err error) {
	return instance.SetProperty("EconomicalAdminPinningEnabled", (value))
}

// GetEconomicalAdminPinningEnabled gets the value of EconomicalAdminPinningEnabled for the instance
func (instance *Win32_OfflineFilesMachineConfiguration) GetPropertyEconomicalAdminPinningEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("EconomicalAdminPinningEnabled")
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

// SetEnabled sets the value of Enabled for the instance
func (instance *Win32_OfflineFilesMachineConfiguration) SetPropertyEnabled(value bool) (err error) {
	return instance.SetProperty("Enabled", (value))
}

// GetEnabled gets the value of Enabled for the instance
func (instance *Win32_OfflineFilesMachineConfiguration) GetPropertyEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("Enabled")
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

// SetExcludedFileTypes sets the value of ExcludedFileTypes for the instance
func (instance *Win32_OfflineFilesMachineConfiguration) SetPropertyExcludedFileTypes(value []string) (err error) {
	return instance.SetProperty("ExcludedFileTypes", (value))
}

// GetExcludedFileTypes gets the value of ExcludedFileTypes for the instance
func (instance *Win32_OfflineFilesMachineConfiguration) GetPropertyExcludedFileTypes() (value []string, err error) {
	retValue, err := instance.GetProperty("ExcludedFileTypes")
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

// SetIsConfiguredByWMI sets the value of IsConfiguredByWMI for the instance
func (instance *Win32_OfflineFilesMachineConfiguration) SetPropertyIsConfiguredByWMI(value bool) (err error) {
	return instance.SetProperty("IsConfiguredByWMI", (value))
}

// GetIsConfiguredByWMI gets the value of IsConfiguredByWMI for the instance
func (instance *Win32_OfflineFilesMachineConfiguration) GetPropertyIsConfiguredByWMI() (value bool, err error) {
	retValue, err := instance.GetProperty("IsConfiguredByWMI")
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

// SetMakeAvailableOfflineButtonRemoved sets the value of MakeAvailableOfflineButtonRemoved for the instance
func (instance *Win32_OfflineFilesMachineConfiguration) SetPropertyMakeAvailableOfflineButtonRemoved(value bool) (err error) {
	return instance.SetProperty("MakeAvailableOfflineButtonRemoved", (value))
}

// GetMakeAvailableOfflineButtonRemoved gets the value of MakeAvailableOfflineButtonRemoved for the instance
func (instance *Win32_OfflineFilesMachineConfiguration) GetPropertyMakeAvailableOfflineButtonRemoved() (value bool, err error) {
	retValue, err := instance.GetProperty("MakeAvailableOfflineButtonRemoved")
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

// SetOfflineFilesCacheEncrypted sets the value of OfflineFilesCacheEncrypted for the instance
func (instance *Win32_OfflineFilesMachineConfiguration) SetPropertyOfflineFilesCacheEncrypted(value bool) (err error) {
	return instance.SetProperty("OfflineFilesCacheEncrypted", (value))
}

// GetOfflineFilesCacheEncrypted gets the value of OfflineFilesCacheEncrypted for the instance
func (instance *Win32_OfflineFilesMachineConfiguration) GetPropertyOfflineFilesCacheEncrypted() (value bool, err error) {
	retValue, err := instance.GetProperty("OfflineFilesCacheEncrypted")
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

// SetSlowLinkEnabled sets the value of SlowLinkEnabled for the instance
func (instance *Win32_OfflineFilesMachineConfiguration) SetPropertySlowLinkEnabled(value bool) (err error) {
	return instance.SetProperty("SlowLinkEnabled", (value))
}

// GetSlowLinkEnabled gets the value of SlowLinkEnabled for the instance
func (instance *Win32_OfflineFilesMachineConfiguration) GetPropertySlowLinkEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("SlowLinkEnabled")
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

// SetSlowLinkParams sets the value of SlowLinkParams for the instance
func (instance *Win32_OfflineFilesMachineConfiguration) SetPropertySlowLinkParams(value []string) (err error) {
	return instance.SetProperty("SlowLinkParams", (value))
}

// GetSlowLinkParams gets the value of SlowLinkParams for the instance
func (instance *Win32_OfflineFilesMachineConfiguration) GetPropertySlowLinkParams() (value []string, err error) {
	retValue, err := instance.GetProperty("SlowLinkParams")
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

// SetSyncOnCostedNetworkEnabled sets the value of SyncOnCostedNetworkEnabled for the instance
func (instance *Win32_OfflineFilesMachineConfiguration) SetPropertySyncOnCostedNetworkEnabled(value bool) (err error) {
	return instance.SetProperty("SyncOnCostedNetworkEnabled", (value))
}

// GetSyncOnCostedNetworkEnabled gets the value of SyncOnCostedNetworkEnabled for the instance
func (instance *Win32_OfflineFilesMachineConfiguration) GetPropertySyncOnCostedNetworkEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("SyncOnCostedNetworkEnabled")
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

// SetTransparentCachingLatencyThreshold sets the value of TransparentCachingLatencyThreshold for the instance
func (instance *Win32_OfflineFilesMachineConfiguration) SetPropertyTransparentCachingLatencyThreshold(value uint32) (err error) {
	return instance.SetProperty("TransparentCachingLatencyThreshold", (value))
}

// GetTransparentCachingLatencyThreshold gets the value of TransparentCachingLatencyThreshold for the instance
func (instance *Win32_OfflineFilesMachineConfiguration) GetPropertyTransparentCachingLatencyThreshold() (value uint32, err error) {
	retValue, err := instance.GetProperty("TransparentCachingLatencyThreshold")
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

// SetWorkOfflineButtonRemoved sets the value of WorkOfflineButtonRemoved for the instance
func (instance *Win32_OfflineFilesMachineConfiguration) SetPropertyWorkOfflineButtonRemoved(value bool) (err error) {
	return instance.SetProperty("WorkOfflineButtonRemoved", (value))
}

// GetWorkOfflineButtonRemoved gets the value of WorkOfflineButtonRemoved for the instance
func (instance *Win32_OfflineFilesMachineConfiguration) GetPropertyWorkOfflineButtonRemoved() (value bool, err error) {
	retValue, err := instance.GetProperty("WorkOfflineButtonRemoved")
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
