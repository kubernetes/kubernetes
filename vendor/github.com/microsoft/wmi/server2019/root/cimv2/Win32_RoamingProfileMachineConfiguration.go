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

// Win32_RoamingProfileMachineConfiguration struct
type Win32_RoamingProfileMachineConfiguration struct {
	*cim.WmiInstance

	// If true, add the Administrator group to roaming user profiles.
	AddAdminGroupToRUPEnabled bool

	// If true, allow cross-forest user policy and roaming user profiles. If false, a roaming profile user receives a local profile when logged on to a cross-forest domain.
	AllowCrossForestUserPolicy bool

	// Contains the parameter for the background upload of a roaming user profile's registry file while the user is logged on.
	BackgroundUploadParams Win32_RoamingProfileBackgroundUploadParams

	// If the DeleteRoamingCache property is true, this property specifies the number of days after which a user profile should be deleted. User profiles older than this number of days are deleted when the computer is restarted.
	DeleteProfilesOlderDays uint16

	// If true, cached copies of the roaming profile are deleted at log off
	DeleteRoamingCacheEnabled bool

	// If true, do not detect slow network connections. If false, use the SlowLinkTimeOutParams property to determine whether the computer has a slow network connection.
	DetectSlowLinkDisabled bool

	// If true, do not forcibly unload the user's registry when the user logs off.
	ForceUnloadDisabled bool

	// Indicates if the settings configured through this WMI class are taking affect.
	IsConfiguredByWMI bool

	// The roaming profile path to be set for all users that log on to this computer. The path should be in the form of \\ComputerName\ShareName\%USERNAME%.
	MachineProfilePath string

	// If true, allow only local user profiles.
	OnlyAllowLocalProfiles bool

	// If true, don't check the owners of user profiles.
	OwnerCheckDisabled bool

	// If true, a configured roaming profile will only be downloaded if the machine is a primary computer for the user.
	PrimaryComputerEnabled bool

	// If true, prevent roaming profile changes from being copied to the server.
	ProfileUploadDisabled bool

	// Contains slow network connection timeout parameters to be used for user profiles.
	SlowLinkTimeOutParams Win32_RoamingProfileSlowLinkParams

	// If true, the user is prompted to specify whether his or her profile should be downloaded even when the network connection is slow.
	SlowLinkUIEnabled bool

	// If true, do not allow users to log in with temporary profiles.
	TempProfileLogonBlocked bool

	// The maximum time, in seconds, to wait for the network transport to be available if a user has a roaming user profile. If the network is unavailable after this time has elapsed, the user is logged on, but the profile is not synchronized.
	WaitForNetworkInSec uint16

	// If true, wait for a remote user profile.
	WaitForRemoteProfile bool
}

func NewWin32_RoamingProfileMachineConfigurationEx1(instance *cim.WmiInstance) (newInstance *Win32_RoamingProfileMachineConfiguration, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_RoamingProfileMachineConfiguration{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_RoamingProfileMachineConfigurationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_RoamingProfileMachineConfiguration, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_RoamingProfileMachineConfiguration{
		WmiInstance: tmp,
	}
	return
}

// SetAddAdminGroupToRUPEnabled sets the value of AddAdminGroupToRUPEnabled for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) SetPropertyAddAdminGroupToRUPEnabled(value bool) (err error) {
	return instance.SetProperty("AddAdminGroupToRUPEnabled", (value))
}

// GetAddAdminGroupToRUPEnabled gets the value of AddAdminGroupToRUPEnabled for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) GetPropertyAddAdminGroupToRUPEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("AddAdminGroupToRUPEnabled")
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

// SetAllowCrossForestUserPolicy sets the value of AllowCrossForestUserPolicy for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) SetPropertyAllowCrossForestUserPolicy(value bool) (err error) {
	return instance.SetProperty("AllowCrossForestUserPolicy", (value))
}

// GetAllowCrossForestUserPolicy gets the value of AllowCrossForestUserPolicy for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) GetPropertyAllowCrossForestUserPolicy() (value bool, err error) {
	retValue, err := instance.GetProperty("AllowCrossForestUserPolicy")
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

// SetBackgroundUploadParams sets the value of BackgroundUploadParams for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) SetPropertyBackgroundUploadParams(value Win32_RoamingProfileBackgroundUploadParams) (err error) {
	return instance.SetProperty("BackgroundUploadParams", (value))
}

// GetBackgroundUploadParams gets the value of BackgroundUploadParams for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) GetPropertyBackgroundUploadParams() (value Win32_RoamingProfileBackgroundUploadParams, err error) {
	retValue, err := instance.GetProperty("BackgroundUploadParams")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_RoamingProfileBackgroundUploadParams)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_RoamingProfileBackgroundUploadParams is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_RoamingProfileBackgroundUploadParams(valuetmp)

	return
}

// SetDeleteProfilesOlderDays sets the value of DeleteProfilesOlderDays for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) SetPropertyDeleteProfilesOlderDays(value uint16) (err error) {
	return instance.SetProperty("DeleteProfilesOlderDays", (value))
}

// GetDeleteProfilesOlderDays gets the value of DeleteProfilesOlderDays for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) GetPropertyDeleteProfilesOlderDays() (value uint16, err error) {
	retValue, err := instance.GetProperty("DeleteProfilesOlderDays")
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

// SetDeleteRoamingCacheEnabled sets the value of DeleteRoamingCacheEnabled for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) SetPropertyDeleteRoamingCacheEnabled(value bool) (err error) {
	return instance.SetProperty("DeleteRoamingCacheEnabled", (value))
}

// GetDeleteRoamingCacheEnabled gets the value of DeleteRoamingCacheEnabled for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) GetPropertyDeleteRoamingCacheEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("DeleteRoamingCacheEnabled")
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

// SetDetectSlowLinkDisabled sets the value of DetectSlowLinkDisabled for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) SetPropertyDetectSlowLinkDisabled(value bool) (err error) {
	return instance.SetProperty("DetectSlowLinkDisabled", (value))
}

// GetDetectSlowLinkDisabled gets the value of DetectSlowLinkDisabled for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) GetPropertyDetectSlowLinkDisabled() (value bool, err error) {
	retValue, err := instance.GetProperty("DetectSlowLinkDisabled")
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

// SetForceUnloadDisabled sets the value of ForceUnloadDisabled for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) SetPropertyForceUnloadDisabled(value bool) (err error) {
	return instance.SetProperty("ForceUnloadDisabled", (value))
}

// GetForceUnloadDisabled gets the value of ForceUnloadDisabled for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) GetPropertyForceUnloadDisabled() (value bool, err error) {
	retValue, err := instance.GetProperty("ForceUnloadDisabled")
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

// SetIsConfiguredByWMI sets the value of IsConfiguredByWMI for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) SetPropertyIsConfiguredByWMI(value bool) (err error) {
	return instance.SetProperty("IsConfiguredByWMI", (value))
}

// GetIsConfiguredByWMI gets the value of IsConfiguredByWMI for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) GetPropertyIsConfiguredByWMI() (value bool, err error) {
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

// SetMachineProfilePath sets the value of MachineProfilePath for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) SetPropertyMachineProfilePath(value string) (err error) {
	return instance.SetProperty("MachineProfilePath", (value))
}

// GetMachineProfilePath gets the value of MachineProfilePath for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) GetPropertyMachineProfilePath() (value string, err error) {
	retValue, err := instance.GetProperty("MachineProfilePath")
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

// SetOnlyAllowLocalProfiles sets the value of OnlyAllowLocalProfiles for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) SetPropertyOnlyAllowLocalProfiles(value bool) (err error) {
	return instance.SetProperty("OnlyAllowLocalProfiles", (value))
}

// GetOnlyAllowLocalProfiles gets the value of OnlyAllowLocalProfiles for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) GetPropertyOnlyAllowLocalProfiles() (value bool, err error) {
	retValue, err := instance.GetProperty("OnlyAllowLocalProfiles")
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

// SetOwnerCheckDisabled sets the value of OwnerCheckDisabled for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) SetPropertyOwnerCheckDisabled(value bool) (err error) {
	return instance.SetProperty("OwnerCheckDisabled", (value))
}

// GetOwnerCheckDisabled gets the value of OwnerCheckDisabled for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) GetPropertyOwnerCheckDisabled() (value bool, err error) {
	retValue, err := instance.GetProperty("OwnerCheckDisabled")
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

// SetPrimaryComputerEnabled sets the value of PrimaryComputerEnabled for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) SetPropertyPrimaryComputerEnabled(value bool) (err error) {
	return instance.SetProperty("PrimaryComputerEnabled", (value))
}

// GetPrimaryComputerEnabled gets the value of PrimaryComputerEnabled for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) GetPropertyPrimaryComputerEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("PrimaryComputerEnabled")
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

// SetProfileUploadDisabled sets the value of ProfileUploadDisabled for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) SetPropertyProfileUploadDisabled(value bool) (err error) {
	return instance.SetProperty("ProfileUploadDisabled", (value))
}

// GetProfileUploadDisabled gets the value of ProfileUploadDisabled for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) GetPropertyProfileUploadDisabled() (value bool, err error) {
	retValue, err := instance.GetProperty("ProfileUploadDisabled")
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

// SetSlowLinkTimeOutParams sets the value of SlowLinkTimeOutParams for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) SetPropertySlowLinkTimeOutParams(value Win32_RoamingProfileSlowLinkParams) (err error) {
	return instance.SetProperty("SlowLinkTimeOutParams", (value))
}

// GetSlowLinkTimeOutParams gets the value of SlowLinkTimeOutParams for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) GetPropertySlowLinkTimeOutParams() (value Win32_RoamingProfileSlowLinkParams, err error) {
	retValue, err := instance.GetProperty("SlowLinkTimeOutParams")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_RoamingProfileSlowLinkParams)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_RoamingProfileSlowLinkParams is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_RoamingProfileSlowLinkParams(valuetmp)

	return
}

// SetSlowLinkUIEnabled sets the value of SlowLinkUIEnabled for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) SetPropertySlowLinkUIEnabled(value bool) (err error) {
	return instance.SetProperty("SlowLinkUIEnabled", (value))
}

// GetSlowLinkUIEnabled gets the value of SlowLinkUIEnabled for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) GetPropertySlowLinkUIEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("SlowLinkUIEnabled")
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

// SetTempProfileLogonBlocked sets the value of TempProfileLogonBlocked for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) SetPropertyTempProfileLogonBlocked(value bool) (err error) {
	return instance.SetProperty("TempProfileLogonBlocked", (value))
}

// GetTempProfileLogonBlocked gets the value of TempProfileLogonBlocked for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) GetPropertyTempProfileLogonBlocked() (value bool, err error) {
	retValue, err := instance.GetProperty("TempProfileLogonBlocked")
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

// SetWaitForNetworkInSec sets the value of WaitForNetworkInSec for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) SetPropertyWaitForNetworkInSec(value uint16) (err error) {
	return instance.SetProperty("WaitForNetworkInSec", (value))
}

// GetWaitForNetworkInSec gets the value of WaitForNetworkInSec for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) GetPropertyWaitForNetworkInSec() (value uint16, err error) {
	retValue, err := instance.GetProperty("WaitForNetworkInSec")
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

// SetWaitForRemoteProfile sets the value of WaitForRemoteProfile for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) SetPropertyWaitForRemoteProfile(value bool) (err error) {
	return instance.SetProperty("WaitForRemoteProfile", (value))
}

// GetWaitForRemoteProfile gets the value of WaitForRemoteProfile for the instance
func (instance *Win32_RoamingProfileMachineConfiguration) GetPropertyWaitForRemoteProfile() (value bool, err error) {
	retValue, err := instance.GetProperty("WaitForRemoteProfile")
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
