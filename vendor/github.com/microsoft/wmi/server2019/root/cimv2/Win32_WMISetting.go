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
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// Win32_WMISetting struct
type Win32_WMISetting struct {
	*CIM_Setting

	//
	ASPScriptDefaultNamespace string

	//
	ASPScriptEnabled bool

	//
	AutorecoverMofs []string

	//
	AutoStartWin9X uint32

	//
	BackupInterval uint32

	//
	BackupLastTime string

	//
	BuildVersion string

	//
	DatabaseDirectory string

	//
	DatabaseMaxSize uint32

	//
	EnableAnonWin9xConnections bool

	//
	EnableEvents bool

	//
	EnableStartupHeapPreallocation bool

	//
	HighThresholdOnClientObjects uint32

	//
	HighThresholdOnEvents uint32

	//
	InstallationDirectory string

	//
	LastStartupHeapPreallocation uint32

	//
	LoggingDirectory string

	//
	LoggingLevel uint32

	//
	LowThresholdOnClientObjects uint32

	//
	LowThresholdOnEvents uint32

	//
	MaxLogFileSize uint32

	//
	MaxWaitOnClientObjects uint32

	//
	MaxWaitOnEvents uint32

	//
	MofSelfInstallDirectory string
}

func NewWin32_WMISettingEx1(instance *cim.WmiInstance) (newInstance *Win32_WMISetting, err error) {
	tmp, err := NewCIM_SettingEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_WMISetting{
		CIM_Setting: tmp,
	}
	return
}

func NewWin32_WMISettingEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_WMISetting, err error) {
	tmp, err := NewCIM_SettingEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_WMISetting{
		CIM_Setting: tmp,
	}
	return
}

// SetASPScriptDefaultNamespace sets the value of ASPScriptDefaultNamespace for the instance
func (instance *Win32_WMISetting) SetPropertyASPScriptDefaultNamespace(value string) (err error) {
	return instance.SetProperty("ASPScriptDefaultNamespace", (value))
}

// GetASPScriptDefaultNamespace gets the value of ASPScriptDefaultNamespace for the instance
func (instance *Win32_WMISetting) GetPropertyASPScriptDefaultNamespace() (value string, err error) {
	retValue, err := instance.GetProperty("ASPScriptDefaultNamespace")
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

// SetASPScriptEnabled sets the value of ASPScriptEnabled for the instance
func (instance *Win32_WMISetting) SetPropertyASPScriptEnabled(value bool) (err error) {
	return instance.SetProperty("ASPScriptEnabled", (value))
}

// GetASPScriptEnabled gets the value of ASPScriptEnabled for the instance
func (instance *Win32_WMISetting) GetPropertyASPScriptEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("ASPScriptEnabled")
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

// SetAutorecoverMofs sets the value of AutorecoverMofs for the instance
func (instance *Win32_WMISetting) SetPropertyAutorecoverMofs(value []string) (err error) {
	return instance.SetProperty("AutorecoverMofs", (value))
}

// GetAutorecoverMofs gets the value of AutorecoverMofs for the instance
func (instance *Win32_WMISetting) GetPropertyAutorecoverMofs() (value []string, err error) {
	retValue, err := instance.GetProperty("AutorecoverMofs")
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

// SetAutoStartWin9X sets the value of AutoStartWin9X for the instance
func (instance *Win32_WMISetting) SetPropertyAutoStartWin9X(value uint32) (err error) {
	return instance.SetProperty("AutoStartWin9X", (value))
}

// GetAutoStartWin9X gets the value of AutoStartWin9X for the instance
func (instance *Win32_WMISetting) GetPropertyAutoStartWin9X() (value uint32, err error) {
	retValue, err := instance.GetProperty("AutoStartWin9X")
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

// SetBackupInterval sets the value of BackupInterval for the instance
func (instance *Win32_WMISetting) SetPropertyBackupInterval(value uint32) (err error) {
	return instance.SetProperty("BackupInterval", (value))
}

// GetBackupInterval gets the value of BackupInterval for the instance
func (instance *Win32_WMISetting) GetPropertyBackupInterval() (value uint32, err error) {
	retValue, err := instance.GetProperty("BackupInterval")
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

// SetBackupLastTime sets the value of BackupLastTime for the instance
func (instance *Win32_WMISetting) SetPropertyBackupLastTime(value string) (err error) {
	return instance.SetProperty("BackupLastTime", (value))
}

// GetBackupLastTime gets the value of BackupLastTime for the instance
func (instance *Win32_WMISetting) GetPropertyBackupLastTime() (value string, err error) {
	retValue, err := instance.GetProperty("BackupLastTime")
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

// SetBuildVersion sets the value of BuildVersion for the instance
func (instance *Win32_WMISetting) SetPropertyBuildVersion(value string) (err error) {
	return instance.SetProperty("BuildVersion", (value))
}

// GetBuildVersion gets the value of BuildVersion for the instance
func (instance *Win32_WMISetting) GetPropertyBuildVersion() (value string, err error) {
	retValue, err := instance.GetProperty("BuildVersion")
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

// SetDatabaseDirectory sets the value of DatabaseDirectory for the instance
func (instance *Win32_WMISetting) SetPropertyDatabaseDirectory(value string) (err error) {
	return instance.SetProperty("DatabaseDirectory", (value))
}

// GetDatabaseDirectory gets the value of DatabaseDirectory for the instance
func (instance *Win32_WMISetting) GetPropertyDatabaseDirectory() (value string, err error) {
	retValue, err := instance.GetProperty("DatabaseDirectory")
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

// SetDatabaseMaxSize sets the value of DatabaseMaxSize for the instance
func (instance *Win32_WMISetting) SetPropertyDatabaseMaxSize(value uint32) (err error) {
	return instance.SetProperty("DatabaseMaxSize", (value))
}

// GetDatabaseMaxSize gets the value of DatabaseMaxSize for the instance
func (instance *Win32_WMISetting) GetPropertyDatabaseMaxSize() (value uint32, err error) {
	retValue, err := instance.GetProperty("DatabaseMaxSize")
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

// SetEnableAnonWin9xConnections sets the value of EnableAnonWin9xConnections for the instance
func (instance *Win32_WMISetting) SetPropertyEnableAnonWin9xConnections(value bool) (err error) {
	return instance.SetProperty("EnableAnonWin9xConnections", (value))
}

// GetEnableAnonWin9xConnections gets the value of EnableAnonWin9xConnections for the instance
func (instance *Win32_WMISetting) GetPropertyEnableAnonWin9xConnections() (value bool, err error) {
	retValue, err := instance.GetProperty("EnableAnonWin9xConnections")
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

// SetEnableEvents sets the value of EnableEvents for the instance
func (instance *Win32_WMISetting) SetPropertyEnableEvents(value bool) (err error) {
	return instance.SetProperty("EnableEvents", (value))
}

// GetEnableEvents gets the value of EnableEvents for the instance
func (instance *Win32_WMISetting) GetPropertyEnableEvents() (value bool, err error) {
	retValue, err := instance.GetProperty("EnableEvents")
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

// SetEnableStartupHeapPreallocation sets the value of EnableStartupHeapPreallocation for the instance
func (instance *Win32_WMISetting) SetPropertyEnableStartupHeapPreallocation(value bool) (err error) {
	return instance.SetProperty("EnableStartupHeapPreallocation", (value))
}

// GetEnableStartupHeapPreallocation gets the value of EnableStartupHeapPreallocation for the instance
func (instance *Win32_WMISetting) GetPropertyEnableStartupHeapPreallocation() (value bool, err error) {
	retValue, err := instance.GetProperty("EnableStartupHeapPreallocation")
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

// SetHighThresholdOnClientObjects sets the value of HighThresholdOnClientObjects for the instance
func (instance *Win32_WMISetting) SetPropertyHighThresholdOnClientObjects(value uint32) (err error) {
	return instance.SetProperty("HighThresholdOnClientObjects", (value))
}

// GetHighThresholdOnClientObjects gets the value of HighThresholdOnClientObjects for the instance
func (instance *Win32_WMISetting) GetPropertyHighThresholdOnClientObjects() (value uint32, err error) {
	retValue, err := instance.GetProperty("HighThresholdOnClientObjects")
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

// SetHighThresholdOnEvents sets the value of HighThresholdOnEvents for the instance
func (instance *Win32_WMISetting) SetPropertyHighThresholdOnEvents(value uint32) (err error) {
	return instance.SetProperty("HighThresholdOnEvents", (value))
}

// GetHighThresholdOnEvents gets the value of HighThresholdOnEvents for the instance
func (instance *Win32_WMISetting) GetPropertyHighThresholdOnEvents() (value uint32, err error) {
	retValue, err := instance.GetProperty("HighThresholdOnEvents")
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

// SetInstallationDirectory sets the value of InstallationDirectory for the instance
func (instance *Win32_WMISetting) SetPropertyInstallationDirectory(value string) (err error) {
	return instance.SetProperty("InstallationDirectory", (value))
}

// GetInstallationDirectory gets the value of InstallationDirectory for the instance
func (instance *Win32_WMISetting) GetPropertyInstallationDirectory() (value string, err error) {
	retValue, err := instance.GetProperty("InstallationDirectory")
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

// SetLastStartupHeapPreallocation sets the value of LastStartupHeapPreallocation for the instance
func (instance *Win32_WMISetting) SetPropertyLastStartupHeapPreallocation(value uint32) (err error) {
	return instance.SetProperty("LastStartupHeapPreallocation", (value))
}

// GetLastStartupHeapPreallocation gets the value of LastStartupHeapPreallocation for the instance
func (instance *Win32_WMISetting) GetPropertyLastStartupHeapPreallocation() (value uint32, err error) {
	retValue, err := instance.GetProperty("LastStartupHeapPreallocation")
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

// SetLoggingDirectory sets the value of LoggingDirectory for the instance
func (instance *Win32_WMISetting) SetPropertyLoggingDirectory(value string) (err error) {
	return instance.SetProperty("LoggingDirectory", (value))
}

// GetLoggingDirectory gets the value of LoggingDirectory for the instance
func (instance *Win32_WMISetting) GetPropertyLoggingDirectory() (value string, err error) {
	retValue, err := instance.GetProperty("LoggingDirectory")
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

// SetLoggingLevel sets the value of LoggingLevel for the instance
func (instance *Win32_WMISetting) SetPropertyLoggingLevel(value uint32) (err error) {
	return instance.SetProperty("LoggingLevel", (value))
}

// GetLoggingLevel gets the value of LoggingLevel for the instance
func (instance *Win32_WMISetting) GetPropertyLoggingLevel() (value uint32, err error) {
	retValue, err := instance.GetProperty("LoggingLevel")
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

// SetLowThresholdOnClientObjects sets the value of LowThresholdOnClientObjects for the instance
func (instance *Win32_WMISetting) SetPropertyLowThresholdOnClientObjects(value uint32) (err error) {
	return instance.SetProperty("LowThresholdOnClientObjects", (value))
}

// GetLowThresholdOnClientObjects gets the value of LowThresholdOnClientObjects for the instance
func (instance *Win32_WMISetting) GetPropertyLowThresholdOnClientObjects() (value uint32, err error) {
	retValue, err := instance.GetProperty("LowThresholdOnClientObjects")
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

// SetLowThresholdOnEvents sets the value of LowThresholdOnEvents for the instance
func (instance *Win32_WMISetting) SetPropertyLowThresholdOnEvents(value uint32) (err error) {
	return instance.SetProperty("LowThresholdOnEvents", (value))
}

// GetLowThresholdOnEvents gets the value of LowThresholdOnEvents for the instance
func (instance *Win32_WMISetting) GetPropertyLowThresholdOnEvents() (value uint32, err error) {
	retValue, err := instance.GetProperty("LowThresholdOnEvents")
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

// SetMaxLogFileSize sets the value of MaxLogFileSize for the instance
func (instance *Win32_WMISetting) SetPropertyMaxLogFileSize(value uint32) (err error) {
	return instance.SetProperty("MaxLogFileSize", (value))
}

// GetMaxLogFileSize gets the value of MaxLogFileSize for the instance
func (instance *Win32_WMISetting) GetPropertyMaxLogFileSize() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaxLogFileSize")
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

// SetMaxWaitOnClientObjects sets the value of MaxWaitOnClientObjects for the instance
func (instance *Win32_WMISetting) SetPropertyMaxWaitOnClientObjects(value uint32) (err error) {
	return instance.SetProperty("MaxWaitOnClientObjects", (value))
}

// GetMaxWaitOnClientObjects gets the value of MaxWaitOnClientObjects for the instance
func (instance *Win32_WMISetting) GetPropertyMaxWaitOnClientObjects() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaxWaitOnClientObjects")
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

// SetMaxWaitOnEvents sets the value of MaxWaitOnEvents for the instance
func (instance *Win32_WMISetting) SetPropertyMaxWaitOnEvents(value uint32) (err error) {
	return instance.SetProperty("MaxWaitOnEvents", (value))
}

// GetMaxWaitOnEvents gets the value of MaxWaitOnEvents for the instance
func (instance *Win32_WMISetting) GetPropertyMaxWaitOnEvents() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaxWaitOnEvents")
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

// SetMofSelfInstallDirectory sets the value of MofSelfInstallDirectory for the instance
func (instance *Win32_WMISetting) SetPropertyMofSelfInstallDirectory(value string) (err error) {
	return instance.SetProperty("MofSelfInstallDirectory", (value))
}

// GetMofSelfInstallDirectory gets the value of MofSelfInstallDirectory for the instance
func (instance *Win32_WMISetting) GetPropertyMofSelfInstallDirectory() (value string, err error) {
	retValue, err := instance.GetProperty("MofSelfInstallDirectory")
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
