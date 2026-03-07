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

// Win32_ClassicCOMClassSetting struct
type Win32_ClassicCOMClassSetting struct {
	*Win32_COMSetting

	//
	AppID string

	//
	AutoConvertToClsid string

	//
	AutoTreatAsClsid string

	//
	ComponentId string

	//
	Control bool

	//
	DefaultIcon string

	//
	InprocHandler string

	//
	InprocHandler32 string

	//
	InprocServer string

	//
	InprocServer32 string

	//
	Insertable bool

	//
	JavaClass bool

	//
	LocalServer string

	//
	LocalServer32 string

	//
	LongDisplayName string

	//
	ProgId string

	//
	ShortDisplayName string

	//
	ThreadingModel string

	//
	ToolBoxBitmap32 string

	//
	TreatAsClsid string

	//
	TypeLibraryId string

	//
	Version string

	//
	VersionIndependentProgId string
}

func NewWin32_ClassicCOMClassSettingEx1(instance *cim.WmiInstance) (newInstance *Win32_ClassicCOMClassSetting, err error) {
	tmp, err := NewWin32_COMSettingEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_ClassicCOMClassSetting{
		Win32_COMSetting: tmp,
	}
	return
}

func NewWin32_ClassicCOMClassSettingEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ClassicCOMClassSetting, err error) {
	tmp, err := NewWin32_COMSettingEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ClassicCOMClassSetting{
		Win32_COMSetting: tmp,
	}
	return
}

// SetAppID sets the value of AppID for the instance
func (instance *Win32_ClassicCOMClassSetting) SetPropertyAppID(value string) (err error) {
	return instance.SetProperty("AppID", (value))
}

// GetAppID gets the value of AppID for the instance
func (instance *Win32_ClassicCOMClassSetting) GetPropertyAppID() (value string, err error) {
	retValue, err := instance.GetProperty("AppID")
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

// SetAutoConvertToClsid sets the value of AutoConvertToClsid for the instance
func (instance *Win32_ClassicCOMClassSetting) SetPropertyAutoConvertToClsid(value string) (err error) {
	return instance.SetProperty("AutoConvertToClsid", (value))
}

// GetAutoConvertToClsid gets the value of AutoConvertToClsid for the instance
func (instance *Win32_ClassicCOMClassSetting) GetPropertyAutoConvertToClsid() (value string, err error) {
	retValue, err := instance.GetProperty("AutoConvertToClsid")
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

// SetAutoTreatAsClsid sets the value of AutoTreatAsClsid for the instance
func (instance *Win32_ClassicCOMClassSetting) SetPropertyAutoTreatAsClsid(value string) (err error) {
	return instance.SetProperty("AutoTreatAsClsid", (value))
}

// GetAutoTreatAsClsid gets the value of AutoTreatAsClsid for the instance
func (instance *Win32_ClassicCOMClassSetting) GetPropertyAutoTreatAsClsid() (value string, err error) {
	retValue, err := instance.GetProperty("AutoTreatAsClsid")
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

// SetComponentId sets the value of ComponentId for the instance
func (instance *Win32_ClassicCOMClassSetting) SetPropertyComponentId(value string) (err error) {
	return instance.SetProperty("ComponentId", (value))
}

// GetComponentId gets the value of ComponentId for the instance
func (instance *Win32_ClassicCOMClassSetting) GetPropertyComponentId() (value string, err error) {
	retValue, err := instance.GetProperty("ComponentId")
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

// SetControl sets the value of Control for the instance
func (instance *Win32_ClassicCOMClassSetting) SetPropertyControl(value bool) (err error) {
	return instance.SetProperty("Control", (value))
}

// GetControl gets the value of Control for the instance
func (instance *Win32_ClassicCOMClassSetting) GetPropertyControl() (value bool, err error) {
	retValue, err := instance.GetProperty("Control")
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

// SetDefaultIcon sets the value of DefaultIcon for the instance
func (instance *Win32_ClassicCOMClassSetting) SetPropertyDefaultIcon(value string) (err error) {
	return instance.SetProperty("DefaultIcon", (value))
}

// GetDefaultIcon gets the value of DefaultIcon for the instance
func (instance *Win32_ClassicCOMClassSetting) GetPropertyDefaultIcon() (value string, err error) {
	retValue, err := instance.GetProperty("DefaultIcon")
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

// SetInprocHandler sets the value of InprocHandler for the instance
func (instance *Win32_ClassicCOMClassSetting) SetPropertyInprocHandler(value string) (err error) {
	return instance.SetProperty("InprocHandler", (value))
}

// GetInprocHandler gets the value of InprocHandler for the instance
func (instance *Win32_ClassicCOMClassSetting) GetPropertyInprocHandler() (value string, err error) {
	retValue, err := instance.GetProperty("InprocHandler")
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

// SetInprocHandler32 sets the value of InprocHandler32 for the instance
func (instance *Win32_ClassicCOMClassSetting) SetPropertyInprocHandler32(value string) (err error) {
	return instance.SetProperty("InprocHandler32", (value))
}

// GetInprocHandler32 gets the value of InprocHandler32 for the instance
func (instance *Win32_ClassicCOMClassSetting) GetPropertyInprocHandler32() (value string, err error) {
	retValue, err := instance.GetProperty("InprocHandler32")
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

// SetInprocServer sets the value of InprocServer for the instance
func (instance *Win32_ClassicCOMClassSetting) SetPropertyInprocServer(value string) (err error) {
	return instance.SetProperty("InprocServer", (value))
}

// GetInprocServer gets the value of InprocServer for the instance
func (instance *Win32_ClassicCOMClassSetting) GetPropertyInprocServer() (value string, err error) {
	retValue, err := instance.GetProperty("InprocServer")
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

// SetInprocServer32 sets the value of InprocServer32 for the instance
func (instance *Win32_ClassicCOMClassSetting) SetPropertyInprocServer32(value string) (err error) {
	return instance.SetProperty("InprocServer32", (value))
}

// GetInprocServer32 gets the value of InprocServer32 for the instance
func (instance *Win32_ClassicCOMClassSetting) GetPropertyInprocServer32() (value string, err error) {
	retValue, err := instance.GetProperty("InprocServer32")
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

// SetInsertable sets the value of Insertable for the instance
func (instance *Win32_ClassicCOMClassSetting) SetPropertyInsertable(value bool) (err error) {
	return instance.SetProperty("Insertable", (value))
}

// GetInsertable gets the value of Insertable for the instance
func (instance *Win32_ClassicCOMClassSetting) GetPropertyInsertable() (value bool, err error) {
	retValue, err := instance.GetProperty("Insertable")
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

// SetJavaClass sets the value of JavaClass for the instance
func (instance *Win32_ClassicCOMClassSetting) SetPropertyJavaClass(value bool) (err error) {
	return instance.SetProperty("JavaClass", (value))
}

// GetJavaClass gets the value of JavaClass for the instance
func (instance *Win32_ClassicCOMClassSetting) GetPropertyJavaClass() (value bool, err error) {
	retValue, err := instance.GetProperty("JavaClass")
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

// SetLocalServer sets the value of LocalServer for the instance
func (instance *Win32_ClassicCOMClassSetting) SetPropertyLocalServer(value string) (err error) {
	return instance.SetProperty("LocalServer", (value))
}

// GetLocalServer gets the value of LocalServer for the instance
func (instance *Win32_ClassicCOMClassSetting) GetPropertyLocalServer() (value string, err error) {
	retValue, err := instance.GetProperty("LocalServer")
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

// SetLocalServer32 sets the value of LocalServer32 for the instance
func (instance *Win32_ClassicCOMClassSetting) SetPropertyLocalServer32(value string) (err error) {
	return instance.SetProperty("LocalServer32", (value))
}

// GetLocalServer32 gets the value of LocalServer32 for the instance
func (instance *Win32_ClassicCOMClassSetting) GetPropertyLocalServer32() (value string, err error) {
	retValue, err := instance.GetProperty("LocalServer32")
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

// SetLongDisplayName sets the value of LongDisplayName for the instance
func (instance *Win32_ClassicCOMClassSetting) SetPropertyLongDisplayName(value string) (err error) {
	return instance.SetProperty("LongDisplayName", (value))
}

// GetLongDisplayName gets the value of LongDisplayName for the instance
func (instance *Win32_ClassicCOMClassSetting) GetPropertyLongDisplayName() (value string, err error) {
	retValue, err := instance.GetProperty("LongDisplayName")
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

// SetProgId sets the value of ProgId for the instance
func (instance *Win32_ClassicCOMClassSetting) SetPropertyProgId(value string) (err error) {
	return instance.SetProperty("ProgId", (value))
}

// GetProgId gets the value of ProgId for the instance
func (instance *Win32_ClassicCOMClassSetting) GetPropertyProgId() (value string, err error) {
	retValue, err := instance.GetProperty("ProgId")
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

// SetShortDisplayName sets the value of ShortDisplayName for the instance
func (instance *Win32_ClassicCOMClassSetting) SetPropertyShortDisplayName(value string) (err error) {
	return instance.SetProperty("ShortDisplayName", (value))
}

// GetShortDisplayName gets the value of ShortDisplayName for the instance
func (instance *Win32_ClassicCOMClassSetting) GetPropertyShortDisplayName() (value string, err error) {
	retValue, err := instance.GetProperty("ShortDisplayName")
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

// SetThreadingModel sets the value of ThreadingModel for the instance
func (instance *Win32_ClassicCOMClassSetting) SetPropertyThreadingModel(value string) (err error) {
	return instance.SetProperty("ThreadingModel", (value))
}

// GetThreadingModel gets the value of ThreadingModel for the instance
func (instance *Win32_ClassicCOMClassSetting) GetPropertyThreadingModel() (value string, err error) {
	retValue, err := instance.GetProperty("ThreadingModel")
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

// SetToolBoxBitmap32 sets the value of ToolBoxBitmap32 for the instance
func (instance *Win32_ClassicCOMClassSetting) SetPropertyToolBoxBitmap32(value string) (err error) {
	return instance.SetProperty("ToolBoxBitmap32", (value))
}

// GetToolBoxBitmap32 gets the value of ToolBoxBitmap32 for the instance
func (instance *Win32_ClassicCOMClassSetting) GetPropertyToolBoxBitmap32() (value string, err error) {
	retValue, err := instance.GetProperty("ToolBoxBitmap32")
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

// SetTreatAsClsid sets the value of TreatAsClsid for the instance
func (instance *Win32_ClassicCOMClassSetting) SetPropertyTreatAsClsid(value string) (err error) {
	return instance.SetProperty("TreatAsClsid", (value))
}

// GetTreatAsClsid gets the value of TreatAsClsid for the instance
func (instance *Win32_ClassicCOMClassSetting) GetPropertyTreatAsClsid() (value string, err error) {
	retValue, err := instance.GetProperty("TreatAsClsid")
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

// SetTypeLibraryId sets the value of TypeLibraryId for the instance
func (instance *Win32_ClassicCOMClassSetting) SetPropertyTypeLibraryId(value string) (err error) {
	return instance.SetProperty("TypeLibraryId", (value))
}

// GetTypeLibraryId gets the value of TypeLibraryId for the instance
func (instance *Win32_ClassicCOMClassSetting) GetPropertyTypeLibraryId() (value string, err error) {
	retValue, err := instance.GetProperty("TypeLibraryId")
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

// SetVersion sets the value of Version for the instance
func (instance *Win32_ClassicCOMClassSetting) SetPropertyVersion(value string) (err error) {
	return instance.SetProperty("Version", (value))
}

// GetVersion gets the value of Version for the instance
func (instance *Win32_ClassicCOMClassSetting) GetPropertyVersion() (value string, err error) {
	retValue, err := instance.GetProperty("Version")
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

// SetVersionIndependentProgId sets the value of VersionIndependentProgId for the instance
func (instance *Win32_ClassicCOMClassSetting) SetPropertyVersionIndependentProgId(value string) (err error) {
	return instance.SetProperty("VersionIndependentProgId", (value))
}

// GetVersionIndependentProgId gets the value of VersionIndependentProgId for the instance
func (instance *Win32_ClassicCOMClassSetting) GetPropertyVersionIndependentProgId() (value string, err error) {
	retValue, err := instance.GetProperty("VersionIndependentProgId")
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
