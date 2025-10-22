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

// Win32_ShadowContext struct
type Win32_ShadowContext struct {
	*CIM_Setting

	//
	ClientAccessible bool

	//
	Differential bool

	//
	ExposedLocally bool

	//
	ExposedRemotely bool

	//
	HardwareAssisted bool

	//
	Imported bool

	//
	Name string

	//
	NoAutoRelease bool

	//
	NotSurfaced bool

	//
	NoWriters bool

	//
	Persistent bool

	//
	Plex bool

	//
	Transportable bool
}

func NewWin32_ShadowContextEx1(instance *cim.WmiInstance) (newInstance *Win32_ShadowContext, err error) {
	tmp, err := NewCIM_SettingEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_ShadowContext{
		CIM_Setting: tmp,
	}
	return
}

func NewWin32_ShadowContextEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ShadowContext, err error) {
	tmp, err := NewCIM_SettingEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ShadowContext{
		CIM_Setting: tmp,
	}
	return
}

// SetClientAccessible sets the value of ClientAccessible for the instance
func (instance *Win32_ShadowContext) SetPropertyClientAccessible(value bool) (err error) {
	return instance.SetProperty("ClientAccessible", (value))
}

// GetClientAccessible gets the value of ClientAccessible for the instance
func (instance *Win32_ShadowContext) GetPropertyClientAccessible() (value bool, err error) {
	retValue, err := instance.GetProperty("ClientAccessible")
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

// SetDifferential sets the value of Differential for the instance
func (instance *Win32_ShadowContext) SetPropertyDifferential(value bool) (err error) {
	return instance.SetProperty("Differential", (value))
}

// GetDifferential gets the value of Differential for the instance
func (instance *Win32_ShadowContext) GetPropertyDifferential() (value bool, err error) {
	retValue, err := instance.GetProperty("Differential")
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

// SetExposedLocally sets the value of ExposedLocally for the instance
func (instance *Win32_ShadowContext) SetPropertyExposedLocally(value bool) (err error) {
	return instance.SetProperty("ExposedLocally", (value))
}

// GetExposedLocally gets the value of ExposedLocally for the instance
func (instance *Win32_ShadowContext) GetPropertyExposedLocally() (value bool, err error) {
	retValue, err := instance.GetProperty("ExposedLocally")
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

// SetExposedRemotely sets the value of ExposedRemotely for the instance
func (instance *Win32_ShadowContext) SetPropertyExposedRemotely(value bool) (err error) {
	return instance.SetProperty("ExposedRemotely", (value))
}

// GetExposedRemotely gets the value of ExposedRemotely for the instance
func (instance *Win32_ShadowContext) GetPropertyExposedRemotely() (value bool, err error) {
	retValue, err := instance.GetProperty("ExposedRemotely")
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

// SetHardwareAssisted sets the value of HardwareAssisted for the instance
func (instance *Win32_ShadowContext) SetPropertyHardwareAssisted(value bool) (err error) {
	return instance.SetProperty("HardwareAssisted", (value))
}

// GetHardwareAssisted gets the value of HardwareAssisted for the instance
func (instance *Win32_ShadowContext) GetPropertyHardwareAssisted() (value bool, err error) {
	retValue, err := instance.GetProperty("HardwareAssisted")
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

// SetImported sets the value of Imported for the instance
func (instance *Win32_ShadowContext) SetPropertyImported(value bool) (err error) {
	return instance.SetProperty("Imported", (value))
}

// GetImported gets the value of Imported for the instance
func (instance *Win32_ShadowContext) GetPropertyImported() (value bool, err error) {
	retValue, err := instance.GetProperty("Imported")
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

// SetName sets the value of Name for the instance
func (instance *Win32_ShadowContext) SetPropertyName(value string) (err error) {
	return instance.SetProperty("Name", (value))
}

// GetName gets the value of Name for the instance
func (instance *Win32_ShadowContext) GetPropertyName() (value string, err error) {
	retValue, err := instance.GetProperty("Name")
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

// SetNoAutoRelease sets the value of NoAutoRelease for the instance
func (instance *Win32_ShadowContext) SetPropertyNoAutoRelease(value bool) (err error) {
	return instance.SetProperty("NoAutoRelease", (value))
}

// GetNoAutoRelease gets the value of NoAutoRelease for the instance
func (instance *Win32_ShadowContext) GetPropertyNoAutoRelease() (value bool, err error) {
	retValue, err := instance.GetProperty("NoAutoRelease")
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

// SetNotSurfaced sets the value of NotSurfaced for the instance
func (instance *Win32_ShadowContext) SetPropertyNotSurfaced(value bool) (err error) {
	return instance.SetProperty("NotSurfaced", (value))
}

// GetNotSurfaced gets the value of NotSurfaced for the instance
func (instance *Win32_ShadowContext) GetPropertyNotSurfaced() (value bool, err error) {
	retValue, err := instance.GetProperty("NotSurfaced")
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

// SetNoWriters sets the value of NoWriters for the instance
func (instance *Win32_ShadowContext) SetPropertyNoWriters(value bool) (err error) {
	return instance.SetProperty("NoWriters", (value))
}

// GetNoWriters gets the value of NoWriters for the instance
func (instance *Win32_ShadowContext) GetPropertyNoWriters() (value bool, err error) {
	retValue, err := instance.GetProperty("NoWriters")
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

// SetPersistent sets the value of Persistent for the instance
func (instance *Win32_ShadowContext) SetPropertyPersistent(value bool) (err error) {
	return instance.SetProperty("Persistent", (value))
}

// GetPersistent gets the value of Persistent for the instance
func (instance *Win32_ShadowContext) GetPropertyPersistent() (value bool, err error) {
	retValue, err := instance.GetProperty("Persistent")
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

// SetPlex sets the value of Plex for the instance
func (instance *Win32_ShadowContext) SetPropertyPlex(value bool) (err error) {
	return instance.SetProperty("Plex", (value))
}

// GetPlex gets the value of Plex for the instance
func (instance *Win32_ShadowContext) GetPropertyPlex() (value bool, err error) {
	retValue, err := instance.GetProperty("Plex")
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

// SetTransportable sets the value of Transportable for the instance
func (instance *Win32_ShadowContext) SetPropertyTransportable(value bool) (err error) {
	return instance.SetProperty("Transportable", (value))
}

// GetTransportable gets the value of Transportable for the instance
func (instance *Win32_ShadowContext) GetPropertyTransportable() (value bool, err error) {
	retValue, err := instance.GetProperty("Transportable")
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
