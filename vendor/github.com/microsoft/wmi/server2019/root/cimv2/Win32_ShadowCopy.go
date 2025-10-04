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

// Win32_ShadowCopy struct
type Win32_ShadowCopy struct {
	*CIM_LogicalElement

	//
	ClientAccessible bool

	//
	Count uint32

	//
	DeviceObject string

	//
	Differential bool

	//
	ExposedLocally bool

	//
	ExposedName string

	//
	ExposedPath string

	//
	ExposedRemotely bool

	//
	HardwareAssisted bool

	//
	ID string

	//
	Imported bool

	//
	NoAutoRelease bool

	//
	NotSurfaced bool

	//
	NoWriters bool

	//
	OriginatingMachine string

	//
	Persistent bool

	//
	Plex bool

	//
	ProviderID string

	//
	ServiceMachine string

	//
	SetID string

	//
	State uint32

	//
	Transportable bool

	//
	VolumeName string
}

func NewWin32_ShadowCopyEx1(instance *cim.WmiInstance) (newInstance *Win32_ShadowCopy, err error) {
	tmp, err := NewCIM_LogicalElementEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_ShadowCopy{
		CIM_LogicalElement: tmp,
	}
	return
}

func NewWin32_ShadowCopyEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ShadowCopy, err error) {
	tmp, err := NewCIM_LogicalElementEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ShadowCopy{
		CIM_LogicalElement: tmp,
	}
	return
}

// SetClientAccessible sets the value of ClientAccessible for the instance
func (instance *Win32_ShadowCopy) SetPropertyClientAccessible(value bool) (err error) {
	return instance.SetProperty("ClientAccessible", (value))
}

// GetClientAccessible gets the value of ClientAccessible for the instance
func (instance *Win32_ShadowCopy) GetPropertyClientAccessible() (value bool, err error) {
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

// SetCount sets the value of Count for the instance
func (instance *Win32_ShadowCopy) SetPropertyCount(value uint32) (err error) {
	return instance.SetProperty("Count", (value))
}

// GetCount gets the value of Count for the instance
func (instance *Win32_ShadowCopy) GetPropertyCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("Count")
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

// SetDeviceObject sets the value of DeviceObject for the instance
func (instance *Win32_ShadowCopy) SetPropertyDeviceObject(value string) (err error) {
	return instance.SetProperty("DeviceObject", (value))
}

// GetDeviceObject gets the value of DeviceObject for the instance
func (instance *Win32_ShadowCopy) GetPropertyDeviceObject() (value string, err error) {
	retValue, err := instance.GetProperty("DeviceObject")
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

// SetDifferential sets the value of Differential for the instance
func (instance *Win32_ShadowCopy) SetPropertyDifferential(value bool) (err error) {
	return instance.SetProperty("Differential", (value))
}

// GetDifferential gets the value of Differential for the instance
func (instance *Win32_ShadowCopy) GetPropertyDifferential() (value bool, err error) {
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
func (instance *Win32_ShadowCopy) SetPropertyExposedLocally(value bool) (err error) {
	return instance.SetProperty("ExposedLocally", (value))
}

// GetExposedLocally gets the value of ExposedLocally for the instance
func (instance *Win32_ShadowCopy) GetPropertyExposedLocally() (value bool, err error) {
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

// SetExposedName sets the value of ExposedName for the instance
func (instance *Win32_ShadowCopy) SetPropertyExposedName(value string) (err error) {
	return instance.SetProperty("ExposedName", (value))
}

// GetExposedName gets the value of ExposedName for the instance
func (instance *Win32_ShadowCopy) GetPropertyExposedName() (value string, err error) {
	retValue, err := instance.GetProperty("ExposedName")
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

// SetExposedPath sets the value of ExposedPath for the instance
func (instance *Win32_ShadowCopy) SetPropertyExposedPath(value string) (err error) {
	return instance.SetProperty("ExposedPath", (value))
}

// GetExposedPath gets the value of ExposedPath for the instance
func (instance *Win32_ShadowCopy) GetPropertyExposedPath() (value string, err error) {
	retValue, err := instance.GetProperty("ExposedPath")
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

// SetExposedRemotely sets the value of ExposedRemotely for the instance
func (instance *Win32_ShadowCopy) SetPropertyExposedRemotely(value bool) (err error) {
	return instance.SetProperty("ExposedRemotely", (value))
}

// GetExposedRemotely gets the value of ExposedRemotely for the instance
func (instance *Win32_ShadowCopy) GetPropertyExposedRemotely() (value bool, err error) {
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
func (instance *Win32_ShadowCopy) SetPropertyHardwareAssisted(value bool) (err error) {
	return instance.SetProperty("HardwareAssisted", (value))
}

// GetHardwareAssisted gets the value of HardwareAssisted for the instance
func (instance *Win32_ShadowCopy) GetPropertyHardwareAssisted() (value bool, err error) {
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

// SetID sets the value of ID for the instance
func (instance *Win32_ShadowCopy) SetPropertyID(value string) (err error) {
	return instance.SetProperty("ID", (value))
}

// GetID gets the value of ID for the instance
func (instance *Win32_ShadowCopy) GetPropertyID() (value string, err error) {
	retValue, err := instance.GetProperty("ID")
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

// SetImported sets the value of Imported for the instance
func (instance *Win32_ShadowCopy) SetPropertyImported(value bool) (err error) {
	return instance.SetProperty("Imported", (value))
}

// GetImported gets the value of Imported for the instance
func (instance *Win32_ShadowCopy) GetPropertyImported() (value bool, err error) {
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

// SetNoAutoRelease sets the value of NoAutoRelease for the instance
func (instance *Win32_ShadowCopy) SetPropertyNoAutoRelease(value bool) (err error) {
	return instance.SetProperty("NoAutoRelease", (value))
}

// GetNoAutoRelease gets the value of NoAutoRelease for the instance
func (instance *Win32_ShadowCopy) GetPropertyNoAutoRelease() (value bool, err error) {
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
func (instance *Win32_ShadowCopy) SetPropertyNotSurfaced(value bool) (err error) {
	return instance.SetProperty("NotSurfaced", (value))
}

// GetNotSurfaced gets the value of NotSurfaced for the instance
func (instance *Win32_ShadowCopy) GetPropertyNotSurfaced() (value bool, err error) {
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
func (instance *Win32_ShadowCopy) SetPropertyNoWriters(value bool) (err error) {
	return instance.SetProperty("NoWriters", (value))
}

// GetNoWriters gets the value of NoWriters for the instance
func (instance *Win32_ShadowCopy) GetPropertyNoWriters() (value bool, err error) {
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

// SetOriginatingMachine sets the value of OriginatingMachine for the instance
func (instance *Win32_ShadowCopy) SetPropertyOriginatingMachine(value string) (err error) {
	return instance.SetProperty("OriginatingMachine", (value))
}

// GetOriginatingMachine gets the value of OriginatingMachine for the instance
func (instance *Win32_ShadowCopy) GetPropertyOriginatingMachine() (value string, err error) {
	retValue, err := instance.GetProperty("OriginatingMachine")
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

// SetPersistent sets the value of Persistent for the instance
func (instance *Win32_ShadowCopy) SetPropertyPersistent(value bool) (err error) {
	return instance.SetProperty("Persistent", (value))
}

// GetPersistent gets the value of Persistent for the instance
func (instance *Win32_ShadowCopy) GetPropertyPersistent() (value bool, err error) {
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
func (instance *Win32_ShadowCopy) SetPropertyPlex(value bool) (err error) {
	return instance.SetProperty("Plex", (value))
}

// GetPlex gets the value of Plex for the instance
func (instance *Win32_ShadowCopy) GetPropertyPlex() (value bool, err error) {
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

// SetProviderID sets the value of ProviderID for the instance
func (instance *Win32_ShadowCopy) SetPropertyProviderID(value string) (err error) {
	return instance.SetProperty("ProviderID", (value))
}

// GetProviderID gets the value of ProviderID for the instance
func (instance *Win32_ShadowCopy) GetPropertyProviderID() (value string, err error) {
	retValue, err := instance.GetProperty("ProviderID")
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

// SetServiceMachine sets the value of ServiceMachine for the instance
func (instance *Win32_ShadowCopy) SetPropertyServiceMachine(value string) (err error) {
	return instance.SetProperty("ServiceMachine", (value))
}

// GetServiceMachine gets the value of ServiceMachine for the instance
func (instance *Win32_ShadowCopy) GetPropertyServiceMachine() (value string, err error) {
	retValue, err := instance.GetProperty("ServiceMachine")
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

// SetSetID sets the value of SetID for the instance
func (instance *Win32_ShadowCopy) SetPropertySetID(value string) (err error) {
	return instance.SetProperty("SetID", (value))
}

// GetSetID gets the value of SetID for the instance
func (instance *Win32_ShadowCopy) GetPropertySetID() (value string, err error) {
	retValue, err := instance.GetProperty("SetID")
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

// SetState sets the value of State for the instance
func (instance *Win32_ShadowCopy) SetPropertyState(value uint32) (err error) {
	return instance.SetProperty("State", (value))
}

// GetState gets the value of State for the instance
func (instance *Win32_ShadowCopy) GetPropertyState() (value uint32, err error) {
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

// SetTransportable sets the value of Transportable for the instance
func (instance *Win32_ShadowCopy) SetPropertyTransportable(value bool) (err error) {
	return instance.SetProperty("Transportable", (value))
}

// GetTransportable gets the value of Transportable for the instance
func (instance *Win32_ShadowCopy) GetPropertyTransportable() (value bool, err error) {
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

// SetVolumeName sets the value of VolumeName for the instance
func (instance *Win32_ShadowCopy) SetPropertyVolumeName(value string) (err error) {
	return instance.SetProperty("VolumeName", (value))
}

// GetVolumeName gets the value of VolumeName for the instance
func (instance *Win32_ShadowCopy) GetPropertyVolumeName() (value string, err error) {
	retValue, err := instance.GetProperty("VolumeName")
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

//

// <param name="Context" type="string "></param>
// <param name="Volume" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
// <param name="ShadowID" type="string "></param>
func (instance *Win32_ShadowCopy) Create( /* IN */ Volume string,
	/* IN */ Context string,
	/* OUT */ ShadowID string) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("Create", Volume, Context)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="ForceDismount" type="bool "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_ShadowCopy) Revert( /* IN */ ForceDismount bool) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Revert", ForceDismount)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}
