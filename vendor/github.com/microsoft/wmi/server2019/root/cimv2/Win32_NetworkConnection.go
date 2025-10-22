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

// Win32_NetworkConnection struct
type Win32_NetworkConnection struct {
	*CIM_LogicalElement

	//
	AccessMask uint32

	//
	Comment string

	//
	ConnectionState string

	//
	ConnectionType string

	//
	DisplayType string

	//
	LocalName string

	//
	Persistent bool

	//
	ProviderName string

	//
	RemoteName string

	//
	RemotePath string

	//
	ResourceType string

	//
	UserName string
}

func NewWin32_NetworkConnectionEx1(instance *cim.WmiInstance) (newInstance *Win32_NetworkConnection, err error) {
	tmp, err := NewCIM_LogicalElementEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_NetworkConnection{
		CIM_LogicalElement: tmp,
	}
	return
}

func NewWin32_NetworkConnectionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_NetworkConnection, err error) {
	tmp, err := NewCIM_LogicalElementEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_NetworkConnection{
		CIM_LogicalElement: tmp,
	}
	return
}

// SetAccessMask sets the value of AccessMask for the instance
func (instance *Win32_NetworkConnection) SetPropertyAccessMask(value uint32) (err error) {
	return instance.SetProperty("AccessMask", (value))
}

// GetAccessMask gets the value of AccessMask for the instance
func (instance *Win32_NetworkConnection) GetPropertyAccessMask() (value uint32, err error) {
	retValue, err := instance.GetProperty("AccessMask")
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

// SetComment sets the value of Comment for the instance
func (instance *Win32_NetworkConnection) SetPropertyComment(value string) (err error) {
	return instance.SetProperty("Comment", (value))
}

// GetComment gets the value of Comment for the instance
func (instance *Win32_NetworkConnection) GetPropertyComment() (value string, err error) {
	retValue, err := instance.GetProperty("Comment")
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

// SetConnectionState sets the value of ConnectionState for the instance
func (instance *Win32_NetworkConnection) SetPropertyConnectionState(value string) (err error) {
	return instance.SetProperty("ConnectionState", (value))
}

// GetConnectionState gets the value of ConnectionState for the instance
func (instance *Win32_NetworkConnection) GetPropertyConnectionState() (value string, err error) {
	retValue, err := instance.GetProperty("ConnectionState")
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

// SetConnectionType sets the value of ConnectionType for the instance
func (instance *Win32_NetworkConnection) SetPropertyConnectionType(value string) (err error) {
	return instance.SetProperty("ConnectionType", (value))
}

// GetConnectionType gets the value of ConnectionType for the instance
func (instance *Win32_NetworkConnection) GetPropertyConnectionType() (value string, err error) {
	retValue, err := instance.GetProperty("ConnectionType")
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

// SetDisplayType sets the value of DisplayType for the instance
func (instance *Win32_NetworkConnection) SetPropertyDisplayType(value string) (err error) {
	return instance.SetProperty("DisplayType", (value))
}

// GetDisplayType gets the value of DisplayType for the instance
func (instance *Win32_NetworkConnection) GetPropertyDisplayType() (value string, err error) {
	retValue, err := instance.GetProperty("DisplayType")
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

// SetLocalName sets the value of LocalName for the instance
func (instance *Win32_NetworkConnection) SetPropertyLocalName(value string) (err error) {
	return instance.SetProperty("LocalName", (value))
}

// GetLocalName gets the value of LocalName for the instance
func (instance *Win32_NetworkConnection) GetPropertyLocalName() (value string, err error) {
	retValue, err := instance.GetProperty("LocalName")
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
func (instance *Win32_NetworkConnection) SetPropertyPersistent(value bool) (err error) {
	return instance.SetProperty("Persistent", (value))
}

// GetPersistent gets the value of Persistent for the instance
func (instance *Win32_NetworkConnection) GetPropertyPersistent() (value bool, err error) {
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

// SetProviderName sets the value of ProviderName for the instance
func (instance *Win32_NetworkConnection) SetPropertyProviderName(value string) (err error) {
	return instance.SetProperty("ProviderName", (value))
}

// GetProviderName gets the value of ProviderName for the instance
func (instance *Win32_NetworkConnection) GetPropertyProviderName() (value string, err error) {
	retValue, err := instance.GetProperty("ProviderName")
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

// SetRemoteName sets the value of RemoteName for the instance
func (instance *Win32_NetworkConnection) SetPropertyRemoteName(value string) (err error) {
	return instance.SetProperty("RemoteName", (value))
}

// GetRemoteName gets the value of RemoteName for the instance
func (instance *Win32_NetworkConnection) GetPropertyRemoteName() (value string, err error) {
	retValue, err := instance.GetProperty("RemoteName")
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

// SetRemotePath sets the value of RemotePath for the instance
func (instance *Win32_NetworkConnection) SetPropertyRemotePath(value string) (err error) {
	return instance.SetProperty("RemotePath", (value))
}

// GetRemotePath gets the value of RemotePath for the instance
func (instance *Win32_NetworkConnection) GetPropertyRemotePath() (value string, err error) {
	retValue, err := instance.GetProperty("RemotePath")
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

// SetResourceType sets the value of ResourceType for the instance
func (instance *Win32_NetworkConnection) SetPropertyResourceType(value string) (err error) {
	return instance.SetProperty("ResourceType", (value))
}

// GetResourceType gets the value of ResourceType for the instance
func (instance *Win32_NetworkConnection) GetPropertyResourceType() (value string, err error) {
	retValue, err := instance.GetProperty("ResourceType")
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

// SetUserName sets the value of UserName for the instance
func (instance *Win32_NetworkConnection) SetPropertyUserName(value string) (err error) {
	return instance.SetProperty("UserName", (value))
}

// GetUserName gets the value of UserName for the instance
func (instance *Win32_NetworkConnection) GetPropertyUserName() (value string, err error) {
	retValue, err := instance.GetProperty("UserName")
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
