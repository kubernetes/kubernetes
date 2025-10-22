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

// MSFT_iSCSISession struct
type MSFT_iSCSISession struct {
	*cim.WmiInstance

	//
	AuthenticationType string

	//
	InitiatorInstanceName string

	//
	InitiatorNodeAddress string

	//
	InitiatorPortalAddress string

	//
	InitiatorSideIdentifier string

	//
	IsConnected bool

	//
	IsDataDigest bool

	//
	IsDiscovered bool

	//
	IsHeaderDigest bool

	//
	IsPersistent bool

	//
	NumberOfConnections uint32

	//
	SessionIdentifier string

	//
	TargetNodeAddress string

	//
	TargetSideIdentifier string
}

func NewMSFT_iSCSISessionEx1(instance *cim.WmiInstance) (newInstance *MSFT_iSCSISession, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_iSCSISession{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_iSCSISessionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_iSCSISession, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_iSCSISession{
		WmiInstance: tmp,
	}
	return
}

// SetAuthenticationType sets the value of AuthenticationType for the instance
func (instance *MSFT_iSCSISession) SetPropertyAuthenticationType(value string) (err error) {
	return instance.SetProperty("AuthenticationType", (value))
}

// GetAuthenticationType gets the value of AuthenticationType for the instance
func (instance *MSFT_iSCSISession) GetPropertyAuthenticationType() (value string, err error) {
	retValue, err := instance.GetProperty("AuthenticationType")
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

// SetInitiatorInstanceName sets the value of InitiatorInstanceName for the instance
func (instance *MSFT_iSCSISession) SetPropertyInitiatorInstanceName(value string) (err error) {
	return instance.SetProperty("InitiatorInstanceName", (value))
}

// GetInitiatorInstanceName gets the value of InitiatorInstanceName for the instance
func (instance *MSFT_iSCSISession) GetPropertyInitiatorInstanceName() (value string, err error) {
	retValue, err := instance.GetProperty("InitiatorInstanceName")
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

// SetInitiatorNodeAddress sets the value of InitiatorNodeAddress for the instance
func (instance *MSFT_iSCSISession) SetPropertyInitiatorNodeAddress(value string) (err error) {
	return instance.SetProperty("InitiatorNodeAddress", (value))
}

// GetInitiatorNodeAddress gets the value of InitiatorNodeAddress for the instance
func (instance *MSFT_iSCSISession) GetPropertyInitiatorNodeAddress() (value string, err error) {
	retValue, err := instance.GetProperty("InitiatorNodeAddress")
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

// SetInitiatorPortalAddress sets the value of InitiatorPortalAddress for the instance
func (instance *MSFT_iSCSISession) SetPropertyInitiatorPortalAddress(value string) (err error) {
	return instance.SetProperty("InitiatorPortalAddress", (value))
}

// GetInitiatorPortalAddress gets the value of InitiatorPortalAddress for the instance
func (instance *MSFT_iSCSISession) GetPropertyInitiatorPortalAddress() (value string, err error) {
	retValue, err := instance.GetProperty("InitiatorPortalAddress")
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

// SetInitiatorSideIdentifier sets the value of InitiatorSideIdentifier for the instance
func (instance *MSFT_iSCSISession) SetPropertyInitiatorSideIdentifier(value string) (err error) {
	return instance.SetProperty("InitiatorSideIdentifier", (value))
}

// GetInitiatorSideIdentifier gets the value of InitiatorSideIdentifier for the instance
func (instance *MSFT_iSCSISession) GetPropertyInitiatorSideIdentifier() (value string, err error) {
	retValue, err := instance.GetProperty("InitiatorSideIdentifier")
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

// SetIsConnected sets the value of IsConnected for the instance
func (instance *MSFT_iSCSISession) SetPropertyIsConnected(value bool) (err error) {
	return instance.SetProperty("IsConnected", (value))
}

// GetIsConnected gets the value of IsConnected for the instance
func (instance *MSFT_iSCSISession) GetPropertyIsConnected() (value bool, err error) {
	retValue, err := instance.GetProperty("IsConnected")
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

// SetIsDataDigest sets the value of IsDataDigest for the instance
func (instance *MSFT_iSCSISession) SetPropertyIsDataDigest(value bool) (err error) {
	return instance.SetProperty("IsDataDigest", (value))
}

// GetIsDataDigest gets the value of IsDataDigest for the instance
func (instance *MSFT_iSCSISession) GetPropertyIsDataDigest() (value bool, err error) {
	retValue, err := instance.GetProperty("IsDataDigest")
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

// SetIsDiscovered sets the value of IsDiscovered for the instance
func (instance *MSFT_iSCSISession) SetPropertyIsDiscovered(value bool) (err error) {
	return instance.SetProperty("IsDiscovered", (value))
}

// GetIsDiscovered gets the value of IsDiscovered for the instance
func (instance *MSFT_iSCSISession) GetPropertyIsDiscovered() (value bool, err error) {
	retValue, err := instance.GetProperty("IsDiscovered")
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

// SetIsHeaderDigest sets the value of IsHeaderDigest for the instance
func (instance *MSFT_iSCSISession) SetPropertyIsHeaderDigest(value bool) (err error) {
	return instance.SetProperty("IsHeaderDigest", (value))
}

// GetIsHeaderDigest gets the value of IsHeaderDigest for the instance
func (instance *MSFT_iSCSISession) GetPropertyIsHeaderDigest() (value bool, err error) {
	retValue, err := instance.GetProperty("IsHeaderDigest")
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

// SetIsPersistent sets the value of IsPersistent for the instance
func (instance *MSFT_iSCSISession) SetPropertyIsPersistent(value bool) (err error) {
	return instance.SetProperty("IsPersistent", (value))
}

// GetIsPersistent gets the value of IsPersistent for the instance
func (instance *MSFT_iSCSISession) GetPropertyIsPersistent() (value bool, err error) {
	retValue, err := instance.GetProperty("IsPersistent")
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

// SetNumberOfConnections sets the value of NumberOfConnections for the instance
func (instance *MSFT_iSCSISession) SetPropertyNumberOfConnections(value uint32) (err error) {
	return instance.SetProperty("NumberOfConnections", (value))
}

// GetNumberOfConnections gets the value of NumberOfConnections for the instance
func (instance *MSFT_iSCSISession) GetPropertyNumberOfConnections() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfConnections")
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

// SetSessionIdentifier sets the value of SessionIdentifier for the instance
func (instance *MSFT_iSCSISession) SetPropertySessionIdentifier(value string) (err error) {
	return instance.SetProperty("SessionIdentifier", (value))
}

// GetSessionIdentifier gets the value of SessionIdentifier for the instance
func (instance *MSFT_iSCSISession) GetPropertySessionIdentifier() (value string, err error) {
	retValue, err := instance.GetProperty("SessionIdentifier")
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

// SetTargetNodeAddress sets the value of TargetNodeAddress for the instance
func (instance *MSFT_iSCSISession) SetPropertyTargetNodeAddress(value string) (err error) {
	return instance.SetProperty("TargetNodeAddress", (value))
}

// GetTargetNodeAddress gets the value of TargetNodeAddress for the instance
func (instance *MSFT_iSCSISession) GetPropertyTargetNodeAddress() (value string, err error) {
	retValue, err := instance.GetProperty("TargetNodeAddress")
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

// SetTargetSideIdentifier sets the value of TargetSideIdentifier for the instance
func (instance *MSFT_iSCSISession) SetPropertyTargetSideIdentifier(value string) (err error) {
	return instance.SetProperty("TargetSideIdentifier", (value))
}

// GetTargetSideIdentifier gets the value of TargetSideIdentifier for the instance
func (instance *MSFT_iSCSISession) GetPropertyTargetSideIdentifier() (value string, err error) {
	retValue, err := instance.GetProperty("TargetSideIdentifier")
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

// <param name="ChapSecret" type="string "></param>
// <param name="ChapUsername" type="string "></param>
// <param name="IsMultipathEnabled" type="bool "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_iSCSISession) Register( /* IN */ IsMultipathEnabled bool,
	/* IN */ ChapUsername string,
	/* IN */ ChapSecret string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Register", IsMultipathEnabled, ChapUsername, ChapSecret)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_iSCSISession) Unregister() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Unregister")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ChapSecret" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_iSCSISession) SetCHAPSecret( /* IN */ ChapSecret string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetCHAPSecret", ChapSecret)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}
