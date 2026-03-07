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

// MSFT_iSCSITargetPortal struct
type MSFT_iSCSITargetPortal struct {
	*cim.WmiInstance

	//
	InitiatorInstanceName string

	//
	InitiatorPortalAddress string

	//
	IsDataDigest bool

	//
	IsHeaderDigest bool

	//
	TargetPortalAddress string

	//
	TargetPortalPortNumber uint16
}

func NewMSFT_iSCSITargetPortalEx1(instance *cim.WmiInstance) (newInstance *MSFT_iSCSITargetPortal, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_iSCSITargetPortal{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_iSCSITargetPortalEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_iSCSITargetPortal, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_iSCSITargetPortal{
		WmiInstance: tmp,
	}
	return
}

// SetInitiatorInstanceName sets the value of InitiatorInstanceName for the instance
func (instance *MSFT_iSCSITargetPortal) SetPropertyInitiatorInstanceName(value string) (err error) {
	return instance.SetProperty("InitiatorInstanceName", (value))
}

// GetInitiatorInstanceName gets the value of InitiatorInstanceName for the instance
func (instance *MSFT_iSCSITargetPortal) GetPropertyInitiatorInstanceName() (value string, err error) {
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

// SetInitiatorPortalAddress sets the value of InitiatorPortalAddress for the instance
func (instance *MSFT_iSCSITargetPortal) SetPropertyInitiatorPortalAddress(value string) (err error) {
	return instance.SetProperty("InitiatorPortalAddress", (value))
}

// GetInitiatorPortalAddress gets the value of InitiatorPortalAddress for the instance
func (instance *MSFT_iSCSITargetPortal) GetPropertyInitiatorPortalAddress() (value string, err error) {
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

// SetIsDataDigest sets the value of IsDataDigest for the instance
func (instance *MSFT_iSCSITargetPortal) SetPropertyIsDataDigest(value bool) (err error) {
	return instance.SetProperty("IsDataDigest", (value))
}

// GetIsDataDigest gets the value of IsDataDigest for the instance
func (instance *MSFT_iSCSITargetPortal) GetPropertyIsDataDigest() (value bool, err error) {
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

// SetIsHeaderDigest sets the value of IsHeaderDigest for the instance
func (instance *MSFT_iSCSITargetPortal) SetPropertyIsHeaderDigest(value bool) (err error) {
	return instance.SetProperty("IsHeaderDigest", (value))
}

// GetIsHeaderDigest gets the value of IsHeaderDigest for the instance
func (instance *MSFT_iSCSITargetPortal) GetPropertyIsHeaderDigest() (value bool, err error) {
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

// SetTargetPortalAddress sets the value of TargetPortalAddress for the instance
func (instance *MSFT_iSCSITargetPortal) SetPropertyTargetPortalAddress(value string) (err error) {
	return instance.SetProperty("TargetPortalAddress", (value))
}

// GetTargetPortalAddress gets the value of TargetPortalAddress for the instance
func (instance *MSFT_iSCSITargetPortal) GetPropertyTargetPortalAddress() (value string, err error) {
	retValue, err := instance.GetProperty("TargetPortalAddress")
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

// SetTargetPortalPortNumber sets the value of TargetPortalPortNumber for the instance
func (instance *MSFT_iSCSITargetPortal) SetPropertyTargetPortalPortNumber(value uint16) (err error) {
	return instance.SetProperty("TargetPortalPortNumber", (value))
}

// GetTargetPortalPortNumber gets the value of TargetPortalPortNumber for the instance
func (instance *MSFT_iSCSITargetPortal) GetPropertyTargetPortalPortNumber() (value uint16, err error) {
	retValue, err := instance.GetProperty("TargetPortalPortNumber")
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

//

// <param name="AuthenticationType" type="string "></param>
// <param name="ChapSecret" type="string "></param>
// <param name="ChapUsername" type="string "></param>
// <param name="InitiatorInstanceName" type="string "></param>
// <param name="InitiatorPortalAddress" type="string "></param>
// <param name="IsDataDigest" type="bool "></param>
// <param name="IsHeaderDigest" type="bool "></param>
// <param name="TargetPortalAddress" type="string "></param>
// <param name="TargetPortalPortNumber" type="uint16 "></param>

// <param name="CreatedTargetPortal" type="MSFT_iSCSITargetPortal "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_iSCSITargetPortal) New( /* IN */ TargetPortalAddress string,
	/* IN */ TargetPortalPortNumber uint16,
	/* IN */ InitiatorInstanceName string,
	/* IN */ InitiatorPortalAddress string,
	/* IN */ AuthenticationType string,
	/* IN */ ChapUsername string,
	/* IN */ ChapSecret string,
	/* IN */ IsHeaderDigest bool,
	/* IN */ IsDataDigest bool,
	/* OUT */ CreatedTargetPortal MSFT_iSCSITargetPortal) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("New", TargetPortalAddress, TargetPortalPortNumber, InitiatorInstanceName, InitiatorPortalAddress, AuthenticationType, ChapUsername, ChapSecret, IsHeaderDigest, IsDataDigest)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="InitiatorInstanceName" type="string "></param>
// <param name="InitiatorPortalAddress" type="string "></param>
// <param name="TargetPortalAddress" type="string "></param>
// <param name="TargetPortalPortNumber" type="uint16 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_iSCSITargetPortal) Remove( /* IN */ InitiatorInstanceName string,
	/* IN */ InitiatorPortalAddress string,
	/* IN */ TargetPortalPortNumber uint16,
	/* IN */ TargetPortalAddress string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Remove", InitiatorInstanceName, InitiatorPortalAddress, TargetPortalPortNumber, TargetPortalAddress)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="InitiatorInstanceName" type="string "></param>
// <param name="InitiatorPortalAddress" type="string "></param>
// <param name="TargetPortalAddress" type="string "></param>
// <param name="TargetPortalPortNumber" type="uint16 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_iSCSITargetPortal) Update( /* IN */ InitiatorInstanceName string,
	/* IN */ InitiatorPortalAddress string,
	/* IN */ TargetPortalAddress string,
	/* IN */ TargetPortalPortNumber uint16) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Update", InitiatorInstanceName, InitiatorPortalAddress, TargetPortalAddress, TargetPortalPortNumber)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}
