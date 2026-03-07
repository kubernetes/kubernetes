// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.Microsoft.Windows.Storage
//////////////////////////////////////////////
package storage

import (
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// MSFT_TargetPortal struct
type MSFT_TargetPortal struct {
	*MSFT_StorageObject

	//
	IPv4Address string

	//
	IPv6Address string

	//
	PortNumber uint32

	//
	SubnetMask string
}

func NewMSFT_TargetPortalEx1(instance *cim.WmiInstance) (newInstance *MSFT_TargetPortal, err error) {
	tmp, err := NewMSFT_StorageObjectEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_TargetPortal{
		MSFT_StorageObject: tmp,
	}
	return
}

func NewMSFT_TargetPortalEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_TargetPortal, err error) {
	tmp, err := NewMSFT_StorageObjectEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_TargetPortal{
		MSFT_StorageObject: tmp,
	}
	return
}

// SetIPv4Address sets the value of IPv4Address for the instance
func (instance *MSFT_TargetPortal) SetPropertyIPv4Address(value string) (err error) {
	return instance.SetProperty("IPv4Address", (value))
}

// GetIPv4Address gets the value of IPv4Address for the instance
func (instance *MSFT_TargetPortal) GetPropertyIPv4Address() (value string, err error) {
	retValue, err := instance.GetProperty("IPv4Address")
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

// SetIPv6Address sets the value of IPv6Address for the instance
func (instance *MSFT_TargetPortal) SetPropertyIPv6Address(value string) (err error) {
	return instance.SetProperty("IPv6Address", (value))
}

// GetIPv6Address gets the value of IPv6Address for the instance
func (instance *MSFT_TargetPortal) GetPropertyIPv6Address() (value string, err error) {
	retValue, err := instance.GetProperty("IPv6Address")
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

// SetPortNumber sets the value of PortNumber for the instance
func (instance *MSFT_TargetPortal) SetPropertyPortNumber(value uint32) (err error) {
	return instance.SetProperty("PortNumber", (value))
}

// GetPortNumber gets the value of PortNumber for the instance
func (instance *MSFT_TargetPortal) GetPropertyPortNumber() (value uint32, err error) {
	retValue, err := instance.GetProperty("PortNumber")
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

// SetSubnetMask sets the value of SubnetMask for the instance
func (instance *MSFT_TargetPortal) SetPropertySubnetMask(value string) (err error) {
	return instance.SetProperty("SubnetMask", (value))
}

// GetSubnetMask gets the value of SubnetMask for the instance
func (instance *MSFT_TargetPortal) GetPropertySubnetMask() (value string, err error) {
	retValue, err := instance.GetProperty("SubnetMask")
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
