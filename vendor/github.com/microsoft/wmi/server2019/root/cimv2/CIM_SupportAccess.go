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

// CIM_SupportAccess struct
type CIM_SupportAccess struct {
	*cim.WmiInstance

	//
	CommunicationInfo string

	//
	CommunicationMode uint16

	//
	Description string

	//
	Locale string

	//
	SupportAccessId string
}

func NewCIM_SupportAccessEx1(instance *cim.WmiInstance) (newInstance *CIM_SupportAccess, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &CIM_SupportAccess{
		WmiInstance: tmp,
	}
	return
}

func NewCIM_SupportAccessEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_SupportAccess, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_SupportAccess{
		WmiInstance: tmp,
	}
	return
}

// SetCommunicationInfo sets the value of CommunicationInfo for the instance
func (instance *CIM_SupportAccess) SetPropertyCommunicationInfo(value string) (err error) {
	return instance.SetProperty("CommunicationInfo", (value))
}

// GetCommunicationInfo gets the value of CommunicationInfo for the instance
func (instance *CIM_SupportAccess) GetPropertyCommunicationInfo() (value string, err error) {
	retValue, err := instance.GetProperty("CommunicationInfo")
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

// SetCommunicationMode sets the value of CommunicationMode for the instance
func (instance *CIM_SupportAccess) SetPropertyCommunicationMode(value uint16) (err error) {
	return instance.SetProperty("CommunicationMode", (value))
}

// GetCommunicationMode gets the value of CommunicationMode for the instance
func (instance *CIM_SupportAccess) GetPropertyCommunicationMode() (value uint16, err error) {
	retValue, err := instance.GetProperty("CommunicationMode")
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

// SetDescription sets the value of Description for the instance
func (instance *CIM_SupportAccess) SetPropertyDescription(value string) (err error) {
	return instance.SetProperty("Description", (value))
}

// GetDescription gets the value of Description for the instance
func (instance *CIM_SupportAccess) GetPropertyDescription() (value string, err error) {
	retValue, err := instance.GetProperty("Description")
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

// SetLocale sets the value of Locale for the instance
func (instance *CIM_SupportAccess) SetPropertyLocale(value string) (err error) {
	return instance.SetProperty("Locale", (value))
}

// GetLocale gets the value of Locale for the instance
func (instance *CIM_SupportAccess) GetPropertyLocale() (value string, err error) {
	retValue, err := instance.GetProperty("Locale")
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

// SetSupportAccessId sets the value of SupportAccessId for the instance
func (instance *CIM_SupportAccess) SetPropertySupportAccessId(value string) (err error) {
	return instance.SetProperty("SupportAccessId", (value))
}

// GetSupportAccessId gets the value of SupportAccessId for the instance
func (instance *CIM_SupportAccess) GetPropertySupportAccessId() (value string, err error) {
	retValue, err := instance.GetProperty("SupportAccessId")
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
