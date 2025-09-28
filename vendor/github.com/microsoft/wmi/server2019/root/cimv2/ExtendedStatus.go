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

// __ExtendedStatus struct
type __ExtendedStatus struct {
	*__NotifyStatus

	//
	Description string

	//
	Operation string

	//
	ParameterInfo string

	//
	ProviderName string
}

func New__ExtendedStatusEx1(instance *cim.WmiInstance) (newInstance *__ExtendedStatus, err error) {
	tmp, err := New__NotifyStatusEx1(instance)

	if err != nil {
		return
	}
	newInstance = &__ExtendedStatus{
		__NotifyStatus: tmp,
	}
	return
}

func New__ExtendedStatusEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *__ExtendedStatus, err error) {
	tmp, err := New__NotifyStatusEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &__ExtendedStatus{
		__NotifyStatus: tmp,
	}
	return
}

// SetDescription sets the value of Description for the instance
func (instance *__ExtendedStatus) SetPropertyDescription(value string) (err error) {
	return instance.SetProperty("Description", (value))
}

// GetDescription gets the value of Description for the instance
func (instance *__ExtendedStatus) GetPropertyDescription() (value string, err error) {
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

// SetOperation sets the value of Operation for the instance
func (instance *__ExtendedStatus) SetPropertyOperation(value string) (err error) {
	return instance.SetProperty("Operation", (value))
}

// GetOperation gets the value of Operation for the instance
func (instance *__ExtendedStatus) GetPropertyOperation() (value string, err error) {
	retValue, err := instance.GetProperty("Operation")
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

// SetParameterInfo sets the value of ParameterInfo for the instance
func (instance *__ExtendedStatus) SetPropertyParameterInfo(value string) (err error) {
	return instance.SetProperty("ParameterInfo", (value))
}

// GetParameterInfo gets the value of ParameterInfo for the instance
func (instance *__ExtendedStatus) GetPropertyParameterInfo() (value string, err error) {
	retValue, err := instance.GetProperty("ParameterInfo")
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

// SetProviderName sets the value of ProviderName for the instance
func (instance *__ExtendedStatus) SetPropertyProviderName(value string) (err error) {
	return instance.SetProperty("ProviderName", (value))
}

// GetProviderName gets the value of ProviderName for the instance
func (instance *__ExtendedStatus) GetPropertyProviderName() (value string, err error) {
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
