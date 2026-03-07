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

// __Trustee struct
type __Trustee struct {
	*__SecurityRelatedClass

	//
	Domain string

	//
	Name string

	//
	SID []uint8

	//
	SidLength uint32

	//
	SIDString string

	//
	TIME_CREATED uint64
}

func New__TrusteeEx1(instance *cim.WmiInstance) (newInstance *__Trustee, err error) {
	tmp, err := New__SecurityRelatedClassEx1(instance)

	if err != nil {
		return
	}
	newInstance = &__Trustee{
		__SecurityRelatedClass: tmp,
	}
	return
}

func New__TrusteeEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *__Trustee, err error) {
	tmp, err := New__SecurityRelatedClassEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &__Trustee{
		__SecurityRelatedClass: tmp,
	}
	return
}

// SetDomain sets the value of Domain for the instance
func (instance *__Trustee) SetPropertyDomain(value string) (err error) {
	return instance.SetProperty("Domain", (value))
}

// GetDomain gets the value of Domain for the instance
func (instance *__Trustee) GetPropertyDomain() (value string, err error) {
	retValue, err := instance.GetProperty("Domain")
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

// SetName sets the value of Name for the instance
func (instance *__Trustee) SetPropertyName(value string) (err error) {
	return instance.SetProperty("Name", (value))
}

// GetName gets the value of Name for the instance
func (instance *__Trustee) GetPropertyName() (value string, err error) {
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

// SetSID sets the value of SID for the instance
func (instance *__Trustee) SetPropertySID(value []uint8) (err error) {
	return instance.SetProperty("SID", (value))
}

// GetSID gets the value of SID for the instance
func (instance *__Trustee) GetPropertySID() (value []uint8, err error) {
	retValue, err := instance.GetProperty("SID")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint8)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint8 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint8(valuetmp))
	}

	return
}

// SetSidLength sets the value of SidLength for the instance
func (instance *__Trustee) SetPropertySidLength(value uint32) (err error) {
	return instance.SetProperty("SidLength", (value))
}

// GetSidLength gets the value of SidLength for the instance
func (instance *__Trustee) GetPropertySidLength() (value uint32, err error) {
	retValue, err := instance.GetProperty("SidLength")
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

// SetSIDString sets the value of SIDString for the instance
func (instance *__Trustee) SetPropertySIDString(value string) (err error) {
	return instance.SetProperty("SIDString", (value))
}

// GetSIDString gets the value of SIDString for the instance
func (instance *__Trustee) GetPropertySIDString() (value string, err error) {
	retValue, err := instance.GetProperty("SIDString")
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

// SetTIME_CREATED sets the value of TIME_CREATED for the instance
func (instance *__Trustee) SetPropertyTIME_CREATED(value uint64) (err error) {
	return instance.SetProperty("TIME_CREATED", (value))
}

// GetTIME_CREATED gets the value of TIME_CREATED for the instance
func (instance *__Trustee) GetPropertyTIME_CREATED() (value uint64, err error) {
	retValue, err := instance.GetProperty("TIME_CREATED")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}
