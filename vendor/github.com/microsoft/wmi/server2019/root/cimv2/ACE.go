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

// __ACE struct
type __ACE struct {
	*__SecurityRelatedClass

	//
	AccessMask uint32

	//
	AceFlags uint32

	//
	AceType uint32

	//
	GuidInheritedObjectType string

	//
	GuidObjectType string

	//
	TIME_CREATED uint64

	//
	Trustee __Trustee
}

func New__ACEEx1(instance *cim.WmiInstance) (newInstance *__ACE, err error) {
	tmp, err := New__SecurityRelatedClassEx1(instance)

	if err != nil {
		return
	}
	newInstance = &__ACE{
		__SecurityRelatedClass: tmp,
	}
	return
}

func New__ACEEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *__ACE, err error) {
	tmp, err := New__SecurityRelatedClassEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &__ACE{
		__SecurityRelatedClass: tmp,
	}
	return
}

// SetAccessMask sets the value of AccessMask for the instance
func (instance *__ACE) SetPropertyAccessMask(value uint32) (err error) {
	return instance.SetProperty("AccessMask", (value))
}

// GetAccessMask gets the value of AccessMask for the instance
func (instance *__ACE) GetPropertyAccessMask() (value uint32, err error) {
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

// SetAceFlags sets the value of AceFlags for the instance
func (instance *__ACE) SetPropertyAceFlags(value uint32) (err error) {
	return instance.SetProperty("AceFlags", (value))
}

// GetAceFlags gets the value of AceFlags for the instance
func (instance *__ACE) GetPropertyAceFlags() (value uint32, err error) {
	retValue, err := instance.GetProperty("AceFlags")
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

// SetAceType sets the value of AceType for the instance
func (instance *__ACE) SetPropertyAceType(value uint32) (err error) {
	return instance.SetProperty("AceType", (value))
}

// GetAceType gets the value of AceType for the instance
func (instance *__ACE) GetPropertyAceType() (value uint32, err error) {
	retValue, err := instance.GetProperty("AceType")
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

// SetGuidInheritedObjectType sets the value of GuidInheritedObjectType for the instance
func (instance *__ACE) SetPropertyGuidInheritedObjectType(value string) (err error) {
	return instance.SetProperty("GuidInheritedObjectType", (value))
}

// GetGuidInheritedObjectType gets the value of GuidInheritedObjectType for the instance
func (instance *__ACE) GetPropertyGuidInheritedObjectType() (value string, err error) {
	retValue, err := instance.GetProperty("GuidInheritedObjectType")
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

// SetGuidObjectType sets the value of GuidObjectType for the instance
func (instance *__ACE) SetPropertyGuidObjectType(value string) (err error) {
	return instance.SetProperty("GuidObjectType", (value))
}

// GetGuidObjectType gets the value of GuidObjectType for the instance
func (instance *__ACE) GetPropertyGuidObjectType() (value string, err error) {
	retValue, err := instance.GetProperty("GuidObjectType")
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
func (instance *__ACE) SetPropertyTIME_CREATED(value uint64) (err error) {
	return instance.SetProperty("TIME_CREATED", (value))
}

// GetTIME_CREATED gets the value of TIME_CREATED for the instance
func (instance *__ACE) GetPropertyTIME_CREATED() (value uint64, err error) {
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

// SetTrustee sets the value of Trustee for the instance
func (instance *__ACE) SetPropertyTrustee(value __Trustee) (err error) {
	return instance.SetProperty("Trustee", (value))
}

// GetTrustee gets the value of Trustee for the instance
func (instance *__ACE) GetPropertyTrustee() (value __Trustee, err error) {
	retValue, err := instance.GetProperty("Trustee")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(__Trustee)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " __Trustee is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = __Trustee(valuetmp)

	return
}
