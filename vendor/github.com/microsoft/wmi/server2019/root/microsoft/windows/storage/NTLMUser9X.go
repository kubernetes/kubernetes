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

// __NTLMUser9X struct
type __NTLMUser9X struct {
	*__SecurityRelatedClass

	//
	Authority string

	//
	Flags int32

	//
	Mask int32

	//
	Name string

	//
	Type int32
}

func New__NTLMUser9XEx1(instance *cim.WmiInstance) (newInstance *__NTLMUser9X, err error) {
	tmp, err := New__SecurityRelatedClassEx1(instance)

	if err != nil {
		return
	}
	newInstance = &__NTLMUser9X{
		__SecurityRelatedClass: tmp,
	}
	return
}

func New__NTLMUser9XEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *__NTLMUser9X, err error) {
	tmp, err := New__SecurityRelatedClassEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &__NTLMUser9X{
		__SecurityRelatedClass: tmp,
	}
	return
}

// SetAuthority sets the value of Authority for the instance
func (instance *__NTLMUser9X) SetPropertyAuthority(value string) (err error) {
	return instance.SetProperty("Authority", (value))
}

// GetAuthority gets the value of Authority for the instance
func (instance *__NTLMUser9X) GetPropertyAuthority() (value string, err error) {
	retValue, err := instance.GetProperty("Authority")
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

// SetFlags sets the value of Flags for the instance
func (instance *__NTLMUser9X) SetPropertyFlags(value int32) (err error) {
	return instance.SetProperty("Flags", (value))
}

// GetFlags gets the value of Flags for the instance
func (instance *__NTLMUser9X) GetPropertyFlags() (value int32, err error) {
	retValue, err := instance.GetProperty("Flags")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = int32(valuetmp)

	return
}

// SetMask sets the value of Mask for the instance
func (instance *__NTLMUser9X) SetPropertyMask(value int32) (err error) {
	return instance.SetProperty("Mask", (value))
}

// GetMask gets the value of Mask for the instance
func (instance *__NTLMUser9X) GetPropertyMask() (value int32, err error) {
	retValue, err := instance.GetProperty("Mask")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = int32(valuetmp)

	return
}

// SetName sets the value of Name for the instance
func (instance *__NTLMUser9X) SetPropertyName(value string) (err error) {
	return instance.SetProperty("Name", (value))
}

// GetName gets the value of Name for the instance
func (instance *__NTLMUser9X) GetPropertyName() (value string, err error) {
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

// SetType sets the value of Type for the instance
func (instance *__NTLMUser9X) SetPropertyType(value int32) (err error) {
	return instance.SetProperty("Type", (value))
}

// GetType gets the value of Type for the instance
func (instance *__NTLMUser9X) GetPropertyType() (value int32, err error) {
	retValue, err := instance.GetProperty("Type")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = int32(valuetmp)

	return
}
