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

// CIM_MemoryMappedIO struct
type CIM_MemoryMappedIO struct {
	*CIM_SystemResource

	//
	CreationClassName string

	//
	CSCreationClassName string

	//
	CSName string

	//
	EndingAddress uint64

	//
	StartingAddress uint64
}

func NewCIM_MemoryMappedIOEx1(instance *cim.WmiInstance) (newInstance *CIM_MemoryMappedIO, err error) {
	tmp, err := NewCIM_SystemResourceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_MemoryMappedIO{
		CIM_SystemResource: tmp,
	}
	return
}

func NewCIM_MemoryMappedIOEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_MemoryMappedIO, err error) {
	tmp, err := NewCIM_SystemResourceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_MemoryMappedIO{
		CIM_SystemResource: tmp,
	}
	return
}

// SetCreationClassName sets the value of CreationClassName for the instance
func (instance *CIM_MemoryMappedIO) SetPropertyCreationClassName(value string) (err error) {
	return instance.SetProperty("CreationClassName", (value))
}

// GetCreationClassName gets the value of CreationClassName for the instance
func (instance *CIM_MemoryMappedIO) GetPropertyCreationClassName() (value string, err error) {
	retValue, err := instance.GetProperty("CreationClassName")
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

// SetCSCreationClassName sets the value of CSCreationClassName for the instance
func (instance *CIM_MemoryMappedIO) SetPropertyCSCreationClassName(value string) (err error) {
	return instance.SetProperty("CSCreationClassName", (value))
}

// GetCSCreationClassName gets the value of CSCreationClassName for the instance
func (instance *CIM_MemoryMappedIO) GetPropertyCSCreationClassName() (value string, err error) {
	retValue, err := instance.GetProperty("CSCreationClassName")
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

// SetCSName sets the value of CSName for the instance
func (instance *CIM_MemoryMappedIO) SetPropertyCSName(value string) (err error) {
	return instance.SetProperty("CSName", (value))
}

// GetCSName gets the value of CSName for the instance
func (instance *CIM_MemoryMappedIO) GetPropertyCSName() (value string, err error) {
	retValue, err := instance.GetProperty("CSName")
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

// SetEndingAddress sets the value of EndingAddress for the instance
func (instance *CIM_MemoryMappedIO) SetPropertyEndingAddress(value uint64) (err error) {
	return instance.SetProperty("EndingAddress", (value))
}

// GetEndingAddress gets the value of EndingAddress for the instance
func (instance *CIM_MemoryMappedIO) GetPropertyEndingAddress() (value uint64, err error) {
	retValue, err := instance.GetProperty("EndingAddress")
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

// SetStartingAddress sets the value of StartingAddress for the instance
func (instance *CIM_MemoryMappedIO) SetPropertyStartingAddress(value uint64) (err error) {
	return instance.SetProperty("StartingAddress", (value))
}

// GetStartingAddress gets the value of StartingAddress for the instance
func (instance *CIM_MemoryMappedIO) GetPropertyStartingAddress() (value uint64, err error) {
	retValue, err := instance.GetProperty("StartingAddress")
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
