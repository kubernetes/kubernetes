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

// CIM_VersionCompatibilityCheck struct
type CIM_VersionCompatibilityCheck struct {
	*CIM_Check

	//
	AllowDownVersion bool

	//
	AllowMultipleVersions bool

	//
	Reinstall bool
}

func NewCIM_VersionCompatibilityCheckEx1(instance *cim.WmiInstance) (newInstance *CIM_VersionCompatibilityCheck, err error) {
	tmp, err := NewCIM_CheckEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_VersionCompatibilityCheck{
		CIM_Check: tmp,
	}
	return
}

func NewCIM_VersionCompatibilityCheckEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_VersionCompatibilityCheck, err error) {
	tmp, err := NewCIM_CheckEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_VersionCompatibilityCheck{
		CIM_Check: tmp,
	}
	return
}

// SetAllowDownVersion sets the value of AllowDownVersion for the instance
func (instance *CIM_VersionCompatibilityCheck) SetPropertyAllowDownVersion(value bool) (err error) {
	return instance.SetProperty("AllowDownVersion", (value))
}

// GetAllowDownVersion gets the value of AllowDownVersion for the instance
func (instance *CIM_VersionCompatibilityCheck) GetPropertyAllowDownVersion() (value bool, err error) {
	retValue, err := instance.GetProperty("AllowDownVersion")
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

// SetAllowMultipleVersions sets the value of AllowMultipleVersions for the instance
func (instance *CIM_VersionCompatibilityCheck) SetPropertyAllowMultipleVersions(value bool) (err error) {
	return instance.SetProperty("AllowMultipleVersions", (value))
}

// GetAllowMultipleVersions gets the value of AllowMultipleVersions for the instance
func (instance *CIM_VersionCompatibilityCheck) GetPropertyAllowMultipleVersions() (value bool, err error) {
	retValue, err := instance.GetProperty("AllowMultipleVersions")
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

// SetReinstall sets the value of Reinstall for the instance
func (instance *CIM_VersionCompatibilityCheck) SetPropertyReinstall(value bool) (err error) {
	return instance.SetProperty("Reinstall", (value))
}

// GetReinstall gets the value of Reinstall for the instance
func (instance *CIM_VersionCompatibilityCheck) GetPropertyReinstall() (value bool, err error) {
	retValue, err := instance.GetProperty("Reinstall")
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
