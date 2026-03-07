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

// CIM_Export struct
type CIM_Export struct {
	*cim.WmiInstance

	//
	Directory CIM_Directory

	//
	ExportedDirectoryName string

	//
	LocalFS CIM_LocalFileSystem
}

func NewCIM_ExportEx1(instance *cim.WmiInstance) (newInstance *CIM_Export, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &CIM_Export{
		WmiInstance: tmp,
	}
	return
}

func NewCIM_ExportEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_Export, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_Export{
		WmiInstance: tmp,
	}
	return
}

// SetDirectory sets the value of Directory for the instance
func (instance *CIM_Export) SetPropertyDirectory(value CIM_Directory) (err error) {
	return instance.SetProperty("Directory", (value))
}

// GetDirectory gets the value of Directory for the instance
func (instance *CIM_Export) GetPropertyDirectory() (value CIM_Directory, err error) {
	retValue, err := instance.GetProperty("Directory")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_Directory)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_Directory is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_Directory(valuetmp)

	return
}

// SetExportedDirectoryName sets the value of ExportedDirectoryName for the instance
func (instance *CIM_Export) SetPropertyExportedDirectoryName(value string) (err error) {
	return instance.SetProperty("ExportedDirectoryName", (value))
}

// GetExportedDirectoryName gets the value of ExportedDirectoryName for the instance
func (instance *CIM_Export) GetPropertyExportedDirectoryName() (value string, err error) {
	retValue, err := instance.GetProperty("ExportedDirectoryName")
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

// SetLocalFS sets the value of LocalFS for the instance
func (instance *CIM_Export) SetPropertyLocalFS(value CIM_LocalFileSystem) (err error) {
	return instance.SetProperty("LocalFS", (value))
}

// GetLocalFS gets the value of LocalFS for the instance
func (instance *CIM_Export) GetPropertyLocalFS() (value CIM_LocalFileSystem, err error) {
	retValue, err := instance.GetProperty("LocalFS")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_LocalFileSystem)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_LocalFileSystem is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_LocalFileSystem(valuetmp)

	return
}
