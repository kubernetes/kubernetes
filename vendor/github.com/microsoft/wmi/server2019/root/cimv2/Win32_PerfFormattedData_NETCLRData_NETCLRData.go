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

// Win32_PerfFormattedData_NETCLRData_NETCLRData struct
type Win32_PerfFormattedData_NETCLRData_NETCLRData struct {
	*Win32_PerfFormattedData

	//
	SqlClientCurrentNumberconnectionpools uint32

	//
	SqlClientCurrentNumberpooledandnonpooledconnections uint32

	//
	SqlClientCurrentNumberpooledconnections uint32

	//
	SqlClientPeakNumberpooledconnections uint32

	//
	SqlClientTotalNumberfailedcommands uint32

	//
	SqlClientTotalNumberfailedconnects uint32
}

func NewWin32_PerfFormattedData_NETCLRData_NETCLRDataEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_NETCLRData_NETCLRData, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_NETCLRData_NETCLRData{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_NETCLRData_NETCLRDataEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_NETCLRData_NETCLRData, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_NETCLRData_NETCLRData{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetSqlClientCurrentNumberconnectionpools sets the value of SqlClientCurrentNumberconnectionpools for the instance
func (instance *Win32_PerfFormattedData_NETCLRData_NETCLRData) SetPropertySqlClientCurrentNumberconnectionpools(value uint32) (err error) {
	return instance.SetProperty("SqlClientCurrentNumberconnectionpools", (value))
}

// GetSqlClientCurrentNumberconnectionpools gets the value of SqlClientCurrentNumberconnectionpools for the instance
func (instance *Win32_PerfFormattedData_NETCLRData_NETCLRData) GetPropertySqlClientCurrentNumberconnectionpools() (value uint32, err error) {
	retValue, err := instance.GetProperty("SqlClientCurrentNumberconnectionpools")
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

// SetSqlClientCurrentNumberpooledandnonpooledconnections sets the value of SqlClientCurrentNumberpooledandnonpooledconnections for the instance
func (instance *Win32_PerfFormattedData_NETCLRData_NETCLRData) SetPropertySqlClientCurrentNumberpooledandnonpooledconnections(value uint32) (err error) {
	return instance.SetProperty("SqlClientCurrentNumberpooledandnonpooledconnections", (value))
}

// GetSqlClientCurrentNumberpooledandnonpooledconnections gets the value of SqlClientCurrentNumberpooledandnonpooledconnections for the instance
func (instance *Win32_PerfFormattedData_NETCLRData_NETCLRData) GetPropertySqlClientCurrentNumberpooledandnonpooledconnections() (value uint32, err error) {
	retValue, err := instance.GetProperty("SqlClientCurrentNumberpooledandnonpooledconnections")
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

// SetSqlClientCurrentNumberpooledconnections sets the value of SqlClientCurrentNumberpooledconnections for the instance
func (instance *Win32_PerfFormattedData_NETCLRData_NETCLRData) SetPropertySqlClientCurrentNumberpooledconnections(value uint32) (err error) {
	return instance.SetProperty("SqlClientCurrentNumberpooledconnections", (value))
}

// GetSqlClientCurrentNumberpooledconnections gets the value of SqlClientCurrentNumberpooledconnections for the instance
func (instance *Win32_PerfFormattedData_NETCLRData_NETCLRData) GetPropertySqlClientCurrentNumberpooledconnections() (value uint32, err error) {
	retValue, err := instance.GetProperty("SqlClientCurrentNumberpooledconnections")
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

// SetSqlClientPeakNumberpooledconnections sets the value of SqlClientPeakNumberpooledconnections for the instance
func (instance *Win32_PerfFormattedData_NETCLRData_NETCLRData) SetPropertySqlClientPeakNumberpooledconnections(value uint32) (err error) {
	return instance.SetProperty("SqlClientPeakNumberpooledconnections", (value))
}

// GetSqlClientPeakNumberpooledconnections gets the value of SqlClientPeakNumberpooledconnections for the instance
func (instance *Win32_PerfFormattedData_NETCLRData_NETCLRData) GetPropertySqlClientPeakNumberpooledconnections() (value uint32, err error) {
	retValue, err := instance.GetProperty("SqlClientPeakNumberpooledconnections")
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

// SetSqlClientTotalNumberfailedcommands sets the value of SqlClientTotalNumberfailedcommands for the instance
func (instance *Win32_PerfFormattedData_NETCLRData_NETCLRData) SetPropertySqlClientTotalNumberfailedcommands(value uint32) (err error) {
	return instance.SetProperty("SqlClientTotalNumberfailedcommands", (value))
}

// GetSqlClientTotalNumberfailedcommands gets the value of SqlClientTotalNumberfailedcommands for the instance
func (instance *Win32_PerfFormattedData_NETCLRData_NETCLRData) GetPropertySqlClientTotalNumberfailedcommands() (value uint32, err error) {
	retValue, err := instance.GetProperty("SqlClientTotalNumberfailedcommands")
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

// SetSqlClientTotalNumberfailedconnects sets the value of SqlClientTotalNumberfailedconnects for the instance
func (instance *Win32_PerfFormattedData_NETCLRData_NETCLRData) SetPropertySqlClientTotalNumberfailedconnects(value uint32) (err error) {
	return instance.SetProperty("SqlClientTotalNumberfailedconnects", (value))
}

// GetSqlClientTotalNumberfailedconnects gets the value of SqlClientTotalNumberfailedconnects for the instance
func (instance *Win32_PerfFormattedData_NETCLRData_NETCLRData) GetPropertySqlClientTotalNumberfailedconnects() (value uint32, err error) {
	retValue, err := instance.GetProperty("SqlClientTotalNumberfailedconnects")
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
