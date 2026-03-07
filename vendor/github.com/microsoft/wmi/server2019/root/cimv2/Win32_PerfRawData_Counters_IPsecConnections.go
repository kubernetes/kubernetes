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

// Win32_PerfRawData_Counters_IPsecConnections struct
type Win32_PerfRawData_Counters_IPsecConnections struct {
	*Win32_PerfRawData

	//
	Maxnumberofconnectionssinceboot uint32

	//
	Numberoffailedauthentications uint64

	//
	TotalBytesInsincestart uint64

	//
	TotalBytesOutsincestart uint64

	//
	TotalNumbercurrentConnections uint32

	//
	Totalnumberofcumulativeconnectionssinceboot uint64
}

func NewWin32_PerfRawData_Counters_IPsecConnectionsEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_Counters_IPsecConnections, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_IPsecConnections{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_Counters_IPsecConnectionsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_Counters_IPsecConnections, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_IPsecConnections{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetMaxnumberofconnectionssinceboot sets the value of Maxnumberofconnectionssinceboot for the instance
func (instance *Win32_PerfRawData_Counters_IPsecConnections) SetPropertyMaxnumberofconnectionssinceboot(value uint32) (err error) {
	return instance.SetProperty("Maxnumberofconnectionssinceboot", (value))
}

// GetMaxnumberofconnectionssinceboot gets the value of Maxnumberofconnectionssinceboot for the instance
func (instance *Win32_PerfRawData_Counters_IPsecConnections) GetPropertyMaxnumberofconnectionssinceboot() (value uint32, err error) {
	retValue, err := instance.GetProperty("Maxnumberofconnectionssinceboot")
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

// SetNumberoffailedauthentications sets the value of Numberoffailedauthentications for the instance
func (instance *Win32_PerfRawData_Counters_IPsecConnections) SetPropertyNumberoffailedauthentications(value uint64) (err error) {
	return instance.SetProperty("Numberoffailedauthentications", (value))
}

// GetNumberoffailedauthentications gets the value of Numberoffailedauthentications for the instance
func (instance *Win32_PerfRawData_Counters_IPsecConnections) GetPropertyNumberoffailedauthentications() (value uint64, err error) {
	retValue, err := instance.GetProperty("Numberoffailedauthentications")
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

// SetTotalBytesInsincestart sets the value of TotalBytesInsincestart for the instance
func (instance *Win32_PerfRawData_Counters_IPsecConnections) SetPropertyTotalBytesInsincestart(value uint64) (err error) {
	return instance.SetProperty("TotalBytesInsincestart", (value))
}

// GetTotalBytesInsincestart gets the value of TotalBytesInsincestart for the instance
func (instance *Win32_PerfRawData_Counters_IPsecConnections) GetPropertyTotalBytesInsincestart() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalBytesInsincestart")
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

// SetTotalBytesOutsincestart sets the value of TotalBytesOutsincestart for the instance
func (instance *Win32_PerfRawData_Counters_IPsecConnections) SetPropertyTotalBytesOutsincestart(value uint64) (err error) {
	return instance.SetProperty("TotalBytesOutsincestart", (value))
}

// GetTotalBytesOutsincestart gets the value of TotalBytesOutsincestart for the instance
func (instance *Win32_PerfRawData_Counters_IPsecConnections) GetPropertyTotalBytesOutsincestart() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalBytesOutsincestart")
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

// SetTotalNumbercurrentConnections sets the value of TotalNumbercurrentConnections for the instance
func (instance *Win32_PerfRawData_Counters_IPsecConnections) SetPropertyTotalNumbercurrentConnections(value uint32) (err error) {
	return instance.SetProperty("TotalNumbercurrentConnections", (value))
}

// GetTotalNumbercurrentConnections gets the value of TotalNumbercurrentConnections for the instance
func (instance *Win32_PerfRawData_Counters_IPsecConnections) GetPropertyTotalNumbercurrentConnections() (value uint32, err error) {
	retValue, err := instance.GetProperty("TotalNumbercurrentConnections")
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

// SetTotalnumberofcumulativeconnectionssinceboot sets the value of Totalnumberofcumulativeconnectionssinceboot for the instance
func (instance *Win32_PerfRawData_Counters_IPsecConnections) SetPropertyTotalnumberofcumulativeconnectionssinceboot(value uint64) (err error) {
	return instance.SetProperty("Totalnumberofcumulativeconnectionssinceboot", (value))
}

// GetTotalnumberofcumulativeconnectionssinceboot gets the value of Totalnumberofcumulativeconnectionssinceboot for the instance
func (instance *Win32_PerfRawData_Counters_IPsecConnections) GetPropertyTotalnumberofcumulativeconnectionssinceboot() (value uint64, err error) {
	retValue, err := instance.GetProperty("Totalnumberofcumulativeconnectionssinceboot")
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
