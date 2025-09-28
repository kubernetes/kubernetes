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

// Win32_PerfFormattedData_ClussvcPerfProvider_ClusterDatabase struct
type Win32_PerfFormattedData_ClussvcPerfProvider_ClusterDatabase struct {
	*Win32_PerfFormattedData

	//
	Flushes uint64

	//
	FlushesPersec uint64
}

func NewWin32_PerfFormattedData_ClussvcPerfProvider_ClusterDatabaseEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterDatabase, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_ClussvcPerfProvider_ClusterDatabase{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_ClussvcPerfProvider_ClusterDatabaseEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterDatabase, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_ClussvcPerfProvider_ClusterDatabase{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetFlushes sets the value of Flushes for the instance
func (instance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterDatabase) SetPropertyFlushes(value uint64) (err error) {
	return instance.SetProperty("Flushes", (value))
}

// GetFlushes gets the value of Flushes for the instance
func (instance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterDatabase) GetPropertyFlushes() (value uint64, err error) {
	retValue, err := instance.GetProperty("Flushes")
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

// SetFlushesPersec sets the value of FlushesPersec for the instance
func (instance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterDatabase) SetPropertyFlushesPersec(value uint64) (err error) {
	return instance.SetProperty("FlushesPersec", (value))
}

// GetFlushesPersec gets the value of FlushesPersec for the instance
func (instance *Win32_PerfFormattedData_ClussvcPerfProvider_ClusterDatabase) GetPropertyFlushesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("FlushesPersec")
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
