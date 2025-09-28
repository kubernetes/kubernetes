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

// Win32_PerfFormattedData_Counters_IPHTTPSGlobal struct
type Win32_PerfFormattedData_Counters_IPHTTPSGlobal struct {
	*Win32_PerfFormattedData

	//
	DropsNeighborresolutiontimeouts uint64

	//
	ErrorsAuthenticationErrors uint64

	//
	ErrorsReceiveerrorsontheserver uint64

	//
	ErrorsTransmiterrorsontheserver uint64

	//
	InTotalbytesreceived uint64

	//
	InTotalpacketsreceived uint64

	//
	OutTotalbytesforwarded uint64

	//
	OutTotalbytessent uint64

	//
	OutTotalpacketssent uint64

	//
	SessionsTotalsessions uint64
}

func NewWin32_PerfFormattedData_Counters_IPHTTPSGlobalEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_IPHTTPSGlobal, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_IPHTTPSGlobal{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_IPHTTPSGlobalEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_IPHTTPSGlobal, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_IPHTTPSGlobal{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetDropsNeighborresolutiontimeouts sets the value of DropsNeighborresolutiontimeouts for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSGlobal) SetPropertyDropsNeighborresolutiontimeouts(value uint64) (err error) {
	return instance.SetProperty("DropsNeighborresolutiontimeouts", (value))
}

// GetDropsNeighborresolutiontimeouts gets the value of DropsNeighborresolutiontimeouts for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSGlobal) GetPropertyDropsNeighborresolutiontimeouts() (value uint64, err error) {
	retValue, err := instance.GetProperty("DropsNeighborresolutiontimeouts")
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

// SetErrorsAuthenticationErrors sets the value of ErrorsAuthenticationErrors for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSGlobal) SetPropertyErrorsAuthenticationErrors(value uint64) (err error) {
	return instance.SetProperty("ErrorsAuthenticationErrors", (value))
}

// GetErrorsAuthenticationErrors gets the value of ErrorsAuthenticationErrors for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSGlobal) GetPropertyErrorsAuthenticationErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("ErrorsAuthenticationErrors")
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

// SetErrorsReceiveerrorsontheserver sets the value of ErrorsReceiveerrorsontheserver for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSGlobal) SetPropertyErrorsReceiveerrorsontheserver(value uint64) (err error) {
	return instance.SetProperty("ErrorsReceiveerrorsontheserver", (value))
}

// GetErrorsReceiveerrorsontheserver gets the value of ErrorsReceiveerrorsontheserver for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSGlobal) GetPropertyErrorsReceiveerrorsontheserver() (value uint64, err error) {
	retValue, err := instance.GetProperty("ErrorsReceiveerrorsontheserver")
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

// SetErrorsTransmiterrorsontheserver sets the value of ErrorsTransmiterrorsontheserver for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSGlobal) SetPropertyErrorsTransmiterrorsontheserver(value uint64) (err error) {
	return instance.SetProperty("ErrorsTransmiterrorsontheserver", (value))
}

// GetErrorsTransmiterrorsontheserver gets the value of ErrorsTransmiterrorsontheserver for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSGlobal) GetPropertyErrorsTransmiterrorsontheserver() (value uint64, err error) {
	retValue, err := instance.GetProperty("ErrorsTransmiterrorsontheserver")
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

// SetInTotalbytesreceived sets the value of InTotalbytesreceived for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSGlobal) SetPropertyInTotalbytesreceived(value uint64) (err error) {
	return instance.SetProperty("InTotalbytesreceived", (value))
}

// GetInTotalbytesreceived gets the value of InTotalbytesreceived for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSGlobal) GetPropertyInTotalbytesreceived() (value uint64, err error) {
	retValue, err := instance.GetProperty("InTotalbytesreceived")
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

// SetInTotalpacketsreceived sets the value of InTotalpacketsreceived for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSGlobal) SetPropertyInTotalpacketsreceived(value uint64) (err error) {
	return instance.SetProperty("InTotalpacketsreceived", (value))
}

// GetInTotalpacketsreceived gets the value of InTotalpacketsreceived for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSGlobal) GetPropertyInTotalpacketsreceived() (value uint64, err error) {
	retValue, err := instance.GetProperty("InTotalpacketsreceived")
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

// SetOutTotalbytesforwarded sets the value of OutTotalbytesforwarded for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSGlobal) SetPropertyOutTotalbytesforwarded(value uint64) (err error) {
	return instance.SetProperty("OutTotalbytesforwarded", (value))
}

// GetOutTotalbytesforwarded gets the value of OutTotalbytesforwarded for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSGlobal) GetPropertyOutTotalbytesforwarded() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutTotalbytesforwarded")
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

// SetOutTotalbytessent sets the value of OutTotalbytessent for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSGlobal) SetPropertyOutTotalbytessent(value uint64) (err error) {
	return instance.SetProperty("OutTotalbytessent", (value))
}

// GetOutTotalbytessent gets the value of OutTotalbytessent for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSGlobal) GetPropertyOutTotalbytessent() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutTotalbytessent")
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

// SetOutTotalpacketssent sets the value of OutTotalpacketssent for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSGlobal) SetPropertyOutTotalpacketssent(value uint64) (err error) {
	return instance.SetProperty("OutTotalpacketssent", (value))
}

// GetOutTotalpacketssent gets the value of OutTotalpacketssent for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSGlobal) GetPropertyOutTotalpacketssent() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutTotalpacketssent")
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

// SetSessionsTotalsessions sets the value of SessionsTotalsessions for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSGlobal) SetPropertySessionsTotalsessions(value uint64) (err error) {
	return instance.SetProperty("SessionsTotalsessions", (value))
}

// GetSessionsTotalsessions gets the value of SessionsTotalsessions for the instance
func (instance *Win32_PerfFormattedData_Counters_IPHTTPSGlobal) GetPropertySessionsTotalsessions() (value uint64, err error) {
	retValue, err := instance.GetProperty("SessionsTotalsessions")
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
