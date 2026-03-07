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

// Win32_PerfRawData_Counters_DNS64Global struct
type Win32_PerfRawData_Counters_DNS64Global struct {
	*Win32_PerfRawData

	//
	AAAAqueriesFailed uint64

	//
	AAAAqueriesSuccessful uint64

	//
	AAAASynthesizedrecords uint64

	//
	IP6ARPAqueriesMatched uint64

	//
	OtherqueriesFailed uint64

	//
	OtherqueriesSuccessful uint64
}

func NewWin32_PerfRawData_Counters_DNS64GlobalEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_Counters_DNS64Global, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_DNS64Global{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_Counters_DNS64GlobalEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_Counters_DNS64Global, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_DNS64Global{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetAAAAqueriesFailed sets the value of AAAAqueriesFailed for the instance
func (instance *Win32_PerfRawData_Counters_DNS64Global) SetPropertyAAAAqueriesFailed(value uint64) (err error) {
	return instance.SetProperty("AAAAqueriesFailed", (value))
}

// GetAAAAqueriesFailed gets the value of AAAAqueriesFailed for the instance
func (instance *Win32_PerfRawData_Counters_DNS64Global) GetPropertyAAAAqueriesFailed() (value uint64, err error) {
	retValue, err := instance.GetProperty("AAAAqueriesFailed")
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

// SetAAAAqueriesSuccessful sets the value of AAAAqueriesSuccessful for the instance
func (instance *Win32_PerfRawData_Counters_DNS64Global) SetPropertyAAAAqueriesSuccessful(value uint64) (err error) {
	return instance.SetProperty("AAAAqueriesSuccessful", (value))
}

// GetAAAAqueriesSuccessful gets the value of AAAAqueriesSuccessful for the instance
func (instance *Win32_PerfRawData_Counters_DNS64Global) GetPropertyAAAAqueriesSuccessful() (value uint64, err error) {
	retValue, err := instance.GetProperty("AAAAqueriesSuccessful")
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

// SetAAAASynthesizedrecords sets the value of AAAASynthesizedrecords for the instance
func (instance *Win32_PerfRawData_Counters_DNS64Global) SetPropertyAAAASynthesizedrecords(value uint64) (err error) {
	return instance.SetProperty("AAAASynthesizedrecords", (value))
}

// GetAAAASynthesizedrecords gets the value of AAAASynthesizedrecords for the instance
func (instance *Win32_PerfRawData_Counters_DNS64Global) GetPropertyAAAASynthesizedrecords() (value uint64, err error) {
	retValue, err := instance.GetProperty("AAAASynthesizedrecords")
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

// SetIP6ARPAqueriesMatched sets the value of IP6ARPAqueriesMatched for the instance
func (instance *Win32_PerfRawData_Counters_DNS64Global) SetPropertyIP6ARPAqueriesMatched(value uint64) (err error) {
	return instance.SetProperty("IP6ARPAqueriesMatched", (value))
}

// GetIP6ARPAqueriesMatched gets the value of IP6ARPAqueriesMatched for the instance
func (instance *Win32_PerfRawData_Counters_DNS64Global) GetPropertyIP6ARPAqueriesMatched() (value uint64, err error) {
	retValue, err := instance.GetProperty("IP6ARPAqueriesMatched")
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

// SetOtherqueriesFailed sets the value of OtherqueriesFailed for the instance
func (instance *Win32_PerfRawData_Counters_DNS64Global) SetPropertyOtherqueriesFailed(value uint64) (err error) {
	return instance.SetProperty("OtherqueriesFailed", (value))
}

// GetOtherqueriesFailed gets the value of OtherqueriesFailed for the instance
func (instance *Win32_PerfRawData_Counters_DNS64Global) GetPropertyOtherqueriesFailed() (value uint64, err error) {
	retValue, err := instance.GetProperty("OtherqueriesFailed")
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

// SetOtherqueriesSuccessful sets the value of OtherqueriesSuccessful for the instance
func (instance *Win32_PerfRawData_Counters_DNS64Global) SetPropertyOtherqueriesSuccessful(value uint64) (err error) {
	return instance.SetProperty("OtherqueriesSuccessful", (value))
}

// GetOtherqueriesSuccessful gets the value of OtherqueriesSuccessful for the instance
func (instance *Win32_PerfRawData_Counters_DNS64Global) GetPropertyOtherqueriesSuccessful() (value uint64, err error) {
	retValue, err := instance.GetProperty("OtherqueriesSuccessful")
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
