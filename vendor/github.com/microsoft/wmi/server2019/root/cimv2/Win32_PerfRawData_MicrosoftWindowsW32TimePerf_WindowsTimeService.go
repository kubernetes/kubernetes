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

// Win32_PerfRawData_MicrosoftWindowsW32TimePerf_WindowsTimeService struct
type Win32_PerfRawData_MicrosoftWindowsW32TimePerf_WindowsTimeService struct {
	*Win32_PerfRawData

	//
	ClockFrequencyAdjustment uint32

	//
	ClockFrequencyAdjustmentPPB uint32

	//
	ComputedTimeOffset uint64

	//
	NTPClientTimeSourceCount uint32

	//
	NTPRoundtripDelay uint32

	//
	NTPServerIncomingRequests uint64

	//
	NTPServerOutgoingResponses uint64
}

func NewWin32_PerfRawData_MicrosoftWindowsW32TimePerf_WindowsTimeServiceEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_MicrosoftWindowsW32TimePerf_WindowsTimeService, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_MicrosoftWindowsW32TimePerf_WindowsTimeService{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_MicrosoftWindowsW32TimePerf_WindowsTimeServiceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_MicrosoftWindowsW32TimePerf_WindowsTimeService, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_MicrosoftWindowsW32TimePerf_WindowsTimeService{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetClockFrequencyAdjustment sets the value of ClockFrequencyAdjustment for the instance
func (instance *Win32_PerfRawData_MicrosoftWindowsW32TimePerf_WindowsTimeService) SetPropertyClockFrequencyAdjustment(value uint32) (err error) {
	return instance.SetProperty("ClockFrequencyAdjustment", (value))
}

// GetClockFrequencyAdjustment gets the value of ClockFrequencyAdjustment for the instance
func (instance *Win32_PerfRawData_MicrosoftWindowsW32TimePerf_WindowsTimeService) GetPropertyClockFrequencyAdjustment() (value uint32, err error) {
	retValue, err := instance.GetProperty("ClockFrequencyAdjustment")
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

// SetClockFrequencyAdjustmentPPB sets the value of ClockFrequencyAdjustmentPPB for the instance
func (instance *Win32_PerfRawData_MicrosoftWindowsW32TimePerf_WindowsTimeService) SetPropertyClockFrequencyAdjustmentPPB(value uint32) (err error) {
	return instance.SetProperty("ClockFrequencyAdjustmentPPB", (value))
}

// GetClockFrequencyAdjustmentPPB gets the value of ClockFrequencyAdjustmentPPB for the instance
func (instance *Win32_PerfRawData_MicrosoftWindowsW32TimePerf_WindowsTimeService) GetPropertyClockFrequencyAdjustmentPPB() (value uint32, err error) {
	retValue, err := instance.GetProperty("ClockFrequencyAdjustmentPPB")
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

// SetComputedTimeOffset sets the value of ComputedTimeOffset for the instance
func (instance *Win32_PerfRawData_MicrosoftWindowsW32TimePerf_WindowsTimeService) SetPropertyComputedTimeOffset(value uint64) (err error) {
	return instance.SetProperty("ComputedTimeOffset", (value))
}

// GetComputedTimeOffset gets the value of ComputedTimeOffset for the instance
func (instance *Win32_PerfRawData_MicrosoftWindowsW32TimePerf_WindowsTimeService) GetPropertyComputedTimeOffset() (value uint64, err error) {
	retValue, err := instance.GetProperty("ComputedTimeOffset")
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

// SetNTPClientTimeSourceCount sets the value of NTPClientTimeSourceCount for the instance
func (instance *Win32_PerfRawData_MicrosoftWindowsW32TimePerf_WindowsTimeService) SetPropertyNTPClientTimeSourceCount(value uint32) (err error) {
	return instance.SetProperty("NTPClientTimeSourceCount", (value))
}

// GetNTPClientTimeSourceCount gets the value of NTPClientTimeSourceCount for the instance
func (instance *Win32_PerfRawData_MicrosoftWindowsW32TimePerf_WindowsTimeService) GetPropertyNTPClientTimeSourceCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("NTPClientTimeSourceCount")
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

// SetNTPRoundtripDelay sets the value of NTPRoundtripDelay for the instance
func (instance *Win32_PerfRawData_MicrosoftWindowsW32TimePerf_WindowsTimeService) SetPropertyNTPRoundtripDelay(value uint32) (err error) {
	return instance.SetProperty("NTPRoundtripDelay", (value))
}

// GetNTPRoundtripDelay gets the value of NTPRoundtripDelay for the instance
func (instance *Win32_PerfRawData_MicrosoftWindowsW32TimePerf_WindowsTimeService) GetPropertyNTPRoundtripDelay() (value uint32, err error) {
	retValue, err := instance.GetProperty("NTPRoundtripDelay")
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

// SetNTPServerIncomingRequests sets the value of NTPServerIncomingRequests for the instance
func (instance *Win32_PerfRawData_MicrosoftWindowsW32TimePerf_WindowsTimeService) SetPropertyNTPServerIncomingRequests(value uint64) (err error) {
	return instance.SetProperty("NTPServerIncomingRequests", (value))
}

// GetNTPServerIncomingRequests gets the value of NTPServerIncomingRequests for the instance
func (instance *Win32_PerfRawData_MicrosoftWindowsW32TimePerf_WindowsTimeService) GetPropertyNTPServerIncomingRequests() (value uint64, err error) {
	retValue, err := instance.GetProperty("NTPServerIncomingRequests")
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

// SetNTPServerOutgoingResponses sets the value of NTPServerOutgoingResponses for the instance
func (instance *Win32_PerfRawData_MicrosoftWindowsW32TimePerf_WindowsTimeService) SetPropertyNTPServerOutgoingResponses(value uint64) (err error) {
	return instance.SetProperty("NTPServerOutgoingResponses", (value))
}

// GetNTPServerOutgoingResponses gets the value of NTPServerOutgoingResponses for the instance
func (instance *Win32_PerfRawData_MicrosoftWindowsW32TimePerf_WindowsTimeService) GetPropertyNTPServerOutgoingResponses() (value uint64, err error) {
	retValue, err := instance.GetProperty("NTPServerOutgoingResponses")
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
