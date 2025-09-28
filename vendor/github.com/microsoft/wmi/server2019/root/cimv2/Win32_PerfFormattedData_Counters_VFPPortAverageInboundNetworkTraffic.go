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

// Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic struct
type Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic struct {
	*Win32_PerfFormattedData

	//
	AverageInboundBytes uint64

	//
	AverageInboundForwardedMulticastPackets uint64

	//
	AverageInboundForwardedUnicastPackets uint64

	//
	AverageInboundGFTCopyFINPackets uint64

	//
	AverageInboundGFTCopyPackets uint64

	//
	AverageInboundGFTCopyResetPackets uint64

	//
	AverageInboundGFTExceptionPackets uint64

	//
	AverageInboundGFTExceptionUFOffloadBlockedPackets uint64

	//
	AverageInboundGFTExceptionUFOffloadDeferredPackets uint64

	//
	AverageInboundGFTExceptionUFOffloadedTCPPackets uint64

	//
	AverageInboundGFTExceptionUFOffloadedUDPPackets uint64

	//
	AverageInboundGFTExceptionUFOffloadFailedPackets uint64

	//
	AverageInboundGFTExceptionUFOffloadPendingPackets uint64

	//
	AverageInboundGFTExceptionUFOffloadRetryAwaitingPackets uint64

	//
	AverageInboundGFTExceptionUFPackets uint64

	//
	AverageInboundGFTTotalBytes uint64

	//
	AverageInboundGFTTotalPackets uint64

	//
	AverageInboundHairPinnedPackets uint64

	//
	AverageInboundInterceptedPackets uint64

	//
	AverageInboundMissedInterceptedPackets uint64

	//
	AverageInboundNonIPPackets uint64

	//
	AverageInboundPackets uint64

	//
	AverageInboundPendingPackets uint64

	//
	AverageInboundTCPSYNACKPackets uint64

	//
	AverageInboundTCPSYNPackets uint64

	//
	AverageInboundThrottledPackets uint64

	//
	AverageInboundUnicastForwardedGFTExceptionPackets uint64
}

func NewWin32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTrafficEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTrafficEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetAverageInboundBytes sets the value of AverageInboundBytes for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) SetPropertyAverageInboundBytes(value uint64) (err error) {
	return instance.SetProperty("AverageInboundBytes", (value))
}

// GetAverageInboundBytes gets the value of AverageInboundBytes for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) GetPropertyAverageInboundBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundBytes")
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

// SetAverageInboundForwardedMulticastPackets sets the value of AverageInboundForwardedMulticastPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) SetPropertyAverageInboundForwardedMulticastPackets(value uint64) (err error) {
	return instance.SetProperty("AverageInboundForwardedMulticastPackets", (value))
}

// GetAverageInboundForwardedMulticastPackets gets the value of AverageInboundForwardedMulticastPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) GetPropertyAverageInboundForwardedMulticastPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundForwardedMulticastPackets")
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

// SetAverageInboundForwardedUnicastPackets sets the value of AverageInboundForwardedUnicastPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) SetPropertyAverageInboundForwardedUnicastPackets(value uint64) (err error) {
	return instance.SetProperty("AverageInboundForwardedUnicastPackets", (value))
}

// GetAverageInboundForwardedUnicastPackets gets the value of AverageInboundForwardedUnicastPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) GetPropertyAverageInboundForwardedUnicastPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundForwardedUnicastPackets")
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

// SetAverageInboundGFTCopyFINPackets sets the value of AverageInboundGFTCopyFINPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) SetPropertyAverageInboundGFTCopyFINPackets(value uint64) (err error) {
	return instance.SetProperty("AverageInboundGFTCopyFINPackets", (value))
}

// GetAverageInboundGFTCopyFINPackets gets the value of AverageInboundGFTCopyFINPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) GetPropertyAverageInboundGFTCopyFINPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundGFTCopyFINPackets")
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

// SetAverageInboundGFTCopyPackets sets the value of AverageInboundGFTCopyPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) SetPropertyAverageInboundGFTCopyPackets(value uint64) (err error) {
	return instance.SetProperty("AverageInboundGFTCopyPackets", (value))
}

// GetAverageInboundGFTCopyPackets gets the value of AverageInboundGFTCopyPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) GetPropertyAverageInboundGFTCopyPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundGFTCopyPackets")
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

// SetAverageInboundGFTCopyResetPackets sets the value of AverageInboundGFTCopyResetPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) SetPropertyAverageInboundGFTCopyResetPackets(value uint64) (err error) {
	return instance.SetProperty("AverageInboundGFTCopyResetPackets", (value))
}

// GetAverageInboundGFTCopyResetPackets gets the value of AverageInboundGFTCopyResetPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) GetPropertyAverageInboundGFTCopyResetPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundGFTCopyResetPackets")
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

// SetAverageInboundGFTExceptionPackets sets the value of AverageInboundGFTExceptionPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) SetPropertyAverageInboundGFTExceptionPackets(value uint64) (err error) {
	return instance.SetProperty("AverageInboundGFTExceptionPackets", (value))
}

// GetAverageInboundGFTExceptionPackets gets the value of AverageInboundGFTExceptionPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) GetPropertyAverageInboundGFTExceptionPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundGFTExceptionPackets")
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

// SetAverageInboundGFTExceptionUFOffloadBlockedPackets sets the value of AverageInboundGFTExceptionUFOffloadBlockedPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) SetPropertyAverageInboundGFTExceptionUFOffloadBlockedPackets(value uint64) (err error) {
	return instance.SetProperty("AverageInboundGFTExceptionUFOffloadBlockedPackets", (value))
}

// GetAverageInboundGFTExceptionUFOffloadBlockedPackets gets the value of AverageInboundGFTExceptionUFOffloadBlockedPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) GetPropertyAverageInboundGFTExceptionUFOffloadBlockedPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundGFTExceptionUFOffloadBlockedPackets")
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

// SetAverageInboundGFTExceptionUFOffloadDeferredPackets sets the value of AverageInboundGFTExceptionUFOffloadDeferredPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) SetPropertyAverageInboundGFTExceptionUFOffloadDeferredPackets(value uint64) (err error) {
	return instance.SetProperty("AverageInboundGFTExceptionUFOffloadDeferredPackets", (value))
}

// GetAverageInboundGFTExceptionUFOffloadDeferredPackets gets the value of AverageInboundGFTExceptionUFOffloadDeferredPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) GetPropertyAverageInboundGFTExceptionUFOffloadDeferredPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundGFTExceptionUFOffloadDeferredPackets")
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

// SetAverageInboundGFTExceptionUFOffloadedTCPPackets sets the value of AverageInboundGFTExceptionUFOffloadedTCPPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) SetPropertyAverageInboundGFTExceptionUFOffloadedTCPPackets(value uint64) (err error) {
	return instance.SetProperty("AverageInboundGFTExceptionUFOffloadedTCPPackets", (value))
}

// GetAverageInboundGFTExceptionUFOffloadedTCPPackets gets the value of AverageInboundGFTExceptionUFOffloadedTCPPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) GetPropertyAverageInboundGFTExceptionUFOffloadedTCPPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundGFTExceptionUFOffloadedTCPPackets")
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

// SetAverageInboundGFTExceptionUFOffloadedUDPPackets sets the value of AverageInboundGFTExceptionUFOffloadedUDPPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) SetPropertyAverageInboundGFTExceptionUFOffloadedUDPPackets(value uint64) (err error) {
	return instance.SetProperty("AverageInboundGFTExceptionUFOffloadedUDPPackets", (value))
}

// GetAverageInboundGFTExceptionUFOffloadedUDPPackets gets the value of AverageInboundGFTExceptionUFOffloadedUDPPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) GetPropertyAverageInboundGFTExceptionUFOffloadedUDPPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundGFTExceptionUFOffloadedUDPPackets")
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

// SetAverageInboundGFTExceptionUFOffloadFailedPackets sets the value of AverageInboundGFTExceptionUFOffloadFailedPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) SetPropertyAverageInboundGFTExceptionUFOffloadFailedPackets(value uint64) (err error) {
	return instance.SetProperty("AverageInboundGFTExceptionUFOffloadFailedPackets", (value))
}

// GetAverageInboundGFTExceptionUFOffloadFailedPackets gets the value of AverageInboundGFTExceptionUFOffloadFailedPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) GetPropertyAverageInboundGFTExceptionUFOffloadFailedPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundGFTExceptionUFOffloadFailedPackets")
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

// SetAverageInboundGFTExceptionUFOffloadPendingPackets sets the value of AverageInboundGFTExceptionUFOffloadPendingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) SetPropertyAverageInboundGFTExceptionUFOffloadPendingPackets(value uint64) (err error) {
	return instance.SetProperty("AverageInboundGFTExceptionUFOffloadPendingPackets", (value))
}

// GetAverageInboundGFTExceptionUFOffloadPendingPackets gets the value of AverageInboundGFTExceptionUFOffloadPendingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) GetPropertyAverageInboundGFTExceptionUFOffloadPendingPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundGFTExceptionUFOffloadPendingPackets")
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

// SetAverageInboundGFTExceptionUFOffloadRetryAwaitingPackets sets the value of AverageInboundGFTExceptionUFOffloadRetryAwaitingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) SetPropertyAverageInboundGFTExceptionUFOffloadRetryAwaitingPackets(value uint64) (err error) {
	return instance.SetProperty("AverageInboundGFTExceptionUFOffloadRetryAwaitingPackets", (value))
}

// GetAverageInboundGFTExceptionUFOffloadRetryAwaitingPackets gets the value of AverageInboundGFTExceptionUFOffloadRetryAwaitingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) GetPropertyAverageInboundGFTExceptionUFOffloadRetryAwaitingPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundGFTExceptionUFOffloadRetryAwaitingPackets")
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

// SetAverageInboundGFTExceptionUFPackets sets the value of AverageInboundGFTExceptionUFPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) SetPropertyAverageInboundGFTExceptionUFPackets(value uint64) (err error) {
	return instance.SetProperty("AverageInboundGFTExceptionUFPackets", (value))
}

// GetAverageInboundGFTExceptionUFPackets gets the value of AverageInboundGFTExceptionUFPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) GetPropertyAverageInboundGFTExceptionUFPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundGFTExceptionUFPackets")
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

// SetAverageInboundGFTTotalBytes sets the value of AverageInboundGFTTotalBytes for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) SetPropertyAverageInboundGFTTotalBytes(value uint64) (err error) {
	return instance.SetProperty("AverageInboundGFTTotalBytes", (value))
}

// GetAverageInboundGFTTotalBytes gets the value of AverageInboundGFTTotalBytes for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) GetPropertyAverageInboundGFTTotalBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundGFTTotalBytes")
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

// SetAverageInboundGFTTotalPackets sets the value of AverageInboundGFTTotalPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) SetPropertyAverageInboundGFTTotalPackets(value uint64) (err error) {
	return instance.SetProperty("AverageInboundGFTTotalPackets", (value))
}

// GetAverageInboundGFTTotalPackets gets the value of AverageInboundGFTTotalPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) GetPropertyAverageInboundGFTTotalPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundGFTTotalPackets")
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

// SetAverageInboundHairPinnedPackets sets the value of AverageInboundHairPinnedPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) SetPropertyAverageInboundHairPinnedPackets(value uint64) (err error) {
	return instance.SetProperty("AverageInboundHairPinnedPackets", (value))
}

// GetAverageInboundHairPinnedPackets gets the value of AverageInboundHairPinnedPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) GetPropertyAverageInboundHairPinnedPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundHairPinnedPackets")
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

// SetAverageInboundInterceptedPackets sets the value of AverageInboundInterceptedPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) SetPropertyAverageInboundInterceptedPackets(value uint64) (err error) {
	return instance.SetProperty("AverageInboundInterceptedPackets", (value))
}

// GetAverageInboundInterceptedPackets gets the value of AverageInboundInterceptedPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) GetPropertyAverageInboundInterceptedPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundInterceptedPackets")
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

// SetAverageInboundMissedInterceptedPackets sets the value of AverageInboundMissedInterceptedPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) SetPropertyAverageInboundMissedInterceptedPackets(value uint64) (err error) {
	return instance.SetProperty("AverageInboundMissedInterceptedPackets", (value))
}

// GetAverageInboundMissedInterceptedPackets gets the value of AverageInboundMissedInterceptedPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) GetPropertyAverageInboundMissedInterceptedPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundMissedInterceptedPackets")
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

// SetAverageInboundNonIPPackets sets the value of AverageInboundNonIPPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) SetPropertyAverageInboundNonIPPackets(value uint64) (err error) {
	return instance.SetProperty("AverageInboundNonIPPackets", (value))
}

// GetAverageInboundNonIPPackets gets the value of AverageInboundNonIPPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) GetPropertyAverageInboundNonIPPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundNonIPPackets")
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

// SetAverageInboundPackets sets the value of AverageInboundPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) SetPropertyAverageInboundPackets(value uint64) (err error) {
	return instance.SetProperty("AverageInboundPackets", (value))
}

// GetAverageInboundPackets gets the value of AverageInboundPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) GetPropertyAverageInboundPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundPackets")
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

// SetAverageInboundPendingPackets sets the value of AverageInboundPendingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) SetPropertyAverageInboundPendingPackets(value uint64) (err error) {
	return instance.SetProperty("AverageInboundPendingPackets", (value))
}

// GetAverageInboundPendingPackets gets the value of AverageInboundPendingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) GetPropertyAverageInboundPendingPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundPendingPackets")
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

// SetAverageInboundTCPSYNACKPackets sets the value of AverageInboundTCPSYNACKPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) SetPropertyAverageInboundTCPSYNACKPackets(value uint64) (err error) {
	return instance.SetProperty("AverageInboundTCPSYNACKPackets", (value))
}

// GetAverageInboundTCPSYNACKPackets gets the value of AverageInboundTCPSYNACKPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) GetPropertyAverageInboundTCPSYNACKPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundTCPSYNACKPackets")
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

// SetAverageInboundTCPSYNPackets sets the value of AverageInboundTCPSYNPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) SetPropertyAverageInboundTCPSYNPackets(value uint64) (err error) {
	return instance.SetProperty("AverageInboundTCPSYNPackets", (value))
}

// GetAverageInboundTCPSYNPackets gets the value of AverageInboundTCPSYNPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) GetPropertyAverageInboundTCPSYNPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundTCPSYNPackets")
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

// SetAverageInboundThrottledPackets sets the value of AverageInboundThrottledPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) SetPropertyAverageInboundThrottledPackets(value uint64) (err error) {
	return instance.SetProperty("AverageInboundThrottledPackets", (value))
}

// GetAverageInboundThrottledPackets gets the value of AverageInboundThrottledPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) GetPropertyAverageInboundThrottledPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundThrottledPackets")
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

// SetAverageInboundUnicastForwardedGFTExceptionPackets sets the value of AverageInboundUnicastForwardedGFTExceptionPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) SetPropertyAverageInboundUnicastForwardedGFTExceptionPackets(value uint64) (err error) {
	return instance.SetProperty("AverageInboundUnicastForwardedGFTExceptionPackets", (value))
}

// GetAverageInboundUnicastForwardedGFTExceptionPackets gets the value of AverageInboundUnicastForwardedGFTExceptionPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortAverageInboundNetworkTraffic) GetPropertyAverageInboundUnicastForwardedGFTExceptionPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageInboundUnicastForwardedGFTExceptionPackets")
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
