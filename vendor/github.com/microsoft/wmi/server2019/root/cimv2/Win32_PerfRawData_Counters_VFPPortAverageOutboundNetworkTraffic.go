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

// Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic struct
type Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic struct {
	*Win32_PerfRawData

	//
	AverageGFTOutboundBytes uint64

	//
	AverageOutboundBytes uint64

	//
	AverageOutboundForwardedMulticastPackets uint64

	//
	AverageOutboundForwardedUnicastPackets uint64

	//
	AverageOutboundGFTCopyFINPackets uint64

	//
	AverageOutboundGFTCopyPackets uint64

	//
	AverageOutboundGFTCopyResetPackets uint64

	//
	AverageOutboundGFTExceptionPackets uint64

	//
	AverageOutboundGFTExceptionUFOffloadBlockedPackets uint64

	//
	AverageOutboundGFTExceptionUFOffloadDeferredPackets uint64

	//
	AverageOutboundGFTExceptionUFOffloadedTCPPackets uint64

	//
	AverageOutboundGFTExceptionUFOffloadedUDPPackets uint64

	//
	AverageOutboundGFTExceptionUFOffloadFailedPackets uint64

	//
	AverageOutboundGFTExceptionUFOffloadPendingPackets uint64

	//
	AverageOutboundGFTExceptionUFOffloadRetryAwaitingPackets uint64

	//
	AverageOutboundGFTExceptionUFPackets uint64

	//
	AverageOutboundGFTPackets uint64

	//
	AverageOutboundHairpinnedPackets uint64

	//
	AverageOutboundInterceptedPackets uint64

	//
	AverageOutboundMissedInterceptedPackets uint64

	//
	AverageOutboundNonIPPackets uint64

	//
	AverageOutboundPackets uint64

	//
	AverageOutboundPendingPackets uint64

	//
	AverageOutboundTCPSYNACKPackets uint64

	//
	AverageOutboundTCPSYNPackets uint64

	//
	AverageOutboundThrottledPackets uint64

	//
	AverageOutboundUnicastForwardedGFTExceptionPackets uint64
}

func NewWin32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTrafficEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTrafficEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetAverageGFTOutboundBytes sets the value of AverageGFTOutboundBytes for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) SetPropertyAverageGFTOutboundBytes(value uint64) (err error) {
	return instance.SetProperty("AverageGFTOutboundBytes", (value))
}

// GetAverageGFTOutboundBytes gets the value of AverageGFTOutboundBytes for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) GetPropertyAverageGFTOutboundBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageGFTOutboundBytes")
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

// SetAverageOutboundBytes sets the value of AverageOutboundBytes for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) SetPropertyAverageOutboundBytes(value uint64) (err error) {
	return instance.SetProperty("AverageOutboundBytes", (value))
}

// GetAverageOutboundBytes gets the value of AverageOutboundBytes for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) GetPropertyAverageOutboundBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageOutboundBytes")
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

// SetAverageOutboundForwardedMulticastPackets sets the value of AverageOutboundForwardedMulticastPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) SetPropertyAverageOutboundForwardedMulticastPackets(value uint64) (err error) {
	return instance.SetProperty("AverageOutboundForwardedMulticastPackets", (value))
}

// GetAverageOutboundForwardedMulticastPackets gets the value of AverageOutboundForwardedMulticastPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) GetPropertyAverageOutboundForwardedMulticastPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageOutboundForwardedMulticastPackets")
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

// SetAverageOutboundForwardedUnicastPackets sets the value of AverageOutboundForwardedUnicastPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) SetPropertyAverageOutboundForwardedUnicastPackets(value uint64) (err error) {
	return instance.SetProperty("AverageOutboundForwardedUnicastPackets", (value))
}

// GetAverageOutboundForwardedUnicastPackets gets the value of AverageOutboundForwardedUnicastPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) GetPropertyAverageOutboundForwardedUnicastPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageOutboundForwardedUnicastPackets")
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

// SetAverageOutboundGFTCopyFINPackets sets the value of AverageOutboundGFTCopyFINPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) SetPropertyAverageOutboundGFTCopyFINPackets(value uint64) (err error) {
	return instance.SetProperty("AverageOutboundGFTCopyFINPackets", (value))
}

// GetAverageOutboundGFTCopyFINPackets gets the value of AverageOutboundGFTCopyFINPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) GetPropertyAverageOutboundGFTCopyFINPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageOutboundGFTCopyFINPackets")
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

// SetAverageOutboundGFTCopyPackets sets the value of AverageOutboundGFTCopyPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) SetPropertyAverageOutboundGFTCopyPackets(value uint64) (err error) {
	return instance.SetProperty("AverageOutboundGFTCopyPackets", (value))
}

// GetAverageOutboundGFTCopyPackets gets the value of AverageOutboundGFTCopyPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) GetPropertyAverageOutboundGFTCopyPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageOutboundGFTCopyPackets")
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

// SetAverageOutboundGFTCopyResetPackets sets the value of AverageOutboundGFTCopyResetPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) SetPropertyAverageOutboundGFTCopyResetPackets(value uint64) (err error) {
	return instance.SetProperty("AverageOutboundGFTCopyResetPackets", (value))
}

// GetAverageOutboundGFTCopyResetPackets gets the value of AverageOutboundGFTCopyResetPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) GetPropertyAverageOutboundGFTCopyResetPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageOutboundGFTCopyResetPackets")
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

// SetAverageOutboundGFTExceptionPackets sets the value of AverageOutboundGFTExceptionPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) SetPropertyAverageOutboundGFTExceptionPackets(value uint64) (err error) {
	return instance.SetProperty("AverageOutboundGFTExceptionPackets", (value))
}

// GetAverageOutboundGFTExceptionPackets gets the value of AverageOutboundGFTExceptionPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) GetPropertyAverageOutboundGFTExceptionPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageOutboundGFTExceptionPackets")
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

// SetAverageOutboundGFTExceptionUFOffloadBlockedPackets sets the value of AverageOutboundGFTExceptionUFOffloadBlockedPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) SetPropertyAverageOutboundGFTExceptionUFOffloadBlockedPackets(value uint64) (err error) {
	return instance.SetProperty("AverageOutboundGFTExceptionUFOffloadBlockedPackets", (value))
}

// GetAverageOutboundGFTExceptionUFOffloadBlockedPackets gets the value of AverageOutboundGFTExceptionUFOffloadBlockedPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) GetPropertyAverageOutboundGFTExceptionUFOffloadBlockedPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageOutboundGFTExceptionUFOffloadBlockedPackets")
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

// SetAverageOutboundGFTExceptionUFOffloadDeferredPackets sets the value of AverageOutboundGFTExceptionUFOffloadDeferredPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) SetPropertyAverageOutboundGFTExceptionUFOffloadDeferredPackets(value uint64) (err error) {
	return instance.SetProperty("AverageOutboundGFTExceptionUFOffloadDeferredPackets", (value))
}

// GetAverageOutboundGFTExceptionUFOffloadDeferredPackets gets the value of AverageOutboundGFTExceptionUFOffloadDeferredPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) GetPropertyAverageOutboundGFTExceptionUFOffloadDeferredPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageOutboundGFTExceptionUFOffloadDeferredPackets")
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

// SetAverageOutboundGFTExceptionUFOffloadedTCPPackets sets the value of AverageOutboundGFTExceptionUFOffloadedTCPPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) SetPropertyAverageOutboundGFTExceptionUFOffloadedTCPPackets(value uint64) (err error) {
	return instance.SetProperty("AverageOutboundGFTExceptionUFOffloadedTCPPackets", (value))
}

// GetAverageOutboundGFTExceptionUFOffloadedTCPPackets gets the value of AverageOutboundGFTExceptionUFOffloadedTCPPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) GetPropertyAverageOutboundGFTExceptionUFOffloadedTCPPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageOutboundGFTExceptionUFOffloadedTCPPackets")
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

// SetAverageOutboundGFTExceptionUFOffloadedUDPPackets sets the value of AverageOutboundGFTExceptionUFOffloadedUDPPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) SetPropertyAverageOutboundGFTExceptionUFOffloadedUDPPackets(value uint64) (err error) {
	return instance.SetProperty("AverageOutboundGFTExceptionUFOffloadedUDPPackets", (value))
}

// GetAverageOutboundGFTExceptionUFOffloadedUDPPackets gets the value of AverageOutboundGFTExceptionUFOffloadedUDPPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) GetPropertyAverageOutboundGFTExceptionUFOffloadedUDPPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageOutboundGFTExceptionUFOffloadedUDPPackets")
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

// SetAverageOutboundGFTExceptionUFOffloadFailedPackets sets the value of AverageOutboundGFTExceptionUFOffloadFailedPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) SetPropertyAverageOutboundGFTExceptionUFOffloadFailedPackets(value uint64) (err error) {
	return instance.SetProperty("AverageOutboundGFTExceptionUFOffloadFailedPackets", (value))
}

// GetAverageOutboundGFTExceptionUFOffloadFailedPackets gets the value of AverageOutboundGFTExceptionUFOffloadFailedPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) GetPropertyAverageOutboundGFTExceptionUFOffloadFailedPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageOutboundGFTExceptionUFOffloadFailedPackets")
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

// SetAverageOutboundGFTExceptionUFOffloadPendingPackets sets the value of AverageOutboundGFTExceptionUFOffloadPendingPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) SetPropertyAverageOutboundGFTExceptionUFOffloadPendingPackets(value uint64) (err error) {
	return instance.SetProperty("AverageOutboundGFTExceptionUFOffloadPendingPackets", (value))
}

// GetAverageOutboundGFTExceptionUFOffloadPendingPackets gets the value of AverageOutboundGFTExceptionUFOffloadPendingPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) GetPropertyAverageOutboundGFTExceptionUFOffloadPendingPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageOutboundGFTExceptionUFOffloadPendingPackets")
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

// SetAverageOutboundGFTExceptionUFOffloadRetryAwaitingPackets sets the value of AverageOutboundGFTExceptionUFOffloadRetryAwaitingPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) SetPropertyAverageOutboundGFTExceptionUFOffloadRetryAwaitingPackets(value uint64) (err error) {
	return instance.SetProperty("AverageOutboundGFTExceptionUFOffloadRetryAwaitingPackets", (value))
}

// GetAverageOutboundGFTExceptionUFOffloadRetryAwaitingPackets gets the value of AverageOutboundGFTExceptionUFOffloadRetryAwaitingPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) GetPropertyAverageOutboundGFTExceptionUFOffloadRetryAwaitingPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageOutboundGFTExceptionUFOffloadRetryAwaitingPackets")
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

// SetAverageOutboundGFTExceptionUFPackets sets the value of AverageOutboundGFTExceptionUFPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) SetPropertyAverageOutboundGFTExceptionUFPackets(value uint64) (err error) {
	return instance.SetProperty("AverageOutboundGFTExceptionUFPackets", (value))
}

// GetAverageOutboundGFTExceptionUFPackets gets the value of AverageOutboundGFTExceptionUFPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) GetPropertyAverageOutboundGFTExceptionUFPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageOutboundGFTExceptionUFPackets")
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

// SetAverageOutboundGFTPackets sets the value of AverageOutboundGFTPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) SetPropertyAverageOutboundGFTPackets(value uint64) (err error) {
	return instance.SetProperty("AverageOutboundGFTPackets", (value))
}

// GetAverageOutboundGFTPackets gets the value of AverageOutboundGFTPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) GetPropertyAverageOutboundGFTPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageOutboundGFTPackets")
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

// SetAverageOutboundHairpinnedPackets sets the value of AverageOutboundHairpinnedPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) SetPropertyAverageOutboundHairpinnedPackets(value uint64) (err error) {
	return instance.SetProperty("AverageOutboundHairpinnedPackets", (value))
}

// GetAverageOutboundHairpinnedPackets gets the value of AverageOutboundHairpinnedPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) GetPropertyAverageOutboundHairpinnedPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageOutboundHairpinnedPackets")
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

// SetAverageOutboundInterceptedPackets sets the value of AverageOutboundInterceptedPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) SetPropertyAverageOutboundInterceptedPackets(value uint64) (err error) {
	return instance.SetProperty("AverageOutboundInterceptedPackets", (value))
}

// GetAverageOutboundInterceptedPackets gets the value of AverageOutboundInterceptedPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) GetPropertyAverageOutboundInterceptedPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageOutboundInterceptedPackets")
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

// SetAverageOutboundMissedInterceptedPackets sets the value of AverageOutboundMissedInterceptedPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) SetPropertyAverageOutboundMissedInterceptedPackets(value uint64) (err error) {
	return instance.SetProperty("AverageOutboundMissedInterceptedPackets", (value))
}

// GetAverageOutboundMissedInterceptedPackets gets the value of AverageOutboundMissedInterceptedPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) GetPropertyAverageOutboundMissedInterceptedPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageOutboundMissedInterceptedPackets")
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

// SetAverageOutboundNonIPPackets sets the value of AverageOutboundNonIPPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) SetPropertyAverageOutboundNonIPPackets(value uint64) (err error) {
	return instance.SetProperty("AverageOutboundNonIPPackets", (value))
}

// GetAverageOutboundNonIPPackets gets the value of AverageOutboundNonIPPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) GetPropertyAverageOutboundNonIPPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageOutboundNonIPPackets")
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

// SetAverageOutboundPackets sets the value of AverageOutboundPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) SetPropertyAverageOutboundPackets(value uint64) (err error) {
	return instance.SetProperty("AverageOutboundPackets", (value))
}

// GetAverageOutboundPackets gets the value of AverageOutboundPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) GetPropertyAverageOutboundPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageOutboundPackets")
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

// SetAverageOutboundPendingPackets sets the value of AverageOutboundPendingPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) SetPropertyAverageOutboundPendingPackets(value uint64) (err error) {
	return instance.SetProperty("AverageOutboundPendingPackets", (value))
}

// GetAverageOutboundPendingPackets gets the value of AverageOutboundPendingPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) GetPropertyAverageOutboundPendingPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageOutboundPendingPackets")
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

// SetAverageOutboundTCPSYNACKPackets sets the value of AverageOutboundTCPSYNACKPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) SetPropertyAverageOutboundTCPSYNACKPackets(value uint64) (err error) {
	return instance.SetProperty("AverageOutboundTCPSYNACKPackets", (value))
}

// GetAverageOutboundTCPSYNACKPackets gets the value of AverageOutboundTCPSYNACKPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) GetPropertyAverageOutboundTCPSYNACKPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageOutboundTCPSYNACKPackets")
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

// SetAverageOutboundTCPSYNPackets sets the value of AverageOutboundTCPSYNPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) SetPropertyAverageOutboundTCPSYNPackets(value uint64) (err error) {
	return instance.SetProperty("AverageOutboundTCPSYNPackets", (value))
}

// GetAverageOutboundTCPSYNPackets gets the value of AverageOutboundTCPSYNPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) GetPropertyAverageOutboundTCPSYNPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageOutboundTCPSYNPackets")
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

// SetAverageOutboundThrottledPackets sets the value of AverageOutboundThrottledPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) SetPropertyAverageOutboundThrottledPackets(value uint64) (err error) {
	return instance.SetProperty("AverageOutboundThrottledPackets", (value))
}

// GetAverageOutboundThrottledPackets gets the value of AverageOutboundThrottledPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) GetPropertyAverageOutboundThrottledPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageOutboundThrottledPackets")
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

// SetAverageOutboundUnicastForwardedGFTExceptionPackets sets the value of AverageOutboundUnicastForwardedGFTExceptionPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) SetPropertyAverageOutboundUnicastForwardedGFTExceptionPackets(value uint64) (err error) {
	return instance.SetProperty("AverageOutboundUnicastForwardedGFTExceptionPackets", (value))
}

// GetAverageOutboundUnicastForwardedGFTExceptionPackets gets the value of AverageOutboundUnicastForwardedGFTExceptionPackets for the instance
func (instance *Win32_PerfRawData_Counters_VFPPortAverageOutboundNetworkTraffic) GetPropertyAverageOutboundUnicastForwardedGFTExceptionPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("AverageOutboundUnicastForwardedGFTExceptionPackets")
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
