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

// Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic struct
type Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic struct {
	*Win32_PerfFormattedData

	//
	TotalInboundBytes uint64

	//
	TotalInboundForwardedMulticastPackets uint64

	//
	TotalInboundForwardedUnicastPackets uint64

	//
	TotalInboundGFTBytes uint64

	//
	TotalInboundGFTCopyFINPackets uint64

	//
	TotalInboundGFTCopyPackets uint64

	//
	TotalInboundGFTCopyResetPackets uint64

	//
	TotalInboundGFTExceptionPackets uint64

	//
	TotalInboundGFTExceptionUFOffloadBlockedPackets uint64

	//
	TotalInboundGFTExceptionUFOffloadDeferredPackets uint64

	//
	TotalInboundGFTExceptionUFOffloadedTCPPackets uint64

	//
	TotalInboundGFTExceptionUFOffloadedUDPPackets uint64

	//
	TotalInboundGFTExceptionUFOffloadFailedPackets uint64

	//
	TotalInboundGFTExceptionUFOffloadPendingPackets uint64

	//
	TotalInboundGFTExceptionUFPackets uint64

	//
	TotalInboundGFTExceptionUFRetryAwaitingPackets uint64

	//
	TotalInboundGFTPackets uint64

	//
	TotalInboundHairpinnedPackets uint64

	//
	TotalInboundInterceptedPackets uint64

	//
	TotalInboundMissedInterceptedPackets uint64

	//
	TotalInboundNonIPPackets uint64

	//
	TotalInboundPackets uint64

	//
	TotalInboundPendingPackets uint64

	//
	TotalInboundTCPSYNACKPackets uint64

	//
	TotalInboundTCPSYNPackets uint64

	//
	TotalInboundThrottledPackets uint64

	//
	TotalInboundUnicastForwardedGFTExceptionPackets uint64
}

func NewWin32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTrafficEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTrafficEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetTotalInboundBytes sets the value of TotalInboundBytes for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) SetPropertyTotalInboundBytes(value uint64) (err error) {
	return instance.SetProperty("TotalInboundBytes", (value))
}

// GetTotalInboundBytes gets the value of TotalInboundBytes for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) GetPropertyTotalInboundBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundBytes")
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

// SetTotalInboundForwardedMulticastPackets sets the value of TotalInboundForwardedMulticastPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) SetPropertyTotalInboundForwardedMulticastPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundForwardedMulticastPackets", (value))
}

// GetTotalInboundForwardedMulticastPackets gets the value of TotalInboundForwardedMulticastPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) GetPropertyTotalInboundForwardedMulticastPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundForwardedMulticastPackets")
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

// SetTotalInboundForwardedUnicastPackets sets the value of TotalInboundForwardedUnicastPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) SetPropertyTotalInboundForwardedUnicastPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundForwardedUnicastPackets", (value))
}

// GetTotalInboundForwardedUnicastPackets gets the value of TotalInboundForwardedUnicastPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) GetPropertyTotalInboundForwardedUnicastPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundForwardedUnicastPackets")
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

// SetTotalInboundGFTBytes sets the value of TotalInboundGFTBytes for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) SetPropertyTotalInboundGFTBytes(value uint64) (err error) {
	return instance.SetProperty("TotalInboundGFTBytes", (value))
}

// GetTotalInboundGFTBytes gets the value of TotalInboundGFTBytes for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) GetPropertyTotalInboundGFTBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundGFTBytes")
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

// SetTotalInboundGFTCopyFINPackets sets the value of TotalInboundGFTCopyFINPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) SetPropertyTotalInboundGFTCopyFINPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundGFTCopyFINPackets", (value))
}

// GetTotalInboundGFTCopyFINPackets gets the value of TotalInboundGFTCopyFINPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) GetPropertyTotalInboundGFTCopyFINPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundGFTCopyFINPackets")
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

// SetTotalInboundGFTCopyPackets sets the value of TotalInboundGFTCopyPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) SetPropertyTotalInboundGFTCopyPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundGFTCopyPackets", (value))
}

// GetTotalInboundGFTCopyPackets gets the value of TotalInboundGFTCopyPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) GetPropertyTotalInboundGFTCopyPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundGFTCopyPackets")
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

// SetTotalInboundGFTCopyResetPackets sets the value of TotalInboundGFTCopyResetPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) SetPropertyTotalInboundGFTCopyResetPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundGFTCopyResetPackets", (value))
}

// GetTotalInboundGFTCopyResetPackets gets the value of TotalInboundGFTCopyResetPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) GetPropertyTotalInboundGFTCopyResetPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundGFTCopyResetPackets")
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

// SetTotalInboundGFTExceptionPackets sets the value of TotalInboundGFTExceptionPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) SetPropertyTotalInboundGFTExceptionPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundGFTExceptionPackets", (value))
}

// GetTotalInboundGFTExceptionPackets gets the value of TotalInboundGFTExceptionPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) GetPropertyTotalInboundGFTExceptionPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundGFTExceptionPackets")
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

// SetTotalInboundGFTExceptionUFOffloadBlockedPackets sets the value of TotalInboundGFTExceptionUFOffloadBlockedPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) SetPropertyTotalInboundGFTExceptionUFOffloadBlockedPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundGFTExceptionUFOffloadBlockedPackets", (value))
}

// GetTotalInboundGFTExceptionUFOffloadBlockedPackets gets the value of TotalInboundGFTExceptionUFOffloadBlockedPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) GetPropertyTotalInboundGFTExceptionUFOffloadBlockedPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundGFTExceptionUFOffloadBlockedPackets")
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

// SetTotalInboundGFTExceptionUFOffloadDeferredPackets sets the value of TotalInboundGFTExceptionUFOffloadDeferredPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) SetPropertyTotalInboundGFTExceptionUFOffloadDeferredPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundGFTExceptionUFOffloadDeferredPackets", (value))
}

// GetTotalInboundGFTExceptionUFOffloadDeferredPackets gets the value of TotalInboundGFTExceptionUFOffloadDeferredPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) GetPropertyTotalInboundGFTExceptionUFOffloadDeferredPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundGFTExceptionUFOffloadDeferredPackets")
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

// SetTotalInboundGFTExceptionUFOffloadedTCPPackets sets the value of TotalInboundGFTExceptionUFOffloadedTCPPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) SetPropertyTotalInboundGFTExceptionUFOffloadedTCPPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundGFTExceptionUFOffloadedTCPPackets", (value))
}

// GetTotalInboundGFTExceptionUFOffloadedTCPPackets gets the value of TotalInboundGFTExceptionUFOffloadedTCPPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) GetPropertyTotalInboundGFTExceptionUFOffloadedTCPPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundGFTExceptionUFOffloadedTCPPackets")
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

// SetTotalInboundGFTExceptionUFOffloadedUDPPackets sets the value of TotalInboundGFTExceptionUFOffloadedUDPPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) SetPropertyTotalInboundGFTExceptionUFOffloadedUDPPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundGFTExceptionUFOffloadedUDPPackets", (value))
}

// GetTotalInboundGFTExceptionUFOffloadedUDPPackets gets the value of TotalInboundGFTExceptionUFOffloadedUDPPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) GetPropertyTotalInboundGFTExceptionUFOffloadedUDPPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundGFTExceptionUFOffloadedUDPPackets")
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

// SetTotalInboundGFTExceptionUFOffloadFailedPackets sets the value of TotalInboundGFTExceptionUFOffloadFailedPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) SetPropertyTotalInboundGFTExceptionUFOffloadFailedPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundGFTExceptionUFOffloadFailedPackets", (value))
}

// GetTotalInboundGFTExceptionUFOffloadFailedPackets gets the value of TotalInboundGFTExceptionUFOffloadFailedPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) GetPropertyTotalInboundGFTExceptionUFOffloadFailedPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundGFTExceptionUFOffloadFailedPackets")
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

// SetTotalInboundGFTExceptionUFOffloadPendingPackets sets the value of TotalInboundGFTExceptionUFOffloadPendingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) SetPropertyTotalInboundGFTExceptionUFOffloadPendingPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundGFTExceptionUFOffloadPendingPackets", (value))
}

// GetTotalInboundGFTExceptionUFOffloadPendingPackets gets the value of TotalInboundGFTExceptionUFOffloadPendingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) GetPropertyTotalInboundGFTExceptionUFOffloadPendingPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundGFTExceptionUFOffloadPendingPackets")
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

// SetTotalInboundGFTExceptionUFPackets sets the value of TotalInboundGFTExceptionUFPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) SetPropertyTotalInboundGFTExceptionUFPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundGFTExceptionUFPackets", (value))
}

// GetTotalInboundGFTExceptionUFPackets gets the value of TotalInboundGFTExceptionUFPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) GetPropertyTotalInboundGFTExceptionUFPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundGFTExceptionUFPackets")
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

// SetTotalInboundGFTExceptionUFRetryAwaitingPackets sets the value of TotalInboundGFTExceptionUFRetryAwaitingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) SetPropertyTotalInboundGFTExceptionUFRetryAwaitingPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundGFTExceptionUFRetryAwaitingPackets", (value))
}

// GetTotalInboundGFTExceptionUFRetryAwaitingPackets gets the value of TotalInboundGFTExceptionUFRetryAwaitingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) GetPropertyTotalInboundGFTExceptionUFRetryAwaitingPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundGFTExceptionUFRetryAwaitingPackets")
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

// SetTotalInboundGFTPackets sets the value of TotalInboundGFTPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) SetPropertyTotalInboundGFTPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundGFTPackets", (value))
}

// GetTotalInboundGFTPackets gets the value of TotalInboundGFTPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) GetPropertyTotalInboundGFTPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundGFTPackets")
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

// SetTotalInboundHairpinnedPackets sets the value of TotalInboundHairpinnedPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) SetPropertyTotalInboundHairpinnedPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundHairpinnedPackets", (value))
}

// GetTotalInboundHairpinnedPackets gets the value of TotalInboundHairpinnedPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) GetPropertyTotalInboundHairpinnedPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundHairpinnedPackets")
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

// SetTotalInboundInterceptedPackets sets the value of TotalInboundInterceptedPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) SetPropertyTotalInboundInterceptedPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundInterceptedPackets", (value))
}

// GetTotalInboundInterceptedPackets gets the value of TotalInboundInterceptedPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) GetPropertyTotalInboundInterceptedPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundInterceptedPackets")
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

// SetTotalInboundMissedInterceptedPackets sets the value of TotalInboundMissedInterceptedPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) SetPropertyTotalInboundMissedInterceptedPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundMissedInterceptedPackets", (value))
}

// GetTotalInboundMissedInterceptedPackets gets the value of TotalInboundMissedInterceptedPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) GetPropertyTotalInboundMissedInterceptedPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundMissedInterceptedPackets")
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

// SetTotalInboundNonIPPackets sets the value of TotalInboundNonIPPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) SetPropertyTotalInboundNonIPPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundNonIPPackets", (value))
}

// GetTotalInboundNonIPPackets gets the value of TotalInboundNonIPPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) GetPropertyTotalInboundNonIPPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundNonIPPackets")
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

// SetTotalInboundPackets sets the value of TotalInboundPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) SetPropertyTotalInboundPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundPackets", (value))
}

// GetTotalInboundPackets gets the value of TotalInboundPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) GetPropertyTotalInboundPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundPackets")
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

// SetTotalInboundPendingPackets sets the value of TotalInboundPendingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) SetPropertyTotalInboundPendingPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundPendingPackets", (value))
}

// GetTotalInboundPendingPackets gets the value of TotalInboundPendingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) GetPropertyTotalInboundPendingPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundPendingPackets")
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

// SetTotalInboundTCPSYNACKPackets sets the value of TotalInboundTCPSYNACKPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) SetPropertyTotalInboundTCPSYNACKPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundTCPSYNACKPackets", (value))
}

// GetTotalInboundTCPSYNACKPackets gets the value of TotalInboundTCPSYNACKPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) GetPropertyTotalInboundTCPSYNACKPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundTCPSYNACKPackets")
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

// SetTotalInboundTCPSYNPackets sets the value of TotalInboundTCPSYNPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) SetPropertyTotalInboundTCPSYNPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundTCPSYNPackets", (value))
}

// GetTotalInboundTCPSYNPackets gets the value of TotalInboundTCPSYNPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) GetPropertyTotalInboundTCPSYNPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundTCPSYNPackets")
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

// SetTotalInboundThrottledPackets sets the value of TotalInboundThrottledPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) SetPropertyTotalInboundThrottledPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundThrottledPackets", (value))
}

// GetTotalInboundThrottledPackets gets the value of TotalInboundThrottledPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) GetPropertyTotalInboundThrottledPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundThrottledPackets")
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

// SetTotalInboundUnicastForwardedGFTExceptionPackets sets the value of TotalInboundUnicastForwardedGFTExceptionPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) SetPropertyTotalInboundUnicastForwardedGFTExceptionPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundUnicastForwardedGFTExceptionPackets", (value))
}

// GetTotalInboundUnicastForwardedGFTExceptionPackets gets the value of TotalInboundUnicastForwardedGFTExceptionPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundNetworkTraffic) GetPropertyTotalInboundUnicastForwardedGFTExceptionPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundUnicastForwardedGFTExceptionPackets")
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
