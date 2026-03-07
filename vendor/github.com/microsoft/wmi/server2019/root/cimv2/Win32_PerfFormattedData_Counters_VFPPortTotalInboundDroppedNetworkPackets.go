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

// Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets struct
type Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets struct {
	*Win32_PerfFormattedData

	//
	TotalInboundDroppedACLPackets uint64

	//
	TotalInboundDroppedARPFilterPackets uint64

	//
	TotalInboundDroppedARPGuardPackets uint64

	//
	TotalInboundDroppedARPLimiterPackets uint64

	//
	TotalInboundDroppedBlockedPackets uint64

	//
	TotalInboundDroppedBroadcastPackets uint64

	//
	TotalInboundDroppedDHCPGuardPackets uint64

	//
	TotalInboundDroppedDHCPLimiterPackets uint64

	//
	TotalInboundDroppedForwardingPackets uint64

	//
	TotalInboundDroppedGFTCopyPackets uint64

	//
	TotalInboundDroppedGFTExceptionPackets uint64

	//
	TotalInboundDroppedInvalidPackets uint64

	//
	TotalInboundDroppedInvalidRuleMatchPackets uint64

	//
	TotalInboundDroppedIPV4SpoofingPackets uint64

	//
	TotalInboundDroppedIPV6SpoofingPackets uint64

	//
	TotalInboundDroppedMACSpoofingPackets uint64

	//
	TotalInboundDroppedMalformedPackets uint64

	//
	TotalInboundDroppedMonitoringPingPackets uint64

	//
	TotalInboundDroppedNonIPPackets uint64

	//
	TotalInboundDroppedNoResourcePackets uint64

	//
	TotalInboundDroppedPackets uint64

	//
	TotalInboundDroppedPendingPackets uint64

	//
	TotalInboundDroppedSimulationPackets uint64
}

func NewWin32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPacketsEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPacketsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetTotalInboundDroppedACLPackets sets the value of TotalInboundDroppedACLPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) SetPropertyTotalInboundDroppedACLPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundDroppedACLPackets", (value))
}

// GetTotalInboundDroppedACLPackets gets the value of TotalInboundDroppedACLPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) GetPropertyTotalInboundDroppedACLPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundDroppedACLPackets")
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

// SetTotalInboundDroppedARPFilterPackets sets the value of TotalInboundDroppedARPFilterPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) SetPropertyTotalInboundDroppedARPFilterPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundDroppedARPFilterPackets", (value))
}

// GetTotalInboundDroppedARPFilterPackets gets the value of TotalInboundDroppedARPFilterPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) GetPropertyTotalInboundDroppedARPFilterPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundDroppedARPFilterPackets")
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

// SetTotalInboundDroppedARPGuardPackets sets the value of TotalInboundDroppedARPGuardPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) SetPropertyTotalInboundDroppedARPGuardPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundDroppedARPGuardPackets", (value))
}

// GetTotalInboundDroppedARPGuardPackets gets the value of TotalInboundDroppedARPGuardPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) GetPropertyTotalInboundDroppedARPGuardPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundDroppedARPGuardPackets")
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

// SetTotalInboundDroppedARPLimiterPackets sets the value of TotalInboundDroppedARPLimiterPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) SetPropertyTotalInboundDroppedARPLimiterPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundDroppedARPLimiterPackets", (value))
}

// GetTotalInboundDroppedARPLimiterPackets gets the value of TotalInboundDroppedARPLimiterPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) GetPropertyTotalInboundDroppedARPLimiterPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundDroppedARPLimiterPackets")
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

// SetTotalInboundDroppedBlockedPackets sets the value of TotalInboundDroppedBlockedPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) SetPropertyTotalInboundDroppedBlockedPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundDroppedBlockedPackets", (value))
}

// GetTotalInboundDroppedBlockedPackets gets the value of TotalInboundDroppedBlockedPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) GetPropertyTotalInboundDroppedBlockedPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundDroppedBlockedPackets")
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

// SetTotalInboundDroppedBroadcastPackets sets the value of TotalInboundDroppedBroadcastPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) SetPropertyTotalInboundDroppedBroadcastPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundDroppedBroadcastPackets", (value))
}

// GetTotalInboundDroppedBroadcastPackets gets the value of TotalInboundDroppedBroadcastPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) GetPropertyTotalInboundDroppedBroadcastPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundDroppedBroadcastPackets")
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

// SetTotalInboundDroppedDHCPGuardPackets sets the value of TotalInboundDroppedDHCPGuardPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) SetPropertyTotalInboundDroppedDHCPGuardPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundDroppedDHCPGuardPackets", (value))
}

// GetTotalInboundDroppedDHCPGuardPackets gets the value of TotalInboundDroppedDHCPGuardPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) GetPropertyTotalInboundDroppedDHCPGuardPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundDroppedDHCPGuardPackets")
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

// SetTotalInboundDroppedDHCPLimiterPackets sets the value of TotalInboundDroppedDHCPLimiterPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) SetPropertyTotalInboundDroppedDHCPLimiterPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundDroppedDHCPLimiterPackets", (value))
}

// GetTotalInboundDroppedDHCPLimiterPackets gets the value of TotalInboundDroppedDHCPLimiterPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) GetPropertyTotalInboundDroppedDHCPLimiterPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundDroppedDHCPLimiterPackets")
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

// SetTotalInboundDroppedForwardingPackets sets the value of TotalInboundDroppedForwardingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) SetPropertyTotalInboundDroppedForwardingPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundDroppedForwardingPackets", (value))
}

// GetTotalInboundDroppedForwardingPackets gets the value of TotalInboundDroppedForwardingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) GetPropertyTotalInboundDroppedForwardingPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundDroppedForwardingPackets")
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

// SetTotalInboundDroppedGFTCopyPackets sets the value of TotalInboundDroppedGFTCopyPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) SetPropertyTotalInboundDroppedGFTCopyPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundDroppedGFTCopyPackets", (value))
}

// GetTotalInboundDroppedGFTCopyPackets gets the value of TotalInboundDroppedGFTCopyPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) GetPropertyTotalInboundDroppedGFTCopyPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundDroppedGFTCopyPackets")
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

// SetTotalInboundDroppedGFTExceptionPackets sets the value of TotalInboundDroppedGFTExceptionPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) SetPropertyTotalInboundDroppedGFTExceptionPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundDroppedGFTExceptionPackets", (value))
}

// GetTotalInboundDroppedGFTExceptionPackets gets the value of TotalInboundDroppedGFTExceptionPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) GetPropertyTotalInboundDroppedGFTExceptionPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundDroppedGFTExceptionPackets")
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

// SetTotalInboundDroppedInvalidPackets sets the value of TotalInboundDroppedInvalidPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) SetPropertyTotalInboundDroppedInvalidPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundDroppedInvalidPackets", (value))
}

// GetTotalInboundDroppedInvalidPackets gets the value of TotalInboundDroppedInvalidPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) GetPropertyTotalInboundDroppedInvalidPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundDroppedInvalidPackets")
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

// SetTotalInboundDroppedInvalidRuleMatchPackets sets the value of TotalInboundDroppedInvalidRuleMatchPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) SetPropertyTotalInboundDroppedInvalidRuleMatchPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundDroppedInvalidRuleMatchPackets", (value))
}

// GetTotalInboundDroppedInvalidRuleMatchPackets gets the value of TotalInboundDroppedInvalidRuleMatchPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) GetPropertyTotalInboundDroppedInvalidRuleMatchPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundDroppedInvalidRuleMatchPackets")
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

// SetTotalInboundDroppedIPV4SpoofingPackets sets the value of TotalInboundDroppedIPV4SpoofingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) SetPropertyTotalInboundDroppedIPV4SpoofingPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundDroppedIPV4SpoofingPackets", (value))
}

// GetTotalInboundDroppedIPV4SpoofingPackets gets the value of TotalInboundDroppedIPV4SpoofingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) GetPropertyTotalInboundDroppedIPV4SpoofingPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundDroppedIPV4SpoofingPackets")
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

// SetTotalInboundDroppedIPV6SpoofingPackets sets the value of TotalInboundDroppedIPV6SpoofingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) SetPropertyTotalInboundDroppedIPV6SpoofingPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundDroppedIPV6SpoofingPackets", (value))
}

// GetTotalInboundDroppedIPV6SpoofingPackets gets the value of TotalInboundDroppedIPV6SpoofingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) GetPropertyTotalInboundDroppedIPV6SpoofingPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundDroppedIPV6SpoofingPackets")
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

// SetTotalInboundDroppedMACSpoofingPackets sets the value of TotalInboundDroppedMACSpoofingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) SetPropertyTotalInboundDroppedMACSpoofingPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundDroppedMACSpoofingPackets", (value))
}

// GetTotalInboundDroppedMACSpoofingPackets gets the value of TotalInboundDroppedMACSpoofingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) GetPropertyTotalInboundDroppedMACSpoofingPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundDroppedMACSpoofingPackets")
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

// SetTotalInboundDroppedMalformedPackets sets the value of TotalInboundDroppedMalformedPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) SetPropertyTotalInboundDroppedMalformedPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundDroppedMalformedPackets", (value))
}

// GetTotalInboundDroppedMalformedPackets gets the value of TotalInboundDroppedMalformedPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) GetPropertyTotalInboundDroppedMalformedPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundDroppedMalformedPackets")
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

// SetTotalInboundDroppedMonitoringPingPackets sets the value of TotalInboundDroppedMonitoringPingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) SetPropertyTotalInboundDroppedMonitoringPingPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundDroppedMonitoringPingPackets", (value))
}

// GetTotalInboundDroppedMonitoringPingPackets gets the value of TotalInboundDroppedMonitoringPingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) GetPropertyTotalInboundDroppedMonitoringPingPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundDroppedMonitoringPingPackets")
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

// SetTotalInboundDroppedNonIPPackets sets the value of TotalInboundDroppedNonIPPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) SetPropertyTotalInboundDroppedNonIPPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundDroppedNonIPPackets", (value))
}

// GetTotalInboundDroppedNonIPPackets gets the value of TotalInboundDroppedNonIPPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) GetPropertyTotalInboundDroppedNonIPPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundDroppedNonIPPackets")
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

// SetTotalInboundDroppedNoResourcePackets sets the value of TotalInboundDroppedNoResourcePackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) SetPropertyTotalInboundDroppedNoResourcePackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundDroppedNoResourcePackets", (value))
}

// GetTotalInboundDroppedNoResourcePackets gets the value of TotalInboundDroppedNoResourcePackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) GetPropertyTotalInboundDroppedNoResourcePackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundDroppedNoResourcePackets")
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

// SetTotalInboundDroppedPackets sets the value of TotalInboundDroppedPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) SetPropertyTotalInboundDroppedPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundDroppedPackets", (value))
}

// GetTotalInboundDroppedPackets gets the value of TotalInboundDroppedPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) GetPropertyTotalInboundDroppedPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundDroppedPackets")
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

// SetTotalInboundDroppedPendingPackets sets the value of TotalInboundDroppedPendingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) SetPropertyTotalInboundDroppedPendingPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundDroppedPendingPackets", (value))
}

// GetTotalInboundDroppedPendingPackets gets the value of TotalInboundDroppedPendingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) GetPropertyTotalInboundDroppedPendingPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundDroppedPendingPackets")
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

// SetTotalInboundDroppedSimulationPackets sets the value of TotalInboundDroppedSimulationPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) SetPropertyTotalInboundDroppedSimulationPackets(value uint64) (err error) {
	return instance.SetProperty("TotalInboundDroppedSimulationPackets", (value))
}

// GetTotalInboundDroppedSimulationPackets gets the value of TotalInboundDroppedSimulationPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalInboundDroppedNetworkPackets) GetPropertyTotalInboundDroppedSimulationPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalInboundDroppedSimulationPackets")
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
