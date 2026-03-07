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

// Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets struct
type Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets struct {
	*Win32_PerfFormattedData

	//
	TotalOutboundDroppedACLPackets uint64

	//
	TotalOutboundDroppedARPFilterPackets uint64

	//
	TotalOutboundDroppedARPGuardPackets uint64

	//
	TotalOutboundDroppedARPLimiterPackets uint64

	//
	TotalOutboundDroppedBlockedPackets uint64

	//
	TotalOutboundDroppedBroadcastPackets uint64

	//
	TotalOutboundDroppedDHCPGuardPackets uint64

	//
	TotalOutboundDroppedDHCPLimiterPackets uint64

	//
	TotalOutboundDroppedForwardingPackets uint64

	//
	TotalOutboundDroppedGFTCopyPackets uint64

	//
	TotalOutboundDroppedGFTExceptionPackets uint64

	//
	TotalOutboundDroppedInvalidPackets uint64

	//
	TotalOutboundDroppedInvalidRuleMatchPackets uint64

	//
	TotalOutboundDroppedIPV4SpoofingPackets uint64

	//
	TotalOutboundDroppedIPV6SpoofingPackets uint64

	//
	TotalOutboundDroppedMACSpoofingPackets uint64

	//
	TotalOutboundDroppedMalformedPackets uint64

	//
	TotalOutboundDroppedMonitoringPingPackets uint64

	//
	TotalOutboundDroppedNonIPPackets uint64

	//
	TotalOutboundDroppedNoResourcePackets uint64

	//
	TotalOutboundDroppedPackets uint64

	//
	TotalOutboundDroppedPendingPackets uint64

	//
	TotalOutboundDroppedSimulationPackets uint64
}

func NewWin32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPacketsEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPacketsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetTotalOutboundDroppedACLPackets sets the value of TotalOutboundDroppedACLPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) SetPropertyTotalOutboundDroppedACLPackets(value uint64) (err error) {
	return instance.SetProperty("TotalOutboundDroppedACLPackets", (value))
}

// GetTotalOutboundDroppedACLPackets gets the value of TotalOutboundDroppedACLPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) GetPropertyTotalOutboundDroppedACLPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalOutboundDroppedACLPackets")
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

// SetTotalOutboundDroppedARPFilterPackets sets the value of TotalOutboundDroppedARPFilterPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) SetPropertyTotalOutboundDroppedARPFilterPackets(value uint64) (err error) {
	return instance.SetProperty("TotalOutboundDroppedARPFilterPackets", (value))
}

// GetTotalOutboundDroppedARPFilterPackets gets the value of TotalOutboundDroppedARPFilterPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) GetPropertyTotalOutboundDroppedARPFilterPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalOutboundDroppedARPFilterPackets")
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

// SetTotalOutboundDroppedARPGuardPackets sets the value of TotalOutboundDroppedARPGuardPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) SetPropertyTotalOutboundDroppedARPGuardPackets(value uint64) (err error) {
	return instance.SetProperty("TotalOutboundDroppedARPGuardPackets", (value))
}

// GetTotalOutboundDroppedARPGuardPackets gets the value of TotalOutboundDroppedARPGuardPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) GetPropertyTotalOutboundDroppedARPGuardPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalOutboundDroppedARPGuardPackets")
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

// SetTotalOutboundDroppedARPLimiterPackets sets the value of TotalOutboundDroppedARPLimiterPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) SetPropertyTotalOutboundDroppedARPLimiterPackets(value uint64) (err error) {
	return instance.SetProperty("TotalOutboundDroppedARPLimiterPackets", (value))
}

// GetTotalOutboundDroppedARPLimiterPackets gets the value of TotalOutboundDroppedARPLimiterPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) GetPropertyTotalOutboundDroppedARPLimiterPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalOutboundDroppedARPLimiterPackets")
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

// SetTotalOutboundDroppedBlockedPackets sets the value of TotalOutboundDroppedBlockedPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) SetPropertyTotalOutboundDroppedBlockedPackets(value uint64) (err error) {
	return instance.SetProperty("TotalOutboundDroppedBlockedPackets", (value))
}

// GetTotalOutboundDroppedBlockedPackets gets the value of TotalOutboundDroppedBlockedPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) GetPropertyTotalOutboundDroppedBlockedPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalOutboundDroppedBlockedPackets")
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

// SetTotalOutboundDroppedBroadcastPackets sets the value of TotalOutboundDroppedBroadcastPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) SetPropertyTotalOutboundDroppedBroadcastPackets(value uint64) (err error) {
	return instance.SetProperty("TotalOutboundDroppedBroadcastPackets", (value))
}

// GetTotalOutboundDroppedBroadcastPackets gets the value of TotalOutboundDroppedBroadcastPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) GetPropertyTotalOutboundDroppedBroadcastPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalOutboundDroppedBroadcastPackets")
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

// SetTotalOutboundDroppedDHCPGuardPackets sets the value of TotalOutboundDroppedDHCPGuardPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) SetPropertyTotalOutboundDroppedDHCPGuardPackets(value uint64) (err error) {
	return instance.SetProperty("TotalOutboundDroppedDHCPGuardPackets", (value))
}

// GetTotalOutboundDroppedDHCPGuardPackets gets the value of TotalOutboundDroppedDHCPGuardPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) GetPropertyTotalOutboundDroppedDHCPGuardPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalOutboundDroppedDHCPGuardPackets")
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

// SetTotalOutboundDroppedDHCPLimiterPackets sets the value of TotalOutboundDroppedDHCPLimiterPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) SetPropertyTotalOutboundDroppedDHCPLimiterPackets(value uint64) (err error) {
	return instance.SetProperty("TotalOutboundDroppedDHCPLimiterPackets", (value))
}

// GetTotalOutboundDroppedDHCPLimiterPackets gets the value of TotalOutboundDroppedDHCPLimiterPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) GetPropertyTotalOutboundDroppedDHCPLimiterPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalOutboundDroppedDHCPLimiterPackets")
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

// SetTotalOutboundDroppedForwardingPackets sets the value of TotalOutboundDroppedForwardingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) SetPropertyTotalOutboundDroppedForwardingPackets(value uint64) (err error) {
	return instance.SetProperty("TotalOutboundDroppedForwardingPackets", (value))
}

// GetTotalOutboundDroppedForwardingPackets gets the value of TotalOutboundDroppedForwardingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) GetPropertyTotalOutboundDroppedForwardingPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalOutboundDroppedForwardingPackets")
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

// SetTotalOutboundDroppedGFTCopyPackets sets the value of TotalOutboundDroppedGFTCopyPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) SetPropertyTotalOutboundDroppedGFTCopyPackets(value uint64) (err error) {
	return instance.SetProperty("TotalOutboundDroppedGFTCopyPackets", (value))
}

// GetTotalOutboundDroppedGFTCopyPackets gets the value of TotalOutboundDroppedGFTCopyPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) GetPropertyTotalOutboundDroppedGFTCopyPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalOutboundDroppedGFTCopyPackets")
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

// SetTotalOutboundDroppedGFTExceptionPackets sets the value of TotalOutboundDroppedGFTExceptionPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) SetPropertyTotalOutboundDroppedGFTExceptionPackets(value uint64) (err error) {
	return instance.SetProperty("TotalOutboundDroppedGFTExceptionPackets", (value))
}

// GetTotalOutboundDroppedGFTExceptionPackets gets the value of TotalOutboundDroppedGFTExceptionPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) GetPropertyTotalOutboundDroppedGFTExceptionPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalOutboundDroppedGFTExceptionPackets")
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

// SetTotalOutboundDroppedInvalidPackets sets the value of TotalOutboundDroppedInvalidPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) SetPropertyTotalOutboundDroppedInvalidPackets(value uint64) (err error) {
	return instance.SetProperty("TotalOutboundDroppedInvalidPackets", (value))
}

// GetTotalOutboundDroppedInvalidPackets gets the value of TotalOutboundDroppedInvalidPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) GetPropertyTotalOutboundDroppedInvalidPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalOutboundDroppedInvalidPackets")
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

// SetTotalOutboundDroppedInvalidRuleMatchPackets sets the value of TotalOutboundDroppedInvalidRuleMatchPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) SetPropertyTotalOutboundDroppedInvalidRuleMatchPackets(value uint64) (err error) {
	return instance.SetProperty("TotalOutboundDroppedInvalidRuleMatchPackets", (value))
}

// GetTotalOutboundDroppedInvalidRuleMatchPackets gets the value of TotalOutboundDroppedInvalidRuleMatchPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) GetPropertyTotalOutboundDroppedInvalidRuleMatchPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalOutboundDroppedInvalidRuleMatchPackets")
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

// SetTotalOutboundDroppedIPV4SpoofingPackets sets the value of TotalOutboundDroppedIPV4SpoofingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) SetPropertyTotalOutboundDroppedIPV4SpoofingPackets(value uint64) (err error) {
	return instance.SetProperty("TotalOutboundDroppedIPV4SpoofingPackets", (value))
}

// GetTotalOutboundDroppedIPV4SpoofingPackets gets the value of TotalOutboundDroppedIPV4SpoofingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) GetPropertyTotalOutboundDroppedIPV4SpoofingPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalOutboundDroppedIPV4SpoofingPackets")
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

// SetTotalOutboundDroppedIPV6SpoofingPackets sets the value of TotalOutboundDroppedIPV6SpoofingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) SetPropertyTotalOutboundDroppedIPV6SpoofingPackets(value uint64) (err error) {
	return instance.SetProperty("TotalOutboundDroppedIPV6SpoofingPackets", (value))
}

// GetTotalOutboundDroppedIPV6SpoofingPackets gets the value of TotalOutboundDroppedIPV6SpoofingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) GetPropertyTotalOutboundDroppedIPV6SpoofingPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalOutboundDroppedIPV6SpoofingPackets")
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

// SetTotalOutboundDroppedMACSpoofingPackets sets the value of TotalOutboundDroppedMACSpoofingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) SetPropertyTotalOutboundDroppedMACSpoofingPackets(value uint64) (err error) {
	return instance.SetProperty("TotalOutboundDroppedMACSpoofingPackets", (value))
}

// GetTotalOutboundDroppedMACSpoofingPackets gets the value of TotalOutboundDroppedMACSpoofingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) GetPropertyTotalOutboundDroppedMACSpoofingPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalOutboundDroppedMACSpoofingPackets")
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

// SetTotalOutboundDroppedMalformedPackets sets the value of TotalOutboundDroppedMalformedPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) SetPropertyTotalOutboundDroppedMalformedPackets(value uint64) (err error) {
	return instance.SetProperty("TotalOutboundDroppedMalformedPackets", (value))
}

// GetTotalOutboundDroppedMalformedPackets gets the value of TotalOutboundDroppedMalformedPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) GetPropertyTotalOutboundDroppedMalformedPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalOutboundDroppedMalformedPackets")
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

// SetTotalOutboundDroppedMonitoringPingPackets sets the value of TotalOutboundDroppedMonitoringPingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) SetPropertyTotalOutboundDroppedMonitoringPingPackets(value uint64) (err error) {
	return instance.SetProperty("TotalOutboundDroppedMonitoringPingPackets", (value))
}

// GetTotalOutboundDroppedMonitoringPingPackets gets the value of TotalOutboundDroppedMonitoringPingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) GetPropertyTotalOutboundDroppedMonitoringPingPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalOutboundDroppedMonitoringPingPackets")
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

// SetTotalOutboundDroppedNonIPPackets sets the value of TotalOutboundDroppedNonIPPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) SetPropertyTotalOutboundDroppedNonIPPackets(value uint64) (err error) {
	return instance.SetProperty("TotalOutboundDroppedNonIPPackets", (value))
}

// GetTotalOutboundDroppedNonIPPackets gets the value of TotalOutboundDroppedNonIPPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) GetPropertyTotalOutboundDroppedNonIPPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalOutboundDroppedNonIPPackets")
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

// SetTotalOutboundDroppedNoResourcePackets sets the value of TotalOutboundDroppedNoResourcePackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) SetPropertyTotalOutboundDroppedNoResourcePackets(value uint64) (err error) {
	return instance.SetProperty("TotalOutboundDroppedNoResourcePackets", (value))
}

// GetTotalOutboundDroppedNoResourcePackets gets the value of TotalOutboundDroppedNoResourcePackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) GetPropertyTotalOutboundDroppedNoResourcePackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalOutboundDroppedNoResourcePackets")
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

// SetTotalOutboundDroppedPackets sets the value of TotalOutboundDroppedPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) SetPropertyTotalOutboundDroppedPackets(value uint64) (err error) {
	return instance.SetProperty("TotalOutboundDroppedPackets", (value))
}

// GetTotalOutboundDroppedPackets gets the value of TotalOutboundDroppedPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) GetPropertyTotalOutboundDroppedPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalOutboundDroppedPackets")
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

// SetTotalOutboundDroppedPendingPackets sets the value of TotalOutboundDroppedPendingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) SetPropertyTotalOutboundDroppedPendingPackets(value uint64) (err error) {
	return instance.SetProperty("TotalOutboundDroppedPendingPackets", (value))
}

// GetTotalOutboundDroppedPendingPackets gets the value of TotalOutboundDroppedPendingPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) GetPropertyTotalOutboundDroppedPendingPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalOutboundDroppedPendingPackets")
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

// SetTotalOutboundDroppedSimulationPackets sets the value of TotalOutboundDroppedSimulationPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) SetPropertyTotalOutboundDroppedSimulationPackets(value uint64) (err error) {
	return instance.SetProperty("TotalOutboundDroppedSimulationPackets", (value))
}

// GetTotalOutboundDroppedSimulationPackets gets the value of TotalOutboundDroppedSimulationPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_VFPPortTotalOutboundDroppedNetworkPackets) GetPropertyTotalOutboundDroppedSimulationPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalOutboundDroppedSimulationPackets")
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
