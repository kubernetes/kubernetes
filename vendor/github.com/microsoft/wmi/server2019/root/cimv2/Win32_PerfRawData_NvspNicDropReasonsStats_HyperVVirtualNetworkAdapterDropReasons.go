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

// Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons struct
type Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons struct {
	*Win32_PerfRawData

	//
	IncomingBridgeReserved uint64

	//
	IncomingBusy uint64

	//
	IncomingDhcpGuard uint64

	//
	IncomingDisconnected uint64

	//
	IncomingFailedDestinationListUpdate uint64

	//
	IncomingFailedPacketFilter uint64

	//
	IncomingFailedPvlanSetting uint64

	//
	IncomingFailedSecurityPolicy uint64

	//
	IncomingFiltered uint64

	//
	IncomingFilteredIsolationUntagged uint64

	//
	IncomingFilteredVLAN uint64

	//
	IncomingInjectedIcmp uint64

	//
	IncomingInvalidConfig uint64

	//
	IncomingInvalidData uint64

	//
	IncomingInvalidDestMac uint64

	//
	IncomingInvalidFirstNBTooSmall uint64

	//
	IncomingInvalidPacket uint64

	//
	IncomingInvalidPDQueue uint64

	//
	IncomingInvalidSourceMac uint64

	//
	IncomingInvalidVlanFormat uint64

	//
	IncomingIpsec uint64

	//
	IncomingLowPowerPacketFilter uint64

	//
	IncomingMacSpoofing uint64

	//
	IncomingMTUMismatch uint64

	//
	IncomingNativeFwdingReq uint64

	//
	IncomingNicDisabled uint64

	//
	IncomingNotAccepted uint64

	//
	IncomingNotReady uint64

	//
	IncomingQos uint64

	//
	IncomingRequiredExtensionMissing uint64

	//
	IncomingResources uint64

	//
	IncomingRouterGuard uint64

	//
	IncomingStormLimit uint64

	//
	IncomingSwitchDataFlowDisabled uint64

	//
	IncomingUnauthorizedMAC uint64

	//
	IncomingUnauthorizedVLAN uint64

	//
	IncomingUnknown uint64

	//
	IncomingVirtualSubnetId uint64

	//
	IncomingWnv uint64

	//
	OutgoingBridgeReserved uint64

	//
	OutgoingBusy uint64

	//
	OutgoingDhcpGuard uint64

	//
	OutgoingDisconnected uint64

	//
	OutgoingFailedDestinationListUpdate uint64

	//
	OutgoingFailedPacketFilter uint64

	//
	OutgoingFailedPvlanSetting uint64

	//
	OutgoingFailedSecurityPolicy uint64

	//
	OutgoingFiltered uint64

	//
	OutgoingFilteredIsolationUntagged uint64

	//
	OutgoingFilteredVLAN uint64

	//
	OutgoingInjectedIcmp uint64

	//
	OutgoingInvalidConfig uint64

	//
	OutgoingInvalidData uint64

	//
	OutgoingInvalidDestMac uint64

	//
	OutgoingInvalidFirstNBTooSmall uint64

	//
	OutgoingInvalidPacket uint64

	//
	OutgoingInvalidPDQueue uint64

	//
	OutgoingInvalidSourceMac uint64

	//
	OutgoingInvalidVlanFormat uint64

	//
	OutgoingIpsec uint64

	//
	OutgoingLowPowerPacketFilter uint64

	//
	OutgoingMacSpoofing uint64

	//
	OutgoingMTUMismatch uint64

	//
	OutgoingNativeFwdingReq uint64

	//
	OutgoingNicDisabled uint64

	//
	OutgoingNotAccepted uint64

	//
	OutgoingNotReady uint64

	//
	OutgoingQos uint64

	//
	OutgoingRequiredExtensionMissing uint64

	//
	OutgoingResources uint64

	//
	OutgoingRouterGuard uint64

	//
	OutgoingStormLimit uint64

	//
	OutgoingSwitchDataFlowDisabled uint64

	//
	OutgoingUnauthorizedMAC uint64

	//
	OutgoingUnauthorizedVLAN uint64

	//
	OutgoingUnknown uint64

	//
	OutgoingVirtualSubnetId uint64

	//
	OutgoingWnv uint64
}

func NewWin32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasonsEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasonsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetIncomingBridgeReserved sets the value of IncomingBridgeReserved for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingBridgeReserved(value uint64) (err error) {
	return instance.SetProperty("IncomingBridgeReserved", (value))
}

// GetIncomingBridgeReserved gets the value of IncomingBridgeReserved for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingBridgeReserved() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingBridgeReserved")
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

// SetIncomingBusy sets the value of IncomingBusy for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingBusy(value uint64) (err error) {
	return instance.SetProperty("IncomingBusy", (value))
}

// GetIncomingBusy gets the value of IncomingBusy for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingBusy() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingBusy")
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

// SetIncomingDhcpGuard sets the value of IncomingDhcpGuard for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingDhcpGuard(value uint64) (err error) {
	return instance.SetProperty("IncomingDhcpGuard", (value))
}

// GetIncomingDhcpGuard gets the value of IncomingDhcpGuard for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingDhcpGuard() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingDhcpGuard")
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

// SetIncomingDisconnected sets the value of IncomingDisconnected for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingDisconnected(value uint64) (err error) {
	return instance.SetProperty("IncomingDisconnected", (value))
}

// GetIncomingDisconnected gets the value of IncomingDisconnected for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingDisconnected() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingDisconnected")
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

// SetIncomingFailedDestinationListUpdate sets the value of IncomingFailedDestinationListUpdate for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingFailedDestinationListUpdate(value uint64) (err error) {
	return instance.SetProperty("IncomingFailedDestinationListUpdate", (value))
}

// GetIncomingFailedDestinationListUpdate gets the value of IncomingFailedDestinationListUpdate for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingFailedDestinationListUpdate() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingFailedDestinationListUpdate")
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

// SetIncomingFailedPacketFilter sets the value of IncomingFailedPacketFilter for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingFailedPacketFilter(value uint64) (err error) {
	return instance.SetProperty("IncomingFailedPacketFilter", (value))
}

// GetIncomingFailedPacketFilter gets the value of IncomingFailedPacketFilter for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingFailedPacketFilter() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingFailedPacketFilter")
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

// SetIncomingFailedPvlanSetting sets the value of IncomingFailedPvlanSetting for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingFailedPvlanSetting(value uint64) (err error) {
	return instance.SetProperty("IncomingFailedPvlanSetting", (value))
}

// GetIncomingFailedPvlanSetting gets the value of IncomingFailedPvlanSetting for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingFailedPvlanSetting() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingFailedPvlanSetting")
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

// SetIncomingFailedSecurityPolicy sets the value of IncomingFailedSecurityPolicy for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingFailedSecurityPolicy(value uint64) (err error) {
	return instance.SetProperty("IncomingFailedSecurityPolicy", (value))
}

// GetIncomingFailedSecurityPolicy gets the value of IncomingFailedSecurityPolicy for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingFailedSecurityPolicy() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingFailedSecurityPolicy")
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

// SetIncomingFiltered sets the value of IncomingFiltered for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingFiltered(value uint64) (err error) {
	return instance.SetProperty("IncomingFiltered", (value))
}

// GetIncomingFiltered gets the value of IncomingFiltered for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingFiltered() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingFiltered")
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

// SetIncomingFilteredIsolationUntagged sets the value of IncomingFilteredIsolationUntagged for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingFilteredIsolationUntagged(value uint64) (err error) {
	return instance.SetProperty("IncomingFilteredIsolationUntagged", (value))
}

// GetIncomingFilteredIsolationUntagged gets the value of IncomingFilteredIsolationUntagged for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingFilteredIsolationUntagged() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingFilteredIsolationUntagged")
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

// SetIncomingFilteredVLAN sets the value of IncomingFilteredVLAN for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingFilteredVLAN(value uint64) (err error) {
	return instance.SetProperty("IncomingFilteredVLAN", (value))
}

// GetIncomingFilteredVLAN gets the value of IncomingFilteredVLAN for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingFilteredVLAN() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingFilteredVLAN")
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

// SetIncomingInjectedIcmp sets the value of IncomingInjectedIcmp for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingInjectedIcmp(value uint64) (err error) {
	return instance.SetProperty("IncomingInjectedIcmp", (value))
}

// GetIncomingInjectedIcmp gets the value of IncomingInjectedIcmp for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingInjectedIcmp() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingInjectedIcmp")
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

// SetIncomingInvalidConfig sets the value of IncomingInvalidConfig for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingInvalidConfig(value uint64) (err error) {
	return instance.SetProperty("IncomingInvalidConfig", (value))
}

// GetIncomingInvalidConfig gets the value of IncomingInvalidConfig for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingInvalidConfig() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingInvalidConfig")
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

// SetIncomingInvalidData sets the value of IncomingInvalidData for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingInvalidData(value uint64) (err error) {
	return instance.SetProperty("IncomingInvalidData", (value))
}

// GetIncomingInvalidData gets the value of IncomingInvalidData for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingInvalidData() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingInvalidData")
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

// SetIncomingInvalidDestMac sets the value of IncomingInvalidDestMac for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingInvalidDestMac(value uint64) (err error) {
	return instance.SetProperty("IncomingInvalidDestMac", (value))
}

// GetIncomingInvalidDestMac gets the value of IncomingInvalidDestMac for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingInvalidDestMac() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingInvalidDestMac")
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

// SetIncomingInvalidFirstNBTooSmall sets the value of IncomingInvalidFirstNBTooSmall for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingInvalidFirstNBTooSmall(value uint64) (err error) {
	return instance.SetProperty("IncomingInvalidFirstNBTooSmall", (value))
}

// GetIncomingInvalidFirstNBTooSmall gets the value of IncomingInvalidFirstNBTooSmall for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingInvalidFirstNBTooSmall() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingInvalidFirstNBTooSmall")
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

// SetIncomingInvalidPacket sets the value of IncomingInvalidPacket for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingInvalidPacket(value uint64) (err error) {
	return instance.SetProperty("IncomingInvalidPacket", (value))
}

// GetIncomingInvalidPacket gets the value of IncomingInvalidPacket for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingInvalidPacket() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingInvalidPacket")
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

// SetIncomingInvalidPDQueue sets the value of IncomingInvalidPDQueue for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingInvalidPDQueue(value uint64) (err error) {
	return instance.SetProperty("IncomingInvalidPDQueue", (value))
}

// GetIncomingInvalidPDQueue gets the value of IncomingInvalidPDQueue for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingInvalidPDQueue() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingInvalidPDQueue")
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

// SetIncomingInvalidSourceMac sets the value of IncomingInvalidSourceMac for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingInvalidSourceMac(value uint64) (err error) {
	return instance.SetProperty("IncomingInvalidSourceMac", (value))
}

// GetIncomingInvalidSourceMac gets the value of IncomingInvalidSourceMac for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingInvalidSourceMac() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingInvalidSourceMac")
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

// SetIncomingInvalidVlanFormat sets the value of IncomingInvalidVlanFormat for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingInvalidVlanFormat(value uint64) (err error) {
	return instance.SetProperty("IncomingInvalidVlanFormat", (value))
}

// GetIncomingInvalidVlanFormat gets the value of IncomingInvalidVlanFormat for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingInvalidVlanFormat() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingInvalidVlanFormat")
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

// SetIncomingIpsec sets the value of IncomingIpsec for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingIpsec(value uint64) (err error) {
	return instance.SetProperty("IncomingIpsec", (value))
}

// GetIncomingIpsec gets the value of IncomingIpsec for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingIpsec() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingIpsec")
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

// SetIncomingLowPowerPacketFilter sets the value of IncomingLowPowerPacketFilter for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingLowPowerPacketFilter(value uint64) (err error) {
	return instance.SetProperty("IncomingLowPowerPacketFilter", (value))
}

// GetIncomingLowPowerPacketFilter gets the value of IncomingLowPowerPacketFilter for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingLowPowerPacketFilter() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingLowPowerPacketFilter")
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

// SetIncomingMacSpoofing sets the value of IncomingMacSpoofing for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingMacSpoofing(value uint64) (err error) {
	return instance.SetProperty("IncomingMacSpoofing", (value))
}

// GetIncomingMacSpoofing gets the value of IncomingMacSpoofing for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingMacSpoofing() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingMacSpoofing")
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

// SetIncomingMTUMismatch sets the value of IncomingMTUMismatch for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingMTUMismatch(value uint64) (err error) {
	return instance.SetProperty("IncomingMTUMismatch", (value))
}

// GetIncomingMTUMismatch gets the value of IncomingMTUMismatch for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingMTUMismatch() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingMTUMismatch")
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

// SetIncomingNativeFwdingReq sets the value of IncomingNativeFwdingReq for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingNativeFwdingReq(value uint64) (err error) {
	return instance.SetProperty("IncomingNativeFwdingReq", (value))
}

// GetIncomingNativeFwdingReq gets the value of IncomingNativeFwdingReq for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingNativeFwdingReq() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingNativeFwdingReq")
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

// SetIncomingNicDisabled sets the value of IncomingNicDisabled for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingNicDisabled(value uint64) (err error) {
	return instance.SetProperty("IncomingNicDisabled", (value))
}

// GetIncomingNicDisabled gets the value of IncomingNicDisabled for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingNicDisabled() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingNicDisabled")
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

// SetIncomingNotAccepted sets the value of IncomingNotAccepted for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingNotAccepted(value uint64) (err error) {
	return instance.SetProperty("IncomingNotAccepted", (value))
}

// GetIncomingNotAccepted gets the value of IncomingNotAccepted for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingNotAccepted() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingNotAccepted")
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

// SetIncomingNotReady sets the value of IncomingNotReady for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingNotReady(value uint64) (err error) {
	return instance.SetProperty("IncomingNotReady", (value))
}

// GetIncomingNotReady gets the value of IncomingNotReady for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingNotReady() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingNotReady")
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

// SetIncomingQos sets the value of IncomingQos for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingQos(value uint64) (err error) {
	return instance.SetProperty("IncomingQos", (value))
}

// GetIncomingQos gets the value of IncomingQos for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingQos() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingQos")
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

// SetIncomingRequiredExtensionMissing sets the value of IncomingRequiredExtensionMissing for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingRequiredExtensionMissing(value uint64) (err error) {
	return instance.SetProperty("IncomingRequiredExtensionMissing", (value))
}

// GetIncomingRequiredExtensionMissing gets the value of IncomingRequiredExtensionMissing for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingRequiredExtensionMissing() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingRequiredExtensionMissing")
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

// SetIncomingResources sets the value of IncomingResources for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingResources(value uint64) (err error) {
	return instance.SetProperty("IncomingResources", (value))
}

// GetIncomingResources gets the value of IncomingResources for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingResources() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingResources")
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

// SetIncomingRouterGuard sets the value of IncomingRouterGuard for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingRouterGuard(value uint64) (err error) {
	return instance.SetProperty("IncomingRouterGuard", (value))
}

// GetIncomingRouterGuard gets the value of IncomingRouterGuard for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingRouterGuard() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingRouterGuard")
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

// SetIncomingStormLimit sets the value of IncomingStormLimit for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingStormLimit(value uint64) (err error) {
	return instance.SetProperty("IncomingStormLimit", (value))
}

// GetIncomingStormLimit gets the value of IncomingStormLimit for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingStormLimit() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingStormLimit")
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

// SetIncomingSwitchDataFlowDisabled sets the value of IncomingSwitchDataFlowDisabled for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingSwitchDataFlowDisabled(value uint64) (err error) {
	return instance.SetProperty("IncomingSwitchDataFlowDisabled", (value))
}

// GetIncomingSwitchDataFlowDisabled gets the value of IncomingSwitchDataFlowDisabled for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingSwitchDataFlowDisabled() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingSwitchDataFlowDisabled")
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

// SetIncomingUnauthorizedMAC sets the value of IncomingUnauthorizedMAC for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingUnauthorizedMAC(value uint64) (err error) {
	return instance.SetProperty("IncomingUnauthorizedMAC", (value))
}

// GetIncomingUnauthorizedMAC gets the value of IncomingUnauthorizedMAC for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingUnauthorizedMAC() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingUnauthorizedMAC")
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

// SetIncomingUnauthorizedVLAN sets the value of IncomingUnauthorizedVLAN for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingUnauthorizedVLAN(value uint64) (err error) {
	return instance.SetProperty("IncomingUnauthorizedVLAN", (value))
}

// GetIncomingUnauthorizedVLAN gets the value of IncomingUnauthorizedVLAN for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingUnauthorizedVLAN() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingUnauthorizedVLAN")
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

// SetIncomingUnknown sets the value of IncomingUnknown for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingUnknown(value uint64) (err error) {
	return instance.SetProperty("IncomingUnknown", (value))
}

// GetIncomingUnknown gets the value of IncomingUnknown for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingUnknown() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingUnknown")
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

// SetIncomingVirtualSubnetId sets the value of IncomingVirtualSubnetId for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingVirtualSubnetId(value uint64) (err error) {
	return instance.SetProperty("IncomingVirtualSubnetId", (value))
}

// GetIncomingVirtualSubnetId gets the value of IncomingVirtualSubnetId for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingVirtualSubnetId() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingVirtualSubnetId")
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

// SetIncomingWnv sets the value of IncomingWnv for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyIncomingWnv(value uint64) (err error) {
	return instance.SetProperty("IncomingWnv", (value))
}

// GetIncomingWnv gets the value of IncomingWnv for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyIncomingWnv() (value uint64, err error) {
	retValue, err := instance.GetProperty("IncomingWnv")
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

// SetOutgoingBridgeReserved sets the value of OutgoingBridgeReserved for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingBridgeReserved(value uint64) (err error) {
	return instance.SetProperty("OutgoingBridgeReserved", (value))
}

// GetOutgoingBridgeReserved gets the value of OutgoingBridgeReserved for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingBridgeReserved() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingBridgeReserved")
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

// SetOutgoingBusy sets the value of OutgoingBusy for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingBusy(value uint64) (err error) {
	return instance.SetProperty("OutgoingBusy", (value))
}

// GetOutgoingBusy gets the value of OutgoingBusy for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingBusy() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingBusy")
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

// SetOutgoingDhcpGuard sets the value of OutgoingDhcpGuard for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingDhcpGuard(value uint64) (err error) {
	return instance.SetProperty("OutgoingDhcpGuard", (value))
}

// GetOutgoingDhcpGuard gets the value of OutgoingDhcpGuard for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingDhcpGuard() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingDhcpGuard")
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

// SetOutgoingDisconnected sets the value of OutgoingDisconnected for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingDisconnected(value uint64) (err error) {
	return instance.SetProperty("OutgoingDisconnected", (value))
}

// GetOutgoingDisconnected gets the value of OutgoingDisconnected for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingDisconnected() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingDisconnected")
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

// SetOutgoingFailedDestinationListUpdate sets the value of OutgoingFailedDestinationListUpdate for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingFailedDestinationListUpdate(value uint64) (err error) {
	return instance.SetProperty("OutgoingFailedDestinationListUpdate", (value))
}

// GetOutgoingFailedDestinationListUpdate gets the value of OutgoingFailedDestinationListUpdate for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingFailedDestinationListUpdate() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingFailedDestinationListUpdate")
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

// SetOutgoingFailedPacketFilter sets the value of OutgoingFailedPacketFilter for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingFailedPacketFilter(value uint64) (err error) {
	return instance.SetProperty("OutgoingFailedPacketFilter", (value))
}

// GetOutgoingFailedPacketFilter gets the value of OutgoingFailedPacketFilter for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingFailedPacketFilter() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingFailedPacketFilter")
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

// SetOutgoingFailedPvlanSetting sets the value of OutgoingFailedPvlanSetting for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingFailedPvlanSetting(value uint64) (err error) {
	return instance.SetProperty("OutgoingFailedPvlanSetting", (value))
}

// GetOutgoingFailedPvlanSetting gets the value of OutgoingFailedPvlanSetting for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingFailedPvlanSetting() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingFailedPvlanSetting")
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

// SetOutgoingFailedSecurityPolicy sets the value of OutgoingFailedSecurityPolicy for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingFailedSecurityPolicy(value uint64) (err error) {
	return instance.SetProperty("OutgoingFailedSecurityPolicy", (value))
}

// GetOutgoingFailedSecurityPolicy gets the value of OutgoingFailedSecurityPolicy for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingFailedSecurityPolicy() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingFailedSecurityPolicy")
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

// SetOutgoingFiltered sets the value of OutgoingFiltered for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingFiltered(value uint64) (err error) {
	return instance.SetProperty("OutgoingFiltered", (value))
}

// GetOutgoingFiltered gets the value of OutgoingFiltered for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingFiltered() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingFiltered")
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

// SetOutgoingFilteredIsolationUntagged sets the value of OutgoingFilteredIsolationUntagged for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingFilteredIsolationUntagged(value uint64) (err error) {
	return instance.SetProperty("OutgoingFilteredIsolationUntagged", (value))
}

// GetOutgoingFilteredIsolationUntagged gets the value of OutgoingFilteredIsolationUntagged for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingFilteredIsolationUntagged() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingFilteredIsolationUntagged")
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

// SetOutgoingFilteredVLAN sets the value of OutgoingFilteredVLAN for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingFilteredVLAN(value uint64) (err error) {
	return instance.SetProperty("OutgoingFilteredVLAN", (value))
}

// GetOutgoingFilteredVLAN gets the value of OutgoingFilteredVLAN for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingFilteredVLAN() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingFilteredVLAN")
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

// SetOutgoingInjectedIcmp sets the value of OutgoingInjectedIcmp for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingInjectedIcmp(value uint64) (err error) {
	return instance.SetProperty("OutgoingInjectedIcmp", (value))
}

// GetOutgoingInjectedIcmp gets the value of OutgoingInjectedIcmp for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingInjectedIcmp() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingInjectedIcmp")
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

// SetOutgoingInvalidConfig sets the value of OutgoingInvalidConfig for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingInvalidConfig(value uint64) (err error) {
	return instance.SetProperty("OutgoingInvalidConfig", (value))
}

// GetOutgoingInvalidConfig gets the value of OutgoingInvalidConfig for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingInvalidConfig() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingInvalidConfig")
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

// SetOutgoingInvalidData sets the value of OutgoingInvalidData for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingInvalidData(value uint64) (err error) {
	return instance.SetProperty("OutgoingInvalidData", (value))
}

// GetOutgoingInvalidData gets the value of OutgoingInvalidData for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingInvalidData() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingInvalidData")
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

// SetOutgoingInvalidDestMac sets the value of OutgoingInvalidDestMac for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingInvalidDestMac(value uint64) (err error) {
	return instance.SetProperty("OutgoingInvalidDestMac", (value))
}

// GetOutgoingInvalidDestMac gets the value of OutgoingInvalidDestMac for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingInvalidDestMac() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingInvalidDestMac")
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

// SetOutgoingInvalidFirstNBTooSmall sets the value of OutgoingInvalidFirstNBTooSmall for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingInvalidFirstNBTooSmall(value uint64) (err error) {
	return instance.SetProperty("OutgoingInvalidFirstNBTooSmall", (value))
}

// GetOutgoingInvalidFirstNBTooSmall gets the value of OutgoingInvalidFirstNBTooSmall for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingInvalidFirstNBTooSmall() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingInvalidFirstNBTooSmall")
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

// SetOutgoingInvalidPacket sets the value of OutgoingInvalidPacket for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingInvalidPacket(value uint64) (err error) {
	return instance.SetProperty("OutgoingInvalidPacket", (value))
}

// GetOutgoingInvalidPacket gets the value of OutgoingInvalidPacket for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingInvalidPacket() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingInvalidPacket")
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

// SetOutgoingInvalidPDQueue sets the value of OutgoingInvalidPDQueue for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingInvalidPDQueue(value uint64) (err error) {
	return instance.SetProperty("OutgoingInvalidPDQueue", (value))
}

// GetOutgoingInvalidPDQueue gets the value of OutgoingInvalidPDQueue for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingInvalidPDQueue() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingInvalidPDQueue")
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

// SetOutgoingInvalidSourceMac sets the value of OutgoingInvalidSourceMac for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingInvalidSourceMac(value uint64) (err error) {
	return instance.SetProperty("OutgoingInvalidSourceMac", (value))
}

// GetOutgoingInvalidSourceMac gets the value of OutgoingInvalidSourceMac for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingInvalidSourceMac() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingInvalidSourceMac")
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

// SetOutgoingInvalidVlanFormat sets the value of OutgoingInvalidVlanFormat for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingInvalidVlanFormat(value uint64) (err error) {
	return instance.SetProperty("OutgoingInvalidVlanFormat", (value))
}

// GetOutgoingInvalidVlanFormat gets the value of OutgoingInvalidVlanFormat for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingInvalidVlanFormat() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingInvalidVlanFormat")
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

// SetOutgoingIpsec sets the value of OutgoingIpsec for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingIpsec(value uint64) (err error) {
	return instance.SetProperty("OutgoingIpsec", (value))
}

// GetOutgoingIpsec gets the value of OutgoingIpsec for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingIpsec() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingIpsec")
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

// SetOutgoingLowPowerPacketFilter sets the value of OutgoingLowPowerPacketFilter for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingLowPowerPacketFilter(value uint64) (err error) {
	return instance.SetProperty("OutgoingLowPowerPacketFilter", (value))
}

// GetOutgoingLowPowerPacketFilter gets the value of OutgoingLowPowerPacketFilter for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingLowPowerPacketFilter() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingLowPowerPacketFilter")
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

// SetOutgoingMacSpoofing sets the value of OutgoingMacSpoofing for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingMacSpoofing(value uint64) (err error) {
	return instance.SetProperty("OutgoingMacSpoofing", (value))
}

// GetOutgoingMacSpoofing gets the value of OutgoingMacSpoofing for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingMacSpoofing() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingMacSpoofing")
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

// SetOutgoingMTUMismatch sets the value of OutgoingMTUMismatch for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingMTUMismatch(value uint64) (err error) {
	return instance.SetProperty("OutgoingMTUMismatch", (value))
}

// GetOutgoingMTUMismatch gets the value of OutgoingMTUMismatch for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingMTUMismatch() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingMTUMismatch")
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

// SetOutgoingNativeFwdingReq sets the value of OutgoingNativeFwdingReq for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingNativeFwdingReq(value uint64) (err error) {
	return instance.SetProperty("OutgoingNativeFwdingReq", (value))
}

// GetOutgoingNativeFwdingReq gets the value of OutgoingNativeFwdingReq for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingNativeFwdingReq() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingNativeFwdingReq")
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

// SetOutgoingNicDisabled sets the value of OutgoingNicDisabled for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingNicDisabled(value uint64) (err error) {
	return instance.SetProperty("OutgoingNicDisabled", (value))
}

// GetOutgoingNicDisabled gets the value of OutgoingNicDisabled for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingNicDisabled() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingNicDisabled")
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

// SetOutgoingNotAccepted sets the value of OutgoingNotAccepted for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingNotAccepted(value uint64) (err error) {
	return instance.SetProperty("OutgoingNotAccepted", (value))
}

// GetOutgoingNotAccepted gets the value of OutgoingNotAccepted for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingNotAccepted() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingNotAccepted")
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

// SetOutgoingNotReady sets the value of OutgoingNotReady for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingNotReady(value uint64) (err error) {
	return instance.SetProperty("OutgoingNotReady", (value))
}

// GetOutgoingNotReady gets the value of OutgoingNotReady for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingNotReady() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingNotReady")
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

// SetOutgoingQos sets the value of OutgoingQos for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingQos(value uint64) (err error) {
	return instance.SetProperty("OutgoingQos", (value))
}

// GetOutgoingQos gets the value of OutgoingQos for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingQos() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingQos")
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

// SetOutgoingRequiredExtensionMissing sets the value of OutgoingRequiredExtensionMissing for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingRequiredExtensionMissing(value uint64) (err error) {
	return instance.SetProperty("OutgoingRequiredExtensionMissing", (value))
}

// GetOutgoingRequiredExtensionMissing gets the value of OutgoingRequiredExtensionMissing for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingRequiredExtensionMissing() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingRequiredExtensionMissing")
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

// SetOutgoingResources sets the value of OutgoingResources for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingResources(value uint64) (err error) {
	return instance.SetProperty("OutgoingResources", (value))
}

// GetOutgoingResources gets the value of OutgoingResources for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingResources() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingResources")
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

// SetOutgoingRouterGuard sets the value of OutgoingRouterGuard for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingRouterGuard(value uint64) (err error) {
	return instance.SetProperty("OutgoingRouterGuard", (value))
}

// GetOutgoingRouterGuard gets the value of OutgoingRouterGuard for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingRouterGuard() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingRouterGuard")
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

// SetOutgoingStormLimit sets the value of OutgoingStormLimit for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingStormLimit(value uint64) (err error) {
	return instance.SetProperty("OutgoingStormLimit", (value))
}

// GetOutgoingStormLimit gets the value of OutgoingStormLimit for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingStormLimit() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingStormLimit")
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

// SetOutgoingSwitchDataFlowDisabled sets the value of OutgoingSwitchDataFlowDisabled for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingSwitchDataFlowDisabled(value uint64) (err error) {
	return instance.SetProperty("OutgoingSwitchDataFlowDisabled", (value))
}

// GetOutgoingSwitchDataFlowDisabled gets the value of OutgoingSwitchDataFlowDisabled for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingSwitchDataFlowDisabled() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingSwitchDataFlowDisabled")
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

// SetOutgoingUnauthorizedMAC sets the value of OutgoingUnauthorizedMAC for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingUnauthorizedMAC(value uint64) (err error) {
	return instance.SetProperty("OutgoingUnauthorizedMAC", (value))
}

// GetOutgoingUnauthorizedMAC gets the value of OutgoingUnauthorizedMAC for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingUnauthorizedMAC() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingUnauthorizedMAC")
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

// SetOutgoingUnauthorizedVLAN sets the value of OutgoingUnauthorizedVLAN for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingUnauthorizedVLAN(value uint64) (err error) {
	return instance.SetProperty("OutgoingUnauthorizedVLAN", (value))
}

// GetOutgoingUnauthorizedVLAN gets the value of OutgoingUnauthorizedVLAN for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingUnauthorizedVLAN() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingUnauthorizedVLAN")
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

// SetOutgoingUnknown sets the value of OutgoingUnknown for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingUnknown(value uint64) (err error) {
	return instance.SetProperty("OutgoingUnknown", (value))
}

// GetOutgoingUnknown gets the value of OutgoingUnknown for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingUnknown() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingUnknown")
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

// SetOutgoingVirtualSubnetId sets the value of OutgoingVirtualSubnetId for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingVirtualSubnetId(value uint64) (err error) {
	return instance.SetProperty("OutgoingVirtualSubnetId", (value))
}

// GetOutgoingVirtualSubnetId gets the value of OutgoingVirtualSubnetId for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingVirtualSubnetId() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingVirtualSubnetId")
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

// SetOutgoingWnv sets the value of OutgoingWnv for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) SetPropertyOutgoingWnv(value uint64) (err error) {
	return instance.SetProperty("OutgoingWnv", (value))
}

// GetOutgoingWnv gets the value of OutgoingWnv for the instance
func (instance *Win32_PerfRawData_NvspNicDropReasonsStats_HyperVVirtualNetworkAdapterDropReasons) GetPropertyOutgoingWnv() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutgoingWnv")
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
