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

// Win32_PerfRawData_Counters_IPsecDoSProtection struct
type Win32_PerfRawData_Counters_IPsecDoSProtection struct {
	*Win32_PerfRawData

	//
	CurrentStateEntries uint64

	//
	InboundAllowedDefaultBlockExemptPackets uint64

	//
	InboundAllowedDefaultBlockExemptPacketsPersec uint32

	//
	InboundAllowedFilterExemptIPv6Packets uint64

	//
	InboundAllowedFilterExemptIPv6PacketsPersec uint32

	//
	InboundAllowedICMPv6Packets uint64

	//
	InboundAllowedICMPv6PacketsPersec uint32

	//
	InboundAllowedIPv6IPsecAuthenticatedPackets uint64

	//
	InboundAllowedIPv6IPsecAuthenticatedPacketsPersec uint32

	//
	InboundAllowedIPv6IPsecUnauthenticatedPackets uint64

	//
	InboundAllowedIPv6IPsecUnauthenticatedPacketsPersec uint32

	//
	InboundDiscardedDefaultBlockPackets uint64

	//
	InboundDiscardedDefaultBlockPacketsPersec uint32

	//
	InboundDiscardedFilterBlockIPv6Packets uint64

	//
	InboundDiscardedFilterBlockIPv6PacketsPersec uint32

	//
	InboundDiscardedPackets uint64

	//
	InboundDiscardedPacketsPersec uint32

	//
	InboundOtherDiscardedIPv6IPsecAuthenticatedPackets uint64

	//
	InboundOtherDiscardedIPv6IPsecAuthenticatedPacketsPersec uint32

	//
	InboundOtherDiscardedIPv6IPsecUnauthenticatedPackets uint64

	//
	InboundOtherDiscardedIPv6IPsecUnauthenticatedPacketsPersec uint32

	//
	InboundPerIPRateLimitDiscardedIPv6IPsecUnauthenticatedPackets uint64

	//
	InboundPerIPRateLimitDiscardedIPv6IPsecUnauthenticatedPacketsPersec uint32

	//
	InboundRateLimitDiscardedDefaultBlockExemptPackets uint64

	//
	InboundRateLimitDiscardedDefaultBlockExemptPacketsPersec uint32

	//
	InboundRateLimitDiscardedFilterExemptIPv6Packets uint64

	//
	InboundRateLimitDiscardedFilterExemptIPv6PacketsPersec uint32

	//
	InboundRateLimitDiscardedICMPv6Packets uint64

	//
	InboundRateLimitDiscardedICMPv6PacketsPersec uint32

	//
	InboundRateLimitDiscardedIPv6IPsecAuthenticatedPackets uint64

	//
	InboundRateLimitDiscardedIPv6IPsecAuthenticatedPacketsPersec uint32

	//
	InboundRateLimitDiscardedIPv6IPsecUnauthenticatedPackets uint64

	//
	InboundRateLimitDiscardedIPv6IPsecUnauthenticatedPacketsPersec uint32

	//
	PerIPRateLimitQueues uint64

	//
	StateEntries uint64

	//
	StateEntriesPersec uint32
}

func NewWin32_PerfRawData_Counters_IPsecDoSProtectionEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_Counters_IPsecDoSProtection, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_IPsecDoSProtection{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_Counters_IPsecDoSProtectionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_Counters_IPsecDoSProtection, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_IPsecDoSProtection{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetCurrentStateEntries sets the value of CurrentStateEntries for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyCurrentStateEntries(value uint64) (err error) {
	return instance.SetProperty("CurrentStateEntries", (value))
}

// GetCurrentStateEntries gets the value of CurrentStateEntries for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyCurrentStateEntries() (value uint64, err error) {
	retValue, err := instance.GetProperty("CurrentStateEntries")
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

// SetInboundAllowedDefaultBlockExemptPackets sets the value of InboundAllowedDefaultBlockExemptPackets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyInboundAllowedDefaultBlockExemptPackets(value uint64) (err error) {
	return instance.SetProperty("InboundAllowedDefaultBlockExemptPackets", (value))
}

// GetInboundAllowedDefaultBlockExemptPackets gets the value of InboundAllowedDefaultBlockExemptPackets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyInboundAllowedDefaultBlockExemptPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("InboundAllowedDefaultBlockExemptPackets")
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

// SetInboundAllowedDefaultBlockExemptPacketsPersec sets the value of InboundAllowedDefaultBlockExemptPacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyInboundAllowedDefaultBlockExemptPacketsPersec(value uint32) (err error) {
	return instance.SetProperty("InboundAllowedDefaultBlockExemptPacketsPersec", (value))
}

// GetInboundAllowedDefaultBlockExemptPacketsPersec gets the value of InboundAllowedDefaultBlockExemptPacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyInboundAllowedDefaultBlockExemptPacketsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("InboundAllowedDefaultBlockExemptPacketsPersec")
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

// SetInboundAllowedFilterExemptIPv6Packets sets the value of InboundAllowedFilterExemptIPv6Packets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyInboundAllowedFilterExemptIPv6Packets(value uint64) (err error) {
	return instance.SetProperty("InboundAllowedFilterExemptIPv6Packets", (value))
}

// GetInboundAllowedFilterExemptIPv6Packets gets the value of InboundAllowedFilterExemptIPv6Packets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyInboundAllowedFilterExemptIPv6Packets() (value uint64, err error) {
	retValue, err := instance.GetProperty("InboundAllowedFilterExemptIPv6Packets")
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

// SetInboundAllowedFilterExemptIPv6PacketsPersec sets the value of InboundAllowedFilterExemptIPv6PacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyInboundAllowedFilterExemptIPv6PacketsPersec(value uint32) (err error) {
	return instance.SetProperty("InboundAllowedFilterExemptIPv6PacketsPersec", (value))
}

// GetInboundAllowedFilterExemptIPv6PacketsPersec gets the value of InboundAllowedFilterExemptIPv6PacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyInboundAllowedFilterExemptIPv6PacketsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("InboundAllowedFilterExemptIPv6PacketsPersec")
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

// SetInboundAllowedICMPv6Packets sets the value of InboundAllowedICMPv6Packets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyInboundAllowedICMPv6Packets(value uint64) (err error) {
	return instance.SetProperty("InboundAllowedICMPv6Packets", (value))
}

// GetInboundAllowedICMPv6Packets gets the value of InboundAllowedICMPv6Packets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyInboundAllowedICMPv6Packets() (value uint64, err error) {
	retValue, err := instance.GetProperty("InboundAllowedICMPv6Packets")
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

// SetInboundAllowedICMPv6PacketsPersec sets the value of InboundAllowedICMPv6PacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyInboundAllowedICMPv6PacketsPersec(value uint32) (err error) {
	return instance.SetProperty("InboundAllowedICMPv6PacketsPersec", (value))
}

// GetInboundAllowedICMPv6PacketsPersec gets the value of InboundAllowedICMPv6PacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyInboundAllowedICMPv6PacketsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("InboundAllowedICMPv6PacketsPersec")
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

// SetInboundAllowedIPv6IPsecAuthenticatedPackets sets the value of InboundAllowedIPv6IPsecAuthenticatedPackets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyInboundAllowedIPv6IPsecAuthenticatedPackets(value uint64) (err error) {
	return instance.SetProperty("InboundAllowedIPv6IPsecAuthenticatedPackets", (value))
}

// GetInboundAllowedIPv6IPsecAuthenticatedPackets gets the value of InboundAllowedIPv6IPsecAuthenticatedPackets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyInboundAllowedIPv6IPsecAuthenticatedPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("InboundAllowedIPv6IPsecAuthenticatedPackets")
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

// SetInboundAllowedIPv6IPsecAuthenticatedPacketsPersec sets the value of InboundAllowedIPv6IPsecAuthenticatedPacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyInboundAllowedIPv6IPsecAuthenticatedPacketsPersec(value uint32) (err error) {
	return instance.SetProperty("InboundAllowedIPv6IPsecAuthenticatedPacketsPersec", (value))
}

// GetInboundAllowedIPv6IPsecAuthenticatedPacketsPersec gets the value of InboundAllowedIPv6IPsecAuthenticatedPacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyInboundAllowedIPv6IPsecAuthenticatedPacketsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("InboundAllowedIPv6IPsecAuthenticatedPacketsPersec")
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

// SetInboundAllowedIPv6IPsecUnauthenticatedPackets sets the value of InboundAllowedIPv6IPsecUnauthenticatedPackets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyInboundAllowedIPv6IPsecUnauthenticatedPackets(value uint64) (err error) {
	return instance.SetProperty("InboundAllowedIPv6IPsecUnauthenticatedPackets", (value))
}

// GetInboundAllowedIPv6IPsecUnauthenticatedPackets gets the value of InboundAllowedIPv6IPsecUnauthenticatedPackets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyInboundAllowedIPv6IPsecUnauthenticatedPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("InboundAllowedIPv6IPsecUnauthenticatedPackets")
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

// SetInboundAllowedIPv6IPsecUnauthenticatedPacketsPersec sets the value of InboundAllowedIPv6IPsecUnauthenticatedPacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyInboundAllowedIPv6IPsecUnauthenticatedPacketsPersec(value uint32) (err error) {
	return instance.SetProperty("InboundAllowedIPv6IPsecUnauthenticatedPacketsPersec", (value))
}

// GetInboundAllowedIPv6IPsecUnauthenticatedPacketsPersec gets the value of InboundAllowedIPv6IPsecUnauthenticatedPacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyInboundAllowedIPv6IPsecUnauthenticatedPacketsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("InboundAllowedIPv6IPsecUnauthenticatedPacketsPersec")
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

// SetInboundDiscardedDefaultBlockPackets sets the value of InboundDiscardedDefaultBlockPackets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyInboundDiscardedDefaultBlockPackets(value uint64) (err error) {
	return instance.SetProperty("InboundDiscardedDefaultBlockPackets", (value))
}

// GetInboundDiscardedDefaultBlockPackets gets the value of InboundDiscardedDefaultBlockPackets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyInboundDiscardedDefaultBlockPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("InboundDiscardedDefaultBlockPackets")
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

// SetInboundDiscardedDefaultBlockPacketsPersec sets the value of InboundDiscardedDefaultBlockPacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyInboundDiscardedDefaultBlockPacketsPersec(value uint32) (err error) {
	return instance.SetProperty("InboundDiscardedDefaultBlockPacketsPersec", (value))
}

// GetInboundDiscardedDefaultBlockPacketsPersec gets the value of InboundDiscardedDefaultBlockPacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyInboundDiscardedDefaultBlockPacketsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("InboundDiscardedDefaultBlockPacketsPersec")
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

// SetInboundDiscardedFilterBlockIPv6Packets sets the value of InboundDiscardedFilterBlockIPv6Packets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyInboundDiscardedFilterBlockIPv6Packets(value uint64) (err error) {
	return instance.SetProperty("InboundDiscardedFilterBlockIPv6Packets", (value))
}

// GetInboundDiscardedFilterBlockIPv6Packets gets the value of InboundDiscardedFilterBlockIPv6Packets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyInboundDiscardedFilterBlockIPv6Packets() (value uint64, err error) {
	retValue, err := instance.GetProperty("InboundDiscardedFilterBlockIPv6Packets")
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

// SetInboundDiscardedFilterBlockIPv6PacketsPersec sets the value of InboundDiscardedFilterBlockIPv6PacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyInboundDiscardedFilterBlockIPv6PacketsPersec(value uint32) (err error) {
	return instance.SetProperty("InboundDiscardedFilterBlockIPv6PacketsPersec", (value))
}

// GetInboundDiscardedFilterBlockIPv6PacketsPersec gets the value of InboundDiscardedFilterBlockIPv6PacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyInboundDiscardedFilterBlockIPv6PacketsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("InboundDiscardedFilterBlockIPv6PacketsPersec")
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

// SetInboundDiscardedPackets sets the value of InboundDiscardedPackets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyInboundDiscardedPackets(value uint64) (err error) {
	return instance.SetProperty("InboundDiscardedPackets", (value))
}

// GetInboundDiscardedPackets gets the value of InboundDiscardedPackets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyInboundDiscardedPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("InboundDiscardedPackets")
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

// SetInboundDiscardedPacketsPersec sets the value of InboundDiscardedPacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyInboundDiscardedPacketsPersec(value uint32) (err error) {
	return instance.SetProperty("InboundDiscardedPacketsPersec", (value))
}

// GetInboundDiscardedPacketsPersec gets the value of InboundDiscardedPacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyInboundDiscardedPacketsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("InboundDiscardedPacketsPersec")
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

// SetInboundOtherDiscardedIPv6IPsecAuthenticatedPackets sets the value of InboundOtherDiscardedIPv6IPsecAuthenticatedPackets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyInboundOtherDiscardedIPv6IPsecAuthenticatedPackets(value uint64) (err error) {
	return instance.SetProperty("InboundOtherDiscardedIPv6IPsecAuthenticatedPackets", (value))
}

// GetInboundOtherDiscardedIPv6IPsecAuthenticatedPackets gets the value of InboundOtherDiscardedIPv6IPsecAuthenticatedPackets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyInboundOtherDiscardedIPv6IPsecAuthenticatedPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("InboundOtherDiscardedIPv6IPsecAuthenticatedPackets")
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

// SetInboundOtherDiscardedIPv6IPsecAuthenticatedPacketsPersec sets the value of InboundOtherDiscardedIPv6IPsecAuthenticatedPacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyInboundOtherDiscardedIPv6IPsecAuthenticatedPacketsPersec(value uint32) (err error) {
	return instance.SetProperty("InboundOtherDiscardedIPv6IPsecAuthenticatedPacketsPersec", (value))
}

// GetInboundOtherDiscardedIPv6IPsecAuthenticatedPacketsPersec gets the value of InboundOtherDiscardedIPv6IPsecAuthenticatedPacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyInboundOtherDiscardedIPv6IPsecAuthenticatedPacketsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("InboundOtherDiscardedIPv6IPsecAuthenticatedPacketsPersec")
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

// SetInboundOtherDiscardedIPv6IPsecUnauthenticatedPackets sets the value of InboundOtherDiscardedIPv6IPsecUnauthenticatedPackets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyInboundOtherDiscardedIPv6IPsecUnauthenticatedPackets(value uint64) (err error) {
	return instance.SetProperty("InboundOtherDiscardedIPv6IPsecUnauthenticatedPackets", (value))
}

// GetInboundOtherDiscardedIPv6IPsecUnauthenticatedPackets gets the value of InboundOtherDiscardedIPv6IPsecUnauthenticatedPackets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyInboundOtherDiscardedIPv6IPsecUnauthenticatedPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("InboundOtherDiscardedIPv6IPsecUnauthenticatedPackets")
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

// SetInboundOtherDiscardedIPv6IPsecUnauthenticatedPacketsPersec sets the value of InboundOtherDiscardedIPv6IPsecUnauthenticatedPacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyInboundOtherDiscardedIPv6IPsecUnauthenticatedPacketsPersec(value uint32) (err error) {
	return instance.SetProperty("InboundOtherDiscardedIPv6IPsecUnauthenticatedPacketsPersec", (value))
}

// GetInboundOtherDiscardedIPv6IPsecUnauthenticatedPacketsPersec gets the value of InboundOtherDiscardedIPv6IPsecUnauthenticatedPacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyInboundOtherDiscardedIPv6IPsecUnauthenticatedPacketsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("InboundOtherDiscardedIPv6IPsecUnauthenticatedPacketsPersec")
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

// SetInboundPerIPRateLimitDiscardedIPv6IPsecUnauthenticatedPackets sets the value of InboundPerIPRateLimitDiscardedIPv6IPsecUnauthenticatedPackets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyInboundPerIPRateLimitDiscardedIPv6IPsecUnauthenticatedPackets(value uint64) (err error) {
	return instance.SetProperty("InboundPerIPRateLimitDiscardedIPv6IPsecUnauthenticatedPackets", (value))
}

// GetInboundPerIPRateLimitDiscardedIPv6IPsecUnauthenticatedPackets gets the value of InboundPerIPRateLimitDiscardedIPv6IPsecUnauthenticatedPackets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyInboundPerIPRateLimitDiscardedIPv6IPsecUnauthenticatedPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("InboundPerIPRateLimitDiscardedIPv6IPsecUnauthenticatedPackets")
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

// SetInboundPerIPRateLimitDiscardedIPv6IPsecUnauthenticatedPacketsPersec sets the value of InboundPerIPRateLimitDiscardedIPv6IPsecUnauthenticatedPacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyInboundPerIPRateLimitDiscardedIPv6IPsecUnauthenticatedPacketsPersec(value uint32) (err error) {
	return instance.SetProperty("InboundPerIPRateLimitDiscardedIPv6IPsecUnauthenticatedPacketsPersec", (value))
}

// GetInboundPerIPRateLimitDiscardedIPv6IPsecUnauthenticatedPacketsPersec gets the value of InboundPerIPRateLimitDiscardedIPv6IPsecUnauthenticatedPacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyInboundPerIPRateLimitDiscardedIPv6IPsecUnauthenticatedPacketsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("InboundPerIPRateLimitDiscardedIPv6IPsecUnauthenticatedPacketsPersec")
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

// SetInboundRateLimitDiscardedDefaultBlockExemptPackets sets the value of InboundRateLimitDiscardedDefaultBlockExemptPackets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyInboundRateLimitDiscardedDefaultBlockExemptPackets(value uint64) (err error) {
	return instance.SetProperty("InboundRateLimitDiscardedDefaultBlockExemptPackets", (value))
}

// GetInboundRateLimitDiscardedDefaultBlockExemptPackets gets the value of InboundRateLimitDiscardedDefaultBlockExemptPackets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyInboundRateLimitDiscardedDefaultBlockExemptPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("InboundRateLimitDiscardedDefaultBlockExemptPackets")
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

// SetInboundRateLimitDiscardedDefaultBlockExemptPacketsPersec sets the value of InboundRateLimitDiscardedDefaultBlockExemptPacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyInboundRateLimitDiscardedDefaultBlockExemptPacketsPersec(value uint32) (err error) {
	return instance.SetProperty("InboundRateLimitDiscardedDefaultBlockExemptPacketsPersec", (value))
}

// GetInboundRateLimitDiscardedDefaultBlockExemptPacketsPersec gets the value of InboundRateLimitDiscardedDefaultBlockExemptPacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyInboundRateLimitDiscardedDefaultBlockExemptPacketsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("InboundRateLimitDiscardedDefaultBlockExemptPacketsPersec")
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

// SetInboundRateLimitDiscardedFilterExemptIPv6Packets sets the value of InboundRateLimitDiscardedFilterExemptIPv6Packets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyInboundRateLimitDiscardedFilterExemptIPv6Packets(value uint64) (err error) {
	return instance.SetProperty("InboundRateLimitDiscardedFilterExemptIPv6Packets", (value))
}

// GetInboundRateLimitDiscardedFilterExemptIPv6Packets gets the value of InboundRateLimitDiscardedFilterExemptIPv6Packets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyInboundRateLimitDiscardedFilterExemptIPv6Packets() (value uint64, err error) {
	retValue, err := instance.GetProperty("InboundRateLimitDiscardedFilterExemptIPv6Packets")
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

// SetInboundRateLimitDiscardedFilterExemptIPv6PacketsPersec sets the value of InboundRateLimitDiscardedFilterExemptIPv6PacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyInboundRateLimitDiscardedFilterExemptIPv6PacketsPersec(value uint32) (err error) {
	return instance.SetProperty("InboundRateLimitDiscardedFilterExemptIPv6PacketsPersec", (value))
}

// GetInboundRateLimitDiscardedFilterExemptIPv6PacketsPersec gets the value of InboundRateLimitDiscardedFilterExemptIPv6PacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyInboundRateLimitDiscardedFilterExemptIPv6PacketsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("InboundRateLimitDiscardedFilterExemptIPv6PacketsPersec")
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

// SetInboundRateLimitDiscardedICMPv6Packets sets the value of InboundRateLimitDiscardedICMPv6Packets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyInboundRateLimitDiscardedICMPv6Packets(value uint64) (err error) {
	return instance.SetProperty("InboundRateLimitDiscardedICMPv6Packets", (value))
}

// GetInboundRateLimitDiscardedICMPv6Packets gets the value of InboundRateLimitDiscardedICMPv6Packets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyInboundRateLimitDiscardedICMPv6Packets() (value uint64, err error) {
	retValue, err := instance.GetProperty("InboundRateLimitDiscardedICMPv6Packets")
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

// SetInboundRateLimitDiscardedICMPv6PacketsPersec sets the value of InboundRateLimitDiscardedICMPv6PacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyInboundRateLimitDiscardedICMPv6PacketsPersec(value uint32) (err error) {
	return instance.SetProperty("InboundRateLimitDiscardedICMPv6PacketsPersec", (value))
}

// GetInboundRateLimitDiscardedICMPv6PacketsPersec gets the value of InboundRateLimitDiscardedICMPv6PacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyInboundRateLimitDiscardedICMPv6PacketsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("InboundRateLimitDiscardedICMPv6PacketsPersec")
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

// SetInboundRateLimitDiscardedIPv6IPsecAuthenticatedPackets sets the value of InboundRateLimitDiscardedIPv6IPsecAuthenticatedPackets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyInboundRateLimitDiscardedIPv6IPsecAuthenticatedPackets(value uint64) (err error) {
	return instance.SetProperty("InboundRateLimitDiscardedIPv6IPsecAuthenticatedPackets", (value))
}

// GetInboundRateLimitDiscardedIPv6IPsecAuthenticatedPackets gets the value of InboundRateLimitDiscardedIPv6IPsecAuthenticatedPackets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyInboundRateLimitDiscardedIPv6IPsecAuthenticatedPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("InboundRateLimitDiscardedIPv6IPsecAuthenticatedPackets")
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

// SetInboundRateLimitDiscardedIPv6IPsecAuthenticatedPacketsPersec sets the value of InboundRateLimitDiscardedIPv6IPsecAuthenticatedPacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyInboundRateLimitDiscardedIPv6IPsecAuthenticatedPacketsPersec(value uint32) (err error) {
	return instance.SetProperty("InboundRateLimitDiscardedIPv6IPsecAuthenticatedPacketsPersec", (value))
}

// GetInboundRateLimitDiscardedIPv6IPsecAuthenticatedPacketsPersec gets the value of InboundRateLimitDiscardedIPv6IPsecAuthenticatedPacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyInboundRateLimitDiscardedIPv6IPsecAuthenticatedPacketsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("InboundRateLimitDiscardedIPv6IPsecAuthenticatedPacketsPersec")
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

// SetInboundRateLimitDiscardedIPv6IPsecUnauthenticatedPackets sets the value of InboundRateLimitDiscardedIPv6IPsecUnauthenticatedPackets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyInboundRateLimitDiscardedIPv6IPsecUnauthenticatedPackets(value uint64) (err error) {
	return instance.SetProperty("InboundRateLimitDiscardedIPv6IPsecUnauthenticatedPackets", (value))
}

// GetInboundRateLimitDiscardedIPv6IPsecUnauthenticatedPackets gets the value of InboundRateLimitDiscardedIPv6IPsecUnauthenticatedPackets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyInboundRateLimitDiscardedIPv6IPsecUnauthenticatedPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("InboundRateLimitDiscardedIPv6IPsecUnauthenticatedPackets")
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

// SetInboundRateLimitDiscardedIPv6IPsecUnauthenticatedPacketsPersec sets the value of InboundRateLimitDiscardedIPv6IPsecUnauthenticatedPacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyInboundRateLimitDiscardedIPv6IPsecUnauthenticatedPacketsPersec(value uint32) (err error) {
	return instance.SetProperty("InboundRateLimitDiscardedIPv6IPsecUnauthenticatedPacketsPersec", (value))
}

// GetInboundRateLimitDiscardedIPv6IPsecUnauthenticatedPacketsPersec gets the value of InboundRateLimitDiscardedIPv6IPsecUnauthenticatedPacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyInboundRateLimitDiscardedIPv6IPsecUnauthenticatedPacketsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("InboundRateLimitDiscardedIPv6IPsecUnauthenticatedPacketsPersec")
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

// SetPerIPRateLimitQueues sets the value of PerIPRateLimitQueues for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyPerIPRateLimitQueues(value uint64) (err error) {
	return instance.SetProperty("PerIPRateLimitQueues", (value))
}

// GetPerIPRateLimitQueues gets the value of PerIPRateLimitQueues for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyPerIPRateLimitQueues() (value uint64, err error) {
	retValue, err := instance.GetProperty("PerIPRateLimitQueues")
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

// SetStateEntries sets the value of StateEntries for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyStateEntries(value uint64) (err error) {
	return instance.SetProperty("StateEntries", (value))
}

// GetStateEntries gets the value of StateEntries for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyStateEntries() (value uint64, err error) {
	retValue, err := instance.GetProperty("StateEntries")
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

// SetStateEntriesPersec sets the value of StateEntriesPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) SetPropertyStateEntriesPersec(value uint32) (err error) {
	return instance.SetProperty("StateEntriesPersec", (value))
}

// GetStateEntriesPersec gets the value of StateEntriesPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDoSProtection) GetPropertyStateEntriesPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("StateEntriesPersec")
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
