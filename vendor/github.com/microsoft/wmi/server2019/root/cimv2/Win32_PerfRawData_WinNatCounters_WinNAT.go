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

// Win32_PerfRawData_WinNatCounters_WinNAT struct
type Win32_PerfRawData_WinNatCounters_WinNAT struct {
	*Win32_PerfRawData

	//
	CurrentSessionCount uint32

	//
	DroppedICMPerrorpackets uint32

	//
	DroppedICMPerrorpacketsPersec uint32

	//
	DroppedPackets uint32

	//
	DroppedPacketsPersec uint32

	//
	InterRoutingDomainHairpinnedPackets uint32

	//
	InterRoutingDomainHairpinnedPacketsPersec uint32

	//
	IntraRoutingDomainHairpinnedPackets uint32

	//
	IntraRoutingDomainHairpinnedPacketsPersec uint32

	//
	PacketsExternaltoInternal uint32

	//
	PacketsInternaltoExternal uint32

	//
	PacketsPersecExternaltoInternal uint32

	//
	PacketsPersecInternaltoExternal uint32

	//
	SessionsPersec uint32
}

func NewWin32_PerfRawData_WinNatCounters_WinNATEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_WinNatCounters_WinNAT, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_WinNatCounters_WinNAT{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_WinNatCounters_WinNATEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_WinNatCounters_WinNAT, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_WinNatCounters_WinNAT{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetCurrentSessionCount sets the value of CurrentSessionCount for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNAT) SetPropertyCurrentSessionCount(value uint32) (err error) {
	return instance.SetProperty("CurrentSessionCount", (value))
}

// GetCurrentSessionCount gets the value of CurrentSessionCount for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNAT) GetPropertyCurrentSessionCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("CurrentSessionCount")
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

// SetDroppedICMPerrorpackets sets the value of DroppedICMPerrorpackets for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNAT) SetPropertyDroppedICMPerrorpackets(value uint32) (err error) {
	return instance.SetProperty("DroppedICMPerrorpackets", (value))
}

// GetDroppedICMPerrorpackets gets the value of DroppedICMPerrorpackets for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNAT) GetPropertyDroppedICMPerrorpackets() (value uint32, err error) {
	retValue, err := instance.GetProperty("DroppedICMPerrorpackets")
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

// SetDroppedICMPerrorpacketsPersec sets the value of DroppedICMPerrorpacketsPersec for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNAT) SetPropertyDroppedICMPerrorpacketsPersec(value uint32) (err error) {
	return instance.SetProperty("DroppedICMPerrorpacketsPersec", (value))
}

// GetDroppedICMPerrorpacketsPersec gets the value of DroppedICMPerrorpacketsPersec for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNAT) GetPropertyDroppedICMPerrorpacketsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("DroppedICMPerrorpacketsPersec")
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

// SetDroppedPackets sets the value of DroppedPackets for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNAT) SetPropertyDroppedPackets(value uint32) (err error) {
	return instance.SetProperty("DroppedPackets", (value))
}

// GetDroppedPackets gets the value of DroppedPackets for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNAT) GetPropertyDroppedPackets() (value uint32, err error) {
	retValue, err := instance.GetProperty("DroppedPackets")
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

// SetDroppedPacketsPersec sets the value of DroppedPacketsPersec for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNAT) SetPropertyDroppedPacketsPersec(value uint32) (err error) {
	return instance.SetProperty("DroppedPacketsPersec", (value))
}

// GetDroppedPacketsPersec gets the value of DroppedPacketsPersec for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNAT) GetPropertyDroppedPacketsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("DroppedPacketsPersec")
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

// SetInterRoutingDomainHairpinnedPackets sets the value of InterRoutingDomainHairpinnedPackets for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNAT) SetPropertyInterRoutingDomainHairpinnedPackets(value uint32) (err error) {
	return instance.SetProperty("InterRoutingDomainHairpinnedPackets", (value))
}

// GetInterRoutingDomainHairpinnedPackets gets the value of InterRoutingDomainHairpinnedPackets for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNAT) GetPropertyInterRoutingDomainHairpinnedPackets() (value uint32, err error) {
	retValue, err := instance.GetProperty("InterRoutingDomainHairpinnedPackets")
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

// SetInterRoutingDomainHairpinnedPacketsPersec sets the value of InterRoutingDomainHairpinnedPacketsPersec for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNAT) SetPropertyInterRoutingDomainHairpinnedPacketsPersec(value uint32) (err error) {
	return instance.SetProperty("InterRoutingDomainHairpinnedPacketsPersec", (value))
}

// GetInterRoutingDomainHairpinnedPacketsPersec gets the value of InterRoutingDomainHairpinnedPacketsPersec for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNAT) GetPropertyInterRoutingDomainHairpinnedPacketsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("InterRoutingDomainHairpinnedPacketsPersec")
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

// SetIntraRoutingDomainHairpinnedPackets sets the value of IntraRoutingDomainHairpinnedPackets for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNAT) SetPropertyIntraRoutingDomainHairpinnedPackets(value uint32) (err error) {
	return instance.SetProperty("IntraRoutingDomainHairpinnedPackets", (value))
}

// GetIntraRoutingDomainHairpinnedPackets gets the value of IntraRoutingDomainHairpinnedPackets for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNAT) GetPropertyIntraRoutingDomainHairpinnedPackets() (value uint32, err error) {
	retValue, err := instance.GetProperty("IntraRoutingDomainHairpinnedPackets")
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

// SetIntraRoutingDomainHairpinnedPacketsPersec sets the value of IntraRoutingDomainHairpinnedPacketsPersec for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNAT) SetPropertyIntraRoutingDomainHairpinnedPacketsPersec(value uint32) (err error) {
	return instance.SetProperty("IntraRoutingDomainHairpinnedPacketsPersec", (value))
}

// GetIntraRoutingDomainHairpinnedPacketsPersec gets the value of IntraRoutingDomainHairpinnedPacketsPersec for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNAT) GetPropertyIntraRoutingDomainHairpinnedPacketsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("IntraRoutingDomainHairpinnedPacketsPersec")
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

// SetPacketsExternaltoInternal sets the value of PacketsExternaltoInternal for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNAT) SetPropertyPacketsExternaltoInternal(value uint32) (err error) {
	return instance.SetProperty("PacketsExternaltoInternal", (value))
}

// GetPacketsExternaltoInternal gets the value of PacketsExternaltoInternal for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNAT) GetPropertyPacketsExternaltoInternal() (value uint32, err error) {
	retValue, err := instance.GetProperty("PacketsExternaltoInternal")
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

// SetPacketsInternaltoExternal sets the value of PacketsInternaltoExternal for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNAT) SetPropertyPacketsInternaltoExternal(value uint32) (err error) {
	return instance.SetProperty("PacketsInternaltoExternal", (value))
}

// GetPacketsInternaltoExternal gets the value of PacketsInternaltoExternal for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNAT) GetPropertyPacketsInternaltoExternal() (value uint32, err error) {
	retValue, err := instance.GetProperty("PacketsInternaltoExternal")
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

// SetPacketsPersecExternaltoInternal sets the value of PacketsPersecExternaltoInternal for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNAT) SetPropertyPacketsPersecExternaltoInternal(value uint32) (err error) {
	return instance.SetProperty("PacketsPersecExternaltoInternal", (value))
}

// GetPacketsPersecExternaltoInternal gets the value of PacketsPersecExternaltoInternal for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNAT) GetPropertyPacketsPersecExternaltoInternal() (value uint32, err error) {
	retValue, err := instance.GetProperty("PacketsPersecExternaltoInternal")
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

// SetPacketsPersecInternaltoExternal sets the value of PacketsPersecInternaltoExternal for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNAT) SetPropertyPacketsPersecInternaltoExternal(value uint32) (err error) {
	return instance.SetProperty("PacketsPersecInternaltoExternal", (value))
}

// GetPacketsPersecInternaltoExternal gets the value of PacketsPersecInternaltoExternal for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNAT) GetPropertyPacketsPersecInternaltoExternal() (value uint32, err error) {
	retValue, err := instance.GetProperty("PacketsPersecInternaltoExternal")
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

// SetSessionsPersec sets the value of SessionsPersec for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNAT) SetPropertySessionsPersec(value uint32) (err error) {
	return instance.SetProperty("SessionsPersec", (value))
}

// GetSessionsPersec gets the value of SessionsPersec for the instance
func (instance *Win32_PerfRawData_WinNatCounters_WinNAT) GetPropertySessionsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("SessionsPersec")
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
