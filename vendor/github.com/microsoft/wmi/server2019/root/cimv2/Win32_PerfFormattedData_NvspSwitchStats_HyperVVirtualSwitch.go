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

// Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch struct
type Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch struct {
	*Win32_PerfFormattedData

	//
	BroadcastPacketsReceivedPersec uint64

	//
	BroadcastPacketsSentPersec uint64

	//
	BytesPersec uint64

	//
	BytesReceivedPersec uint64

	//
	BytesSentPersec uint64

	//
	DirectedPacketsReceivedPersec uint64

	//
	DirectedPacketsSentPersec uint64

	//
	DroppedPacketsIncomingPersec uint64

	//
	DroppedPacketsOutgoingPersec uint64

	//
	ExtensionsDroppedPacketsIncomingPersec uint64

	//
	ExtensionsDroppedPacketsOutgoingPersec uint64

	//
	LearnedMacAddresses uint64

	//
	LearnedMacAddressesPersec uint64

	//
	MulticastPacketsReceivedPersec uint64

	//
	MulticastPacketsSentPersec uint64

	//
	NumberofSendChannelMovesPersec uint64

	//
	NumberofVMQMovesPersec uint64

	//
	PacketsFlooded uint64

	//
	PacketsFloodedPersec uint64

	//
	PacketsPersec uint64

	//
	PacketsReceivedPersec uint64

	//
	PacketsSentPersec uint64

	//
	PurgedMacAddresses uint64

	//
	PurgedMacAddressesPersec uint64

	//
	RSCCoalescedBytes uint64

	//
	RSCCoalescedEventBucket10To1 uint64

	//
	RSCCoalescedEventBucket22To3 uint64

	//
	RSCCoalescedEventBucket34To7 uint64

	//
	RSCCoalescedEventBucket48To15 uint64

	//
	RSCCoalescedEventBucket516To31 uint64

	//
	RSCCoalescedEventBucket632To63 uint64

	//
	RSCCoalescedPacketBucket10To1 uint64

	//
	RSCCoalescedPacketBucket22To3 uint64

	//
	RSCCoalescedPacketBucket34To7 uint64

	//
	RSCCoalescedPacketBucket48To15 uint64

	//
	RSCCoalescedPacketBucket516To31 uint64

	//
	RSCCoalescedPacketBucket632To63 uint64

	//
	RSCCoalescedPackets uint64

	//
	RSCCoalesceEvents uint64

	//
	RSCPacketsProcessed uint64
}

func NewWin32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitchEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitchEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetBroadcastPacketsReceivedPersec sets the value of BroadcastPacketsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyBroadcastPacketsReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("BroadcastPacketsReceivedPersec", (value))
}

// GetBroadcastPacketsReceivedPersec gets the value of BroadcastPacketsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyBroadcastPacketsReceivedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("BroadcastPacketsReceivedPersec")
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

// SetBroadcastPacketsSentPersec sets the value of BroadcastPacketsSentPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyBroadcastPacketsSentPersec(value uint64) (err error) {
	return instance.SetProperty("BroadcastPacketsSentPersec", (value))
}

// GetBroadcastPacketsSentPersec gets the value of BroadcastPacketsSentPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyBroadcastPacketsSentPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("BroadcastPacketsSentPersec")
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

// SetBytesPersec sets the value of BytesPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyBytesPersec(value uint64) (err error) {
	return instance.SetProperty("BytesPersec", (value))
}

// GetBytesPersec gets the value of BytesPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesPersec")
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

// SetBytesReceivedPersec sets the value of BytesReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyBytesReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("BytesReceivedPersec", (value))
}

// GetBytesReceivedPersec gets the value of BytesReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyBytesReceivedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesReceivedPersec")
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

// SetBytesSentPersec sets the value of BytesSentPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyBytesSentPersec(value uint64) (err error) {
	return instance.SetProperty("BytesSentPersec", (value))
}

// GetBytesSentPersec gets the value of BytesSentPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyBytesSentPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesSentPersec")
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

// SetDirectedPacketsReceivedPersec sets the value of DirectedPacketsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyDirectedPacketsReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("DirectedPacketsReceivedPersec", (value))
}

// GetDirectedPacketsReceivedPersec gets the value of DirectedPacketsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyDirectedPacketsReceivedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("DirectedPacketsReceivedPersec")
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

// SetDirectedPacketsSentPersec sets the value of DirectedPacketsSentPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyDirectedPacketsSentPersec(value uint64) (err error) {
	return instance.SetProperty("DirectedPacketsSentPersec", (value))
}

// GetDirectedPacketsSentPersec gets the value of DirectedPacketsSentPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyDirectedPacketsSentPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("DirectedPacketsSentPersec")
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

// SetDroppedPacketsIncomingPersec sets the value of DroppedPacketsIncomingPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyDroppedPacketsIncomingPersec(value uint64) (err error) {
	return instance.SetProperty("DroppedPacketsIncomingPersec", (value))
}

// GetDroppedPacketsIncomingPersec gets the value of DroppedPacketsIncomingPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyDroppedPacketsIncomingPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("DroppedPacketsIncomingPersec")
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

// SetDroppedPacketsOutgoingPersec sets the value of DroppedPacketsOutgoingPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyDroppedPacketsOutgoingPersec(value uint64) (err error) {
	return instance.SetProperty("DroppedPacketsOutgoingPersec", (value))
}

// GetDroppedPacketsOutgoingPersec gets the value of DroppedPacketsOutgoingPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyDroppedPacketsOutgoingPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("DroppedPacketsOutgoingPersec")
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

// SetExtensionsDroppedPacketsIncomingPersec sets the value of ExtensionsDroppedPacketsIncomingPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyExtensionsDroppedPacketsIncomingPersec(value uint64) (err error) {
	return instance.SetProperty("ExtensionsDroppedPacketsIncomingPersec", (value))
}

// GetExtensionsDroppedPacketsIncomingPersec gets the value of ExtensionsDroppedPacketsIncomingPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyExtensionsDroppedPacketsIncomingPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ExtensionsDroppedPacketsIncomingPersec")
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

// SetExtensionsDroppedPacketsOutgoingPersec sets the value of ExtensionsDroppedPacketsOutgoingPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyExtensionsDroppedPacketsOutgoingPersec(value uint64) (err error) {
	return instance.SetProperty("ExtensionsDroppedPacketsOutgoingPersec", (value))
}

// GetExtensionsDroppedPacketsOutgoingPersec gets the value of ExtensionsDroppedPacketsOutgoingPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyExtensionsDroppedPacketsOutgoingPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ExtensionsDroppedPacketsOutgoingPersec")
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

// SetLearnedMacAddresses sets the value of LearnedMacAddresses for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyLearnedMacAddresses(value uint64) (err error) {
	return instance.SetProperty("LearnedMacAddresses", (value))
}

// GetLearnedMacAddresses gets the value of LearnedMacAddresses for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyLearnedMacAddresses() (value uint64, err error) {
	retValue, err := instance.GetProperty("LearnedMacAddresses")
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

// SetLearnedMacAddressesPersec sets the value of LearnedMacAddressesPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyLearnedMacAddressesPersec(value uint64) (err error) {
	return instance.SetProperty("LearnedMacAddressesPersec", (value))
}

// GetLearnedMacAddressesPersec gets the value of LearnedMacAddressesPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyLearnedMacAddressesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("LearnedMacAddressesPersec")
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

// SetMulticastPacketsReceivedPersec sets the value of MulticastPacketsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyMulticastPacketsReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("MulticastPacketsReceivedPersec", (value))
}

// GetMulticastPacketsReceivedPersec gets the value of MulticastPacketsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyMulticastPacketsReceivedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("MulticastPacketsReceivedPersec")
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

// SetMulticastPacketsSentPersec sets the value of MulticastPacketsSentPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyMulticastPacketsSentPersec(value uint64) (err error) {
	return instance.SetProperty("MulticastPacketsSentPersec", (value))
}

// GetMulticastPacketsSentPersec gets the value of MulticastPacketsSentPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyMulticastPacketsSentPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("MulticastPacketsSentPersec")
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

// SetNumberofSendChannelMovesPersec sets the value of NumberofSendChannelMovesPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyNumberofSendChannelMovesPersec(value uint64) (err error) {
	return instance.SetProperty("NumberofSendChannelMovesPersec", (value))
}

// GetNumberofSendChannelMovesPersec gets the value of NumberofSendChannelMovesPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyNumberofSendChannelMovesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("NumberofSendChannelMovesPersec")
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

// SetNumberofVMQMovesPersec sets the value of NumberofVMQMovesPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyNumberofVMQMovesPersec(value uint64) (err error) {
	return instance.SetProperty("NumberofVMQMovesPersec", (value))
}

// GetNumberofVMQMovesPersec gets the value of NumberofVMQMovesPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyNumberofVMQMovesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("NumberofVMQMovesPersec")
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

// SetPacketsFlooded sets the value of PacketsFlooded for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyPacketsFlooded(value uint64) (err error) {
	return instance.SetProperty("PacketsFlooded", (value))
}

// GetPacketsFlooded gets the value of PacketsFlooded for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyPacketsFlooded() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsFlooded")
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

// SetPacketsFloodedPersec sets the value of PacketsFloodedPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyPacketsFloodedPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsFloodedPersec", (value))
}

// GetPacketsFloodedPersec gets the value of PacketsFloodedPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyPacketsFloodedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsFloodedPersec")
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

// SetPacketsPersec sets the value of PacketsPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyPacketsPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsPersec", (value))
}

// GetPacketsPersec gets the value of PacketsPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyPacketsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsPersec")
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

// SetPacketsReceivedPersec sets the value of PacketsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyPacketsReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsReceivedPersec", (value))
}

// GetPacketsReceivedPersec gets the value of PacketsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyPacketsReceivedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsReceivedPersec")
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

// SetPacketsSentPersec sets the value of PacketsSentPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyPacketsSentPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsSentPersec", (value))
}

// GetPacketsSentPersec gets the value of PacketsSentPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyPacketsSentPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsSentPersec")
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

// SetPurgedMacAddresses sets the value of PurgedMacAddresses for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyPurgedMacAddresses(value uint64) (err error) {
	return instance.SetProperty("PurgedMacAddresses", (value))
}

// GetPurgedMacAddresses gets the value of PurgedMacAddresses for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyPurgedMacAddresses() (value uint64, err error) {
	retValue, err := instance.GetProperty("PurgedMacAddresses")
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

// SetPurgedMacAddressesPersec sets the value of PurgedMacAddressesPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyPurgedMacAddressesPersec(value uint64) (err error) {
	return instance.SetProperty("PurgedMacAddressesPersec", (value))
}

// GetPurgedMacAddressesPersec gets the value of PurgedMacAddressesPersec for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyPurgedMacAddressesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("PurgedMacAddressesPersec")
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

// SetRSCCoalescedBytes sets the value of RSCCoalescedBytes for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyRSCCoalescedBytes(value uint64) (err error) {
	return instance.SetProperty("RSCCoalescedBytes", (value))
}

// GetRSCCoalescedBytes gets the value of RSCCoalescedBytes for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyRSCCoalescedBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("RSCCoalescedBytes")
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

// SetRSCCoalescedEventBucket10To1 sets the value of RSCCoalescedEventBucket10To1 for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyRSCCoalescedEventBucket10To1(value uint64) (err error) {
	return instance.SetProperty("RSCCoalescedEventBucket10To1", (value))
}

// GetRSCCoalescedEventBucket10To1 gets the value of RSCCoalescedEventBucket10To1 for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyRSCCoalescedEventBucket10To1() (value uint64, err error) {
	retValue, err := instance.GetProperty("RSCCoalescedEventBucket10To1")
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

// SetRSCCoalescedEventBucket22To3 sets the value of RSCCoalescedEventBucket22To3 for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyRSCCoalescedEventBucket22To3(value uint64) (err error) {
	return instance.SetProperty("RSCCoalescedEventBucket22To3", (value))
}

// GetRSCCoalescedEventBucket22To3 gets the value of RSCCoalescedEventBucket22To3 for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyRSCCoalescedEventBucket22To3() (value uint64, err error) {
	retValue, err := instance.GetProperty("RSCCoalescedEventBucket22To3")
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

// SetRSCCoalescedEventBucket34To7 sets the value of RSCCoalescedEventBucket34To7 for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyRSCCoalescedEventBucket34To7(value uint64) (err error) {
	return instance.SetProperty("RSCCoalescedEventBucket34To7", (value))
}

// GetRSCCoalescedEventBucket34To7 gets the value of RSCCoalescedEventBucket34To7 for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyRSCCoalescedEventBucket34To7() (value uint64, err error) {
	retValue, err := instance.GetProperty("RSCCoalescedEventBucket34To7")
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

// SetRSCCoalescedEventBucket48To15 sets the value of RSCCoalescedEventBucket48To15 for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyRSCCoalescedEventBucket48To15(value uint64) (err error) {
	return instance.SetProperty("RSCCoalescedEventBucket48To15", (value))
}

// GetRSCCoalescedEventBucket48To15 gets the value of RSCCoalescedEventBucket48To15 for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyRSCCoalescedEventBucket48To15() (value uint64, err error) {
	retValue, err := instance.GetProperty("RSCCoalescedEventBucket48To15")
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

// SetRSCCoalescedEventBucket516To31 sets the value of RSCCoalescedEventBucket516To31 for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyRSCCoalescedEventBucket516To31(value uint64) (err error) {
	return instance.SetProperty("RSCCoalescedEventBucket516To31", (value))
}

// GetRSCCoalescedEventBucket516To31 gets the value of RSCCoalescedEventBucket516To31 for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyRSCCoalescedEventBucket516To31() (value uint64, err error) {
	retValue, err := instance.GetProperty("RSCCoalescedEventBucket516To31")
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

// SetRSCCoalescedEventBucket632To63 sets the value of RSCCoalescedEventBucket632To63 for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyRSCCoalescedEventBucket632To63(value uint64) (err error) {
	return instance.SetProperty("RSCCoalescedEventBucket632To63", (value))
}

// GetRSCCoalescedEventBucket632To63 gets the value of RSCCoalescedEventBucket632To63 for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyRSCCoalescedEventBucket632To63() (value uint64, err error) {
	retValue, err := instance.GetProperty("RSCCoalescedEventBucket632To63")
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

// SetRSCCoalescedPacketBucket10To1 sets the value of RSCCoalescedPacketBucket10To1 for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyRSCCoalescedPacketBucket10To1(value uint64) (err error) {
	return instance.SetProperty("RSCCoalescedPacketBucket10To1", (value))
}

// GetRSCCoalescedPacketBucket10To1 gets the value of RSCCoalescedPacketBucket10To1 for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyRSCCoalescedPacketBucket10To1() (value uint64, err error) {
	retValue, err := instance.GetProperty("RSCCoalescedPacketBucket10To1")
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

// SetRSCCoalescedPacketBucket22To3 sets the value of RSCCoalescedPacketBucket22To3 for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyRSCCoalescedPacketBucket22To3(value uint64) (err error) {
	return instance.SetProperty("RSCCoalescedPacketBucket22To3", (value))
}

// GetRSCCoalescedPacketBucket22To3 gets the value of RSCCoalescedPacketBucket22To3 for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyRSCCoalescedPacketBucket22To3() (value uint64, err error) {
	retValue, err := instance.GetProperty("RSCCoalescedPacketBucket22To3")
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

// SetRSCCoalescedPacketBucket34To7 sets the value of RSCCoalescedPacketBucket34To7 for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyRSCCoalescedPacketBucket34To7(value uint64) (err error) {
	return instance.SetProperty("RSCCoalescedPacketBucket34To7", (value))
}

// GetRSCCoalescedPacketBucket34To7 gets the value of RSCCoalescedPacketBucket34To7 for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyRSCCoalescedPacketBucket34To7() (value uint64, err error) {
	retValue, err := instance.GetProperty("RSCCoalescedPacketBucket34To7")
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

// SetRSCCoalescedPacketBucket48To15 sets the value of RSCCoalescedPacketBucket48To15 for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyRSCCoalescedPacketBucket48To15(value uint64) (err error) {
	return instance.SetProperty("RSCCoalescedPacketBucket48To15", (value))
}

// GetRSCCoalescedPacketBucket48To15 gets the value of RSCCoalescedPacketBucket48To15 for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyRSCCoalescedPacketBucket48To15() (value uint64, err error) {
	retValue, err := instance.GetProperty("RSCCoalescedPacketBucket48To15")
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

// SetRSCCoalescedPacketBucket516To31 sets the value of RSCCoalescedPacketBucket516To31 for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyRSCCoalescedPacketBucket516To31(value uint64) (err error) {
	return instance.SetProperty("RSCCoalescedPacketBucket516To31", (value))
}

// GetRSCCoalescedPacketBucket516To31 gets the value of RSCCoalescedPacketBucket516To31 for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyRSCCoalescedPacketBucket516To31() (value uint64, err error) {
	retValue, err := instance.GetProperty("RSCCoalescedPacketBucket516To31")
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

// SetRSCCoalescedPacketBucket632To63 sets the value of RSCCoalescedPacketBucket632To63 for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyRSCCoalescedPacketBucket632To63(value uint64) (err error) {
	return instance.SetProperty("RSCCoalescedPacketBucket632To63", (value))
}

// GetRSCCoalescedPacketBucket632To63 gets the value of RSCCoalescedPacketBucket632To63 for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyRSCCoalescedPacketBucket632To63() (value uint64, err error) {
	retValue, err := instance.GetProperty("RSCCoalescedPacketBucket632To63")
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

// SetRSCCoalescedPackets sets the value of RSCCoalescedPackets for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyRSCCoalescedPackets(value uint64) (err error) {
	return instance.SetProperty("RSCCoalescedPackets", (value))
}

// GetRSCCoalescedPackets gets the value of RSCCoalescedPackets for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyRSCCoalescedPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("RSCCoalescedPackets")
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

// SetRSCCoalesceEvents sets the value of RSCCoalesceEvents for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyRSCCoalesceEvents(value uint64) (err error) {
	return instance.SetProperty("RSCCoalesceEvents", (value))
}

// GetRSCCoalesceEvents gets the value of RSCCoalesceEvents for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyRSCCoalesceEvents() (value uint64, err error) {
	retValue, err := instance.GetProperty("RSCCoalesceEvents")
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

// SetRSCPacketsProcessed sets the value of RSCPacketsProcessed for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) SetPropertyRSCPacketsProcessed(value uint64) (err error) {
	return instance.SetProperty("RSCPacketsProcessed", (value))
}

// GetRSCPacketsProcessed gets the value of RSCPacketsProcessed for the instance
func (instance *Win32_PerfFormattedData_NvspSwitchStats_HyperVVirtualSwitch) GetPropertyRSCPacketsProcessed() (value uint64, err error) {
	retValue, err := instance.GetProperty("RSCPacketsProcessed")
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
