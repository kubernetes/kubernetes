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

// Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort struct
type Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort struct {
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
	IPsecoffloadBytesReceivePersec uint64

	//
	IPsecoffloadBytesSentPersec uint64

	//
	IPsecSAsOffloaded uint64

	//
	MulticastPacketsReceivedPersec uint64

	//
	MulticastPacketsSentPersec uint64

	//
	PacketsPersec uint64

	//
	PacketsReceivedPersec uint64

	//
	PacketsSentPersec uint64

	//
	UnhashedPacketsReceivedPersec uint64

	//
	UnhashedPacketsSendCompletedPersec uint64
}

func NewWin32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPortEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPortEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetBroadcastPacketsReceivedPersec sets the value of BroadcastPacketsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) SetPropertyBroadcastPacketsReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("BroadcastPacketsReceivedPersec", (value))
}

// GetBroadcastPacketsReceivedPersec gets the value of BroadcastPacketsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) GetPropertyBroadcastPacketsReceivedPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) SetPropertyBroadcastPacketsSentPersec(value uint64) (err error) {
	return instance.SetProperty("BroadcastPacketsSentPersec", (value))
}

// GetBroadcastPacketsSentPersec gets the value of BroadcastPacketsSentPersec for the instance
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) GetPropertyBroadcastPacketsSentPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) SetPropertyBytesPersec(value uint64) (err error) {
	return instance.SetProperty("BytesPersec", (value))
}

// GetBytesPersec gets the value of BytesPersec for the instance
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) GetPropertyBytesPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) SetPropertyBytesReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("BytesReceivedPersec", (value))
}

// GetBytesReceivedPersec gets the value of BytesReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) GetPropertyBytesReceivedPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) SetPropertyBytesSentPersec(value uint64) (err error) {
	return instance.SetProperty("BytesSentPersec", (value))
}

// GetBytesSentPersec gets the value of BytesSentPersec for the instance
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) GetPropertyBytesSentPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) SetPropertyDirectedPacketsReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("DirectedPacketsReceivedPersec", (value))
}

// GetDirectedPacketsReceivedPersec gets the value of DirectedPacketsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) GetPropertyDirectedPacketsReceivedPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) SetPropertyDirectedPacketsSentPersec(value uint64) (err error) {
	return instance.SetProperty("DirectedPacketsSentPersec", (value))
}

// GetDirectedPacketsSentPersec gets the value of DirectedPacketsSentPersec for the instance
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) GetPropertyDirectedPacketsSentPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) SetPropertyDroppedPacketsIncomingPersec(value uint64) (err error) {
	return instance.SetProperty("DroppedPacketsIncomingPersec", (value))
}

// GetDroppedPacketsIncomingPersec gets the value of DroppedPacketsIncomingPersec for the instance
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) GetPropertyDroppedPacketsIncomingPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) SetPropertyDroppedPacketsOutgoingPersec(value uint64) (err error) {
	return instance.SetProperty("DroppedPacketsOutgoingPersec", (value))
}

// GetDroppedPacketsOutgoingPersec gets the value of DroppedPacketsOutgoingPersec for the instance
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) GetPropertyDroppedPacketsOutgoingPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) SetPropertyExtensionsDroppedPacketsIncomingPersec(value uint64) (err error) {
	return instance.SetProperty("ExtensionsDroppedPacketsIncomingPersec", (value))
}

// GetExtensionsDroppedPacketsIncomingPersec gets the value of ExtensionsDroppedPacketsIncomingPersec for the instance
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) GetPropertyExtensionsDroppedPacketsIncomingPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) SetPropertyExtensionsDroppedPacketsOutgoingPersec(value uint64) (err error) {
	return instance.SetProperty("ExtensionsDroppedPacketsOutgoingPersec", (value))
}

// GetExtensionsDroppedPacketsOutgoingPersec gets the value of ExtensionsDroppedPacketsOutgoingPersec for the instance
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) GetPropertyExtensionsDroppedPacketsOutgoingPersec() (value uint64, err error) {
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

// SetIPsecoffloadBytesReceivePersec sets the value of IPsecoffloadBytesReceivePersec for the instance
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) SetPropertyIPsecoffloadBytesReceivePersec(value uint64) (err error) {
	return instance.SetProperty("IPsecoffloadBytesReceivePersec", (value))
}

// GetIPsecoffloadBytesReceivePersec gets the value of IPsecoffloadBytesReceivePersec for the instance
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) GetPropertyIPsecoffloadBytesReceivePersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("IPsecoffloadBytesReceivePersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetIPsecoffloadBytesSentPersec sets the value of IPsecoffloadBytesSentPersec for the instance
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) SetPropertyIPsecoffloadBytesSentPersec(value uint64) (err error) {
	return instance.SetProperty("IPsecoffloadBytesSentPersec", (value))
}

// GetIPsecoffloadBytesSentPersec gets the value of IPsecoffloadBytesSentPersec for the instance
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) GetPropertyIPsecoffloadBytesSentPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("IPsecoffloadBytesSentPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetIPsecSAsOffloaded sets the value of IPsecSAsOffloaded for the instance
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) SetPropertyIPsecSAsOffloaded(value uint64) (err error) {
	return instance.SetProperty("IPsecSAsOffloaded", (value))
}

// GetIPsecSAsOffloaded gets the value of IPsecSAsOffloaded for the instance
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) GetPropertyIPsecSAsOffloaded() (value uint64, err error) {
	retValue, err := instance.GetProperty("IPsecSAsOffloaded")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
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
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) SetPropertyMulticastPacketsReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("MulticastPacketsReceivedPersec", (value))
}

// GetMulticastPacketsReceivedPersec gets the value of MulticastPacketsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) GetPropertyMulticastPacketsReceivedPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) SetPropertyMulticastPacketsSentPersec(value uint64) (err error) {
	return instance.SetProperty("MulticastPacketsSentPersec", (value))
}

// GetMulticastPacketsSentPersec gets the value of MulticastPacketsSentPersec for the instance
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) GetPropertyMulticastPacketsSentPersec() (value uint64, err error) {
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

// SetPacketsPersec sets the value of PacketsPersec for the instance
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) SetPropertyPacketsPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsPersec", (value))
}

// GetPacketsPersec gets the value of PacketsPersec for the instance
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) GetPropertyPacketsPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) SetPropertyPacketsReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsReceivedPersec", (value))
}

// GetPacketsReceivedPersec gets the value of PacketsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) GetPropertyPacketsReceivedPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) SetPropertyPacketsSentPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsSentPersec", (value))
}

// GetPacketsSentPersec gets the value of PacketsSentPersec for the instance
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) GetPropertyPacketsSentPersec() (value uint64, err error) {
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

// SetUnhashedPacketsReceivedPersec sets the value of UnhashedPacketsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) SetPropertyUnhashedPacketsReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("UnhashedPacketsReceivedPersec", (value))
}

// GetUnhashedPacketsReceivedPersec gets the value of UnhashedPacketsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) GetPropertyUnhashedPacketsReceivedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("UnhashedPacketsReceivedPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetUnhashedPacketsSendCompletedPersec sets the value of UnhashedPacketsSendCompletedPersec for the instance
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) SetPropertyUnhashedPacketsSendCompletedPersec(value uint64) (err error) {
	return instance.SetProperty("UnhashedPacketsSendCompletedPersec", (value))
}

// GetUnhashedPacketsSendCompletedPersec gets the value of UnhashedPacketsSendCompletedPersec for the instance
func (instance *Win32_PerfFormattedData_NvspPortStats_HyperVVirtualSwitchPort) GetPropertyUnhashedPacketsSendCompletedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("UnhashedPacketsSendCompletedPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}
