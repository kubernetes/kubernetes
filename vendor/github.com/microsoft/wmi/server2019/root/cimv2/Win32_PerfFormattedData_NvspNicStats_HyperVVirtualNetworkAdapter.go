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

// Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter struct
type Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter struct {
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
	MulticastPacketsReceivedPersec uint64

	//
	MulticastPacketsSentPersec uint64

	//
	PacketsFailedSoftwareIPRxCSO uint64

	//
	PacketsFailedSoftwareIPRxCSOPersec uint64

	//
	PacketsFailedSoftwareRxCSOParsingPersec uint64

	//
	PacketsFailedSoftwareTCPRxCSO uint64

	//
	PacketsFailedSoftwareTCPRxCSOPersec uint64

	//
	PacketsFailedSoftwareUDPRxCSO uint64

	//
	PacketsFailedSoftwareUDPRxCSOPersec uint64

	//
	PacketsPassedSoftwareIPRxCSOPersec uint64

	//
	PacketsPassedSoftwareTCPRxCSOPersec uint64

	//
	PacketsPassedSoftwareUDPRxCSOPersec uint64

	//
	PacketsPersec uint64

	//
	PacketsReceivedPersec uint64

	//
	PacketsSentPersec uint64

	//
	PacketsWithSoftwareIPTxCSOPersec uint64

	//
	PacketsWithSoftwareTCPTxCSOPersec uint64

	//
	PacketsWithSoftwareUDPTxCSOPersec uint64

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

func NewWin32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapterEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapterEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetBroadcastPacketsReceivedPersec sets the value of BroadcastPacketsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyBroadcastPacketsReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("BroadcastPacketsReceivedPersec", (value))
}

// GetBroadcastPacketsReceivedPersec gets the value of BroadcastPacketsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyBroadcastPacketsReceivedPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyBroadcastPacketsSentPersec(value uint64) (err error) {
	return instance.SetProperty("BroadcastPacketsSentPersec", (value))
}

// GetBroadcastPacketsSentPersec gets the value of BroadcastPacketsSentPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyBroadcastPacketsSentPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyBytesPersec(value uint64) (err error) {
	return instance.SetProperty("BytesPersec", (value))
}

// GetBytesPersec gets the value of BytesPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyBytesPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyBytesReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("BytesReceivedPersec", (value))
}

// GetBytesReceivedPersec gets the value of BytesReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyBytesReceivedPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyBytesSentPersec(value uint64) (err error) {
	return instance.SetProperty("BytesSentPersec", (value))
}

// GetBytesSentPersec gets the value of BytesSentPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyBytesSentPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyDirectedPacketsReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("DirectedPacketsReceivedPersec", (value))
}

// GetDirectedPacketsReceivedPersec gets the value of DirectedPacketsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyDirectedPacketsReceivedPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyDirectedPacketsSentPersec(value uint64) (err error) {
	return instance.SetProperty("DirectedPacketsSentPersec", (value))
}

// GetDirectedPacketsSentPersec gets the value of DirectedPacketsSentPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyDirectedPacketsSentPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyDroppedPacketsIncomingPersec(value uint64) (err error) {
	return instance.SetProperty("DroppedPacketsIncomingPersec", (value))
}

// GetDroppedPacketsIncomingPersec gets the value of DroppedPacketsIncomingPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyDroppedPacketsIncomingPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyDroppedPacketsOutgoingPersec(value uint64) (err error) {
	return instance.SetProperty("DroppedPacketsOutgoingPersec", (value))
}

// GetDroppedPacketsOutgoingPersec gets the value of DroppedPacketsOutgoingPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyDroppedPacketsOutgoingPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyExtensionsDroppedPacketsIncomingPersec(value uint64) (err error) {
	return instance.SetProperty("ExtensionsDroppedPacketsIncomingPersec", (value))
}

// GetExtensionsDroppedPacketsIncomingPersec gets the value of ExtensionsDroppedPacketsIncomingPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyExtensionsDroppedPacketsIncomingPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyExtensionsDroppedPacketsOutgoingPersec(value uint64) (err error) {
	return instance.SetProperty("ExtensionsDroppedPacketsOutgoingPersec", (value))
}

// GetExtensionsDroppedPacketsOutgoingPersec gets the value of ExtensionsDroppedPacketsOutgoingPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyExtensionsDroppedPacketsOutgoingPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyIPsecoffloadBytesReceivePersec(value uint64) (err error) {
	return instance.SetProperty("IPsecoffloadBytesReceivePersec", (value))
}

// GetIPsecoffloadBytesReceivePersec gets the value of IPsecoffloadBytesReceivePersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyIPsecoffloadBytesReceivePersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyIPsecoffloadBytesSentPersec(value uint64) (err error) {
	return instance.SetProperty("IPsecoffloadBytesSentPersec", (value))
}

// GetIPsecoffloadBytesSentPersec gets the value of IPsecoffloadBytesSentPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyIPsecoffloadBytesSentPersec() (value uint64, err error) {
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

// SetMulticastPacketsReceivedPersec sets the value of MulticastPacketsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyMulticastPacketsReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("MulticastPacketsReceivedPersec", (value))
}

// GetMulticastPacketsReceivedPersec gets the value of MulticastPacketsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyMulticastPacketsReceivedPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyMulticastPacketsSentPersec(value uint64) (err error) {
	return instance.SetProperty("MulticastPacketsSentPersec", (value))
}

// GetMulticastPacketsSentPersec gets the value of MulticastPacketsSentPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyMulticastPacketsSentPersec() (value uint64, err error) {
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

// SetPacketsFailedSoftwareIPRxCSO sets the value of PacketsFailedSoftwareIPRxCSO for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyPacketsFailedSoftwareIPRxCSO(value uint64) (err error) {
	return instance.SetProperty("PacketsFailedSoftwareIPRxCSO", (value))
}

// GetPacketsFailedSoftwareIPRxCSO gets the value of PacketsFailedSoftwareIPRxCSO for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyPacketsFailedSoftwareIPRxCSO() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsFailedSoftwareIPRxCSO")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPacketsFailedSoftwareIPRxCSOPersec sets the value of PacketsFailedSoftwareIPRxCSOPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyPacketsFailedSoftwareIPRxCSOPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsFailedSoftwareIPRxCSOPersec", (value))
}

// GetPacketsFailedSoftwareIPRxCSOPersec gets the value of PacketsFailedSoftwareIPRxCSOPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyPacketsFailedSoftwareIPRxCSOPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsFailedSoftwareIPRxCSOPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPacketsFailedSoftwareRxCSOParsingPersec sets the value of PacketsFailedSoftwareRxCSOParsingPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyPacketsFailedSoftwareRxCSOParsingPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsFailedSoftwareRxCSOParsingPersec", (value))
}

// GetPacketsFailedSoftwareRxCSOParsingPersec gets the value of PacketsFailedSoftwareRxCSOParsingPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyPacketsFailedSoftwareRxCSOParsingPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsFailedSoftwareRxCSOParsingPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPacketsFailedSoftwareTCPRxCSO sets the value of PacketsFailedSoftwareTCPRxCSO for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyPacketsFailedSoftwareTCPRxCSO(value uint64) (err error) {
	return instance.SetProperty("PacketsFailedSoftwareTCPRxCSO", (value))
}

// GetPacketsFailedSoftwareTCPRxCSO gets the value of PacketsFailedSoftwareTCPRxCSO for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyPacketsFailedSoftwareTCPRxCSO() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsFailedSoftwareTCPRxCSO")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPacketsFailedSoftwareTCPRxCSOPersec sets the value of PacketsFailedSoftwareTCPRxCSOPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyPacketsFailedSoftwareTCPRxCSOPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsFailedSoftwareTCPRxCSOPersec", (value))
}

// GetPacketsFailedSoftwareTCPRxCSOPersec gets the value of PacketsFailedSoftwareTCPRxCSOPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyPacketsFailedSoftwareTCPRxCSOPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsFailedSoftwareTCPRxCSOPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPacketsFailedSoftwareUDPRxCSO sets the value of PacketsFailedSoftwareUDPRxCSO for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyPacketsFailedSoftwareUDPRxCSO(value uint64) (err error) {
	return instance.SetProperty("PacketsFailedSoftwareUDPRxCSO", (value))
}

// GetPacketsFailedSoftwareUDPRxCSO gets the value of PacketsFailedSoftwareUDPRxCSO for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyPacketsFailedSoftwareUDPRxCSO() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsFailedSoftwareUDPRxCSO")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPacketsFailedSoftwareUDPRxCSOPersec sets the value of PacketsFailedSoftwareUDPRxCSOPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyPacketsFailedSoftwareUDPRxCSOPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsFailedSoftwareUDPRxCSOPersec", (value))
}

// GetPacketsFailedSoftwareUDPRxCSOPersec gets the value of PacketsFailedSoftwareUDPRxCSOPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyPacketsFailedSoftwareUDPRxCSOPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsFailedSoftwareUDPRxCSOPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPacketsPassedSoftwareIPRxCSOPersec sets the value of PacketsPassedSoftwareIPRxCSOPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyPacketsPassedSoftwareIPRxCSOPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsPassedSoftwareIPRxCSOPersec", (value))
}

// GetPacketsPassedSoftwareIPRxCSOPersec gets the value of PacketsPassedSoftwareIPRxCSOPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyPacketsPassedSoftwareIPRxCSOPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsPassedSoftwareIPRxCSOPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPacketsPassedSoftwareTCPRxCSOPersec sets the value of PacketsPassedSoftwareTCPRxCSOPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyPacketsPassedSoftwareTCPRxCSOPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsPassedSoftwareTCPRxCSOPersec", (value))
}

// GetPacketsPassedSoftwareTCPRxCSOPersec gets the value of PacketsPassedSoftwareTCPRxCSOPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyPacketsPassedSoftwareTCPRxCSOPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsPassedSoftwareTCPRxCSOPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPacketsPassedSoftwareUDPRxCSOPersec sets the value of PacketsPassedSoftwareUDPRxCSOPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyPacketsPassedSoftwareUDPRxCSOPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsPassedSoftwareUDPRxCSOPersec", (value))
}

// GetPacketsPassedSoftwareUDPRxCSOPersec gets the value of PacketsPassedSoftwareUDPRxCSOPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyPacketsPassedSoftwareUDPRxCSOPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsPassedSoftwareUDPRxCSOPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
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
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyPacketsPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsPersec", (value))
}

// GetPacketsPersec gets the value of PacketsPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyPacketsPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyPacketsReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsReceivedPersec", (value))
}

// GetPacketsReceivedPersec gets the value of PacketsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyPacketsReceivedPersec() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyPacketsSentPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsSentPersec", (value))
}

// GetPacketsSentPersec gets the value of PacketsSentPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyPacketsSentPersec() (value uint64, err error) {
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

// SetPacketsWithSoftwareIPTxCSOPersec sets the value of PacketsWithSoftwareIPTxCSOPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyPacketsWithSoftwareIPTxCSOPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsWithSoftwareIPTxCSOPersec", (value))
}

// GetPacketsWithSoftwareIPTxCSOPersec gets the value of PacketsWithSoftwareIPTxCSOPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyPacketsWithSoftwareIPTxCSOPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsWithSoftwareIPTxCSOPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPacketsWithSoftwareTCPTxCSOPersec sets the value of PacketsWithSoftwareTCPTxCSOPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyPacketsWithSoftwareTCPTxCSOPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsWithSoftwareTCPTxCSOPersec", (value))
}

// GetPacketsWithSoftwareTCPTxCSOPersec gets the value of PacketsWithSoftwareTCPTxCSOPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyPacketsWithSoftwareTCPTxCSOPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsWithSoftwareTCPTxCSOPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPacketsWithSoftwareUDPTxCSOPersec sets the value of PacketsWithSoftwareUDPTxCSOPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyPacketsWithSoftwareUDPTxCSOPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsWithSoftwareUDPTxCSOPersec", (value))
}

// GetPacketsWithSoftwareUDPTxCSOPersec gets the value of PacketsWithSoftwareUDPTxCSOPersec for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyPacketsWithSoftwareUDPTxCSOPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsWithSoftwareUDPTxCSOPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
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
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyRSCCoalescedBytes(value uint64) (err error) {
	return instance.SetProperty("RSCCoalescedBytes", (value))
}

// GetRSCCoalescedBytes gets the value of RSCCoalescedBytes for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyRSCCoalescedBytes() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyRSCCoalescedEventBucket10To1(value uint64) (err error) {
	return instance.SetProperty("RSCCoalescedEventBucket10To1", (value))
}

// GetRSCCoalescedEventBucket10To1 gets the value of RSCCoalescedEventBucket10To1 for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyRSCCoalescedEventBucket10To1() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyRSCCoalescedEventBucket22To3(value uint64) (err error) {
	return instance.SetProperty("RSCCoalescedEventBucket22To3", (value))
}

// GetRSCCoalescedEventBucket22To3 gets the value of RSCCoalescedEventBucket22To3 for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyRSCCoalescedEventBucket22To3() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyRSCCoalescedEventBucket34To7(value uint64) (err error) {
	return instance.SetProperty("RSCCoalescedEventBucket34To7", (value))
}

// GetRSCCoalescedEventBucket34To7 gets the value of RSCCoalescedEventBucket34To7 for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyRSCCoalescedEventBucket34To7() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyRSCCoalescedEventBucket48To15(value uint64) (err error) {
	return instance.SetProperty("RSCCoalescedEventBucket48To15", (value))
}

// GetRSCCoalescedEventBucket48To15 gets the value of RSCCoalescedEventBucket48To15 for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyRSCCoalescedEventBucket48To15() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyRSCCoalescedEventBucket516To31(value uint64) (err error) {
	return instance.SetProperty("RSCCoalescedEventBucket516To31", (value))
}

// GetRSCCoalescedEventBucket516To31 gets the value of RSCCoalescedEventBucket516To31 for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyRSCCoalescedEventBucket516To31() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyRSCCoalescedEventBucket632To63(value uint64) (err error) {
	return instance.SetProperty("RSCCoalescedEventBucket632To63", (value))
}

// GetRSCCoalescedEventBucket632To63 gets the value of RSCCoalescedEventBucket632To63 for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyRSCCoalescedEventBucket632To63() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyRSCCoalescedPacketBucket10To1(value uint64) (err error) {
	return instance.SetProperty("RSCCoalescedPacketBucket10To1", (value))
}

// GetRSCCoalescedPacketBucket10To1 gets the value of RSCCoalescedPacketBucket10To1 for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyRSCCoalescedPacketBucket10To1() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyRSCCoalescedPacketBucket22To3(value uint64) (err error) {
	return instance.SetProperty("RSCCoalescedPacketBucket22To3", (value))
}

// GetRSCCoalescedPacketBucket22To3 gets the value of RSCCoalescedPacketBucket22To3 for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyRSCCoalescedPacketBucket22To3() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyRSCCoalescedPacketBucket34To7(value uint64) (err error) {
	return instance.SetProperty("RSCCoalescedPacketBucket34To7", (value))
}

// GetRSCCoalescedPacketBucket34To7 gets the value of RSCCoalescedPacketBucket34To7 for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyRSCCoalescedPacketBucket34To7() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyRSCCoalescedPacketBucket48To15(value uint64) (err error) {
	return instance.SetProperty("RSCCoalescedPacketBucket48To15", (value))
}

// GetRSCCoalescedPacketBucket48To15 gets the value of RSCCoalescedPacketBucket48To15 for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyRSCCoalescedPacketBucket48To15() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyRSCCoalescedPacketBucket516To31(value uint64) (err error) {
	return instance.SetProperty("RSCCoalescedPacketBucket516To31", (value))
}

// GetRSCCoalescedPacketBucket516To31 gets the value of RSCCoalescedPacketBucket516To31 for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyRSCCoalescedPacketBucket516To31() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyRSCCoalescedPacketBucket632To63(value uint64) (err error) {
	return instance.SetProperty("RSCCoalescedPacketBucket632To63", (value))
}

// GetRSCCoalescedPacketBucket632To63 gets the value of RSCCoalescedPacketBucket632To63 for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyRSCCoalescedPacketBucket632To63() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyRSCCoalescedPackets(value uint64) (err error) {
	return instance.SetProperty("RSCCoalescedPackets", (value))
}

// GetRSCCoalescedPackets gets the value of RSCCoalescedPackets for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyRSCCoalescedPackets() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyRSCCoalesceEvents(value uint64) (err error) {
	return instance.SetProperty("RSCCoalesceEvents", (value))
}

// GetRSCCoalesceEvents gets the value of RSCCoalesceEvents for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyRSCCoalesceEvents() (value uint64, err error) {
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
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) SetPropertyRSCPacketsProcessed(value uint64) (err error) {
	return instance.SetProperty("RSCPacketsProcessed", (value))
}

// GetRSCPacketsProcessed gets the value of RSCPacketsProcessed for the instance
func (instance *Win32_PerfFormattedData_NvspNicStats_HyperVVirtualNetworkAdapter) GetPropertyRSCPacketsProcessed() (value uint64, err error) {
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
