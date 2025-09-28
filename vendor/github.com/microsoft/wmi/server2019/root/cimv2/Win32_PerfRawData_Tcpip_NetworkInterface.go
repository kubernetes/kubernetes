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

// Win32_PerfRawData_Tcpip_NetworkInterface struct
type Win32_PerfRawData_Tcpip_NetworkInterface struct {
	*Win32_PerfRawData

	//
	BytesReceivedPersec uint64

	//
	BytesSentPersec uint64

	//
	BytesTotalPersec uint64

	//
	CurrentBandwidth uint64

	//
	OffloadedConnections uint64

	//
	OutputQueueLength uint64

	//
	PacketsOutboundDiscarded uint64

	//
	PacketsOutboundErrors uint64

	//
	PacketsPersec uint64

	//
	PacketsReceivedDiscarded uint64

	//
	PacketsReceivedErrors uint64

	//
	PacketsReceivedNonUnicastPersec uint64

	//
	PacketsReceivedPersec uint64

	//
	PacketsReceivedUnicastPersec uint64

	//
	PacketsReceivedUnknown uint64

	//
	PacketsSentNonUnicastPersec uint64

	//
	PacketsSentPersec uint64

	//
	PacketsSentUnicastPersec uint64

	//
	TCPActiveRSCConnections uint64

	//
	TCPRSCAveragePacketSize uint64

	//
	TCPRSCCoalescedPacketsPersec uint64

	//
	TCPRSCExceptionsPersec uint64
}

func NewWin32_PerfRawData_Tcpip_NetworkInterfaceEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_Tcpip_NetworkInterface, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Tcpip_NetworkInterface{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_Tcpip_NetworkInterfaceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_Tcpip_NetworkInterface, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Tcpip_NetworkInterface{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetBytesReceivedPersec sets the value of BytesReceivedPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) SetPropertyBytesReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("BytesReceivedPersec", (value))
}

// GetBytesReceivedPersec gets the value of BytesReceivedPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) GetPropertyBytesReceivedPersec() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) SetPropertyBytesSentPersec(value uint64) (err error) {
	return instance.SetProperty("BytesSentPersec", (value))
}

// GetBytesSentPersec gets the value of BytesSentPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) GetPropertyBytesSentPersec() (value uint64, err error) {
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

// SetBytesTotalPersec sets the value of BytesTotalPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) SetPropertyBytesTotalPersec(value uint64) (err error) {
	return instance.SetProperty("BytesTotalPersec", (value))
}

// GetBytesTotalPersec gets the value of BytesTotalPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) GetPropertyBytesTotalPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesTotalPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetCurrentBandwidth sets the value of CurrentBandwidth for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) SetPropertyCurrentBandwidth(value uint64) (err error) {
	return instance.SetProperty("CurrentBandwidth", (value))
}

// GetCurrentBandwidth gets the value of CurrentBandwidth for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) GetPropertyCurrentBandwidth() (value uint64, err error) {
	retValue, err := instance.GetProperty("CurrentBandwidth")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetOffloadedConnections sets the value of OffloadedConnections for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) SetPropertyOffloadedConnections(value uint64) (err error) {
	return instance.SetProperty("OffloadedConnections", (value))
}

// GetOffloadedConnections gets the value of OffloadedConnections for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) GetPropertyOffloadedConnections() (value uint64, err error) {
	retValue, err := instance.GetProperty("OffloadedConnections")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetOutputQueueLength sets the value of OutputQueueLength for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) SetPropertyOutputQueueLength(value uint64) (err error) {
	return instance.SetProperty("OutputQueueLength", (value))
}

// GetOutputQueueLength gets the value of OutputQueueLength for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) GetPropertyOutputQueueLength() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutputQueueLength")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPacketsOutboundDiscarded sets the value of PacketsOutboundDiscarded for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) SetPropertyPacketsOutboundDiscarded(value uint64) (err error) {
	return instance.SetProperty("PacketsOutboundDiscarded", (value))
}

// GetPacketsOutboundDiscarded gets the value of PacketsOutboundDiscarded for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) GetPropertyPacketsOutboundDiscarded() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsOutboundDiscarded")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPacketsOutboundErrors sets the value of PacketsOutboundErrors for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) SetPropertyPacketsOutboundErrors(value uint64) (err error) {
	return instance.SetProperty("PacketsOutboundErrors", (value))
}

// GetPacketsOutboundErrors gets the value of PacketsOutboundErrors for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) GetPropertyPacketsOutboundErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsOutboundErrors")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
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
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) SetPropertyPacketsPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsPersec", (value))
}

// GetPacketsPersec gets the value of PacketsPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) GetPropertyPacketsPersec() (value uint64, err error) {
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

// SetPacketsReceivedDiscarded sets the value of PacketsReceivedDiscarded for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) SetPropertyPacketsReceivedDiscarded(value uint64) (err error) {
	return instance.SetProperty("PacketsReceivedDiscarded", (value))
}

// GetPacketsReceivedDiscarded gets the value of PacketsReceivedDiscarded for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) GetPropertyPacketsReceivedDiscarded() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsReceivedDiscarded")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPacketsReceivedErrors sets the value of PacketsReceivedErrors for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) SetPropertyPacketsReceivedErrors(value uint64) (err error) {
	return instance.SetProperty("PacketsReceivedErrors", (value))
}

// GetPacketsReceivedErrors gets the value of PacketsReceivedErrors for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) GetPropertyPacketsReceivedErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsReceivedErrors")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPacketsReceivedNonUnicastPersec sets the value of PacketsReceivedNonUnicastPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) SetPropertyPacketsReceivedNonUnicastPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsReceivedNonUnicastPersec", (value))
}

// GetPacketsReceivedNonUnicastPersec gets the value of PacketsReceivedNonUnicastPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) GetPropertyPacketsReceivedNonUnicastPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsReceivedNonUnicastPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
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
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) SetPropertyPacketsReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsReceivedPersec", (value))
}

// GetPacketsReceivedPersec gets the value of PacketsReceivedPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) GetPropertyPacketsReceivedPersec() (value uint64, err error) {
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

// SetPacketsReceivedUnicastPersec sets the value of PacketsReceivedUnicastPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) SetPropertyPacketsReceivedUnicastPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsReceivedUnicastPersec", (value))
}

// GetPacketsReceivedUnicastPersec gets the value of PacketsReceivedUnicastPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) GetPropertyPacketsReceivedUnicastPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsReceivedUnicastPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPacketsReceivedUnknown sets the value of PacketsReceivedUnknown for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) SetPropertyPacketsReceivedUnknown(value uint64) (err error) {
	return instance.SetProperty("PacketsReceivedUnknown", (value))
}

// GetPacketsReceivedUnknown gets the value of PacketsReceivedUnknown for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) GetPropertyPacketsReceivedUnknown() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsReceivedUnknown")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPacketsSentNonUnicastPersec sets the value of PacketsSentNonUnicastPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) SetPropertyPacketsSentNonUnicastPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsSentNonUnicastPersec", (value))
}

// GetPacketsSentNonUnicastPersec gets the value of PacketsSentNonUnicastPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) GetPropertyPacketsSentNonUnicastPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsSentNonUnicastPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
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
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) SetPropertyPacketsSentPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsSentPersec", (value))
}

// GetPacketsSentPersec gets the value of PacketsSentPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) GetPropertyPacketsSentPersec() (value uint64, err error) {
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

// SetPacketsSentUnicastPersec sets the value of PacketsSentUnicastPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) SetPropertyPacketsSentUnicastPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsSentUnicastPersec", (value))
}

// GetPacketsSentUnicastPersec gets the value of PacketsSentUnicastPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) GetPropertyPacketsSentUnicastPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsSentUnicastPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetTCPActiveRSCConnections sets the value of TCPActiveRSCConnections for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) SetPropertyTCPActiveRSCConnections(value uint64) (err error) {
	return instance.SetProperty("TCPActiveRSCConnections", (value))
}

// GetTCPActiveRSCConnections gets the value of TCPActiveRSCConnections for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) GetPropertyTCPActiveRSCConnections() (value uint64, err error) {
	retValue, err := instance.GetProperty("TCPActiveRSCConnections")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetTCPRSCAveragePacketSize sets the value of TCPRSCAveragePacketSize for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) SetPropertyTCPRSCAveragePacketSize(value uint64) (err error) {
	return instance.SetProperty("TCPRSCAveragePacketSize", (value))
}

// GetTCPRSCAveragePacketSize gets the value of TCPRSCAveragePacketSize for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) GetPropertyTCPRSCAveragePacketSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("TCPRSCAveragePacketSize")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetTCPRSCCoalescedPacketsPersec sets the value of TCPRSCCoalescedPacketsPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) SetPropertyTCPRSCCoalescedPacketsPersec(value uint64) (err error) {
	return instance.SetProperty("TCPRSCCoalescedPacketsPersec", (value))
}

// GetTCPRSCCoalescedPacketsPersec gets the value of TCPRSCCoalescedPacketsPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) GetPropertyTCPRSCCoalescedPacketsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("TCPRSCCoalescedPacketsPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetTCPRSCExceptionsPersec sets the value of TCPRSCExceptionsPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) SetPropertyTCPRSCExceptionsPersec(value uint64) (err error) {
	return instance.SetProperty("TCPRSCExceptionsPersec", (value))
}

// GetTCPRSCExceptionsPersec gets the value of TCPRSCExceptionsPersec for the instance
func (instance *Win32_PerfRawData_Tcpip_NetworkInterface) GetPropertyTCPRSCExceptionsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("TCPRSCExceptionsPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}
