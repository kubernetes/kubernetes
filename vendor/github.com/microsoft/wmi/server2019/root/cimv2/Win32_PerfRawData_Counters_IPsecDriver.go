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

// Win32_PerfRawData_Counters_IPsecDriver struct
type Win32_PerfRawData_Counters_IPsecDriver struct {
	*Win32_PerfRawData

	//
	ActiveSecurityAssociations uint32

	//
	BytesReceivedinTransportModePersec uint32

	//
	BytesReceivedinTunnelModePersec uint32

	//
	BytesSentinTransportModePersec uint32

	//
	BytesSentinTunnelModePersec uint32

	//
	InboundPacketsDroppedPersec uint32

	//
	InboundPacketsReceivedPersec uint32

	//
	IncorrectSPIPackets uint32

	//
	IncorrectSPIPacketsPersec uint32

	//
	OffloadedBytesReceivedPersec uint32

	//
	OffloadedBytesSentPersec uint32

	//
	OffloadedSecurityAssociations uint32

	//
	PacketsNotAuthenticated uint32

	//
	PacketsNotAuthenticatedPersec uint32

	//
	PacketsNotDecrypted uint32

	//
	PacketsNotDecryptedPersec uint32

	//
	PacketsReceivedOverWrongSA uint32

	//
	PacketsReceivedOverWrongSAPersec uint32

	//
	PacketsThatFailedESPValidation uint32

	//
	PacketsThatFailedESPValidationPersec uint32

	//
	PacketsThatFailedReplayDetection uint32

	//
	PacketsThatFailedReplayDetectionPersec uint32

	//
	PacketsThatFailedUDPESPValidation uint32

	//
	PacketsThatFailedUDPESPValidationPersec uint32

	//
	PendingSecurityAssociations uint32

	//
	PlaintextPacketsReceived uint32

	//
	PlaintextPacketsReceivedPersec uint32

	//
	SARekeys uint32

	//
	SecurityAssociationsAdded uint32

	//
	TotalInboundPacketsDropped uint32

	//
	TotalInboundPacketsReceived uint32
}

func NewWin32_PerfRawData_Counters_IPsecDriverEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_Counters_IPsecDriver, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_IPsecDriver{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_Counters_IPsecDriverEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_Counters_IPsecDriver, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_IPsecDriver{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetActiveSecurityAssociations sets the value of ActiveSecurityAssociations for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) SetPropertyActiveSecurityAssociations(value uint32) (err error) {
	return instance.SetProperty("ActiveSecurityAssociations", (value))
}

// GetActiveSecurityAssociations gets the value of ActiveSecurityAssociations for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) GetPropertyActiveSecurityAssociations() (value uint32, err error) {
	retValue, err := instance.GetProperty("ActiveSecurityAssociations")
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

// SetBytesReceivedinTransportModePersec sets the value of BytesReceivedinTransportModePersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) SetPropertyBytesReceivedinTransportModePersec(value uint32) (err error) {
	return instance.SetProperty("BytesReceivedinTransportModePersec", (value))
}

// GetBytesReceivedinTransportModePersec gets the value of BytesReceivedinTransportModePersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) GetPropertyBytesReceivedinTransportModePersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("BytesReceivedinTransportModePersec")
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

// SetBytesReceivedinTunnelModePersec sets the value of BytesReceivedinTunnelModePersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) SetPropertyBytesReceivedinTunnelModePersec(value uint32) (err error) {
	return instance.SetProperty("BytesReceivedinTunnelModePersec", (value))
}

// GetBytesReceivedinTunnelModePersec gets the value of BytesReceivedinTunnelModePersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) GetPropertyBytesReceivedinTunnelModePersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("BytesReceivedinTunnelModePersec")
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

// SetBytesSentinTransportModePersec sets the value of BytesSentinTransportModePersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) SetPropertyBytesSentinTransportModePersec(value uint32) (err error) {
	return instance.SetProperty("BytesSentinTransportModePersec", (value))
}

// GetBytesSentinTransportModePersec gets the value of BytesSentinTransportModePersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) GetPropertyBytesSentinTransportModePersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("BytesSentinTransportModePersec")
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

// SetBytesSentinTunnelModePersec sets the value of BytesSentinTunnelModePersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) SetPropertyBytesSentinTunnelModePersec(value uint32) (err error) {
	return instance.SetProperty("BytesSentinTunnelModePersec", (value))
}

// GetBytesSentinTunnelModePersec gets the value of BytesSentinTunnelModePersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) GetPropertyBytesSentinTunnelModePersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("BytesSentinTunnelModePersec")
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

// SetInboundPacketsDroppedPersec sets the value of InboundPacketsDroppedPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) SetPropertyInboundPacketsDroppedPersec(value uint32) (err error) {
	return instance.SetProperty("InboundPacketsDroppedPersec", (value))
}

// GetInboundPacketsDroppedPersec gets the value of InboundPacketsDroppedPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) GetPropertyInboundPacketsDroppedPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("InboundPacketsDroppedPersec")
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

// SetInboundPacketsReceivedPersec sets the value of InboundPacketsReceivedPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) SetPropertyInboundPacketsReceivedPersec(value uint32) (err error) {
	return instance.SetProperty("InboundPacketsReceivedPersec", (value))
}

// GetInboundPacketsReceivedPersec gets the value of InboundPacketsReceivedPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) GetPropertyInboundPacketsReceivedPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("InboundPacketsReceivedPersec")
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

// SetIncorrectSPIPackets sets the value of IncorrectSPIPackets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) SetPropertyIncorrectSPIPackets(value uint32) (err error) {
	return instance.SetProperty("IncorrectSPIPackets", (value))
}

// GetIncorrectSPIPackets gets the value of IncorrectSPIPackets for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) GetPropertyIncorrectSPIPackets() (value uint32, err error) {
	retValue, err := instance.GetProperty("IncorrectSPIPackets")
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

// SetIncorrectSPIPacketsPersec sets the value of IncorrectSPIPacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) SetPropertyIncorrectSPIPacketsPersec(value uint32) (err error) {
	return instance.SetProperty("IncorrectSPIPacketsPersec", (value))
}

// GetIncorrectSPIPacketsPersec gets the value of IncorrectSPIPacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) GetPropertyIncorrectSPIPacketsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("IncorrectSPIPacketsPersec")
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

// SetOffloadedBytesReceivedPersec sets the value of OffloadedBytesReceivedPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) SetPropertyOffloadedBytesReceivedPersec(value uint32) (err error) {
	return instance.SetProperty("OffloadedBytesReceivedPersec", (value))
}

// GetOffloadedBytesReceivedPersec gets the value of OffloadedBytesReceivedPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) GetPropertyOffloadedBytesReceivedPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("OffloadedBytesReceivedPersec")
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

// SetOffloadedBytesSentPersec sets the value of OffloadedBytesSentPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) SetPropertyOffloadedBytesSentPersec(value uint32) (err error) {
	return instance.SetProperty("OffloadedBytesSentPersec", (value))
}

// GetOffloadedBytesSentPersec gets the value of OffloadedBytesSentPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) GetPropertyOffloadedBytesSentPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("OffloadedBytesSentPersec")
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

// SetOffloadedSecurityAssociations sets the value of OffloadedSecurityAssociations for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) SetPropertyOffloadedSecurityAssociations(value uint32) (err error) {
	return instance.SetProperty("OffloadedSecurityAssociations", (value))
}

// GetOffloadedSecurityAssociations gets the value of OffloadedSecurityAssociations for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) GetPropertyOffloadedSecurityAssociations() (value uint32, err error) {
	retValue, err := instance.GetProperty("OffloadedSecurityAssociations")
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

// SetPacketsNotAuthenticated sets the value of PacketsNotAuthenticated for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) SetPropertyPacketsNotAuthenticated(value uint32) (err error) {
	return instance.SetProperty("PacketsNotAuthenticated", (value))
}

// GetPacketsNotAuthenticated gets the value of PacketsNotAuthenticated for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) GetPropertyPacketsNotAuthenticated() (value uint32, err error) {
	retValue, err := instance.GetProperty("PacketsNotAuthenticated")
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

// SetPacketsNotAuthenticatedPersec sets the value of PacketsNotAuthenticatedPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) SetPropertyPacketsNotAuthenticatedPersec(value uint32) (err error) {
	return instance.SetProperty("PacketsNotAuthenticatedPersec", (value))
}

// GetPacketsNotAuthenticatedPersec gets the value of PacketsNotAuthenticatedPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) GetPropertyPacketsNotAuthenticatedPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("PacketsNotAuthenticatedPersec")
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

// SetPacketsNotDecrypted sets the value of PacketsNotDecrypted for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) SetPropertyPacketsNotDecrypted(value uint32) (err error) {
	return instance.SetProperty("PacketsNotDecrypted", (value))
}

// GetPacketsNotDecrypted gets the value of PacketsNotDecrypted for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) GetPropertyPacketsNotDecrypted() (value uint32, err error) {
	retValue, err := instance.GetProperty("PacketsNotDecrypted")
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

// SetPacketsNotDecryptedPersec sets the value of PacketsNotDecryptedPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) SetPropertyPacketsNotDecryptedPersec(value uint32) (err error) {
	return instance.SetProperty("PacketsNotDecryptedPersec", (value))
}

// GetPacketsNotDecryptedPersec gets the value of PacketsNotDecryptedPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) GetPropertyPacketsNotDecryptedPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("PacketsNotDecryptedPersec")
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

// SetPacketsReceivedOverWrongSA sets the value of PacketsReceivedOverWrongSA for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) SetPropertyPacketsReceivedOverWrongSA(value uint32) (err error) {
	return instance.SetProperty("PacketsReceivedOverWrongSA", (value))
}

// GetPacketsReceivedOverWrongSA gets the value of PacketsReceivedOverWrongSA for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) GetPropertyPacketsReceivedOverWrongSA() (value uint32, err error) {
	retValue, err := instance.GetProperty("PacketsReceivedOverWrongSA")
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

// SetPacketsReceivedOverWrongSAPersec sets the value of PacketsReceivedOverWrongSAPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) SetPropertyPacketsReceivedOverWrongSAPersec(value uint32) (err error) {
	return instance.SetProperty("PacketsReceivedOverWrongSAPersec", (value))
}

// GetPacketsReceivedOverWrongSAPersec gets the value of PacketsReceivedOverWrongSAPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) GetPropertyPacketsReceivedOverWrongSAPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("PacketsReceivedOverWrongSAPersec")
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

// SetPacketsThatFailedESPValidation sets the value of PacketsThatFailedESPValidation for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) SetPropertyPacketsThatFailedESPValidation(value uint32) (err error) {
	return instance.SetProperty("PacketsThatFailedESPValidation", (value))
}

// GetPacketsThatFailedESPValidation gets the value of PacketsThatFailedESPValidation for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) GetPropertyPacketsThatFailedESPValidation() (value uint32, err error) {
	retValue, err := instance.GetProperty("PacketsThatFailedESPValidation")
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

// SetPacketsThatFailedESPValidationPersec sets the value of PacketsThatFailedESPValidationPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) SetPropertyPacketsThatFailedESPValidationPersec(value uint32) (err error) {
	return instance.SetProperty("PacketsThatFailedESPValidationPersec", (value))
}

// GetPacketsThatFailedESPValidationPersec gets the value of PacketsThatFailedESPValidationPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) GetPropertyPacketsThatFailedESPValidationPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("PacketsThatFailedESPValidationPersec")
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

// SetPacketsThatFailedReplayDetection sets the value of PacketsThatFailedReplayDetection for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) SetPropertyPacketsThatFailedReplayDetection(value uint32) (err error) {
	return instance.SetProperty("PacketsThatFailedReplayDetection", (value))
}

// GetPacketsThatFailedReplayDetection gets the value of PacketsThatFailedReplayDetection for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) GetPropertyPacketsThatFailedReplayDetection() (value uint32, err error) {
	retValue, err := instance.GetProperty("PacketsThatFailedReplayDetection")
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

// SetPacketsThatFailedReplayDetectionPersec sets the value of PacketsThatFailedReplayDetectionPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) SetPropertyPacketsThatFailedReplayDetectionPersec(value uint32) (err error) {
	return instance.SetProperty("PacketsThatFailedReplayDetectionPersec", (value))
}

// GetPacketsThatFailedReplayDetectionPersec gets the value of PacketsThatFailedReplayDetectionPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) GetPropertyPacketsThatFailedReplayDetectionPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("PacketsThatFailedReplayDetectionPersec")
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

// SetPacketsThatFailedUDPESPValidation sets the value of PacketsThatFailedUDPESPValidation for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) SetPropertyPacketsThatFailedUDPESPValidation(value uint32) (err error) {
	return instance.SetProperty("PacketsThatFailedUDPESPValidation", (value))
}

// GetPacketsThatFailedUDPESPValidation gets the value of PacketsThatFailedUDPESPValidation for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) GetPropertyPacketsThatFailedUDPESPValidation() (value uint32, err error) {
	retValue, err := instance.GetProperty("PacketsThatFailedUDPESPValidation")
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

// SetPacketsThatFailedUDPESPValidationPersec sets the value of PacketsThatFailedUDPESPValidationPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) SetPropertyPacketsThatFailedUDPESPValidationPersec(value uint32) (err error) {
	return instance.SetProperty("PacketsThatFailedUDPESPValidationPersec", (value))
}

// GetPacketsThatFailedUDPESPValidationPersec gets the value of PacketsThatFailedUDPESPValidationPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) GetPropertyPacketsThatFailedUDPESPValidationPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("PacketsThatFailedUDPESPValidationPersec")
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

// SetPendingSecurityAssociations sets the value of PendingSecurityAssociations for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) SetPropertyPendingSecurityAssociations(value uint32) (err error) {
	return instance.SetProperty("PendingSecurityAssociations", (value))
}

// GetPendingSecurityAssociations gets the value of PendingSecurityAssociations for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) GetPropertyPendingSecurityAssociations() (value uint32, err error) {
	retValue, err := instance.GetProperty("PendingSecurityAssociations")
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

// SetPlaintextPacketsReceived sets the value of PlaintextPacketsReceived for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) SetPropertyPlaintextPacketsReceived(value uint32) (err error) {
	return instance.SetProperty("PlaintextPacketsReceived", (value))
}

// GetPlaintextPacketsReceived gets the value of PlaintextPacketsReceived for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) GetPropertyPlaintextPacketsReceived() (value uint32, err error) {
	retValue, err := instance.GetProperty("PlaintextPacketsReceived")
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

// SetPlaintextPacketsReceivedPersec sets the value of PlaintextPacketsReceivedPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) SetPropertyPlaintextPacketsReceivedPersec(value uint32) (err error) {
	return instance.SetProperty("PlaintextPacketsReceivedPersec", (value))
}

// GetPlaintextPacketsReceivedPersec gets the value of PlaintextPacketsReceivedPersec for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) GetPropertyPlaintextPacketsReceivedPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("PlaintextPacketsReceivedPersec")
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

// SetSARekeys sets the value of SARekeys for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) SetPropertySARekeys(value uint32) (err error) {
	return instance.SetProperty("SARekeys", (value))
}

// GetSARekeys gets the value of SARekeys for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) GetPropertySARekeys() (value uint32, err error) {
	retValue, err := instance.GetProperty("SARekeys")
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

// SetSecurityAssociationsAdded sets the value of SecurityAssociationsAdded for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) SetPropertySecurityAssociationsAdded(value uint32) (err error) {
	return instance.SetProperty("SecurityAssociationsAdded", (value))
}

// GetSecurityAssociationsAdded gets the value of SecurityAssociationsAdded for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) GetPropertySecurityAssociationsAdded() (value uint32, err error) {
	retValue, err := instance.GetProperty("SecurityAssociationsAdded")
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

// SetTotalInboundPacketsDropped sets the value of TotalInboundPacketsDropped for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) SetPropertyTotalInboundPacketsDropped(value uint32) (err error) {
	return instance.SetProperty("TotalInboundPacketsDropped", (value))
}

// GetTotalInboundPacketsDropped gets the value of TotalInboundPacketsDropped for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) GetPropertyTotalInboundPacketsDropped() (value uint32, err error) {
	retValue, err := instance.GetProperty("TotalInboundPacketsDropped")
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

// SetTotalInboundPacketsReceived sets the value of TotalInboundPacketsReceived for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) SetPropertyTotalInboundPacketsReceived(value uint32) (err error) {
	return instance.SetProperty("TotalInboundPacketsReceived", (value))
}

// GetTotalInboundPacketsReceived gets the value of TotalInboundPacketsReceived for the instance
func (instance *Win32_PerfRawData_Counters_IPsecDriver) GetPropertyTotalInboundPacketsReceived() (value uint32, err error) {
	retValue, err := instance.GetProperty("TotalInboundPacketsReceived")
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
