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

// Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters struct
type Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters struct {
	*Win32_PerfRawData

	//
	CQOverflows uint64

	//
	Devicedetectedstalledstate uint64

	//
	DroplessModeEntries uint64

	//
	DroplessModeExits uint64

	//
	Linkdowneventsphy uint64

	//
	Packetdetectedasstalled uint64

	//
	PacketsdiscardedduetoHeadOfQueuelifetimelimit uint64

	//
	PacketsdiscardedduetoTCinstalledstate uint64

	//
	RequesterCQEErrors uint64

	//
	RequesterInvalidRequestErrors uint64

	//
	RequesterLengthErrors uint64

	//
	RequesterOutoforderSequenceNAK uint64

	//
	RequesterProtectionErrors uint64

	//
	RequesterQPOperationErrors uint64

	//
	RequesterQPTransportRetriesExceededErrors uint64

	//
	RequesterRemoteAccessErrors uint64

	//
	RequesterRemoteOperationErrors uint64

	//
	RequesterRNRNAK uint64

	//
	RequesterRNRNAKRetriesExceededErrors uint64

	//
	RequesterTimeoutReceived uint64

	//
	RequesterTransportRetriesExceededErrors uint64

	//
	ResponderCQEErrors uint64

	//
	ResponderDuplicateRequestReceived uint64

	//
	ResponderInvalidRequestErrors uint64

	//
	ResponderLengthErrors uint64

	//
	ResponderOutoforderSequenceReceived uint64

	//
	ResponderProtectionErrors uint64

	//
	ResponderQPOperationErrors uint64

	//
	ResponderRemoteAccessErrors uint64

	//
	ResponderRNRNAK uint64

	//
	RscAborts uint64

	//
	RscCoalesceEvents uint64

	//
	RscCoalesceOctets uint64

	//
	RscCoalescePackets uint64

	//
	TXCopiedPackets uint64

	//
	TXRingIsFullPackets uint64
}

func NewWin32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCountersEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCountersEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetCQOverflows sets the value of CQOverflows for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyCQOverflows(value uint64) (err error) {
	return instance.SetProperty("CQOverflows", (value))
}

// GetCQOverflows gets the value of CQOverflows for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyCQOverflows() (value uint64, err error) {
	retValue, err := instance.GetProperty("CQOverflows")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetDevicedetectedstalledstate sets the value of Devicedetectedstalledstate for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyDevicedetectedstalledstate(value uint64) (err error) {
	return instance.SetProperty("Devicedetectedstalledstate", (value))
}

// GetDevicedetectedstalledstate gets the value of Devicedetectedstalledstate for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyDevicedetectedstalledstate() (value uint64, err error) {
	retValue, err := instance.GetProperty("Devicedetectedstalledstate")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetDroplessModeEntries sets the value of DroplessModeEntries for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyDroplessModeEntries(value uint64) (err error) {
	return instance.SetProperty("DroplessModeEntries", (value))
}

// GetDroplessModeEntries gets the value of DroplessModeEntries for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyDroplessModeEntries() (value uint64, err error) {
	retValue, err := instance.GetProperty("DroplessModeEntries")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetDroplessModeExits sets the value of DroplessModeExits for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyDroplessModeExits(value uint64) (err error) {
	return instance.SetProperty("DroplessModeExits", (value))
}

// GetDroplessModeExits gets the value of DroplessModeExits for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyDroplessModeExits() (value uint64, err error) {
	retValue, err := instance.GetProperty("DroplessModeExits")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetLinkdowneventsphy sets the value of Linkdowneventsphy for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyLinkdowneventsphy(value uint64) (err error) {
	return instance.SetProperty("Linkdowneventsphy", (value))
}

// GetLinkdowneventsphy gets the value of Linkdowneventsphy for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyLinkdowneventsphy() (value uint64, err error) {
	retValue, err := instance.GetProperty("Linkdowneventsphy")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPacketdetectedasstalled sets the value of Packetdetectedasstalled for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyPacketdetectedasstalled(value uint64) (err error) {
	return instance.SetProperty("Packetdetectedasstalled", (value))
}

// GetPacketdetectedasstalled gets the value of Packetdetectedasstalled for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyPacketdetectedasstalled() (value uint64, err error) {
	retValue, err := instance.GetProperty("Packetdetectedasstalled")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPacketsdiscardedduetoHeadOfQueuelifetimelimit sets the value of PacketsdiscardedduetoHeadOfQueuelifetimelimit for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyPacketsdiscardedduetoHeadOfQueuelifetimelimit(value uint64) (err error) {
	return instance.SetProperty("PacketsdiscardedduetoHeadOfQueuelifetimelimit", (value))
}

// GetPacketsdiscardedduetoHeadOfQueuelifetimelimit gets the value of PacketsdiscardedduetoHeadOfQueuelifetimelimit for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyPacketsdiscardedduetoHeadOfQueuelifetimelimit() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsdiscardedduetoHeadOfQueuelifetimelimit")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPacketsdiscardedduetoTCinstalledstate sets the value of PacketsdiscardedduetoTCinstalledstate for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyPacketsdiscardedduetoTCinstalledstate(value uint64) (err error) {
	return instance.SetProperty("PacketsdiscardedduetoTCinstalledstate", (value))
}

// GetPacketsdiscardedduetoTCinstalledstate gets the value of PacketsdiscardedduetoTCinstalledstate for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyPacketsdiscardedduetoTCinstalledstate() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsdiscardedduetoTCinstalledstate")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetRequesterCQEErrors sets the value of RequesterCQEErrors for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyRequesterCQEErrors(value uint64) (err error) {
	return instance.SetProperty("RequesterCQEErrors", (value))
}

// GetRequesterCQEErrors gets the value of RequesterCQEErrors for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyRequesterCQEErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("RequesterCQEErrors")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetRequesterInvalidRequestErrors sets the value of RequesterInvalidRequestErrors for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyRequesterInvalidRequestErrors(value uint64) (err error) {
	return instance.SetProperty("RequesterInvalidRequestErrors", (value))
}

// GetRequesterInvalidRequestErrors gets the value of RequesterInvalidRequestErrors for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyRequesterInvalidRequestErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("RequesterInvalidRequestErrors")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetRequesterLengthErrors sets the value of RequesterLengthErrors for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyRequesterLengthErrors(value uint64) (err error) {
	return instance.SetProperty("RequesterLengthErrors", (value))
}

// GetRequesterLengthErrors gets the value of RequesterLengthErrors for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyRequesterLengthErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("RequesterLengthErrors")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetRequesterOutoforderSequenceNAK sets the value of RequesterOutoforderSequenceNAK for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyRequesterOutoforderSequenceNAK(value uint64) (err error) {
	return instance.SetProperty("RequesterOutoforderSequenceNAK", (value))
}

// GetRequesterOutoforderSequenceNAK gets the value of RequesterOutoforderSequenceNAK for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyRequesterOutoforderSequenceNAK() (value uint64, err error) {
	retValue, err := instance.GetProperty("RequesterOutoforderSequenceNAK")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetRequesterProtectionErrors sets the value of RequesterProtectionErrors for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyRequesterProtectionErrors(value uint64) (err error) {
	return instance.SetProperty("RequesterProtectionErrors", (value))
}

// GetRequesterProtectionErrors gets the value of RequesterProtectionErrors for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyRequesterProtectionErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("RequesterProtectionErrors")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetRequesterQPOperationErrors sets the value of RequesterQPOperationErrors for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyRequesterQPOperationErrors(value uint64) (err error) {
	return instance.SetProperty("RequesterQPOperationErrors", (value))
}

// GetRequesterQPOperationErrors gets the value of RequesterQPOperationErrors for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyRequesterQPOperationErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("RequesterQPOperationErrors")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetRequesterQPTransportRetriesExceededErrors sets the value of RequesterQPTransportRetriesExceededErrors for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyRequesterQPTransportRetriesExceededErrors(value uint64) (err error) {
	return instance.SetProperty("RequesterQPTransportRetriesExceededErrors", (value))
}

// GetRequesterQPTransportRetriesExceededErrors gets the value of RequesterQPTransportRetriesExceededErrors for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyRequesterQPTransportRetriesExceededErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("RequesterQPTransportRetriesExceededErrors")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetRequesterRemoteAccessErrors sets the value of RequesterRemoteAccessErrors for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyRequesterRemoteAccessErrors(value uint64) (err error) {
	return instance.SetProperty("RequesterRemoteAccessErrors", (value))
}

// GetRequesterRemoteAccessErrors gets the value of RequesterRemoteAccessErrors for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyRequesterRemoteAccessErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("RequesterRemoteAccessErrors")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetRequesterRemoteOperationErrors sets the value of RequesterRemoteOperationErrors for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyRequesterRemoteOperationErrors(value uint64) (err error) {
	return instance.SetProperty("RequesterRemoteOperationErrors", (value))
}

// GetRequesterRemoteOperationErrors gets the value of RequesterRemoteOperationErrors for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyRequesterRemoteOperationErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("RequesterRemoteOperationErrors")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetRequesterRNRNAK sets the value of RequesterRNRNAK for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyRequesterRNRNAK(value uint64) (err error) {
	return instance.SetProperty("RequesterRNRNAK", (value))
}

// GetRequesterRNRNAK gets the value of RequesterRNRNAK for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyRequesterRNRNAK() (value uint64, err error) {
	retValue, err := instance.GetProperty("RequesterRNRNAK")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetRequesterRNRNAKRetriesExceededErrors sets the value of RequesterRNRNAKRetriesExceededErrors for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyRequesterRNRNAKRetriesExceededErrors(value uint64) (err error) {
	return instance.SetProperty("RequesterRNRNAKRetriesExceededErrors", (value))
}

// GetRequesterRNRNAKRetriesExceededErrors gets the value of RequesterRNRNAKRetriesExceededErrors for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyRequesterRNRNAKRetriesExceededErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("RequesterRNRNAKRetriesExceededErrors")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetRequesterTimeoutReceived sets the value of RequesterTimeoutReceived for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyRequesterTimeoutReceived(value uint64) (err error) {
	return instance.SetProperty("RequesterTimeoutReceived", (value))
}

// GetRequesterTimeoutReceived gets the value of RequesterTimeoutReceived for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyRequesterTimeoutReceived() (value uint64, err error) {
	retValue, err := instance.GetProperty("RequesterTimeoutReceived")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetRequesterTransportRetriesExceededErrors sets the value of RequesterTransportRetriesExceededErrors for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyRequesterTransportRetriesExceededErrors(value uint64) (err error) {
	return instance.SetProperty("RequesterTransportRetriesExceededErrors", (value))
}

// GetRequesterTransportRetriesExceededErrors gets the value of RequesterTransportRetriesExceededErrors for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyRequesterTransportRetriesExceededErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("RequesterTransportRetriesExceededErrors")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetResponderCQEErrors sets the value of ResponderCQEErrors for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyResponderCQEErrors(value uint64) (err error) {
	return instance.SetProperty("ResponderCQEErrors", (value))
}

// GetResponderCQEErrors gets the value of ResponderCQEErrors for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyResponderCQEErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("ResponderCQEErrors")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetResponderDuplicateRequestReceived sets the value of ResponderDuplicateRequestReceived for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyResponderDuplicateRequestReceived(value uint64) (err error) {
	return instance.SetProperty("ResponderDuplicateRequestReceived", (value))
}

// GetResponderDuplicateRequestReceived gets the value of ResponderDuplicateRequestReceived for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyResponderDuplicateRequestReceived() (value uint64, err error) {
	retValue, err := instance.GetProperty("ResponderDuplicateRequestReceived")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetResponderInvalidRequestErrors sets the value of ResponderInvalidRequestErrors for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyResponderInvalidRequestErrors(value uint64) (err error) {
	return instance.SetProperty("ResponderInvalidRequestErrors", (value))
}

// GetResponderInvalidRequestErrors gets the value of ResponderInvalidRequestErrors for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyResponderInvalidRequestErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("ResponderInvalidRequestErrors")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetResponderLengthErrors sets the value of ResponderLengthErrors for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyResponderLengthErrors(value uint64) (err error) {
	return instance.SetProperty("ResponderLengthErrors", (value))
}

// GetResponderLengthErrors gets the value of ResponderLengthErrors for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyResponderLengthErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("ResponderLengthErrors")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetResponderOutoforderSequenceReceived sets the value of ResponderOutoforderSequenceReceived for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyResponderOutoforderSequenceReceived(value uint64) (err error) {
	return instance.SetProperty("ResponderOutoforderSequenceReceived", (value))
}

// GetResponderOutoforderSequenceReceived gets the value of ResponderOutoforderSequenceReceived for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyResponderOutoforderSequenceReceived() (value uint64, err error) {
	retValue, err := instance.GetProperty("ResponderOutoforderSequenceReceived")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetResponderProtectionErrors sets the value of ResponderProtectionErrors for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyResponderProtectionErrors(value uint64) (err error) {
	return instance.SetProperty("ResponderProtectionErrors", (value))
}

// GetResponderProtectionErrors gets the value of ResponderProtectionErrors for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyResponderProtectionErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("ResponderProtectionErrors")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetResponderQPOperationErrors sets the value of ResponderQPOperationErrors for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyResponderQPOperationErrors(value uint64) (err error) {
	return instance.SetProperty("ResponderQPOperationErrors", (value))
}

// GetResponderQPOperationErrors gets the value of ResponderQPOperationErrors for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyResponderQPOperationErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("ResponderQPOperationErrors")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetResponderRemoteAccessErrors sets the value of ResponderRemoteAccessErrors for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyResponderRemoteAccessErrors(value uint64) (err error) {
	return instance.SetProperty("ResponderRemoteAccessErrors", (value))
}

// GetResponderRemoteAccessErrors gets the value of ResponderRemoteAccessErrors for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyResponderRemoteAccessErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("ResponderRemoteAccessErrors")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetResponderRNRNAK sets the value of ResponderRNRNAK for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyResponderRNRNAK(value uint64) (err error) {
	return instance.SetProperty("ResponderRNRNAK", (value))
}

// GetResponderRNRNAK gets the value of ResponderRNRNAK for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyResponderRNRNAK() (value uint64, err error) {
	retValue, err := instance.GetProperty("ResponderRNRNAK")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetRscAborts sets the value of RscAborts for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyRscAborts(value uint64) (err error) {
	return instance.SetProperty("RscAborts", (value))
}

// GetRscAborts gets the value of RscAborts for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyRscAborts() (value uint64, err error) {
	retValue, err := instance.GetProperty("RscAborts")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetRscCoalesceEvents sets the value of RscCoalesceEvents for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyRscCoalesceEvents(value uint64) (err error) {
	return instance.SetProperty("RscCoalesceEvents", (value))
}

// GetRscCoalesceEvents gets the value of RscCoalesceEvents for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyRscCoalesceEvents() (value uint64, err error) {
	retValue, err := instance.GetProperty("RscCoalesceEvents")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetRscCoalesceOctets sets the value of RscCoalesceOctets for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyRscCoalesceOctets(value uint64) (err error) {
	return instance.SetProperty("RscCoalesceOctets", (value))
}

// GetRscCoalesceOctets gets the value of RscCoalesceOctets for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyRscCoalesceOctets() (value uint64, err error) {
	retValue, err := instance.GetProperty("RscCoalesceOctets")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetRscCoalescePackets sets the value of RscCoalescePackets for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyRscCoalescePackets(value uint64) (err error) {
	return instance.SetProperty("RscCoalescePackets", (value))
}

// GetRscCoalescePackets gets the value of RscCoalescePackets for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyRscCoalescePackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("RscCoalescePackets")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetTXCopiedPackets sets the value of TXCopiedPackets for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyTXCopiedPackets(value uint64) (err error) {
	return instance.SetProperty("TXCopiedPackets", (value))
}

// GetTXCopiedPackets gets the value of TXCopiedPackets for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyTXCopiedPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TXCopiedPackets")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetTXRingIsFullPackets sets the value of TXRingIsFullPackets for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) SetPropertyTXRingIsFullPackets(value uint64) (err error) {
	return instance.SetProperty("TXRingIsFullPackets", (value))
}

// GetTXRingIsFullPackets gets the value of TXRingIsFullPackets for the instance
func (instance *Win32_PerfRawData_Mlx4EthProvider_MellanoxAdapterDiagnosticCounters) GetPropertyTXRingIsFullPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TXRingIsFullPackets")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}
