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

// Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity struct
type Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity struct {
	*Win32_PerfRawData

	//
	BuildScatterGatherListCallsPersec uint64

	//
	DPCsDeferredPersec uint64

	//
	DPCsQueuedonOtherCPUsPersec uint64

	//
	DPCsQueuedPersec uint64

	//
	InterruptsPersec uint64

	//
	LowResourceReceivedPacketsPersec uint64

	//
	LowResourceReceiveIndicationsPersec uint64

	//
	PacketsCoalescedPersec uint64

	//
	ReceivedPacketsPersec uint64

	//
	ReceiveIndicationsPersec uint64

	//
	ReturnedPacketsPersec uint64

	//
	ReturnPacketCallsPersec uint64

	//
	RSSIndirectionTableChangeCallsPersec uint64

	//
	SendCompleteCallsPersec uint64

	//
	SendRequestCallsPersec uint64

	//
	SentCompletePacketsPersec uint64

	//
	SentPacketsPersec uint64

	//
	TcpOffloadReceivebytesPersec uint64

	//
	TcpOffloadReceiveIndicationsPersec uint64

	//
	TcpOffloadSendbytesPersec uint64

	//
	TcpOffloadSendRequestCallsPersec uint64
}

func NewWin32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivityEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivityEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetBuildScatterGatherListCallsPersec sets the value of BuildScatterGatherListCallsPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) SetPropertyBuildScatterGatherListCallsPersec(value uint64) (err error) {
	return instance.SetProperty("BuildScatterGatherListCallsPersec", (value))
}

// GetBuildScatterGatherListCallsPersec gets the value of BuildScatterGatherListCallsPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) GetPropertyBuildScatterGatherListCallsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("BuildScatterGatherListCallsPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetDPCsDeferredPersec sets the value of DPCsDeferredPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) SetPropertyDPCsDeferredPersec(value uint64) (err error) {
	return instance.SetProperty("DPCsDeferredPersec", (value))
}

// GetDPCsDeferredPersec gets the value of DPCsDeferredPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) GetPropertyDPCsDeferredPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("DPCsDeferredPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetDPCsQueuedonOtherCPUsPersec sets the value of DPCsQueuedonOtherCPUsPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) SetPropertyDPCsQueuedonOtherCPUsPersec(value uint64) (err error) {
	return instance.SetProperty("DPCsQueuedonOtherCPUsPersec", (value))
}

// GetDPCsQueuedonOtherCPUsPersec gets the value of DPCsQueuedonOtherCPUsPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) GetPropertyDPCsQueuedonOtherCPUsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("DPCsQueuedonOtherCPUsPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetDPCsQueuedPersec sets the value of DPCsQueuedPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) SetPropertyDPCsQueuedPersec(value uint64) (err error) {
	return instance.SetProperty("DPCsQueuedPersec", (value))
}

// GetDPCsQueuedPersec gets the value of DPCsQueuedPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) GetPropertyDPCsQueuedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("DPCsQueuedPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetInterruptsPersec sets the value of InterruptsPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) SetPropertyInterruptsPersec(value uint64) (err error) {
	return instance.SetProperty("InterruptsPersec", (value))
}

// GetInterruptsPersec gets the value of InterruptsPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) GetPropertyInterruptsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("InterruptsPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetLowResourceReceivedPacketsPersec sets the value of LowResourceReceivedPacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) SetPropertyLowResourceReceivedPacketsPersec(value uint64) (err error) {
	return instance.SetProperty("LowResourceReceivedPacketsPersec", (value))
}

// GetLowResourceReceivedPacketsPersec gets the value of LowResourceReceivedPacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) GetPropertyLowResourceReceivedPacketsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("LowResourceReceivedPacketsPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetLowResourceReceiveIndicationsPersec sets the value of LowResourceReceiveIndicationsPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) SetPropertyLowResourceReceiveIndicationsPersec(value uint64) (err error) {
	return instance.SetProperty("LowResourceReceiveIndicationsPersec", (value))
}

// GetLowResourceReceiveIndicationsPersec gets the value of LowResourceReceiveIndicationsPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) GetPropertyLowResourceReceiveIndicationsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("LowResourceReceiveIndicationsPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPacketsCoalescedPersec sets the value of PacketsCoalescedPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) SetPropertyPacketsCoalescedPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsCoalescedPersec", (value))
}

// GetPacketsCoalescedPersec gets the value of PacketsCoalescedPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) GetPropertyPacketsCoalescedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsCoalescedPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetReceivedPacketsPersec sets the value of ReceivedPacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) SetPropertyReceivedPacketsPersec(value uint64) (err error) {
	return instance.SetProperty("ReceivedPacketsPersec", (value))
}

// GetReceivedPacketsPersec gets the value of ReceivedPacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) GetPropertyReceivedPacketsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReceivedPacketsPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetReceiveIndicationsPersec sets the value of ReceiveIndicationsPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) SetPropertyReceiveIndicationsPersec(value uint64) (err error) {
	return instance.SetProperty("ReceiveIndicationsPersec", (value))
}

// GetReceiveIndicationsPersec gets the value of ReceiveIndicationsPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) GetPropertyReceiveIndicationsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReceiveIndicationsPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetReturnedPacketsPersec sets the value of ReturnedPacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) SetPropertyReturnedPacketsPersec(value uint64) (err error) {
	return instance.SetProperty("ReturnedPacketsPersec", (value))
}

// GetReturnedPacketsPersec gets the value of ReturnedPacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) GetPropertyReturnedPacketsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReturnedPacketsPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetReturnPacketCallsPersec sets the value of ReturnPacketCallsPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) SetPropertyReturnPacketCallsPersec(value uint64) (err error) {
	return instance.SetProperty("ReturnPacketCallsPersec", (value))
}

// GetReturnPacketCallsPersec gets the value of ReturnPacketCallsPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) GetPropertyReturnPacketCallsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReturnPacketCallsPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetRSSIndirectionTableChangeCallsPersec sets the value of RSSIndirectionTableChangeCallsPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) SetPropertyRSSIndirectionTableChangeCallsPersec(value uint64) (err error) {
	return instance.SetProperty("RSSIndirectionTableChangeCallsPersec", (value))
}

// GetRSSIndirectionTableChangeCallsPersec gets the value of RSSIndirectionTableChangeCallsPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) GetPropertyRSSIndirectionTableChangeCallsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("RSSIndirectionTableChangeCallsPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetSendCompleteCallsPersec sets the value of SendCompleteCallsPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) SetPropertySendCompleteCallsPersec(value uint64) (err error) {
	return instance.SetProperty("SendCompleteCallsPersec", (value))
}

// GetSendCompleteCallsPersec gets the value of SendCompleteCallsPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) GetPropertySendCompleteCallsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("SendCompleteCallsPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetSendRequestCallsPersec sets the value of SendRequestCallsPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) SetPropertySendRequestCallsPersec(value uint64) (err error) {
	return instance.SetProperty("SendRequestCallsPersec", (value))
}

// GetSendRequestCallsPersec gets the value of SendRequestCallsPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) GetPropertySendRequestCallsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("SendRequestCallsPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetSentCompletePacketsPersec sets the value of SentCompletePacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) SetPropertySentCompletePacketsPersec(value uint64) (err error) {
	return instance.SetProperty("SentCompletePacketsPersec", (value))
}

// GetSentCompletePacketsPersec gets the value of SentCompletePacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) GetPropertySentCompletePacketsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("SentCompletePacketsPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetSentPacketsPersec sets the value of SentPacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) SetPropertySentPacketsPersec(value uint64) (err error) {
	return instance.SetProperty("SentPacketsPersec", (value))
}

// GetSentPacketsPersec gets the value of SentPacketsPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) GetPropertySentPacketsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("SentPacketsPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetTcpOffloadReceivebytesPersec sets the value of TcpOffloadReceivebytesPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) SetPropertyTcpOffloadReceivebytesPersec(value uint64) (err error) {
	return instance.SetProperty("TcpOffloadReceivebytesPersec", (value))
}

// GetTcpOffloadReceivebytesPersec gets the value of TcpOffloadReceivebytesPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) GetPropertyTcpOffloadReceivebytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("TcpOffloadReceivebytesPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetTcpOffloadReceiveIndicationsPersec sets the value of TcpOffloadReceiveIndicationsPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) SetPropertyTcpOffloadReceiveIndicationsPersec(value uint64) (err error) {
	return instance.SetProperty("TcpOffloadReceiveIndicationsPersec", (value))
}

// GetTcpOffloadReceiveIndicationsPersec gets the value of TcpOffloadReceiveIndicationsPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) GetPropertyTcpOffloadReceiveIndicationsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("TcpOffloadReceiveIndicationsPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetTcpOffloadSendbytesPersec sets the value of TcpOffloadSendbytesPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) SetPropertyTcpOffloadSendbytesPersec(value uint64) (err error) {
	return instance.SetProperty("TcpOffloadSendbytesPersec", (value))
}

// GetTcpOffloadSendbytesPersec gets the value of TcpOffloadSendbytesPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) GetPropertyTcpOffloadSendbytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("TcpOffloadSendbytesPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetTcpOffloadSendRequestCallsPersec sets the value of TcpOffloadSendRequestCallsPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) SetPropertyTcpOffloadSendRequestCallsPersec(value uint64) (err error) {
	return instance.SetProperty("TcpOffloadSendRequestCallsPersec", (value))
}

// GetTcpOffloadSendRequestCallsPersec gets the value of TcpOffloadSendRequestCallsPersec for the instance
func (instance *Win32_PerfRawData_Counters_PerProcessorNetworkInterfaceCardActivity) GetPropertyTcpOffloadSendRequestCallsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("TcpOffloadSendRequestCallsPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}
