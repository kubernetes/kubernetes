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

// Win32_PerfFormattedData_Counters_RemoteFXNetwork struct
type Win32_PerfFormattedData_Counters_RemoteFXNetwork struct {
	*Win32_PerfFormattedData

	//
	BaseTCPRTT uint32

	//
	BaseUDPRTT uint32

	//
	CurrentTCPBandwidth uint32

	//
	CurrentTCPRTT uint32

	//
	CurrentUDPBandwidth uint32

	//
	CurrentUDPRTT uint32

	//
	FECRate uint32

	//
	LossRate uint32

	//
	RetransmissionRate uint32

	//
	SentRateP0 uint32

	//
	SentRateP1 uint32

	//
	SentRateP2 uint32

	//
	SentRateP3 uint32

	//
	TCPReceivedRate uint32

	//
	TCPSentRate uint32

	//
	TotalReceivedBytes uint32

	//
	TotalReceivedRate uint32

	//
	TotalSentBytes uint32

	//
	TotalSentRate uint32

	//
	UDPPacketsReceivedPersec uint32

	//
	UDPPacketsSentPersec uint32

	//
	UDPReceivedRate uint32

	//
	UDPSentRate uint32
}

func NewWin32_PerfFormattedData_Counters_RemoteFXNetworkEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_RemoteFXNetwork, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_RemoteFXNetwork{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_RemoteFXNetworkEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_RemoteFXNetwork, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_RemoteFXNetwork{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetBaseTCPRTT sets the value of BaseTCPRTT for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) SetPropertyBaseTCPRTT(value uint32) (err error) {
	return instance.SetProperty("BaseTCPRTT", (value))
}

// GetBaseTCPRTT gets the value of BaseTCPRTT for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) GetPropertyBaseTCPRTT() (value uint32, err error) {
	retValue, err := instance.GetProperty("BaseTCPRTT")
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

// SetBaseUDPRTT sets the value of BaseUDPRTT for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) SetPropertyBaseUDPRTT(value uint32) (err error) {
	return instance.SetProperty("BaseUDPRTT", (value))
}

// GetBaseUDPRTT gets the value of BaseUDPRTT for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) GetPropertyBaseUDPRTT() (value uint32, err error) {
	retValue, err := instance.GetProperty("BaseUDPRTT")
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

// SetCurrentTCPBandwidth sets the value of CurrentTCPBandwidth for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) SetPropertyCurrentTCPBandwidth(value uint32) (err error) {
	return instance.SetProperty("CurrentTCPBandwidth", (value))
}

// GetCurrentTCPBandwidth gets the value of CurrentTCPBandwidth for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) GetPropertyCurrentTCPBandwidth() (value uint32, err error) {
	retValue, err := instance.GetProperty("CurrentTCPBandwidth")
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

// SetCurrentTCPRTT sets the value of CurrentTCPRTT for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) SetPropertyCurrentTCPRTT(value uint32) (err error) {
	return instance.SetProperty("CurrentTCPRTT", (value))
}

// GetCurrentTCPRTT gets the value of CurrentTCPRTT for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) GetPropertyCurrentTCPRTT() (value uint32, err error) {
	retValue, err := instance.GetProperty("CurrentTCPRTT")
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

// SetCurrentUDPBandwidth sets the value of CurrentUDPBandwidth for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) SetPropertyCurrentUDPBandwidth(value uint32) (err error) {
	return instance.SetProperty("CurrentUDPBandwidth", (value))
}

// GetCurrentUDPBandwidth gets the value of CurrentUDPBandwidth for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) GetPropertyCurrentUDPBandwidth() (value uint32, err error) {
	retValue, err := instance.GetProperty("CurrentUDPBandwidth")
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

// SetCurrentUDPRTT sets the value of CurrentUDPRTT for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) SetPropertyCurrentUDPRTT(value uint32) (err error) {
	return instance.SetProperty("CurrentUDPRTT", (value))
}

// GetCurrentUDPRTT gets the value of CurrentUDPRTT for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) GetPropertyCurrentUDPRTT() (value uint32, err error) {
	retValue, err := instance.GetProperty("CurrentUDPRTT")
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

// SetFECRate sets the value of FECRate for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) SetPropertyFECRate(value uint32) (err error) {
	return instance.SetProperty("FECRate", (value))
}

// GetFECRate gets the value of FECRate for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) GetPropertyFECRate() (value uint32, err error) {
	retValue, err := instance.GetProperty("FECRate")
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

// SetLossRate sets the value of LossRate for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) SetPropertyLossRate(value uint32) (err error) {
	return instance.SetProperty("LossRate", (value))
}

// GetLossRate gets the value of LossRate for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) GetPropertyLossRate() (value uint32, err error) {
	retValue, err := instance.GetProperty("LossRate")
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

// SetRetransmissionRate sets the value of RetransmissionRate for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) SetPropertyRetransmissionRate(value uint32) (err error) {
	return instance.SetProperty("RetransmissionRate", (value))
}

// GetRetransmissionRate gets the value of RetransmissionRate for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) GetPropertyRetransmissionRate() (value uint32, err error) {
	retValue, err := instance.GetProperty("RetransmissionRate")
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

// SetSentRateP0 sets the value of SentRateP0 for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) SetPropertySentRateP0(value uint32) (err error) {
	return instance.SetProperty("SentRateP0", (value))
}

// GetSentRateP0 gets the value of SentRateP0 for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) GetPropertySentRateP0() (value uint32, err error) {
	retValue, err := instance.GetProperty("SentRateP0")
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

// SetSentRateP1 sets the value of SentRateP1 for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) SetPropertySentRateP1(value uint32) (err error) {
	return instance.SetProperty("SentRateP1", (value))
}

// GetSentRateP1 gets the value of SentRateP1 for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) GetPropertySentRateP1() (value uint32, err error) {
	retValue, err := instance.GetProperty("SentRateP1")
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

// SetSentRateP2 sets the value of SentRateP2 for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) SetPropertySentRateP2(value uint32) (err error) {
	return instance.SetProperty("SentRateP2", (value))
}

// GetSentRateP2 gets the value of SentRateP2 for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) GetPropertySentRateP2() (value uint32, err error) {
	retValue, err := instance.GetProperty("SentRateP2")
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

// SetSentRateP3 sets the value of SentRateP3 for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) SetPropertySentRateP3(value uint32) (err error) {
	return instance.SetProperty("SentRateP3", (value))
}

// GetSentRateP3 gets the value of SentRateP3 for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) GetPropertySentRateP3() (value uint32, err error) {
	retValue, err := instance.GetProperty("SentRateP3")
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

// SetTCPReceivedRate sets the value of TCPReceivedRate for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) SetPropertyTCPReceivedRate(value uint32) (err error) {
	return instance.SetProperty("TCPReceivedRate", (value))
}

// GetTCPReceivedRate gets the value of TCPReceivedRate for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) GetPropertyTCPReceivedRate() (value uint32, err error) {
	retValue, err := instance.GetProperty("TCPReceivedRate")
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

// SetTCPSentRate sets the value of TCPSentRate for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) SetPropertyTCPSentRate(value uint32) (err error) {
	return instance.SetProperty("TCPSentRate", (value))
}

// GetTCPSentRate gets the value of TCPSentRate for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) GetPropertyTCPSentRate() (value uint32, err error) {
	retValue, err := instance.GetProperty("TCPSentRate")
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

// SetTotalReceivedBytes sets the value of TotalReceivedBytes for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) SetPropertyTotalReceivedBytes(value uint32) (err error) {
	return instance.SetProperty("TotalReceivedBytes", (value))
}

// GetTotalReceivedBytes gets the value of TotalReceivedBytes for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) GetPropertyTotalReceivedBytes() (value uint32, err error) {
	retValue, err := instance.GetProperty("TotalReceivedBytes")
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

// SetTotalReceivedRate sets the value of TotalReceivedRate for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) SetPropertyTotalReceivedRate(value uint32) (err error) {
	return instance.SetProperty("TotalReceivedRate", (value))
}

// GetTotalReceivedRate gets the value of TotalReceivedRate for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) GetPropertyTotalReceivedRate() (value uint32, err error) {
	retValue, err := instance.GetProperty("TotalReceivedRate")
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

// SetTotalSentBytes sets the value of TotalSentBytes for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) SetPropertyTotalSentBytes(value uint32) (err error) {
	return instance.SetProperty("TotalSentBytes", (value))
}

// GetTotalSentBytes gets the value of TotalSentBytes for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) GetPropertyTotalSentBytes() (value uint32, err error) {
	retValue, err := instance.GetProperty("TotalSentBytes")
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

// SetTotalSentRate sets the value of TotalSentRate for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) SetPropertyTotalSentRate(value uint32) (err error) {
	return instance.SetProperty("TotalSentRate", (value))
}

// GetTotalSentRate gets the value of TotalSentRate for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) GetPropertyTotalSentRate() (value uint32, err error) {
	retValue, err := instance.GetProperty("TotalSentRate")
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

// SetUDPPacketsReceivedPersec sets the value of UDPPacketsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) SetPropertyUDPPacketsReceivedPersec(value uint32) (err error) {
	return instance.SetProperty("UDPPacketsReceivedPersec", (value))
}

// GetUDPPacketsReceivedPersec gets the value of UDPPacketsReceivedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) GetPropertyUDPPacketsReceivedPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("UDPPacketsReceivedPersec")
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

// SetUDPPacketsSentPersec sets the value of UDPPacketsSentPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) SetPropertyUDPPacketsSentPersec(value uint32) (err error) {
	return instance.SetProperty("UDPPacketsSentPersec", (value))
}

// GetUDPPacketsSentPersec gets the value of UDPPacketsSentPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) GetPropertyUDPPacketsSentPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("UDPPacketsSentPersec")
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

// SetUDPReceivedRate sets the value of UDPReceivedRate for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) SetPropertyUDPReceivedRate(value uint32) (err error) {
	return instance.SetProperty("UDPReceivedRate", (value))
}

// GetUDPReceivedRate gets the value of UDPReceivedRate for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) GetPropertyUDPReceivedRate() (value uint32, err error) {
	retValue, err := instance.GetProperty("UDPReceivedRate")
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

// SetUDPSentRate sets the value of UDPSentRate for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) SetPropertyUDPSentRate(value uint32) (err error) {
	return instance.SetProperty("UDPSentRate", (value))
}

// GetUDPSentRate gets the value of UDPSentRate for the instance
func (instance *Win32_PerfFormattedData_Counters_RemoteFXNetwork) GetPropertyUDPSentRate() (value uint32, err error) {
	retValue, err := instance.GetProperty("UDPSentRate")
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
