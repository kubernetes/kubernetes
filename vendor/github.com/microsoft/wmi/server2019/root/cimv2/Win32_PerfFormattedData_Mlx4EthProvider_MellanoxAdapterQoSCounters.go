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

// Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters struct
type Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters struct {
	*Win32_PerfFormattedData

	//
	BytesReceived uint64

	//
	BytesSent uint64

	//
	BytesTotal uint64

	//
	KBytesReceivedPerSec uint32

	//
	KBytesSentPerSec uint32

	//
	KBytesTotalPerSec uint32

	//
	PacketsReceived uint64

	//
	PacketsReceivedPerSec uint32

	//
	PacketsSent uint64

	//
	PacketsSentPerSec uint32

	//
	PacketsTotal uint64

	//
	PacketsTotalPerSec uint32

	//
	RcvPauseDuration uint64

	//
	RcvPauseFrames uint64

	//
	RequesterAllocatedRateLimiters uint64

	//
	RequesterAverageTotalRate uint64

	//
	RequesterCurrentTotalRate uint64

	//
	RequesterIgnoredLimitationRequest uint64

	//
	RequesterSuccessfullyHandledLimitationRequest uint64

	//
	RequesterTotalAllocatedRateLimiters uint32

	//
	RequesterTrafficRateHighPeak uint64

	//
	RequesterTrafficRateLowPeak uint64

	//
	ResponderActiveCNP uint64

	//
	ResponderCNPSentSuccessfully uint64

	//
	ResponderECNHandledSuccessfully uint64

	//
	ResponderIgnoredECN uint64

	//
	ResponderIgnoredECNdueCNPcoalesce uint64

	//
	SentDiscardFrames uint64

	//
	SentPauseDuration uint64

	//
	SentPauseFrames uint64
}

func NewWin32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCountersEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCountersEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetBytesReceived sets the value of BytesReceived for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) SetPropertyBytesReceived(value uint64) (err error) {
	return instance.SetProperty("BytesReceived", (value))
}

// GetBytesReceived gets the value of BytesReceived for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) GetPropertyBytesReceived() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesReceived")
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

// SetBytesSent sets the value of BytesSent for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) SetPropertyBytesSent(value uint64) (err error) {
	return instance.SetProperty("BytesSent", (value))
}

// GetBytesSent gets the value of BytesSent for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) GetPropertyBytesSent() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesSent")
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

// SetBytesTotal sets the value of BytesTotal for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) SetPropertyBytesTotal(value uint64) (err error) {
	return instance.SetProperty("BytesTotal", (value))
}

// GetBytesTotal gets the value of BytesTotal for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) GetPropertyBytesTotal() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesTotal")
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

// SetKBytesReceivedPerSec sets the value of KBytesReceivedPerSec for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) SetPropertyKBytesReceivedPerSec(value uint32) (err error) {
	return instance.SetProperty("KBytesReceivedPerSec", (value))
}

// GetKBytesReceivedPerSec gets the value of KBytesReceivedPerSec for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) GetPropertyKBytesReceivedPerSec() (value uint32, err error) {
	retValue, err := instance.GetProperty("KBytesReceivedPerSec")
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

// SetKBytesSentPerSec sets the value of KBytesSentPerSec for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) SetPropertyKBytesSentPerSec(value uint32) (err error) {
	return instance.SetProperty("KBytesSentPerSec", (value))
}

// GetKBytesSentPerSec gets the value of KBytesSentPerSec for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) GetPropertyKBytesSentPerSec() (value uint32, err error) {
	retValue, err := instance.GetProperty("KBytesSentPerSec")
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

// SetKBytesTotalPerSec sets the value of KBytesTotalPerSec for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) SetPropertyKBytesTotalPerSec(value uint32) (err error) {
	return instance.SetProperty("KBytesTotalPerSec", (value))
}

// GetKBytesTotalPerSec gets the value of KBytesTotalPerSec for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) GetPropertyKBytesTotalPerSec() (value uint32, err error) {
	retValue, err := instance.GetProperty("KBytesTotalPerSec")
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

// SetPacketsReceived sets the value of PacketsReceived for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) SetPropertyPacketsReceived(value uint64) (err error) {
	return instance.SetProperty("PacketsReceived", (value))
}

// GetPacketsReceived gets the value of PacketsReceived for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) GetPropertyPacketsReceived() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsReceived")
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

// SetPacketsReceivedPerSec sets the value of PacketsReceivedPerSec for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) SetPropertyPacketsReceivedPerSec(value uint32) (err error) {
	return instance.SetProperty("PacketsReceivedPerSec", (value))
}

// GetPacketsReceivedPerSec gets the value of PacketsReceivedPerSec for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) GetPropertyPacketsReceivedPerSec() (value uint32, err error) {
	retValue, err := instance.GetProperty("PacketsReceivedPerSec")
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

// SetPacketsSent sets the value of PacketsSent for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) SetPropertyPacketsSent(value uint64) (err error) {
	return instance.SetProperty("PacketsSent", (value))
}

// GetPacketsSent gets the value of PacketsSent for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) GetPropertyPacketsSent() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsSent")
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

// SetPacketsSentPerSec sets the value of PacketsSentPerSec for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) SetPropertyPacketsSentPerSec(value uint32) (err error) {
	return instance.SetProperty("PacketsSentPerSec", (value))
}

// GetPacketsSentPerSec gets the value of PacketsSentPerSec for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) GetPropertyPacketsSentPerSec() (value uint32, err error) {
	retValue, err := instance.GetProperty("PacketsSentPerSec")
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

// SetPacketsTotal sets the value of PacketsTotal for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) SetPropertyPacketsTotal(value uint64) (err error) {
	return instance.SetProperty("PacketsTotal", (value))
}

// GetPacketsTotal gets the value of PacketsTotal for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) GetPropertyPacketsTotal() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsTotal")
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

// SetPacketsTotalPerSec sets the value of PacketsTotalPerSec for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) SetPropertyPacketsTotalPerSec(value uint32) (err error) {
	return instance.SetProperty("PacketsTotalPerSec", (value))
}

// GetPacketsTotalPerSec gets the value of PacketsTotalPerSec for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) GetPropertyPacketsTotalPerSec() (value uint32, err error) {
	retValue, err := instance.GetProperty("PacketsTotalPerSec")
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

// SetRcvPauseDuration sets the value of RcvPauseDuration for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) SetPropertyRcvPauseDuration(value uint64) (err error) {
	return instance.SetProperty("RcvPauseDuration", (value))
}

// GetRcvPauseDuration gets the value of RcvPauseDuration for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) GetPropertyRcvPauseDuration() (value uint64, err error) {
	retValue, err := instance.GetProperty("RcvPauseDuration")
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

// SetRcvPauseFrames sets the value of RcvPauseFrames for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) SetPropertyRcvPauseFrames(value uint64) (err error) {
	return instance.SetProperty("RcvPauseFrames", (value))
}

// GetRcvPauseFrames gets the value of RcvPauseFrames for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) GetPropertyRcvPauseFrames() (value uint64, err error) {
	retValue, err := instance.GetProperty("RcvPauseFrames")
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

// SetRequesterAllocatedRateLimiters sets the value of RequesterAllocatedRateLimiters for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) SetPropertyRequesterAllocatedRateLimiters(value uint64) (err error) {
	return instance.SetProperty("RequesterAllocatedRateLimiters", (value))
}

// GetRequesterAllocatedRateLimiters gets the value of RequesterAllocatedRateLimiters for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) GetPropertyRequesterAllocatedRateLimiters() (value uint64, err error) {
	retValue, err := instance.GetProperty("RequesterAllocatedRateLimiters")
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

// SetRequesterAverageTotalRate sets the value of RequesterAverageTotalRate for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) SetPropertyRequesterAverageTotalRate(value uint64) (err error) {
	return instance.SetProperty("RequesterAverageTotalRate", (value))
}

// GetRequesterAverageTotalRate gets the value of RequesterAverageTotalRate for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) GetPropertyRequesterAverageTotalRate() (value uint64, err error) {
	retValue, err := instance.GetProperty("RequesterAverageTotalRate")
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

// SetRequesterCurrentTotalRate sets the value of RequesterCurrentTotalRate for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) SetPropertyRequesterCurrentTotalRate(value uint64) (err error) {
	return instance.SetProperty("RequesterCurrentTotalRate", (value))
}

// GetRequesterCurrentTotalRate gets the value of RequesterCurrentTotalRate for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) GetPropertyRequesterCurrentTotalRate() (value uint64, err error) {
	retValue, err := instance.GetProperty("RequesterCurrentTotalRate")
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

// SetRequesterIgnoredLimitationRequest sets the value of RequesterIgnoredLimitationRequest for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) SetPropertyRequesterIgnoredLimitationRequest(value uint64) (err error) {
	return instance.SetProperty("RequesterIgnoredLimitationRequest", (value))
}

// GetRequesterIgnoredLimitationRequest gets the value of RequesterIgnoredLimitationRequest for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) GetPropertyRequesterIgnoredLimitationRequest() (value uint64, err error) {
	retValue, err := instance.GetProperty("RequesterIgnoredLimitationRequest")
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

// SetRequesterSuccessfullyHandledLimitationRequest sets the value of RequesterSuccessfullyHandledLimitationRequest for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) SetPropertyRequesterSuccessfullyHandledLimitationRequest(value uint64) (err error) {
	return instance.SetProperty("RequesterSuccessfullyHandledLimitationRequest", (value))
}

// GetRequesterSuccessfullyHandledLimitationRequest gets the value of RequesterSuccessfullyHandledLimitationRequest for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) GetPropertyRequesterSuccessfullyHandledLimitationRequest() (value uint64, err error) {
	retValue, err := instance.GetProperty("RequesterSuccessfullyHandledLimitationRequest")
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

// SetRequesterTotalAllocatedRateLimiters sets the value of RequesterTotalAllocatedRateLimiters for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) SetPropertyRequesterTotalAllocatedRateLimiters(value uint32) (err error) {
	return instance.SetProperty("RequesterTotalAllocatedRateLimiters", (value))
}

// GetRequesterTotalAllocatedRateLimiters gets the value of RequesterTotalAllocatedRateLimiters for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) GetPropertyRequesterTotalAllocatedRateLimiters() (value uint32, err error) {
	retValue, err := instance.GetProperty("RequesterTotalAllocatedRateLimiters")
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

// SetRequesterTrafficRateHighPeak sets the value of RequesterTrafficRateHighPeak for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) SetPropertyRequesterTrafficRateHighPeak(value uint64) (err error) {
	return instance.SetProperty("RequesterTrafficRateHighPeak", (value))
}

// GetRequesterTrafficRateHighPeak gets the value of RequesterTrafficRateHighPeak for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) GetPropertyRequesterTrafficRateHighPeak() (value uint64, err error) {
	retValue, err := instance.GetProperty("RequesterTrafficRateHighPeak")
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

// SetRequesterTrafficRateLowPeak sets the value of RequesterTrafficRateLowPeak for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) SetPropertyRequesterTrafficRateLowPeak(value uint64) (err error) {
	return instance.SetProperty("RequesterTrafficRateLowPeak", (value))
}

// GetRequesterTrafficRateLowPeak gets the value of RequesterTrafficRateLowPeak for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) GetPropertyRequesterTrafficRateLowPeak() (value uint64, err error) {
	retValue, err := instance.GetProperty("RequesterTrafficRateLowPeak")
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

// SetResponderActiveCNP sets the value of ResponderActiveCNP for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) SetPropertyResponderActiveCNP(value uint64) (err error) {
	return instance.SetProperty("ResponderActiveCNP", (value))
}

// GetResponderActiveCNP gets the value of ResponderActiveCNP for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) GetPropertyResponderActiveCNP() (value uint64, err error) {
	retValue, err := instance.GetProperty("ResponderActiveCNP")
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

// SetResponderCNPSentSuccessfully sets the value of ResponderCNPSentSuccessfully for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) SetPropertyResponderCNPSentSuccessfully(value uint64) (err error) {
	return instance.SetProperty("ResponderCNPSentSuccessfully", (value))
}

// GetResponderCNPSentSuccessfully gets the value of ResponderCNPSentSuccessfully for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) GetPropertyResponderCNPSentSuccessfully() (value uint64, err error) {
	retValue, err := instance.GetProperty("ResponderCNPSentSuccessfully")
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

// SetResponderECNHandledSuccessfully sets the value of ResponderECNHandledSuccessfully for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) SetPropertyResponderECNHandledSuccessfully(value uint64) (err error) {
	return instance.SetProperty("ResponderECNHandledSuccessfully", (value))
}

// GetResponderECNHandledSuccessfully gets the value of ResponderECNHandledSuccessfully for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) GetPropertyResponderECNHandledSuccessfully() (value uint64, err error) {
	retValue, err := instance.GetProperty("ResponderECNHandledSuccessfully")
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

// SetResponderIgnoredECN sets the value of ResponderIgnoredECN for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) SetPropertyResponderIgnoredECN(value uint64) (err error) {
	return instance.SetProperty("ResponderIgnoredECN", (value))
}

// GetResponderIgnoredECN gets the value of ResponderIgnoredECN for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) GetPropertyResponderIgnoredECN() (value uint64, err error) {
	retValue, err := instance.GetProperty("ResponderIgnoredECN")
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

// SetResponderIgnoredECNdueCNPcoalesce sets the value of ResponderIgnoredECNdueCNPcoalesce for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) SetPropertyResponderIgnoredECNdueCNPcoalesce(value uint64) (err error) {
	return instance.SetProperty("ResponderIgnoredECNdueCNPcoalesce", (value))
}

// GetResponderIgnoredECNdueCNPcoalesce gets the value of ResponderIgnoredECNdueCNPcoalesce for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) GetPropertyResponderIgnoredECNdueCNPcoalesce() (value uint64, err error) {
	retValue, err := instance.GetProperty("ResponderIgnoredECNdueCNPcoalesce")
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

// SetSentDiscardFrames sets the value of SentDiscardFrames for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) SetPropertySentDiscardFrames(value uint64) (err error) {
	return instance.SetProperty("SentDiscardFrames", (value))
}

// GetSentDiscardFrames gets the value of SentDiscardFrames for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) GetPropertySentDiscardFrames() (value uint64, err error) {
	retValue, err := instance.GetProperty("SentDiscardFrames")
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

// SetSentPauseDuration sets the value of SentPauseDuration for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) SetPropertySentPauseDuration(value uint64) (err error) {
	return instance.SetProperty("SentPauseDuration", (value))
}

// GetSentPauseDuration gets the value of SentPauseDuration for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) GetPropertySentPauseDuration() (value uint64, err error) {
	retValue, err := instance.GetProperty("SentPauseDuration")
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

// SetSentPauseFrames sets the value of SentPauseFrames for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) SetPropertySentPauseFrames(value uint64) (err error) {
	return instance.SetProperty("SentPauseFrames", (value))
}

// GetSentPauseFrames gets the value of SentPauseFrames for the instance
func (instance *Win32_PerfFormattedData_Mlx4EthProvider_MellanoxAdapterQoSCounters) GetPropertySentPauseFrames() (value uint64, err error) {
	retValue, err := instance.GetProperty("SentPauseFrames")
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
