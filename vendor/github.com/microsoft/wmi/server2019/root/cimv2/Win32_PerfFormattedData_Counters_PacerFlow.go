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

// Win32_PerfFormattedData_Counters_PacerFlow struct
type Win32_PerfFormattedData_Counters_PacerFlow struct {
	*Win32_PerfFormattedData

	//
	Averagepacketsinnetcard uint32

	//
	Averagepacketsinsequencer uint32

	//
	Averagepacketsinshaper uint32

	//
	Bytesscheduled uint64

	//
	BytesscheduledPersec uint64

	//
	Bytestransmitted uint64

	//
	BytestransmittedPersec uint64

	//
	Maximumpacketsinnetcard uint32

	//
	Maxpacketsinsequencer uint32

	//
	Maxpacketsinshaper uint32

	//
	Nonconformingpacketsscheduled uint32

	//
	NonconformingpacketsscheduledPersec uint32

	//
	Nonconformingpacketstransmitted uint32

	//
	NonconformingpacketstransmittedPersec uint32

	//
	Packetsdropped uint32

	//
	PacketsdroppedPersec uint32

	//
	Packetsscheduled uint32

	//
	PacketsscheduledPersec uint32

	//
	Packetstransmitted uint32

	//
	PacketstransmittedPersec uint32
}

func NewWin32_PerfFormattedData_Counters_PacerFlowEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_PacerFlow, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_PacerFlow{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_PacerFlowEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_PacerFlow, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_PacerFlow{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetAveragepacketsinnetcard sets the value of Averagepacketsinnetcard for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) SetPropertyAveragepacketsinnetcard(value uint32) (err error) {
	return instance.SetProperty("Averagepacketsinnetcard", (value))
}

// GetAveragepacketsinnetcard gets the value of Averagepacketsinnetcard for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) GetPropertyAveragepacketsinnetcard() (value uint32, err error) {
	retValue, err := instance.GetProperty("Averagepacketsinnetcard")
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

// SetAveragepacketsinsequencer sets the value of Averagepacketsinsequencer for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) SetPropertyAveragepacketsinsequencer(value uint32) (err error) {
	return instance.SetProperty("Averagepacketsinsequencer", (value))
}

// GetAveragepacketsinsequencer gets the value of Averagepacketsinsequencer for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) GetPropertyAveragepacketsinsequencer() (value uint32, err error) {
	retValue, err := instance.GetProperty("Averagepacketsinsequencer")
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

// SetAveragepacketsinshaper sets the value of Averagepacketsinshaper for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) SetPropertyAveragepacketsinshaper(value uint32) (err error) {
	return instance.SetProperty("Averagepacketsinshaper", (value))
}

// GetAveragepacketsinshaper gets the value of Averagepacketsinshaper for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) GetPropertyAveragepacketsinshaper() (value uint32, err error) {
	retValue, err := instance.GetProperty("Averagepacketsinshaper")
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

// SetBytesscheduled sets the value of Bytesscheduled for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) SetPropertyBytesscheduled(value uint64) (err error) {
	return instance.SetProperty("Bytesscheduled", (value))
}

// GetBytesscheduled gets the value of Bytesscheduled for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) GetPropertyBytesscheduled() (value uint64, err error) {
	retValue, err := instance.GetProperty("Bytesscheduled")
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

// SetBytesscheduledPersec sets the value of BytesscheduledPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) SetPropertyBytesscheduledPersec(value uint64) (err error) {
	return instance.SetProperty("BytesscheduledPersec", (value))
}

// GetBytesscheduledPersec gets the value of BytesscheduledPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) GetPropertyBytesscheduledPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesscheduledPersec")
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

// SetBytestransmitted sets the value of Bytestransmitted for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) SetPropertyBytestransmitted(value uint64) (err error) {
	return instance.SetProperty("Bytestransmitted", (value))
}

// GetBytestransmitted gets the value of Bytestransmitted for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) GetPropertyBytestransmitted() (value uint64, err error) {
	retValue, err := instance.GetProperty("Bytestransmitted")
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

// SetBytestransmittedPersec sets the value of BytestransmittedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) SetPropertyBytestransmittedPersec(value uint64) (err error) {
	return instance.SetProperty("BytestransmittedPersec", (value))
}

// GetBytestransmittedPersec gets the value of BytestransmittedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) GetPropertyBytestransmittedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytestransmittedPersec")
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

// SetMaximumpacketsinnetcard sets the value of Maximumpacketsinnetcard for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) SetPropertyMaximumpacketsinnetcard(value uint32) (err error) {
	return instance.SetProperty("Maximumpacketsinnetcard", (value))
}

// GetMaximumpacketsinnetcard gets the value of Maximumpacketsinnetcard for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) GetPropertyMaximumpacketsinnetcard() (value uint32, err error) {
	retValue, err := instance.GetProperty("Maximumpacketsinnetcard")
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

// SetMaxpacketsinsequencer sets the value of Maxpacketsinsequencer for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) SetPropertyMaxpacketsinsequencer(value uint32) (err error) {
	return instance.SetProperty("Maxpacketsinsequencer", (value))
}

// GetMaxpacketsinsequencer gets the value of Maxpacketsinsequencer for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) GetPropertyMaxpacketsinsequencer() (value uint32, err error) {
	retValue, err := instance.GetProperty("Maxpacketsinsequencer")
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

// SetMaxpacketsinshaper sets the value of Maxpacketsinshaper for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) SetPropertyMaxpacketsinshaper(value uint32) (err error) {
	return instance.SetProperty("Maxpacketsinshaper", (value))
}

// GetMaxpacketsinshaper gets the value of Maxpacketsinshaper for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) GetPropertyMaxpacketsinshaper() (value uint32, err error) {
	retValue, err := instance.GetProperty("Maxpacketsinshaper")
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

// SetNonconformingpacketsscheduled sets the value of Nonconformingpacketsscheduled for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) SetPropertyNonconformingpacketsscheduled(value uint32) (err error) {
	return instance.SetProperty("Nonconformingpacketsscheduled", (value))
}

// GetNonconformingpacketsscheduled gets the value of Nonconformingpacketsscheduled for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) GetPropertyNonconformingpacketsscheduled() (value uint32, err error) {
	retValue, err := instance.GetProperty("Nonconformingpacketsscheduled")
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

// SetNonconformingpacketsscheduledPersec sets the value of NonconformingpacketsscheduledPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) SetPropertyNonconformingpacketsscheduledPersec(value uint32) (err error) {
	return instance.SetProperty("NonconformingpacketsscheduledPersec", (value))
}

// GetNonconformingpacketsscheduledPersec gets the value of NonconformingpacketsscheduledPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) GetPropertyNonconformingpacketsscheduledPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("NonconformingpacketsscheduledPersec")
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

// SetNonconformingpacketstransmitted sets the value of Nonconformingpacketstransmitted for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) SetPropertyNonconformingpacketstransmitted(value uint32) (err error) {
	return instance.SetProperty("Nonconformingpacketstransmitted", (value))
}

// GetNonconformingpacketstransmitted gets the value of Nonconformingpacketstransmitted for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) GetPropertyNonconformingpacketstransmitted() (value uint32, err error) {
	retValue, err := instance.GetProperty("Nonconformingpacketstransmitted")
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

// SetNonconformingpacketstransmittedPersec sets the value of NonconformingpacketstransmittedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) SetPropertyNonconformingpacketstransmittedPersec(value uint32) (err error) {
	return instance.SetProperty("NonconformingpacketstransmittedPersec", (value))
}

// GetNonconformingpacketstransmittedPersec gets the value of NonconformingpacketstransmittedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) GetPropertyNonconformingpacketstransmittedPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("NonconformingpacketstransmittedPersec")
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

// SetPacketsdropped sets the value of Packetsdropped for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) SetPropertyPacketsdropped(value uint32) (err error) {
	return instance.SetProperty("Packetsdropped", (value))
}

// GetPacketsdropped gets the value of Packetsdropped for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) GetPropertyPacketsdropped() (value uint32, err error) {
	retValue, err := instance.GetProperty("Packetsdropped")
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

// SetPacketsdroppedPersec sets the value of PacketsdroppedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) SetPropertyPacketsdroppedPersec(value uint32) (err error) {
	return instance.SetProperty("PacketsdroppedPersec", (value))
}

// GetPacketsdroppedPersec gets the value of PacketsdroppedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) GetPropertyPacketsdroppedPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("PacketsdroppedPersec")
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

// SetPacketsscheduled sets the value of Packetsscheduled for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) SetPropertyPacketsscheduled(value uint32) (err error) {
	return instance.SetProperty("Packetsscheduled", (value))
}

// GetPacketsscheduled gets the value of Packetsscheduled for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) GetPropertyPacketsscheduled() (value uint32, err error) {
	retValue, err := instance.GetProperty("Packetsscheduled")
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

// SetPacketsscheduledPersec sets the value of PacketsscheduledPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) SetPropertyPacketsscheduledPersec(value uint32) (err error) {
	return instance.SetProperty("PacketsscheduledPersec", (value))
}

// GetPacketsscheduledPersec gets the value of PacketsscheduledPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) GetPropertyPacketsscheduledPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("PacketsscheduledPersec")
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

// SetPacketstransmitted sets the value of Packetstransmitted for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) SetPropertyPacketstransmitted(value uint32) (err error) {
	return instance.SetProperty("Packetstransmitted", (value))
}

// GetPacketstransmitted gets the value of Packetstransmitted for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) GetPropertyPacketstransmitted() (value uint32, err error) {
	retValue, err := instance.GetProperty("Packetstransmitted")
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

// SetPacketstransmittedPersec sets the value of PacketstransmittedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) SetPropertyPacketstransmittedPersec(value uint32) (err error) {
	return instance.SetProperty("PacketstransmittedPersec", (value))
}

// GetPacketstransmittedPersec gets the value of PacketstransmittedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerFlow) GetPropertyPacketstransmittedPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("PacketstransmittedPersec")
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
