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

// Win32_PerfFormattedData_Counters_TeredoRelay struct
type Win32_PerfFormattedData_Counters_TeredoRelay struct {
	*Win32_PerfFormattedData

	//
	InTeredoRelayErrorPacketsDestinationError uint32

	//
	InTeredoRelayErrorPacketsHeaderError uint32

	//
	InTeredoRelayErrorPacketsSourceError uint32

	//
	InTeredoRelayErrorPacketsTotal uint32

	//
	InTeredoRelaySuccessPacketsBubbles uint32

	//
	InTeredoRelaySuccessPacketsDataPackets uint64

	//
	InTeredoRelaySuccessPacketsDataPacketsKernelMode uint64

	//
	InTeredoRelaySuccessPacketsDataPacketsUserMode uint64

	//
	InTeredoRelaySuccessPacketsTotal uint64

	//
	InTeredoRelayTotalPacketsSuccessError uint32

	//
	InTeredoRelayTotalPacketsSuccessErrorPersec uint32

	//
	OutTeredoRelayErrorPackets uint32

	//
	OutTeredoRelayErrorPacketsDestinationError uint32

	//
	OutTeredoRelayErrorPacketsHeaderError uint32

	//
	OutTeredoRelayErrorPacketsSourceError uint32

	//
	OutTeredoRelaySuccessPackets uint64

	//
	OutTeredoRelaySuccessPacketsBubbles uint32

	//
	OutTeredoRelaySuccessPacketsDataPackets uint64

	//
	OutTeredoRelaySuccessPacketsDataPacketsKernelMode uint64

	//
	OutTeredoRelaySuccessPacketsDataPacketsUserMode uint64

	//
	OutTeredoRelayTotalPacketsSuccessError uint32

	//
	OutTeredoRelayTotalPacketsSuccessErrorPersec uint32
}

func NewWin32_PerfFormattedData_Counters_TeredoRelayEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_TeredoRelay, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_TeredoRelay{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_TeredoRelayEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_TeredoRelay, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_TeredoRelay{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetInTeredoRelayErrorPacketsDestinationError sets the value of InTeredoRelayErrorPacketsDestinationError for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) SetPropertyInTeredoRelayErrorPacketsDestinationError(value uint32) (err error) {
	return instance.SetProperty("InTeredoRelayErrorPacketsDestinationError", (value))
}

// GetInTeredoRelayErrorPacketsDestinationError gets the value of InTeredoRelayErrorPacketsDestinationError for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) GetPropertyInTeredoRelayErrorPacketsDestinationError() (value uint32, err error) {
	retValue, err := instance.GetProperty("InTeredoRelayErrorPacketsDestinationError")
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

// SetInTeredoRelayErrorPacketsHeaderError sets the value of InTeredoRelayErrorPacketsHeaderError for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) SetPropertyInTeredoRelayErrorPacketsHeaderError(value uint32) (err error) {
	return instance.SetProperty("InTeredoRelayErrorPacketsHeaderError", (value))
}

// GetInTeredoRelayErrorPacketsHeaderError gets the value of InTeredoRelayErrorPacketsHeaderError for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) GetPropertyInTeredoRelayErrorPacketsHeaderError() (value uint32, err error) {
	retValue, err := instance.GetProperty("InTeredoRelayErrorPacketsHeaderError")
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

// SetInTeredoRelayErrorPacketsSourceError sets the value of InTeredoRelayErrorPacketsSourceError for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) SetPropertyInTeredoRelayErrorPacketsSourceError(value uint32) (err error) {
	return instance.SetProperty("InTeredoRelayErrorPacketsSourceError", (value))
}

// GetInTeredoRelayErrorPacketsSourceError gets the value of InTeredoRelayErrorPacketsSourceError for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) GetPropertyInTeredoRelayErrorPacketsSourceError() (value uint32, err error) {
	retValue, err := instance.GetProperty("InTeredoRelayErrorPacketsSourceError")
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

// SetInTeredoRelayErrorPacketsTotal sets the value of InTeredoRelayErrorPacketsTotal for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) SetPropertyInTeredoRelayErrorPacketsTotal(value uint32) (err error) {
	return instance.SetProperty("InTeredoRelayErrorPacketsTotal", (value))
}

// GetInTeredoRelayErrorPacketsTotal gets the value of InTeredoRelayErrorPacketsTotal for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) GetPropertyInTeredoRelayErrorPacketsTotal() (value uint32, err error) {
	retValue, err := instance.GetProperty("InTeredoRelayErrorPacketsTotal")
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

// SetInTeredoRelaySuccessPacketsBubbles sets the value of InTeredoRelaySuccessPacketsBubbles for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) SetPropertyInTeredoRelaySuccessPacketsBubbles(value uint32) (err error) {
	return instance.SetProperty("InTeredoRelaySuccessPacketsBubbles", (value))
}

// GetInTeredoRelaySuccessPacketsBubbles gets the value of InTeredoRelaySuccessPacketsBubbles for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) GetPropertyInTeredoRelaySuccessPacketsBubbles() (value uint32, err error) {
	retValue, err := instance.GetProperty("InTeredoRelaySuccessPacketsBubbles")
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

// SetInTeredoRelaySuccessPacketsDataPackets sets the value of InTeredoRelaySuccessPacketsDataPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) SetPropertyInTeredoRelaySuccessPacketsDataPackets(value uint64) (err error) {
	return instance.SetProperty("InTeredoRelaySuccessPacketsDataPackets", (value))
}

// GetInTeredoRelaySuccessPacketsDataPackets gets the value of InTeredoRelaySuccessPacketsDataPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) GetPropertyInTeredoRelaySuccessPacketsDataPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("InTeredoRelaySuccessPacketsDataPackets")
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

// SetInTeredoRelaySuccessPacketsDataPacketsKernelMode sets the value of InTeredoRelaySuccessPacketsDataPacketsKernelMode for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) SetPropertyInTeredoRelaySuccessPacketsDataPacketsKernelMode(value uint64) (err error) {
	return instance.SetProperty("InTeredoRelaySuccessPacketsDataPacketsKernelMode", (value))
}

// GetInTeredoRelaySuccessPacketsDataPacketsKernelMode gets the value of InTeredoRelaySuccessPacketsDataPacketsKernelMode for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) GetPropertyInTeredoRelaySuccessPacketsDataPacketsKernelMode() (value uint64, err error) {
	retValue, err := instance.GetProperty("InTeredoRelaySuccessPacketsDataPacketsKernelMode")
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

// SetInTeredoRelaySuccessPacketsDataPacketsUserMode sets the value of InTeredoRelaySuccessPacketsDataPacketsUserMode for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) SetPropertyInTeredoRelaySuccessPacketsDataPacketsUserMode(value uint64) (err error) {
	return instance.SetProperty("InTeredoRelaySuccessPacketsDataPacketsUserMode", (value))
}

// GetInTeredoRelaySuccessPacketsDataPacketsUserMode gets the value of InTeredoRelaySuccessPacketsDataPacketsUserMode for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) GetPropertyInTeredoRelaySuccessPacketsDataPacketsUserMode() (value uint64, err error) {
	retValue, err := instance.GetProperty("InTeredoRelaySuccessPacketsDataPacketsUserMode")
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

// SetInTeredoRelaySuccessPacketsTotal sets the value of InTeredoRelaySuccessPacketsTotal for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) SetPropertyInTeredoRelaySuccessPacketsTotal(value uint64) (err error) {
	return instance.SetProperty("InTeredoRelaySuccessPacketsTotal", (value))
}

// GetInTeredoRelaySuccessPacketsTotal gets the value of InTeredoRelaySuccessPacketsTotal for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) GetPropertyInTeredoRelaySuccessPacketsTotal() (value uint64, err error) {
	retValue, err := instance.GetProperty("InTeredoRelaySuccessPacketsTotal")
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

// SetInTeredoRelayTotalPacketsSuccessError sets the value of InTeredoRelayTotalPacketsSuccessError for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) SetPropertyInTeredoRelayTotalPacketsSuccessError(value uint32) (err error) {
	return instance.SetProperty("InTeredoRelayTotalPacketsSuccessError", (value))
}

// GetInTeredoRelayTotalPacketsSuccessError gets the value of InTeredoRelayTotalPacketsSuccessError for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) GetPropertyInTeredoRelayTotalPacketsSuccessError() (value uint32, err error) {
	retValue, err := instance.GetProperty("InTeredoRelayTotalPacketsSuccessError")
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

// SetInTeredoRelayTotalPacketsSuccessErrorPersec sets the value of InTeredoRelayTotalPacketsSuccessErrorPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) SetPropertyInTeredoRelayTotalPacketsSuccessErrorPersec(value uint32) (err error) {
	return instance.SetProperty("InTeredoRelayTotalPacketsSuccessErrorPersec", (value))
}

// GetInTeredoRelayTotalPacketsSuccessErrorPersec gets the value of InTeredoRelayTotalPacketsSuccessErrorPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) GetPropertyInTeredoRelayTotalPacketsSuccessErrorPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("InTeredoRelayTotalPacketsSuccessErrorPersec")
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

// SetOutTeredoRelayErrorPackets sets the value of OutTeredoRelayErrorPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) SetPropertyOutTeredoRelayErrorPackets(value uint32) (err error) {
	return instance.SetProperty("OutTeredoRelayErrorPackets", (value))
}

// GetOutTeredoRelayErrorPackets gets the value of OutTeredoRelayErrorPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) GetPropertyOutTeredoRelayErrorPackets() (value uint32, err error) {
	retValue, err := instance.GetProperty("OutTeredoRelayErrorPackets")
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

// SetOutTeredoRelayErrorPacketsDestinationError sets the value of OutTeredoRelayErrorPacketsDestinationError for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) SetPropertyOutTeredoRelayErrorPacketsDestinationError(value uint32) (err error) {
	return instance.SetProperty("OutTeredoRelayErrorPacketsDestinationError", (value))
}

// GetOutTeredoRelayErrorPacketsDestinationError gets the value of OutTeredoRelayErrorPacketsDestinationError for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) GetPropertyOutTeredoRelayErrorPacketsDestinationError() (value uint32, err error) {
	retValue, err := instance.GetProperty("OutTeredoRelayErrorPacketsDestinationError")
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

// SetOutTeredoRelayErrorPacketsHeaderError sets the value of OutTeredoRelayErrorPacketsHeaderError for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) SetPropertyOutTeredoRelayErrorPacketsHeaderError(value uint32) (err error) {
	return instance.SetProperty("OutTeredoRelayErrorPacketsHeaderError", (value))
}

// GetOutTeredoRelayErrorPacketsHeaderError gets the value of OutTeredoRelayErrorPacketsHeaderError for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) GetPropertyOutTeredoRelayErrorPacketsHeaderError() (value uint32, err error) {
	retValue, err := instance.GetProperty("OutTeredoRelayErrorPacketsHeaderError")
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

// SetOutTeredoRelayErrorPacketsSourceError sets the value of OutTeredoRelayErrorPacketsSourceError for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) SetPropertyOutTeredoRelayErrorPacketsSourceError(value uint32) (err error) {
	return instance.SetProperty("OutTeredoRelayErrorPacketsSourceError", (value))
}

// GetOutTeredoRelayErrorPacketsSourceError gets the value of OutTeredoRelayErrorPacketsSourceError for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) GetPropertyOutTeredoRelayErrorPacketsSourceError() (value uint32, err error) {
	retValue, err := instance.GetProperty("OutTeredoRelayErrorPacketsSourceError")
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

// SetOutTeredoRelaySuccessPackets sets the value of OutTeredoRelaySuccessPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) SetPropertyOutTeredoRelaySuccessPackets(value uint64) (err error) {
	return instance.SetProperty("OutTeredoRelaySuccessPackets", (value))
}

// GetOutTeredoRelaySuccessPackets gets the value of OutTeredoRelaySuccessPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) GetPropertyOutTeredoRelaySuccessPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutTeredoRelaySuccessPackets")
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

// SetOutTeredoRelaySuccessPacketsBubbles sets the value of OutTeredoRelaySuccessPacketsBubbles for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) SetPropertyOutTeredoRelaySuccessPacketsBubbles(value uint32) (err error) {
	return instance.SetProperty("OutTeredoRelaySuccessPacketsBubbles", (value))
}

// GetOutTeredoRelaySuccessPacketsBubbles gets the value of OutTeredoRelaySuccessPacketsBubbles for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) GetPropertyOutTeredoRelaySuccessPacketsBubbles() (value uint32, err error) {
	retValue, err := instance.GetProperty("OutTeredoRelaySuccessPacketsBubbles")
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

// SetOutTeredoRelaySuccessPacketsDataPackets sets the value of OutTeredoRelaySuccessPacketsDataPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) SetPropertyOutTeredoRelaySuccessPacketsDataPackets(value uint64) (err error) {
	return instance.SetProperty("OutTeredoRelaySuccessPacketsDataPackets", (value))
}

// GetOutTeredoRelaySuccessPacketsDataPackets gets the value of OutTeredoRelaySuccessPacketsDataPackets for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) GetPropertyOutTeredoRelaySuccessPacketsDataPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutTeredoRelaySuccessPacketsDataPackets")
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

// SetOutTeredoRelaySuccessPacketsDataPacketsKernelMode sets the value of OutTeredoRelaySuccessPacketsDataPacketsKernelMode for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) SetPropertyOutTeredoRelaySuccessPacketsDataPacketsKernelMode(value uint64) (err error) {
	return instance.SetProperty("OutTeredoRelaySuccessPacketsDataPacketsKernelMode", (value))
}

// GetOutTeredoRelaySuccessPacketsDataPacketsKernelMode gets the value of OutTeredoRelaySuccessPacketsDataPacketsKernelMode for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) GetPropertyOutTeredoRelaySuccessPacketsDataPacketsKernelMode() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutTeredoRelaySuccessPacketsDataPacketsKernelMode")
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

// SetOutTeredoRelaySuccessPacketsDataPacketsUserMode sets the value of OutTeredoRelaySuccessPacketsDataPacketsUserMode for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) SetPropertyOutTeredoRelaySuccessPacketsDataPacketsUserMode(value uint64) (err error) {
	return instance.SetProperty("OutTeredoRelaySuccessPacketsDataPacketsUserMode", (value))
}

// GetOutTeredoRelaySuccessPacketsDataPacketsUserMode gets the value of OutTeredoRelaySuccessPacketsDataPacketsUserMode for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) GetPropertyOutTeredoRelaySuccessPacketsDataPacketsUserMode() (value uint64, err error) {
	retValue, err := instance.GetProperty("OutTeredoRelaySuccessPacketsDataPacketsUserMode")
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

// SetOutTeredoRelayTotalPacketsSuccessError sets the value of OutTeredoRelayTotalPacketsSuccessError for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) SetPropertyOutTeredoRelayTotalPacketsSuccessError(value uint32) (err error) {
	return instance.SetProperty("OutTeredoRelayTotalPacketsSuccessError", (value))
}

// GetOutTeredoRelayTotalPacketsSuccessError gets the value of OutTeredoRelayTotalPacketsSuccessError for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) GetPropertyOutTeredoRelayTotalPacketsSuccessError() (value uint32, err error) {
	retValue, err := instance.GetProperty("OutTeredoRelayTotalPacketsSuccessError")
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

// SetOutTeredoRelayTotalPacketsSuccessErrorPersec sets the value of OutTeredoRelayTotalPacketsSuccessErrorPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) SetPropertyOutTeredoRelayTotalPacketsSuccessErrorPersec(value uint32) (err error) {
	return instance.SetProperty("OutTeredoRelayTotalPacketsSuccessErrorPersec", (value))
}

// GetOutTeredoRelayTotalPacketsSuccessErrorPersec gets the value of OutTeredoRelayTotalPacketsSuccessErrorPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoRelay) GetPropertyOutTeredoRelayTotalPacketsSuccessErrorPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("OutTeredoRelayTotalPacketsSuccessErrorPersec")
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
