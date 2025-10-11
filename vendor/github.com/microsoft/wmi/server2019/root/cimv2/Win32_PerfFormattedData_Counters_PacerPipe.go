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

// Win32_PerfFormattedData_Counters_PacerPipe struct
type Win32_PerfFormattedData_Counters_PacerPipe struct {
	*Win32_PerfFormattedData

	//
	Averagepacketsinnetcard uint32

	//
	Averagepacketsinsequencer uint32

	//
	Averagepacketsinshaper uint32

	//
	Flowmodsrejected uint32

	//
	Flowsclosed uint32

	//
	Flowsmodified uint32

	//
	Flowsopened uint32

	//
	Flowsrejected uint32

	//
	Maxpacketsinnetcard uint32

	//
	Maxpacketsinsequencer uint32

	//
	Maxpacketsinshaper uint32

	//
	Maxsimultaneousflows uint32

	//
	Nonconformingpacketsscheduled uint32

	//
	NonconformingpacketsscheduledPersec uint32

	//
	Nonconformingpacketstransmitted uint32

	//
	NonconformingpacketstransmittedPersec uint32

	//
	Outofpackets uint32
}

func NewWin32_PerfFormattedData_Counters_PacerPipeEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_PacerPipe, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_PacerPipe{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_PacerPipeEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_PacerPipe, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_PacerPipe{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetAveragepacketsinnetcard sets the value of Averagepacketsinnetcard for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) SetPropertyAveragepacketsinnetcard(value uint32) (err error) {
	return instance.SetProperty("Averagepacketsinnetcard", (value))
}

// GetAveragepacketsinnetcard gets the value of Averagepacketsinnetcard for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) GetPropertyAveragepacketsinnetcard() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) SetPropertyAveragepacketsinsequencer(value uint32) (err error) {
	return instance.SetProperty("Averagepacketsinsequencer", (value))
}

// GetAveragepacketsinsequencer gets the value of Averagepacketsinsequencer for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) GetPropertyAveragepacketsinsequencer() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) SetPropertyAveragepacketsinshaper(value uint32) (err error) {
	return instance.SetProperty("Averagepacketsinshaper", (value))
}

// GetAveragepacketsinshaper gets the value of Averagepacketsinshaper for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) GetPropertyAveragepacketsinshaper() (value uint32, err error) {
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

// SetFlowmodsrejected sets the value of Flowmodsrejected for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) SetPropertyFlowmodsrejected(value uint32) (err error) {
	return instance.SetProperty("Flowmodsrejected", (value))
}

// GetFlowmodsrejected gets the value of Flowmodsrejected for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) GetPropertyFlowmodsrejected() (value uint32, err error) {
	retValue, err := instance.GetProperty("Flowmodsrejected")
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

// SetFlowsclosed sets the value of Flowsclosed for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) SetPropertyFlowsclosed(value uint32) (err error) {
	return instance.SetProperty("Flowsclosed", (value))
}

// GetFlowsclosed gets the value of Flowsclosed for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) GetPropertyFlowsclosed() (value uint32, err error) {
	retValue, err := instance.GetProperty("Flowsclosed")
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

// SetFlowsmodified sets the value of Flowsmodified for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) SetPropertyFlowsmodified(value uint32) (err error) {
	return instance.SetProperty("Flowsmodified", (value))
}

// GetFlowsmodified gets the value of Flowsmodified for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) GetPropertyFlowsmodified() (value uint32, err error) {
	retValue, err := instance.GetProperty("Flowsmodified")
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

// SetFlowsopened sets the value of Flowsopened for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) SetPropertyFlowsopened(value uint32) (err error) {
	return instance.SetProperty("Flowsopened", (value))
}

// GetFlowsopened gets the value of Flowsopened for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) GetPropertyFlowsopened() (value uint32, err error) {
	retValue, err := instance.GetProperty("Flowsopened")
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

// SetFlowsrejected sets the value of Flowsrejected for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) SetPropertyFlowsrejected(value uint32) (err error) {
	return instance.SetProperty("Flowsrejected", (value))
}

// GetFlowsrejected gets the value of Flowsrejected for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) GetPropertyFlowsrejected() (value uint32, err error) {
	retValue, err := instance.GetProperty("Flowsrejected")
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

// SetMaxpacketsinnetcard sets the value of Maxpacketsinnetcard for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) SetPropertyMaxpacketsinnetcard(value uint32) (err error) {
	return instance.SetProperty("Maxpacketsinnetcard", (value))
}

// GetMaxpacketsinnetcard gets the value of Maxpacketsinnetcard for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) GetPropertyMaxpacketsinnetcard() (value uint32, err error) {
	retValue, err := instance.GetProperty("Maxpacketsinnetcard")
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
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) SetPropertyMaxpacketsinsequencer(value uint32) (err error) {
	return instance.SetProperty("Maxpacketsinsequencer", (value))
}

// GetMaxpacketsinsequencer gets the value of Maxpacketsinsequencer for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) GetPropertyMaxpacketsinsequencer() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) SetPropertyMaxpacketsinshaper(value uint32) (err error) {
	return instance.SetProperty("Maxpacketsinshaper", (value))
}

// GetMaxpacketsinshaper gets the value of Maxpacketsinshaper for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) GetPropertyMaxpacketsinshaper() (value uint32, err error) {
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

// SetMaxsimultaneousflows sets the value of Maxsimultaneousflows for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) SetPropertyMaxsimultaneousflows(value uint32) (err error) {
	return instance.SetProperty("Maxsimultaneousflows", (value))
}

// GetMaxsimultaneousflows gets the value of Maxsimultaneousflows for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) GetPropertyMaxsimultaneousflows() (value uint32, err error) {
	retValue, err := instance.GetProperty("Maxsimultaneousflows")
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
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) SetPropertyNonconformingpacketsscheduled(value uint32) (err error) {
	return instance.SetProperty("Nonconformingpacketsscheduled", (value))
}

// GetNonconformingpacketsscheduled gets the value of Nonconformingpacketsscheduled for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) GetPropertyNonconformingpacketsscheduled() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) SetPropertyNonconformingpacketsscheduledPersec(value uint32) (err error) {
	return instance.SetProperty("NonconformingpacketsscheduledPersec", (value))
}

// GetNonconformingpacketsscheduledPersec gets the value of NonconformingpacketsscheduledPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) GetPropertyNonconformingpacketsscheduledPersec() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) SetPropertyNonconformingpacketstransmitted(value uint32) (err error) {
	return instance.SetProperty("Nonconformingpacketstransmitted", (value))
}

// GetNonconformingpacketstransmitted gets the value of Nonconformingpacketstransmitted for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) GetPropertyNonconformingpacketstransmitted() (value uint32, err error) {
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
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) SetPropertyNonconformingpacketstransmittedPersec(value uint32) (err error) {
	return instance.SetProperty("NonconformingpacketstransmittedPersec", (value))
}

// GetNonconformingpacketstransmittedPersec gets the value of NonconformingpacketstransmittedPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) GetPropertyNonconformingpacketstransmittedPersec() (value uint32, err error) {
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

// SetOutofpackets sets the value of Outofpackets for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) SetPropertyOutofpackets(value uint32) (err error) {
	return instance.SetProperty("Outofpackets", (value))
}

// GetOutofpackets gets the value of Outofpackets for the instance
func (instance *Win32_PerfFormattedData_Counters_PacerPipe) GetPropertyOutofpackets() (value uint32, err error) {
	retValue, err := instance.GetProperty("Outofpackets")
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
