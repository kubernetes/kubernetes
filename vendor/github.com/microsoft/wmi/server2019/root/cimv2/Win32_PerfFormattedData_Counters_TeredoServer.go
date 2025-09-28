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

// Win32_PerfFormattedData_Counters_TeredoServer struct
type Win32_PerfFormattedData_Counters_TeredoServer struct {
	*Win32_PerfFormattedData

	//
	InTeredoServerErrorPacketsAuthenticationError uint32

	//
	InTeredoServerErrorPacketsDestinationError uint32

	//
	InTeredoServerErrorPacketsHeaderError uint32

	//
	InTeredoServerErrorPacketsSourceError uint32

	//
	InTeredoServerErrorPacketsTotal uint32

	//
	InTeredoServerSuccessPacketsBubbles uint32

	//
	InTeredoServerSuccessPacketsEcho uint32

	//
	InTeredoServerSuccessPacketsRSPrimary uint32

	//
	InTeredoServerSuccessPacketsRSSecondary uint32

	//
	InTeredoServerSuccessPacketsTotal uint32

	//
	InTeredoServerTotalPacketsSuccessError uint32

	//
	InTeredoServerTotalPacketsSuccessErrorPersec uint32

	//
	OutTeredoServerRAPrimary uint32

	//
	OutTeredoServerRASecondary uint32
}

func NewWin32_PerfFormattedData_Counters_TeredoServerEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_TeredoServer, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_TeredoServer{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_TeredoServerEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_TeredoServer, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_TeredoServer{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetInTeredoServerErrorPacketsAuthenticationError sets the value of InTeredoServerErrorPacketsAuthenticationError for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoServer) SetPropertyInTeredoServerErrorPacketsAuthenticationError(value uint32) (err error) {
	return instance.SetProperty("InTeredoServerErrorPacketsAuthenticationError", (value))
}

// GetInTeredoServerErrorPacketsAuthenticationError gets the value of InTeredoServerErrorPacketsAuthenticationError for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoServer) GetPropertyInTeredoServerErrorPacketsAuthenticationError() (value uint32, err error) {
	retValue, err := instance.GetProperty("InTeredoServerErrorPacketsAuthenticationError")
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

// SetInTeredoServerErrorPacketsDestinationError sets the value of InTeredoServerErrorPacketsDestinationError for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoServer) SetPropertyInTeredoServerErrorPacketsDestinationError(value uint32) (err error) {
	return instance.SetProperty("InTeredoServerErrorPacketsDestinationError", (value))
}

// GetInTeredoServerErrorPacketsDestinationError gets the value of InTeredoServerErrorPacketsDestinationError for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoServer) GetPropertyInTeredoServerErrorPacketsDestinationError() (value uint32, err error) {
	retValue, err := instance.GetProperty("InTeredoServerErrorPacketsDestinationError")
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

// SetInTeredoServerErrorPacketsHeaderError sets the value of InTeredoServerErrorPacketsHeaderError for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoServer) SetPropertyInTeredoServerErrorPacketsHeaderError(value uint32) (err error) {
	return instance.SetProperty("InTeredoServerErrorPacketsHeaderError", (value))
}

// GetInTeredoServerErrorPacketsHeaderError gets the value of InTeredoServerErrorPacketsHeaderError for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoServer) GetPropertyInTeredoServerErrorPacketsHeaderError() (value uint32, err error) {
	retValue, err := instance.GetProperty("InTeredoServerErrorPacketsHeaderError")
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

// SetInTeredoServerErrorPacketsSourceError sets the value of InTeredoServerErrorPacketsSourceError for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoServer) SetPropertyInTeredoServerErrorPacketsSourceError(value uint32) (err error) {
	return instance.SetProperty("InTeredoServerErrorPacketsSourceError", (value))
}

// GetInTeredoServerErrorPacketsSourceError gets the value of InTeredoServerErrorPacketsSourceError for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoServer) GetPropertyInTeredoServerErrorPacketsSourceError() (value uint32, err error) {
	retValue, err := instance.GetProperty("InTeredoServerErrorPacketsSourceError")
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

// SetInTeredoServerErrorPacketsTotal sets the value of InTeredoServerErrorPacketsTotal for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoServer) SetPropertyInTeredoServerErrorPacketsTotal(value uint32) (err error) {
	return instance.SetProperty("InTeredoServerErrorPacketsTotal", (value))
}

// GetInTeredoServerErrorPacketsTotal gets the value of InTeredoServerErrorPacketsTotal for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoServer) GetPropertyInTeredoServerErrorPacketsTotal() (value uint32, err error) {
	retValue, err := instance.GetProperty("InTeredoServerErrorPacketsTotal")
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

// SetInTeredoServerSuccessPacketsBubbles sets the value of InTeredoServerSuccessPacketsBubbles for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoServer) SetPropertyInTeredoServerSuccessPacketsBubbles(value uint32) (err error) {
	return instance.SetProperty("InTeredoServerSuccessPacketsBubbles", (value))
}

// GetInTeredoServerSuccessPacketsBubbles gets the value of InTeredoServerSuccessPacketsBubbles for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoServer) GetPropertyInTeredoServerSuccessPacketsBubbles() (value uint32, err error) {
	retValue, err := instance.GetProperty("InTeredoServerSuccessPacketsBubbles")
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

// SetInTeredoServerSuccessPacketsEcho sets the value of InTeredoServerSuccessPacketsEcho for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoServer) SetPropertyInTeredoServerSuccessPacketsEcho(value uint32) (err error) {
	return instance.SetProperty("InTeredoServerSuccessPacketsEcho", (value))
}

// GetInTeredoServerSuccessPacketsEcho gets the value of InTeredoServerSuccessPacketsEcho for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoServer) GetPropertyInTeredoServerSuccessPacketsEcho() (value uint32, err error) {
	retValue, err := instance.GetProperty("InTeredoServerSuccessPacketsEcho")
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

// SetInTeredoServerSuccessPacketsRSPrimary sets the value of InTeredoServerSuccessPacketsRSPrimary for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoServer) SetPropertyInTeredoServerSuccessPacketsRSPrimary(value uint32) (err error) {
	return instance.SetProperty("InTeredoServerSuccessPacketsRSPrimary", (value))
}

// GetInTeredoServerSuccessPacketsRSPrimary gets the value of InTeredoServerSuccessPacketsRSPrimary for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoServer) GetPropertyInTeredoServerSuccessPacketsRSPrimary() (value uint32, err error) {
	retValue, err := instance.GetProperty("InTeredoServerSuccessPacketsRSPrimary")
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

// SetInTeredoServerSuccessPacketsRSSecondary sets the value of InTeredoServerSuccessPacketsRSSecondary for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoServer) SetPropertyInTeredoServerSuccessPacketsRSSecondary(value uint32) (err error) {
	return instance.SetProperty("InTeredoServerSuccessPacketsRSSecondary", (value))
}

// GetInTeredoServerSuccessPacketsRSSecondary gets the value of InTeredoServerSuccessPacketsRSSecondary for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoServer) GetPropertyInTeredoServerSuccessPacketsRSSecondary() (value uint32, err error) {
	retValue, err := instance.GetProperty("InTeredoServerSuccessPacketsRSSecondary")
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

// SetInTeredoServerSuccessPacketsTotal sets the value of InTeredoServerSuccessPacketsTotal for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoServer) SetPropertyInTeredoServerSuccessPacketsTotal(value uint32) (err error) {
	return instance.SetProperty("InTeredoServerSuccessPacketsTotal", (value))
}

// GetInTeredoServerSuccessPacketsTotal gets the value of InTeredoServerSuccessPacketsTotal for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoServer) GetPropertyInTeredoServerSuccessPacketsTotal() (value uint32, err error) {
	retValue, err := instance.GetProperty("InTeredoServerSuccessPacketsTotal")
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

// SetInTeredoServerTotalPacketsSuccessError sets the value of InTeredoServerTotalPacketsSuccessError for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoServer) SetPropertyInTeredoServerTotalPacketsSuccessError(value uint32) (err error) {
	return instance.SetProperty("InTeredoServerTotalPacketsSuccessError", (value))
}

// GetInTeredoServerTotalPacketsSuccessError gets the value of InTeredoServerTotalPacketsSuccessError for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoServer) GetPropertyInTeredoServerTotalPacketsSuccessError() (value uint32, err error) {
	retValue, err := instance.GetProperty("InTeredoServerTotalPacketsSuccessError")
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

// SetInTeredoServerTotalPacketsSuccessErrorPersec sets the value of InTeredoServerTotalPacketsSuccessErrorPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoServer) SetPropertyInTeredoServerTotalPacketsSuccessErrorPersec(value uint32) (err error) {
	return instance.SetProperty("InTeredoServerTotalPacketsSuccessErrorPersec", (value))
}

// GetInTeredoServerTotalPacketsSuccessErrorPersec gets the value of InTeredoServerTotalPacketsSuccessErrorPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoServer) GetPropertyInTeredoServerTotalPacketsSuccessErrorPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("InTeredoServerTotalPacketsSuccessErrorPersec")
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

// SetOutTeredoServerRAPrimary sets the value of OutTeredoServerRAPrimary for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoServer) SetPropertyOutTeredoServerRAPrimary(value uint32) (err error) {
	return instance.SetProperty("OutTeredoServerRAPrimary", (value))
}

// GetOutTeredoServerRAPrimary gets the value of OutTeredoServerRAPrimary for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoServer) GetPropertyOutTeredoServerRAPrimary() (value uint32, err error) {
	retValue, err := instance.GetProperty("OutTeredoServerRAPrimary")
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

// SetOutTeredoServerRASecondary sets the value of OutTeredoServerRASecondary for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoServer) SetPropertyOutTeredoServerRASecondary(value uint32) (err error) {
	return instance.SetProperty("OutTeredoServerRASecondary", (value))
}

// GetOutTeredoServerRASecondary gets the value of OutTeredoServerRASecondary for the instance
func (instance *Win32_PerfFormattedData_Counters_TeredoServer) GetPropertyOutTeredoServerRASecondary() (value uint32, err error) {
	retValue, err := instance.GetProperty("OutTeredoServerRASecondary")
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
