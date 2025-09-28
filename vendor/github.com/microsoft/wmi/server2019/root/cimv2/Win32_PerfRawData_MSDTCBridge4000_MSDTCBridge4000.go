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

// Win32_PerfRawData_MSDTCBridge4000_MSDTCBridge4000 struct
type Win32_PerfRawData_MSDTCBridge4000_MSDTCBridge4000 struct {
	*Win32_PerfRawData

	//
	Averageparticipantcommitresponsetime uint32

	//
	Averageparticipantcommitresponsetime_Base uint32

	//
	Averageparticipantprepareresponsetime uint32

	//
	Averageparticipantprepareresponsetime_Base uint32

	//
	CommitretrycountPersec uint32

	//
	FaultsreceivedcountPersec uint32

	//
	FaultssentcountPersec uint32

	//
	MessagesendfailuresPersec uint32

	//
	PreparedretrycountPersec uint32

	//
	PrepareretrycountPersec uint32

	//
	ReplayretrycountPersec uint32
}

func NewWin32_PerfRawData_MSDTCBridge4000_MSDTCBridge4000Ex1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_MSDTCBridge4000_MSDTCBridge4000, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_MSDTCBridge4000_MSDTCBridge4000{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_MSDTCBridge4000_MSDTCBridge4000Ex6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_MSDTCBridge4000_MSDTCBridge4000, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_MSDTCBridge4000_MSDTCBridge4000{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetAverageparticipantcommitresponsetime sets the value of Averageparticipantcommitresponsetime for the instance
func (instance *Win32_PerfRawData_MSDTCBridge4000_MSDTCBridge4000) SetPropertyAverageparticipantcommitresponsetime(value uint32) (err error) {
	return instance.SetProperty("Averageparticipantcommitresponsetime", (value))
}

// GetAverageparticipantcommitresponsetime gets the value of Averageparticipantcommitresponsetime for the instance
func (instance *Win32_PerfRawData_MSDTCBridge4000_MSDTCBridge4000) GetPropertyAverageparticipantcommitresponsetime() (value uint32, err error) {
	retValue, err := instance.GetProperty("Averageparticipantcommitresponsetime")
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

// SetAverageparticipantcommitresponsetime_Base sets the value of Averageparticipantcommitresponsetime_Base for the instance
func (instance *Win32_PerfRawData_MSDTCBridge4000_MSDTCBridge4000) SetPropertyAverageparticipantcommitresponsetime_Base(value uint32) (err error) {
	return instance.SetProperty("Averageparticipantcommitresponsetime_Base", (value))
}

// GetAverageparticipantcommitresponsetime_Base gets the value of Averageparticipantcommitresponsetime_Base for the instance
func (instance *Win32_PerfRawData_MSDTCBridge4000_MSDTCBridge4000) GetPropertyAverageparticipantcommitresponsetime_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("Averageparticipantcommitresponsetime_Base")
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

// SetAverageparticipantprepareresponsetime sets the value of Averageparticipantprepareresponsetime for the instance
func (instance *Win32_PerfRawData_MSDTCBridge4000_MSDTCBridge4000) SetPropertyAverageparticipantprepareresponsetime(value uint32) (err error) {
	return instance.SetProperty("Averageparticipantprepareresponsetime", (value))
}

// GetAverageparticipantprepareresponsetime gets the value of Averageparticipantprepareresponsetime for the instance
func (instance *Win32_PerfRawData_MSDTCBridge4000_MSDTCBridge4000) GetPropertyAverageparticipantprepareresponsetime() (value uint32, err error) {
	retValue, err := instance.GetProperty("Averageparticipantprepareresponsetime")
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

// SetAverageparticipantprepareresponsetime_Base sets the value of Averageparticipantprepareresponsetime_Base for the instance
func (instance *Win32_PerfRawData_MSDTCBridge4000_MSDTCBridge4000) SetPropertyAverageparticipantprepareresponsetime_Base(value uint32) (err error) {
	return instance.SetProperty("Averageparticipantprepareresponsetime_Base", (value))
}

// GetAverageparticipantprepareresponsetime_Base gets the value of Averageparticipantprepareresponsetime_Base for the instance
func (instance *Win32_PerfRawData_MSDTCBridge4000_MSDTCBridge4000) GetPropertyAverageparticipantprepareresponsetime_Base() (value uint32, err error) {
	retValue, err := instance.GetProperty("Averageparticipantprepareresponsetime_Base")
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

// SetCommitretrycountPersec sets the value of CommitretrycountPersec for the instance
func (instance *Win32_PerfRawData_MSDTCBridge4000_MSDTCBridge4000) SetPropertyCommitretrycountPersec(value uint32) (err error) {
	return instance.SetProperty("CommitretrycountPersec", (value))
}

// GetCommitretrycountPersec gets the value of CommitretrycountPersec for the instance
func (instance *Win32_PerfRawData_MSDTCBridge4000_MSDTCBridge4000) GetPropertyCommitretrycountPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("CommitretrycountPersec")
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

// SetFaultsreceivedcountPersec sets the value of FaultsreceivedcountPersec for the instance
func (instance *Win32_PerfRawData_MSDTCBridge4000_MSDTCBridge4000) SetPropertyFaultsreceivedcountPersec(value uint32) (err error) {
	return instance.SetProperty("FaultsreceivedcountPersec", (value))
}

// GetFaultsreceivedcountPersec gets the value of FaultsreceivedcountPersec for the instance
func (instance *Win32_PerfRawData_MSDTCBridge4000_MSDTCBridge4000) GetPropertyFaultsreceivedcountPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("FaultsreceivedcountPersec")
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

// SetFaultssentcountPersec sets the value of FaultssentcountPersec for the instance
func (instance *Win32_PerfRawData_MSDTCBridge4000_MSDTCBridge4000) SetPropertyFaultssentcountPersec(value uint32) (err error) {
	return instance.SetProperty("FaultssentcountPersec", (value))
}

// GetFaultssentcountPersec gets the value of FaultssentcountPersec for the instance
func (instance *Win32_PerfRawData_MSDTCBridge4000_MSDTCBridge4000) GetPropertyFaultssentcountPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("FaultssentcountPersec")
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

// SetMessagesendfailuresPersec sets the value of MessagesendfailuresPersec for the instance
func (instance *Win32_PerfRawData_MSDTCBridge4000_MSDTCBridge4000) SetPropertyMessagesendfailuresPersec(value uint32) (err error) {
	return instance.SetProperty("MessagesendfailuresPersec", (value))
}

// GetMessagesendfailuresPersec gets the value of MessagesendfailuresPersec for the instance
func (instance *Win32_PerfRawData_MSDTCBridge4000_MSDTCBridge4000) GetPropertyMessagesendfailuresPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("MessagesendfailuresPersec")
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

// SetPreparedretrycountPersec sets the value of PreparedretrycountPersec for the instance
func (instance *Win32_PerfRawData_MSDTCBridge4000_MSDTCBridge4000) SetPropertyPreparedretrycountPersec(value uint32) (err error) {
	return instance.SetProperty("PreparedretrycountPersec", (value))
}

// GetPreparedretrycountPersec gets the value of PreparedretrycountPersec for the instance
func (instance *Win32_PerfRawData_MSDTCBridge4000_MSDTCBridge4000) GetPropertyPreparedretrycountPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("PreparedretrycountPersec")
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

// SetPrepareretrycountPersec sets the value of PrepareretrycountPersec for the instance
func (instance *Win32_PerfRawData_MSDTCBridge4000_MSDTCBridge4000) SetPropertyPrepareretrycountPersec(value uint32) (err error) {
	return instance.SetProperty("PrepareretrycountPersec", (value))
}

// GetPrepareretrycountPersec gets the value of PrepareretrycountPersec for the instance
func (instance *Win32_PerfRawData_MSDTCBridge4000_MSDTCBridge4000) GetPropertyPrepareretrycountPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("PrepareretrycountPersec")
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

// SetReplayretrycountPersec sets the value of ReplayretrycountPersec for the instance
func (instance *Win32_PerfRawData_MSDTCBridge4000_MSDTCBridge4000) SetPropertyReplayretrycountPersec(value uint32) (err error) {
	return instance.SetProperty("ReplayretrycountPersec", (value))
}

// GetReplayretrycountPersec gets the value of ReplayretrycountPersec for the instance
func (instance *Win32_PerfRawData_MSDTCBridge4000_MSDTCBridge4000) GetPropertyReplayretrycountPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReplayretrycountPersec")
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
