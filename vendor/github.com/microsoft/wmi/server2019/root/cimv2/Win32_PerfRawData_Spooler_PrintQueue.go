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

// Win32_PerfRawData_Spooler_PrintQueue struct
type Win32_PerfRawData_Spooler_PrintQueue struct {
	*Win32_PerfRawData

	//
	AddNetworkPrinterCalls uint32

	//
	BytesPrintedPersec uint64

	//
	EnumerateNetworkPrinterCalls uint32

	//
	JobErrors uint32

	//
	Jobs uint32

	//
	JobsSpooling uint32

	//
	MaxJobsSpooling uint32

	//
	MaxReferences uint32

	//
	NotReadyErrors uint32

	//
	OutofPaperErrors uint32

	//
	References uint32

	//
	TotalJobsPrinted uint32

	//
	TotalPagesPrinted uint32
}

func NewWin32_PerfRawData_Spooler_PrintQueueEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_Spooler_PrintQueue, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Spooler_PrintQueue{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_Spooler_PrintQueueEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_Spooler_PrintQueue, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Spooler_PrintQueue{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetAddNetworkPrinterCalls sets the value of AddNetworkPrinterCalls for the instance
func (instance *Win32_PerfRawData_Spooler_PrintQueue) SetPropertyAddNetworkPrinterCalls(value uint32) (err error) {
	return instance.SetProperty("AddNetworkPrinterCalls", (value))
}

// GetAddNetworkPrinterCalls gets the value of AddNetworkPrinterCalls for the instance
func (instance *Win32_PerfRawData_Spooler_PrintQueue) GetPropertyAddNetworkPrinterCalls() (value uint32, err error) {
	retValue, err := instance.GetProperty("AddNetworkPrinterCalls")
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

// SetBytesPrintedPersec sets the value of BytesPrintedPersec for the instance
func (instance *Win32_PerfRawData_Spooler_PrintQueue) SetPropertyBytesPrintedPersec(value uint64) (err error) {
	return instance.SetProperty("BytesPrintedPersec", (value))
}

// GetBytesPrintedPersec gets the value of BytesPrintedPersec for the instance
func (instance *Win32_PerfRawData_Spooler_PrintQueue) GetPropertyBytesPrintedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesPrintedPersec")
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

// SetEnumerateNetworkPrinterCalls sets the value of EnumerateNetworkPrinterCalls for the instance
func (instance *Win32_PerfRawData_Spooler_PrintQueue) SetPropertyEnumerateNetworkPrinterCalls(value uint32) (err error) {
	return instance.SetProperty("EnumerateNetworkPrinterCalls", (value))
}

// GetEnumerateNetworkPrinterCalls gets the value of EnumerateNetworkPrinterCalls for the instance
func (instance *Win32_PerfRawData_Spooler_PrintQueue) GetPropertyEnumerateNetworkPrinterCalls() (value uint32, err error) {
	retValue, err := instance.GetProperty("EnumerateNetworkPrinterCalls")
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

// SetJobErrors sets the value of JobErrors for the instance
func (instance *Win32_PerfRawData_Spooler_PrintQueue) SetPropertyJobErrors(value uint32) (err error) {
	return instance.SetProperty("JobErrors", (value))
}

// GetJobErrors gets the value of JobErrors for the instance
func (instance *Win32_PerfRawData_Spooler_PrintQueue) GetPropertyJobErrors() (value uint32, err error) {
	retValue, err := instance.GetProperty("JobErrors")
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

// SetJobs sets the value of Jobs for the instance
func (instance *Win32_PerfRawData_Spooler_PrintQueue) SetPropertyJobs(value uint32) (err error) {
	return instance.SetProperty("Jobs", (value))
}

// GetJobs gets the value of Jobs for the instance
func (instance *Win32_PerfRawData_Spooler_PrintQueue) GetPropertyJobs() (value uint32, err error) {
	retValue, err := instance.GetProperty("Jobs")
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

// SetJobsSpooling sets the value of JobsSpooling for the instance
func (instance *Win32_PerfRawData_Spooler_PrintQueue) SetPropertyJobsSpooling(value uint32) (err error) {
	return instance.SetProperty("JobsSpooling", (value))
}

// GetJobsSpooling gets the value of JobsSpooling for the instance
func (instance *Win32_PerfRawData_Spooler_PrintQueue) GetPropertyJobsSpooling() (value uint32, err error) {
	retValue, err := instance.GetProperty("JobsSpooling")
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

// SetMaxJobsSpooling sets the value of MaxJobsSpooling for the instance
func (instance *Win32_PerfRawData_Spooler_PrintQueue) SetPropertyMaxJobsSpooling(value uint32) (err error) {
	return instance.SetProperty("MaxJobsSpooling", (value))
}

// GetMaxJobsSpooling gets the value of MaxJobsSpooling for the instance
func (instance *Win32_PerfRawData_Spooler_PrintQueue) GetPropertyMaxJobsSpooling() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaxJobsSpooling")
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

// SetMaxReferences sets the value of MaxReferences for the instance
func (instance *Win32_PerfRawData_Spooler_PrintQueue) SetPropertyMaxReferences(value uint32) (err error) {
	return instance.SetProperty("MaxReferences", (value))
}

// GetMaxReferences gets the value of MaxReferences for the instance
func (instance *Win32_PerfRawData_Spooler_PrintQueue) GetPropertyMaxReferences() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaxReferences")
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

// SetNotReadyErrors sets the value of NotReadyErrors for the instance
func (instance *Win32_PerfRawData_Spooler_PrintQueue) SetPropertyNotReadyErrors(value uint32) (err error) {
	return instance.SetProperty("NotReadyErrors", (value))
}

// GetNotReadyErrors gets the value of NotReadyErrors for the instance
func (instance *Win32_PerfRawData_Spooler_PrintQueue) GetPropertyNotReadyErrors() (value uint32, err error) {
	retValue, err := instance.GetProperty("NotReadyErrors")
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

// SetOutofPaperErrors sets the value of OutofPaperErrors for the instance
func (instance *Win32_PerfRawData_Spooler_PrintQueue) SetPropertyOutofPaperErrors(value uint32) (err error) {
	return instance.SetProperty("OutofPaperErrors", (value))
}

// GetOutofPaperErrors gets the value of OutofPaperErrors for the instance
func (instance *Win32_PerfRawData_Spooler_PrintQueue) GetPropertyOutofPaperErrors() (value uint32, err error) {
	retValue, err := instance.GetProperty("OutofPaperErrors")
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

// SetReferences sets the value of References for the instance
func (instance *Win32_PerfRawData_Spooler_PrintQueue) SetPropertyReferences(value uint32) (err error) {
	return instance.SetProperty("References", (value))
}

// GetReferences gets the value of References for the instance
func (instance *Win32_PerfRawData_Spooler_PrintQueue) GetPropertyReferences() (value uint32, err error) {
	retValue, err := instance.GetProperty("References")
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

// SetTotalJobsPrinted sets the value of TotalJobsPrinted for the instance
func (instance *Win32_PerfRawData_Spooler_PrintQueue) SetPropertyTotalJobsPrinted(value uint32) (err error) {
	return instance.SetProperty("TotalJobsPrinted", (value))
}

// GetTotalJobsPrinted gets the value of TotalJobsPrinted for the instance
func (instance *Win32_PerfRawData_Spooler_PrintQueue) GetPropertyTotalJobsPrinted() (value uint32, err error) {
	retValue, err := instance.GetProperty("TotalJobsPrinted")
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

// SetTotalPagesPrinted sets the value of TotalPagesPrinted for the instance
func (instance *Win32_PerfRawData_Spooler_PrintQueue) SetPropertyTotalPagesPrinted(value uint32) (err error) {
	return instance.SetProperty("TotalPagesPrinted", (value))
}

// GetTotalPagesPrinted gets the value of TotalPagesPrinted for the instance
func (instance *Win32_PerfRawData_Spooler_PrintQueue) GetPropertyTotalPagesPrinted() (value uint32, err error) {
	retValue, err := instance.GetProperty("TotalPagesPrinted")
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
