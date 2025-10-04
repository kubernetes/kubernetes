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

// Win32_PerfFormattedData_Counters_PowerShellWorkflow struct
type Win32_PerfFormattedData_Counters_PowerShellWorkflow struct {
	*Win32_PerfFormattedData

	//
	ActivityHostManagerhostprocessespoolsize uint32

	//
	ActivityHostManagerNumberofbusyhostprocesses uint32

	//
	ActivityHostManagerNumberofcreatedhostprocesses uint32

	//
	ActivityHostManagerNumberofdisposedhostprocesses uint32

	//
	ActivityHostManagerNumberoffailedrequestsinqueue uint32

	//
	ActivityHostManagerNumberoffailedrequestsPersec uint32

	//
	ActivityHostManagerNumberofincomingrequestsPersec uint32

	//
	ActivityHostManagerNumberofpendingrequestsinqueue uint32

	//
	Numberoffailedworkflowjobs uint32

	//
	NumberoffailedworkflowjobsPersec uint32

	//
	Numberofresumedworkflowjobs uint32

	//
	NumberofresumedworkflowjobsPersec uint32

	//
	Numberofrunningworkflowjobs uint32

	//
	NumberofrunningworkflowjobsPersec uint32

	//
	Numberofstoppedworkflowjobs uint32

	//
	NumberofstoppedworkflowjobsPersec uint32

	//
	Numberofsucceededworkflowjobs uint32

	//
	NumberofsucceededworkflowjobsPersec uint32

	//
	Numberofsuspendedworkflowjobs uint32

	//
	NumberofsuspendedworkflowjobsPersec uint32

	//
	Numberofterminatedworkflowjobs uint32

	//
	NumberofterminatedworkflowjobsPersec uint32

	//
	Numberofwaitingworkflowjobs uint32

	//
	PowerShellRemotingNumberofconnectionsclosedreopened uint32

	//
	PowerShellRemotingNumberofcreatedconnections uint32

	//
	PowerShellRemotingNumberofdisposedconnections uint32

	//
	PowerShellRemotingNumberofforcedtowaitrequestsinqueue uint32

	//
	PowerShellRemotingNumberofpendingrequestsinqueue uint32

	//
	PowerShellRemotingNumberofrequestsbeingserviced uint32
}

func NewWin32_PerfFormattedData_Counters_PowerShellWorkflowEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_PowerShellWorkflow, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_PowerShellWorkflow{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_PowerShellWorkflowEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_PowerShellWorkflow, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_PowerShellWorkflow{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetActivityHostManagerhostprocessespoolsize sets the value of ActivityHostManagerhostprocessespoolsize for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) SetPropertyActivityHostManagerhostprocessespoolsize(value uint32) (err error) {
	return instance.SetProperty("ActivityHostManagerhostprocessespoolsize", (value))
}

// GetActivityHostManagerhostprocessespoolsize gets the value of ActivityHostManagerhostprocessespoolsize for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) GetPropertyActivityHostManagerhostprocessespoolsize() (value uint32, err error) {
	retValue, err := instance.GetProperty("ActivityHostManagerhostprocessespoolsize")
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

// SetActivityHostManagerNumberofbusyhostprocesses sets the value of ActivityHostManagerNumberofbusyhostprocesses for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) SetPropertyActivityHostManagerNumberofbusyhostprocesses(value uint32) (err error) {
	return instance.SetProperty("ActivityHostManagerNumberofbusyhostprocesses", (value))
}

// GetActivityHostManagerNumberofbusyhostprocesses gets the value of ActivityHostManagerNumberofbusyhostprocesses for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) GetPropertyActivityHostManagerNumberofbusyhostprocesses() (value uint32, err error) {
	retValue, err := instance.GetProperty("ActivityHostManagerNumberofbusyhostprocesses")
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

// SetActivityHostManagerNumberofcreatedhostprocesses sets the value of ActivityHostManagerNumberofcreatedhostprocesses for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) SetPropertyActivityHostManagerNumberofcreatedhostprocesses(value uint32) (err error) {
	return instance.SetProperty("ActivityHostManagerNumberofcreatedhostprocesses", (value))
}

// GetActivityHostManagerNumberofcreatedhostprocesses gets the value of ActivityHostManagerNumberofcreatedhostprocesses for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) GetPropertyActivityHostManagerNumberofcreatedhostprocesses() (value uint32, err error) {
	retValue, err := instance.GetProperty("ActivityHostManagerNumberofcreatedhostprocesses")
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

// SetActivityHostManagerNumberofdisposedhostprocesses sets the value of ActivityHostManagerNumberofdisposedhostprocesses for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) SetPropertyActivityHostManagerNumberofdisposedhostprocesses(value uint32) (err error) {
	return instance.SetProperty("ActivityHostManagerNumberofdisposedhostprocesses", (value))
}

// GetActivityHostManagerNumberofdisposedhostprocesses gets the value of ActivityHostManagerNumberofdisposedhostprocesses for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) GetPropertyActivityHostManagerNumberofdisposedhostprocesses() (value uint32, err error) {
	retValue, err := instance.GetProperty("ActivityHostManagerNumberofdisposedhostprocesses")
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

// SetActivityHostManagerNumberoffailedrequestsinqueue sets the value of ActivityHostManagerNumberoffailedrequestsinqueue for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) SetPropertyActivityHostManagerNumberoffailedrequestsinqueue(value uint32) (err error) {
	return instance.SetProperty("ActivityHostManagerNumberoffailedrequestsinqueue", (value))
}

// GetActivityHostManagerNumberoffailedrequestsinqueue gets the value of ActivityHostManagerNumberoffailedrequestsinqueue for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) GetPropertyActivityHostManagerNumberoffailedrequestsinqueue() (value uint32, err error) {
	retValue, err := instance.GetProperty("ActivityHostManagerNumberoffailedrequestsinqueue")
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

// SetActivityHostManagerNumberoffailedrequestsPersec sets the value of ActivityHostManagerNumberoffailedrequestsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) SetPropertyActivityHostManagerNumberoffailedrequestsPersec(value uint32) (err error) {
	return instance.SetProperty("ActivityHostManagerNumberoffailedrequestsPersec", (value))
}

// GetActivityHostManagerNumberoffailedrequestsPersec gets the value of ActivityHostManagerNumberoffailedrequestsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) GetPropertyActivityHostManagerNumberoffailedrequestsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ActivityHostManagerNumberoffailedrequestsPersec")
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

// SetActivityHostManagerNumberofincomingrequestsPersec sets the value of ActivityHostManagerNumberofincomingrequestsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) SetPropertyActivityHostManagerNumberofincomingrequestsPersec(value uint32) (err error) {
	return instance.SetProperty("ActivityHostManagerNumberofincomingrequestsPersec", (value))
}

// GetActivityHostManagerNumberofincomingrequestsPersec gets the value of ActivityHostManagerNumberofincomingrequestsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) GetPropertyActivityHostManagerNumberofincomingrequestsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ActivityHostManagerNumberofincomingrequestsPersec")
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

// SetActivityHostManagerNumberofpendingrequestsinqueue sets the value of ActivityHostManagerNumberofpendingrequestsinqueue for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) SetPropertyActivityHostManagerNumberofpendingrequestsinqueue(value uint32) (err error) {
	return instance.SetProperty("ActivityHostManagerNumberofpendingrequestsinqueue", (value))
}

// GetActivityHostManagerNumberofpendingrequestsinqueue gets the value of ActivityHostManagerNumberofpendingrequestsinqueue for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) GetPropertyActivityHostManagerNumberofpendingrequestsinqueue() (value uint32, err error) {
	retValue, err := instance.GetProperty("ActivityHostManagerNumberofpendingrequestsinqueue")
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

// SetNumberoffailedworkflowjobs sets the value of Numberoffailedworkflowjobs for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) SetPropertyNumberoffailedworkflowjobs(value uint32) (err error) {
	return instance.SetProperty("Numberoffailedworkflowjobs", (value))
}

// GetNumberoffailedworkflowjobs gets the value of Numberoffailedworkflowjobs for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) GetPropertyNumberoffailedworkflowjobs() (value uint32, err error) {
	retValue, err := instance.GetProperty("Numberoffailedworkflowjobs")
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

// SetNumberoffailedworkflowjobsPersec sets the value of NumberoffailedworkflowjobsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) SetPropertyNumberoffailedworkflowjobsPersec(value uint32) (err error) {
	return instance.SetProperty("NumberoffailedworkflowjobsPersec", (value))
}

// GetNumberoffailedworkflowjobsPersec gets the value of NumberoffailedworkflowjobsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) GetPropertyNumberoffailedworkflowjobsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberoffailedworkflowjobsPersec")
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

// SetNumberofresumedworkflowjobs sets the value of Numberofresumedworkflowjobs for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) SetPropertyNumberofresumedworkflowjobs(value uint32) (err error) {
	return instance.SetProperty("Numberofresumedworkflowjobs", (value))
}

// GetNumberofresumedworkflowjobs gets the value of Numberofresumedworkflowjobs for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) GetPropertyNumberofresumedworkflowjobs() (value uint32, err error) {
	retValue, err := instance.GetProperty("Numberofresumedworkflowjobs")
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

// SetNumberofresumedworkflowjobsPersec sets the value of NumberofresumedworkflowjobsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) SetPropertyNumberofresumedworkflowjobsPersec(value uint32) (err error) {
	return instance.SetProperty("NumberofresumedworkflowjobsPersec", (value))
}

// GetNumberofresumedworkflowjobsPersec gets the value of NumberofresumedworkflowjobsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) GetPropertyNumberofresumedworkflowjobsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberofresumedworkflowjobsPersec")
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

// SetNumberofrunningworkflowjobs sets the value of Numberofrunningworkflowjobs for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) SetPropertyNumberofrunningworkflowjobs(value uint32) (err error) {
	return instance.SetProperty("Numberofrunningworkflowjobs", (value))
}

// GetNumberofrunningworkflowjobs gets the value of Numberofrunningworkflowjobs for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) GetPropertyNumberofrunningworkflowjobs() (value uint32, err error) {
	retValue, err := instance.GetProperty("Numberofrunningworkflowjobs")
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

// SetNumberofrunningworkflowjobsPersec sets the value of NumberofrunningworkflowjobsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) SetPropertyNumberofrunningworkflowjobsPersec(value uint32) (err error) {
	return instance.SetProperty("NumberofrunningworkflowjobsPersec", (value))
}

// GetNumberofrunningworkflowjobsPersec gets the value of NumberofrunningworkflowjobsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) GetPropertyNumberofrunningworkflowjobsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberofrunningworkflowjobsPersec")
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

// SetNumberofstoppedworkflowjobs sets the value of Numberofstoppedworkflowjobs for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) SetPropertyNumberofstoppedworkflowjobs(value uint32) (err error) {
	return instance.SetProperty("Numberofstoppedworkflowjobs", (value))
}

// GetNumberofstoppedworkflowjobs gets the value of Numberofstoppedworkflowjobs for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) GetPropertyNumberofstoppedworkflowjobs() (value uint32, err error) {
	retValue, err := instance.GetProperty("Numberofstoppedworkflowjobs")
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

// SetNumberofstoppedworkflowjobsPersec sets the value of NumberofstoppedworkflowjobsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) SetPropertyNumberofstoppedworkflowjobsPersec(value uint32) (err error) {
	return instance.SetProperty("NumberofstoppedworkflowjobsPersec", (value))
}

// GetNumberofstoppedworkflowjobsPersec gets the value of NumberofstoppedworkflowjobsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) GetPropertyNumberofstoppedworkflowjobsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberofstoppedworkflowjobsPersec")
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

// SetNumberofsucceededworkflowjobs sets the value of Numberofsucceededworkflowjobs for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) SetPropertyNumberofsucceededworkflowjobs(value uint32) (err error) {
	return instance.SetProperty("Numberofsucceededworkflowjobs", (value))
}

// GetNumberofsucceededworkflowjobs gets the value of Numberofsucceededworkflowjobs for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) GetPropertyNumberofsucceededworkflowjobs() (value uint32, err error) {
	retValue, err := instance.GetProperty("Numberofsucceededworkflowjobs")
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

// SetNumberofsucceededworkflowjobsPersec sets the value of NumberofsucceededworkflowjobsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) SetPropertyNumberofsucceededworkflowjobsPersec(value uint32) (err error) {
	return instance.SetProperty("NumberofsucceededworkflowjobsPersec", (value))
}

// GetNumberofsucceededworkflowjobsPersec gets the value of NumberofsucceededworkflowjobsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) GetPropertyNumberofsucceededworkflowjobsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberofsucceededworkflowjobsPersec")
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

// SetNumberofsuspendedworkflowjobs sets the value of Numberofsuspendedworkflowjobs for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) SetPropertyNumberofsuspendedworkflowjobs(value uint32) (err error) {
	return instance.SetProperty("Numberofsuspendedworkflowjobs", (value))
}

// GetNumberofsuspendedworkflowjobs gets the value of Numberofsuspendedworkflowjobs for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) GetPropertyNumberofsuspendedworkflowjobs() (value uint32, err error) {
	retValue, err := instance.GetProperty("Numberofsuspendedworkflowjobs")
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

// SetNumberofsuspendedworkflowjobsPersec sets the value of NumberofsuspendedworkflowjobsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) SetPropertyNumberofsuspendedworkflowjobsPersec(value uint32) (err error) {
	return instance.SetProperty("NumberofsuspendedworkflowjobsPersec", (value))
}

// GetNumberofsuspendedworkflowjobsPersec gets the value of NumberofsuspendedworkflowjobsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) GetPropertyNumberofsuspendedworkflowjobsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberofsuspendedworkflowjobsPersec")
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

// SetNumberofterminatedworkflowjobs sets the value of Numberofterminatedworkflowjobs for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) SetPropertyNumberofterminatedworkflowjobs(value uint32) (err error) {
	return instance.SetProperty("Numberofterminatedworkflowjobs", (value))
}

// GetNumberofterminatedworkflowjobs gets the value of Numberofterminatedworkflowjobs for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) GetPropertyNumberofterminatedworkflowjobs() (value uint32, err error) {
	retValue, err := instance.GetProperty("Numberofterminatedworkflowjobs")
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

// SetNumberofterminatedworkflowjobsPersec sets the value of NumberofterminatedworkflowjobsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) SetPropertyNumberofterminatedworkflowjobsPersec(value uint32) (err error) {
	return instance.SetProperty("NumberofterminatedworkflowjobsPersec", (value))
}

// GetNumberofterminatedworkflowjobsPersec gets the value of NumberofterminatedworkflowjobsPersec for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) GetPropertyNumberofterminatedworkflowjobsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberofterminatedworkflowjobsPersec")
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

// SetNumberofwaitingworkflowjobs sets the value of Numberofwaitingworkflowjobs for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) SetPropertyNumberofwaitingworkflowjobs(value uint32) (err error) {
	return instance.SetProperty("Numberofwaitingworkflowjobs", (value))
}

// GetNumberofwaitingworkflowjobs gets the value of Numberofwaitingworkflowjobs for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) GetPropertyNumberofwaitingworkflowjobs() (value uint32, err error) {
	retValue, err := instance.GetProperty("Numberofwaitingworkflowjobs")
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

// SetPowerShellRemotingNumberofconnectionsclosedreopened sets the value of PowerShellRemotingNumberofconnectionsclosedreopened for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) SetPropertyPowerShellRemotingNumberofconnectionsclosedreopened(value uint32) (err error) {
	return instance.SetProperty("PowerShellRemotingNumberofconnectionsclosedreopened", (value))
}

// GetPowerShellRemotingNumberofconnectionsclosedreopened gets the value of PowerShellRemotingNumberofconnectionsclosedreopened for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) GetPropertyPowerShellRemotingNumberofconnectionsclosedreopened() (value uint32, err error) {
	retValue, err := instance.GetProperty("PowerShellRemotingNumberofconnectionsclosedreopened")
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

// SetPowerShellRemotingNumberofcreatedconnections sets the value of PowerShellRemotingNumberofcreatedconnections for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) SetPropertyPowerShellRemotingNumberofcreatedconnections(value uint32) (err error) {
	return instance.SetProperty("PowerShellRemotingNumberofcreatedconnections", (value))
}

// GetPowerShellRemotingNumberofcreatedconnections gets the value of PowerShellRemotingNumberofcreatedconnections for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) GetPropertyPowerShellRemotingNumberofcreatedconnections() (value uint32, err error) {
	retValue, err := instance.GetProperty("PowerShellRemotingNumberofcreatedconnections")
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

// SetPowerShellRemotingNumberofdisposedconnections sets the value of PowerShellRemotingNumberofdisposedconnections for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) SetPropertyPowerShellRemotingNumberofdisposedconnections(value uint32) (err error) {
	return instance.SetProperty("PowerShellRemotingNumberofdisposedconnections", (value))
}

// GetPowerShellRemotingNumberofdisposedconnections gets the value of PowerShellRemotingNumberofdisposedconnections for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) GetPropertyPowerShellRemotingNumberofdisposedconnections() (value uint32, err error) {
	retValue, err := instance.GetProperty("PowerShellRemotingNumberofdisposedconnections")
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

// SetPowerShellRemotingNumberofforcedtowaitrequestsinqueue sets the value of PowerShellRemotingNumberofforcedtowaitrequestsinqueue for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) SetPropertyPowerShellRemotingNumberofforcedtowaitrequestsinqueue(value uint32) (err error) {
	return instance.SetProperty("PowerShellRemotingNumberofforcedtowaitrequestsinqueue", (value))
}

// GetPowerShellRemotingNumberofforcedtowaitrequestsinqueue gets the value of PowerShellRemotingNumberofforcedtowaitrequestsinqueue for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) GetPropertyPowerShellRemotingNumberofforcedtowaitrequestsinqueue() (value uint32, err error) {
	retValue, err := instance.GetProperty("PowerShellRemotingNumberofforcedtowaitrequestsinqueue")
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

// SetPowerShellRemotingNumberofpendingrequestsinqueue sets the value of PowerShellRemotingNumberofpendingrequestsinqueue for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) SetPropertyPowerShellRemotingNumberofpendingrequestsinqueue(value uint32) (err error) {
	return instance.SetProperty("PowerShellRemotingNumberofpendingrequestsinqueue", (value))
}

// GetPowerShellRemotingNumberofpendingrequestsinqueue gets the value of PowerShellRemotingNumberofpendingrequestsinqueue for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) GetPropertyPowerShellRemotingNumberofpendingrequestsinqueue() (value uint32, err error) {
	retValue, err := instance.GetProperty("PowerShellRemotingNumberofpendingrequestsinqueue")
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

// SetPowerShellRemotingNumberofrequestsbeingserviced sets the value of PowerShellRemotingNumberofrequestsbeingserviced for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) SetPropertyPowerShellRemotingNumberofrequestsbeingserviced(value uint32) (err error) {
	return instance.SetProperty("PowerShellRemotingNumberofrequestsbeingserviced", (value))
}

// GetPowerShellRemotingNumberofrequestsbeingserviced gets the value of PowerShellRemotingNumberofrequestsbeingserviced for the instance
func (instance *Win32_PerfFormattedData_Counters_PowerShellWorkflow) GetPropertyPowerShellRemotingNumberofrequestsbeingserviced() (value uint32, err error) {
	retValue, err := instance.GetProperty("PowerShellRemotingNumberofrequestsbeingserviced")
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
