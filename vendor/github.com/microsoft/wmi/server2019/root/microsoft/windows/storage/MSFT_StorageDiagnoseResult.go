// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.Microsoft.Windows.Storage
//////////////////////////////////////////////
package storage

import (
	"github.com/microsoft/wmi/pkg/base/instance"
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// MSFT_StorageDiagnoseResult struct
type MSFT_StorageDiagnoseResult struct {
	*cim.WmiInstance

	//
	FaultId string

	//
	FaultingObjectDescription string

	//
	FaultingObjectLocation string

	//
	FaultingObjectType string

	//
	FaultingObjectUniqueId string

	//
	FaultTime string

	//
	FaultType string

	//
	PerceivedSeverity uint16

	//
	Reason string

	//
	RecommendedActions []string
}

func NewMSFT_StorageDiagnoseResultEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageDiagnoseResult, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageDiagnoseResult{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StorageDiagnoseResultEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageDiagnoseResult, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageDiagnoseResult{
		WmiInstance: tmp,
	}
	return
}

// SetFaultId sets the value of FaultId for the instance
func (instance *MSFT_StorageDiagnoseResult) SetPropertyFaultId(value string) (err error) {
	return instance.SetProperty("FaultId", (value))
}

// GetFaultId gets the value of FaultId for the instance
func (instance *MSFT_StorageDiagnoseResult) GetPropertyFaultId() (value string, err error) {
	retValue, err := instance.GetProperty("FaultId")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetFaultingObjectDescription sets the value of FaultingObjectDescription for the instance
func (instance *MSFT_StorageDiagnoseResult) SetPropertyFaultingObjectDescription(value string) (err error) {
	return instance.SetProperty("FaultingObjectDescription", (value))
}

// GetFaultingObjectDescription gets the value of FaultingObjectDescription for the instance
func (instance *MSFT_StorageDiagnoseResult) GetPropertyFaultingObjectDescription() (value string, err error) {
	retValue, err := instance.GetProperty("FaultingObjectDescription")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetFaultingObjectLocation sets the value of FaultingObjectLocation for the instance
func (instance *MSFT_StorageDiagnoseResult) SetPropertyFaultingObjectLocation(value string) (err error) {
	return instance.SetProperty("FaultingObjectLocation", (value))
}

// GetFaultingObjectLocation gets the value of FaultingObjectLocation for the instance
func (instance *MSFT_StorageDiagnoseResult) GetPropertyFaultingObjectLocation() (value string, err error) {
	retValue, err := instance.GetProperty("FaultingObjectLocation")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetFaultingObjectType sets the value of FaultingObjectType for the instance
func (instance *MSFT_StorageDiagnoseResult) SetPropertyFaultingObjectType(value string) (err error) {
	return instance.SetProperty("FaultingObjectType", (value))
}

// GetFaultingObjectType gets the value of FaultingObjectType for the instance
func (instance *MSFT_StorageDiagnoseResult) GetPropertyFaultingObjectType() (value string, err error) {
	retValue, err := instance.GetProperty("FaultingObjectType")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetFaultingObjectUniqueId sets the value of FaultingObjectUniqueId for the instance
func (instance *MSFT_StorageDiagnoseResult) SetPropertyFaultingObjectUniqueId(value string) (err error) {
	return instance.SetProperty("FaultingObjectUniqueId", (value))
}

// GetFaultingObjectUniqueId gets the value of FaultingObjectUniqueId for the instance
func (instance *MSFT_StorageDiagnoseResult) GetPropertyFaultingObjectUniqueId() (value string, err error) {
	retValue, err := instance.GetProperty("FaultingObjectUniqueId")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetFaultTime sets the value of FaultTime for the instance
func (instance *MSFT_StorageDiagnoseResult) SetPropertyFaultTime(value string) (err error) {
	return instance.SetProperty("FaultTime", (value))
}

// GetFaultTime gets the value of FaultTime for the instance
func (instance *MSFT_StorageDiagnoseResult) GetPropertyFaultTime() (value string, err error) {
	retValue, err := instance.GetProperty("FaultTime")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetFaultType sets the value of FaultType for the instance
func (instance *MSFT_StorageDiagnoseResult) SetPropertyFaultType(value string) (err error) {
	return instance.SetProperty("FaultType", (value))
}

// GetFaultType gets the value of FaultType for the instance
func (instance *MSFT_StorageDiagnoseResult) GetPropertyFaultType() (value string, err error) {
	retValue, err := instance.GetProperty("FaultType")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetPerceivedSeverity sets the value of PerceivedSeverity for the instance
func (instance *MSFT_StorageDiagnoseResult) SetPropertyPerceivedSeverity(value uint16) (err error) {
	return instance.SetProperty("PerceivedSeverity", (value))
}

// GetPerceivedSeverity gets the value of PerceivedSeverity for the instance
func (instance *MSFT_StorageDiagnoseResult) GetPropertyPerceivedSeverity() (value uint16, err error) {
	retValue, err := instance.GetProperty("PerceivedSeverity")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetReason sets the value of Reason for the instance
func (instance *MSFT_StorageDiagnoseResult) SetPropertyReason(value string) (err error) {
	return instance.SetProperty("Reason", (value))
}

// GetReason gets the value of Reason for the instance
func (instance *MSFT_StorageDiagnoseResult) GetPropertyReason() (value string, err error) {
	retValue, err := instance.GetProperty("Reason")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetRecommendedActions sets the value of RecommendedActions for the instance
func (instance *MSFT_StorageDiagnoseResult) SetPropertyRecommendedActions(value []string) (err error) {
	return instance.SetProperty("RecommendedActions", (value))
}

// GetRecommendedActions gets the value of RecommendedActions for the instance
func (instance *MSFT_StorageDiagnoseResult) GetPropertyRecommendedActions() (value []string, err error) {
	retValue, err := instance.GetProperty("RecommendedActions")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}
