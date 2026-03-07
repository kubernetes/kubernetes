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

// CIM_ModifySettingAction struct
type CIM_ModifySettingAction struct {
	*CIM_Action

	//
	ActionType uint16

	//
	EntryName string

	//
	EntryValue string

	//
	FileName string

	//
	SectionKey string
}

func NewCIM_ModifySettingActionEx1(instance *cim.WmiInstance) (newInstance *CIM_ModifySettingAction, err error) {
	tmp, err := NewCIM_ActionEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_ModifySettingAction{
		CIM_Action: tmp,
	}
	return
}

func NewCIM_ModifySettingActionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_ModifySettingAction, err error) {
	tmp, err := NewCIM_ActionEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_ModifySettingAction{
		CIM_Action: tmp,
	}
	return
}

// SetActionType sets the value of ActionType for the instance
func (instance *CIM_ModifySettingAction) SetPropertyActionType(value uint16) (err error) {
	return instance.SetProperty("ActionType", (value))
}

// GetActionType gets the value of ActionType for the instance
func (instance *CIM_ModifySettingAction) GetPropertyActionType() (value uint16, err error) {
	retValue, err := instance.GetProperty("ActionType")
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

// SetEntryName sets the value of EntryName for the instance
func (instance *CIM_ModifySettingAction) SetPropertyEntryName(value string) (err error) {
	return instance.SetProperty("EntryName", (value))
}

// GetEntryName gets the value of EntryName for the instance
func (instance *CIM_ModifySettingAction) GetPropertyEntryName() (value string, err error) {
	retValue, err := instance.GetProperty("EntryName")
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

// SetEntryValue sets the value of EntryValue for the instance
func (instance *CIM_ModifySettingAction) SetPropertyEntryValue(value string) (err error) {
	return instance.SetProperty("EntryValue", (value))
}

// GetEntryValue gets the value of EntryValue for the instance
func (instance *CIM_ModifySettingAction) GetPropertyEntryValue() (value string, err error) {
	retValue, err := instance.GetProperty("EntryValue")
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

// SetFileName sets the value of FileName for the instance
func (instance *CIM_ModifySettingAction) SetPropertyFileName(value string) (err error) {
	return instance.SetProperty("FileName", (value))
}

// GetFileName gets the value of FileName for the instance
func (instance *CIM_ModifySettingAction) GetPropertyFileName() (value string, err error) {
	retValue, err := instance.GetProperty("FileName")
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

// SetSectionKey sets the value of SectionKey for the instance
func (instance *CIM_ModifySettingAction) SetPropertySectionKey(value string) (err error) {
	return instance.SetProperty("SectionKey", (value))
}

// GetSectionKey gets the value of SectionKey for the instance
func (instance *CIM_ModifySettingAction) GetPropertySectionKey() (value string, err error) {
	retValue, err := instance.GetProperty("SectionKey")
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
