// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.Microsoft.Windows.Storage
//////////////////////////////////////////////
package storage

import (
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// __EventFilter struct
type __EventFilter struct {
	*__IndicationRelated

	//
	CreatorSID []uint8

	//
	EventAccess string

	//
	EventNamespace string

	//
	Name string

	//
	Query string

	//
	QueryLanguage string
}

func New__EventFilterEx1(instance *cim.WmiInstance) (newInstance *__EventFilter, err error) {
	tmp, err := New__IndicationRelatedEx1(instance)

	if err != nil {
		return
	}
	newInstance = &__EventFilter{
		__IndicationRelated: tmp,
	}
	return
}

func New__EventFilterEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *__EventFilter, err error) {
	tmp, err := New__IndicationRelatedEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &__EventFilter{
		__IndicationRelated: tmp,
	}
	return
}

// SetCreatorSID sets the value of CreatorSID for the instance
func (instance *__EventFilter) SetPropertyCreatorSID(value []uint8) (err error) {
	return instance.SetProperty("CreatorSID", (value))
}

// GetCreatorSID gets the value of CreatorSID for the instance
func (instance *__EventFilter) GetPropertyCreatorSID() (value []uint8, err error) {
	retValue, err := instance.GetProperty("CreatorSID")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint8)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint8 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint8(valuetmp))
	}

	return
}

// SetEventAccess sets the value of EventAccess for the instance
func (instance *__EventFilter) SetPropertyEventAccess(value string) (err error) {
	return instance.SetProperty("EventAccess", (value))
}

// GetEventAccess gets the value of EventAccess for the instance
func (instance *__EventFilter) GetPropertyEventAccess() (value string, err error) {
	retValue, err := instance.GetProperty("EventAccess")
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

// SetEventNamespace sets the value of EventNamespace for the instance
func (instance *__EventFilter) SetPropertyEventNamespace(value string) (err error) {
	return instance.SetProperty("EventNamespace", (value))
}

// GetEventNamespace gets the value of EventNamespace for the instance
func (instance *__EventFilter) GetPropertyEventNamespace() (value string, err error) {
	retValue, err := instance.GetProperty("EventNamespace")
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

// SetName sets the value of Name for the instance
func (instance *__EventFilter) SetPropertyName(value string) (err error) {
	return instance.SetProperty("Name", (value))
}

// GetName gets the value of Name for the instance
func (instance *__EventFilter) GetPropertyName() (value string, err error) {
	retValue, err := instance.GetProperty("Name")
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

// SetQuery sets the value of Query for the instance
func (instance *__EventFilter) SetPropertyQuery(value string) (err error) {
	return instance.SetProperty("Query", (value))
}

// GetQuery gets the value of Query for the instance
func (instance *__EventFilter) GetPropertyQuery() (value string, err error) {
	retValue, err := instance.GetProperty("Query")
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

// SetQueryLanguage sets the value of QueryLanguage for the instance
func (instance *__EventFilter) SetPropertyQueryLanguage(value string) (err error) {
	return instance.SetProperty("QueryLanguage", (value))
}

// GetQueryLanguage gets the value of QueryLanguage for the instance
func (instance *__EventFilter) GetPropertyQueryLanguage() (value string, err error) {
	retValue, err := instance.GetProperty("QueryLanguage")
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
