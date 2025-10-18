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

// CIM_Error struct
type CIM_Error struct {
	*cim.WmiInstance

	//
	CIMStatusCode uint32

	//
	CIMStatusCodeDescription string

	//
	ErrorSource string

	//
	ErrorSourceFormat uint16

	//
	ErrorType uint16

	//
	Message string

	//
	MessageArguments []string

	//
	MessageID string

	//
	OtherErrorSourceFormat string

	//
	OtherErrorType string

	//
	OwningEntity string

	//
	PerceivedSeverity uint16

	//
	ProbableCause uint16

	//
	ProbableCauseDescription string

	//
	RecommendedActions []string
}

func NewCIM_ErrorEx1(instance *cim.WmiInstance) (newInstance *CIM_Error, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &CIM_Error{
		WmiInstance: tmp,
	}
	return
}

func NewCIM_ErrorEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_Error, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_Error{
		WmiInstance: tmp,
	}
	return
}

// SetCIMStatusCode sets the value of CIMStatusCode for the instance
func (instance *CIM_Error) SetPropertyCIMStatusCode(value uint32) (err error) {
	return instance.SetProperty("CIMStatusCode", (value))
}

// GetCIMStatusCode gets the value of CIMStatusCode for the instance
func (instance *CIM_Error) GetPropertyCIMStatusCode() (value uint32, err error) {
	retValue, err := instance.GetProperty("CIMStatusCode")
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

// SetCIMStatusCodeDescription sets the value of CIMStatusCodeDescription for the instance
func (instance *CIM_Error) SetPropertyCIMStatusCodeDescription(value string) (err error) {
	return instance.SetProperty("CIMStatusCodeDescription", (value))
}

// GetCIMStatusCodeDescription gets the value of CIMStatusCodeDescription for the instance
func (instance *CIM_Error) GetPropertyCIMStatusCodeDescription() (value string, err error) {
	retValue, err := instance.GetProperty("CIMStatusCodeDescription")
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

// SetErrorSource sets the value of ErrorSource for the instance
func (instance *CIM_Error) SetPropertyErrorSource(value string) (err error) {
	return instance.SetProperty("ErrorSource", (value))
}

// GetErrorSource gets the value of ErrorSource for the instance
func (instance *CIM_Error) GetPropertyErrorSource() (value string, err error) {
	retValue, err := instance.GetProperty("ErrorSource")
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

// SetErrorSourceFormat sets the value of ErrorSourceFormat for the instance
func (instance *CIM_Error) SetPropertyErrorSourceFormat(value uint16) (err error) {
	return instance.SetProperty("ErrorSourceFormat", (value))
}

// GetErrorSourceFormat gets the value of ErrorSourceFormat for the instance
func (instance *CIM_Error) GetPropertyErrorSourceFormat() (value uint16, err error) {
	retValue, err := instance.GetProperty("ErrorSourceFormat")
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

// SetErrorType sets the value of ErrorType for the instance
func (instance *CIM_Error) SetPropertyErrorType(value uint16) (err error) {
	return instance.SetProperty("ErrorType", (value))
}

// GetErrorType gets the value of ErrorType for the instance
func (instance *CIM_Error) GetPropertyErrorType() (value uint16, err error) {
	retValue, err := instance.GetProperty("ErrorType")
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

// SetMessage sets the value of Message for the instance
func (instance *CIM_Error) SetPropertyMessage(value string) (err error) {
	return instance.SetProperty("Message", (value))
}

// GetMessage gets the value of Message for the instance
func (instance *CIM_Error) GetPropertyMessage() (value string, err error) {
	retValue, err := instance.GetProperty("Message")
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

// SetMessageArguments sets the value of MessageArguments for the instance
func (instance *CIM_Error) SetPropertyMessageArguments(value []string) (err error) {
	return instance.SetProperty("MessageArguments", (value))
}

// GetMessageArguments gets the value of MessageArguments for the instance
func (instance *CIM_Error) GetPropertyMessageArguments() (value []string, err error) {
	retValue, err := instance.GetProperty("MessageArguments")
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

// SetMessageID sets the value of MessageID for the instance
func (instance *CIM_Error) SetPropertyMessageID(value string) (err error) {
	return instance.SetProperty("MessageID", (value))
}

// GetMessageID gets the value of MessageID for the instance
func (instance *CIM_Error) GetPropertyMessageID() (value string, err error) {
	retValue, err := instance.GetProperty("MessageID")
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

// SetOtherErrorSourceFormat sets the value of OtherErrorSourceFormat for the instance
func (instance *CIM_Error) SetPropertyOtherErrorSourceFormat(value string) (err error) {
	return instance.SetProperty("OtherErrorSourceFormat", (value))
}

// GetOtherErrorSourceFormat gets the value of OtherErrorSourceFormat for the instance
func (instance *CIM_Error) GetPropertyOtherErrorSourceFormat() (value string, err error) {
	retValue, err := instance.GetProperty("OtherErrorSourceFormat")
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

// SetOtherErrorType sets the value of OtherErrorType for the instance
func (instance *CIM_Error) SetPropertyOtherErrorType(value string) (err error) {
	return instance.SetProperty("OtherErrorType", (value))
}

// GetOtherErrorType gets the value of OtherErrorType for the instance
func (instance *CIM_Error) GetPropertyOtherErrorType() (value string, err error) {
	retValue, err := instance.GetProperty("OtherErrorType")
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

// SetOwningEntity sets the value of OwningEntity for the instance
func (instance *CIM_Error) SetPropertyOwningEntity(value string) (err error) {
	return instance.SetProperty("OwningEntity", (value))
}

// GetOwningEntity gets the value of OwningEntity for the instance
func (instance *CIM_Error) GetPropertyOwningEntity() (value string, err error) {
	retValue, err := instance.GetProperty("OwningEntity")
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
func (instance *CIM_Error) SetPropertyPerceivedSeverity(value uint16) (err error) {
	return instance.SetProperty("PerceivedSeverity", (value))
}

// GetPerceivedSeverity gets the value of PerceivedSeverity for the instance
func (instance *CIM_Error) GetPropertyPerceivedSeverity() (value uint16, err error) {
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

// SetProbableCause sets the value of ProbableCause for the instance
func (instance *CIM_Error) SetPropertyProbableCause(value uint16) (err error) {
	return instance.SetProperty("ProbableCause", (value))
}

// GetProbableCause gets the value of ProbableCause for the instance
func (instance *CIM_Error) GetPropertyProbableCause() (value uint16, err error) {
	retValue, err := instance.GetProperty("ProbableCause")
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

// SetProbableCauseDescription sets the value of ProbableCauseDescription for the instance
func (instance *CIM_Error) SetPropertyProbableCauseDescription(value string) (err error) {
	return instance.SetProperty("ProbableCauseDescription", (value))
}

// GetProbableCauseDescription gets the value of ProbableCauseDescription for the instance
func (instance *CIM_Error) GetPropertyProbableCauseDescription() (value string, err error) {
	retValue, err := instance.GetProperty("ProbableCauseDescription")
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
func (instance *CIM_Error) SetPropertyRecommendedActions(value []string) (err error) {
	return instance.SetProperty("RecommendedActions", (value))
}

// GetRecommendedActions gets the value of RecommendedActions for the instance
func (instance *CIM_Error) GetPropertyRecommendedActions() (value []string, err error) {
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
