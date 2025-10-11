// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.CIMV2
//////////////////////////////////////////////
package cimv2

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

	// The CIM status code that characterizes this instance.
	///This property defines the status codes that MAY be return by a conforming CIM Server or Listener. Note that not all status codes are valid for each operation. The specification for each operation SHOULD define the status codes that may be returned by that operation.
	///The following values for CIM status code are defined:
	///1 - CIM_ERR_FAILED. A general error occurred that is not covered by a more specific error code.
	///2 - CIM_ERR_ACCESS_DENIED. Access to a CIM resource was not available to the client.
	///3 - CIM_ERR_INVALID_NAMESPACE. The target namespace does not exist.
	///4 - CIM_ERR_INVALID_PARAMETER. One or more parameter values passed to the method were invalid.
	///5 - CIM_ERR_INVALID_CLASS. The specified Class does not exist.
	///6 - CIM_ERR_NOT_FOUND. The requested object could not be found.
	///7 - CIM_ERR_NOT_SUPPORTED. The requested operation is not supported.
	///8 - CIM_ERR_CLASS_HAS_CHILDREN. Operation cannot be carried out on this class since it has instances.
	///9 - CIM_ERR_CLASS_HAS_INSTANCES. Operation cannot be carried out on this class since it has instances.
	///10 - CIM_ERR_INVALID_SUPERCLASS. Operation cannot be carried out since the specified superclass does not exist.
	///11 - CIM_ERR_ALREADY_EXISTS. Operation cannot be carried out because an object already exists.
	///12 - CIM_ERR_NO_SUCH_PROPERTY. The specified Property does not exist.
	///13 - CIM_ERR_TYPE_MISMATCH. The value supplied is incompatible with the type.
	///14 - CIM_ERR_QUERY_LANGUAGE_NOT_SUPPORTED. The query language is not recognized or supported.
	///15 - CIM_ERR_INVALID_QUERY. The query is not valid for the specified query language.
	///16 - CIM_ERR_METHOD_NOT_AVAILABLE. The extrinsic Method could not be executed.
	///17 - CIM_ERR_METHOD_NOT_FOUND. The specified extrinsic Method does not exist.
	///18 - CIM_ERR_UNEXPECTED_RESPONSE. The returned response to the asynchronous operation was not expected.
	///19 - CIM_ERR_INVALID_RESPONSE_DESTINATION. The specified destination for the asynchronous response is not valid.
	///20 - CIM_ERR_NAMESPACE_NOT_EMPTY. The specified Namespace is not empty.
	///21 - CIM_ERR_INVALID_ENUMERATION_CONTEXT. The enumeration context supplied is not valid.
	///22 - CIM_ERR_INVALID_OPERATION_TIMEOUT. The specified Namespace is not empty.
	///23 - CIM_ERR_PULL_HAS_BEEN_ABANDONED. The specified Namespace is not empty.
	///24 - CIM_ERR_PULL_CANNOT_BE_ABANDONED. The attempt to abandon a pull operation has failed.
	///25 - CIM_ERR_FILTERED_ENUMERATION_NOT_SUPPORTED. Filtered Enumeratrions are not supported.
	///26 - CIM_ERR_CONTINUATION_ON_ERROR_NOT_SUPPORTED. Continue on error is not supported.
	///27 - CIM_ERR_SERVER_LIMITS_EXCEEDED. The WBEM Server limits have been exceeded (e.g. memory, connections, ...).
	///28 - CIM_ERR_SERVER_IS_SHUTTING_DOWN. The WBEM Server is shutting down.
	///29 - CIM_ERR_QUERY_FEATURE_NOT_SUPPORTED. The specified Query Feature is not supported.
	CIMStatusCode Error_CIMStatusCode

	// A free-form string containing a human-readable description of CIMStatusCode. This description MAY extend, but MUST be consistent with, the definition of CIMStatusCode.
	CIMStatusCodeDescription string

	// The identifying information of the entity (i.e., the instance) generating the error. If this entity is modeled in the CIM Schema, this property contains the path of the instance encoded as a string parameter. If not modeled, the property contains some identifying string that names the entity that generated the error. The path or identifying string is formatted per the ErrorSourceFormat property.
	ErrorSource string

	// An array containing the dynamic content of the message.
	ErrorSourceFormat Error_ErrorSourceFormat

	// Primary classification of the error. The following values are defined:
	///2 - Communications Error. Errors of this type are principally associated with the procedures and/or processes required to convey information from one point to another.
	///3 - Quality of Service Error. Errors of this type are principally associated with failures that result in reduced functionality or performance.
	///4 - Software Error. Error of this type are principally associated with a software or processing fault.
	///5 - Hardware Error. Errors of this type are principally associated with an equipment or hardware failure.
	///6 - Environmental Error. Errors of this type are principally associated with a failure condition relating the to facility, or other environmental considerations.
	///7 - Security Error. Errors of this type are associated with security violations, detection of viruses, and similar issues.
	///8 - Oversubscription Error. Errors of this type are principally associated with the failure to allocate sufficient resources to complete the operation.
	///9 - Unavailable Resource Error. Errors of this type are principally associated with the failure to access a required resource.
	///10 -Unsupported Operation Error. Errors of this type are principally associated with requests that are not supported.
	ErrorType Error_ErrorType

	// The formatted message. This message is constructed by combining some or all of the dynamic elements specified in the MessageArguments property with the static elements uniquely identified by the MessageID in a message registry or other catalog associated with the OwningEntity.
	Message string

	// An array containing the dynamic content of the message.
	MessageArguments []string

	// An opaque string that uniquely identifies, within the scope of the OwningEntity, the format of the Message.
	MessageID string

	// A string defining "Other" values for ErrorSourceFormat. This value MUST be set to a non NULL value when ErrorSourceFormat is set to a value of 1 ("Other"). For all other values of ErrorSourceFormat, the value of this string must be set to NULL.
	OtherErrorSourceFormat string

	// A free-form string describing the ErrorType when 1, "Other", is specified as the ErrorType.
	OtherErrorType string

	// A string that uniquely identifies the entity that owns the definition of the format of the Message described in this instance. OwningEntity MUST include a copyrighted, trademarked or otherwise unique name that is owned by the business entity or standards body defining the format.
	OWningEntity string

	// An enumerated value that describes the severity of the Indication from the notifier's point of view:
	///0 - the Perceived Severity of the indication is unknown or indeterminate.
	///1 - Other, by CIM convention, is used to indicate that the Severity's value can be found in the OtherSeverity property.
	///2 - Information should be used when providing an informative response.
	///3 - Degraded/Warning should be used when its appropriate to let the user decide if action is needed.
	///4 - Minor should be used to indicate action is needed, but the situation is not serious at this time.
	///5 - Major should be used to indicate action is needed NOW.
	///6 - Critical should be used to indicate action is needed NOW and the scope is broad (perhaps an imminent outage to a critical resource will result).
	///7 - Fatal/NonRecoverable should be used to indicate an error occurred, but it's too late to take remedial action.
	///2 and 0 - Information and Unknown (respectively) follow common usage. Literally, the Error is purely informational or its severity is simply unknown.
	PerceivedSeverity Error_PerceivedSeverity

	// An enumerated value that describes the probable cause of the error.
	ProbableCause Error_ProbableCause

	// A free-form string describing the probable cause of the error.
	ProbableCauseDescription string

	// A free-form string describing recommended actions to take to resolve the error.
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
func (instance *CIM_Error) SetPropertyCIMStatusCode(value Error_CIMStatusCode) (err error) {
	return instance.SetProperty("CIMStatusCode", (value))
}

// GetCIMStatusCode gets the value of CIMStatusCode for the instance
func (instance *CIM_Error) GetPropertyCIMStatusCode() (value Error_CIMStatusCode, err error) {
	retValue, err := instance.GetProperty("CIMStatusCode")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Error_CIMStatusCode(valuetmp)

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
func (instance *CIM_Error) SetPropertyErrorSourceFormat(value Error_ErrorSourceFormat) (err error) {
	return instance.SetProperty("ErrorSourceFormat", (value))
}

// GetErrorSourceFormat gets the value of ErrorSourceFormat for the instance
func (instance *CIM_Error) GetPropertyErrorSourceFormat() (value Error_ErrorSourceFormat, err error) {
	retValue, err := instance.GetProperty("ErrorSourceFormat")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Error_ErrorSourceFormat(valuetmp)

	return
}

// SetErrorType sets the value of ErrorType for the instance
func (instance *CIM_Error) SetPropertyErrorType(value Error_ErrorType) (err error) {
	return instance.SetProperty("ErrorType", (value))
}

// GetErrorType gets the value of ErrorType for the instance
func (instance *CIM_Error) GetPropertyErrorType() (value Error_ErrorType, err error) {
	retValue, err := instance.GetProperty("ErrorType")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Error_ErrorType(valuetmp)

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

// SetOWningEntity sets the value of OWningEntity for the instance
func (instance *CIM_Error) SetPropertyOWningEntity(value string) (err error) {
	return instance.SetProperty("OWningEntity", (value))
}

// GetOWningEntity gets the value of OWningEntity for the instance
func (instance *CIM_Error) GetPropertyOWningEntity() (value string, err error) {
	retValue, err := instance.GetProperty("OWningEntity")
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
func (instance *CIM_Error) SetPropertyPerceivedSeverity(value Error_PerceivedSeverity) (err error) {
	return instance.SetProperty("PerceivedSeverity", (value))
}

// GetPerceivedSeverity gets the value of PerceivedSeverity for the instance
func (instance *CIM_Error) GetPropertyPerceivedSeverity() (value Error_PerceivedSeverity, err error) {
	retValue, err := instance.GetProperty("PerceivedSeverity")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Error_PerceivedSeverity(valuetmp)

	return
}

// SetProbableCause sets the value of ProbableCause for the instance
func (instance *CIM_Error) SetPropertyProbableCause(value Error_ProbableCause) (err error) {
	return instance.SetProperty("ProbableCause", (value))
}

// GetProbableCause gets the value of ProbableCause for the instance
func (instance *CIM_Error) GetPropertyProbableCause() (value Error_ProbableCause, err error) {
	retValue, err := instance.GetProperty("ProbableCause")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Error_ProbableCause(valuetmp)

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
