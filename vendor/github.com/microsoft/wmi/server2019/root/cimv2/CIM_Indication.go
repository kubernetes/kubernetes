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

// CIM_Indication struct
type CIM_Indication struct {
	*cim.WmiInstance

	// A list of IndicationIdentifiers whose notifications are correlated with (related to) this one.
	CorrelatedIndications []string

	// An identifier for the indication filter that selects this indication and causes it to be sent. This property is to be filled out by the indication sending service. The value shall be correlatable with the Name property of the instance of CIM_IndicationFilter describing the criteria of the indication. The value of the IndicationFilterName should be formatted using the following algorithm: < OrgID > : < LocalID >, where < OrgID > and < LocalID > are separated by a colon (:) and < OrgID > shall include a copyrighted, trademarked, or otherwise unique name that is owned by the business entity that is creating or defining the value or that is a registered ID assigned to the business entity by a recognized global authority. In addition, to ensure uniqueness, < OrgID > shall not contain a colon (:).When using this algorithm, the first colon to appear in the value shall appear between < OrgID > and < LocalID >. < LocalID > is chosen by the business entity and shall be used uniquely.
	IndicationFilterName string

	// An identifier for the Indication. This property is similar to a key value in that it can be used for identification, when correlating Indications (see the CorrelatedIndications array). Its value SHOULD be unique as long as correlations are reported, but MAY be reused or left NULL if no future Indications will reference it in their CorrelatedIndications array.To ensure uniqueness, the value of IndicationIdentifier should be constructed using the following "preferred" algorithm:
	///<OrgID>:<LocalID>
	///Where <OrgID> and <LocalID> are separated by a colon (:), and where <OrgID> must include a copyrighted, trademarked, or otherwise unique name that is owned by the business entity that is creating or defining the IndicationIdentifier or that is a recognized ID that is assigned to the business entity by a recognized global authority. (This requirement is similar to the <Schema Name>_<Class Name> structure of Schema class names.) In addition, to ensure uniqueness <OrgID> must not contain a colon (:). When using this algorithm, the first colon to appear in IndicationIdentifier must appear between <OrgID> and <LocalID>.
	///<LocalID> is chosen by the business entity and should not be re-used to identify different underlying (real-world) elements.
	///If the above "preferred" algorithm is not used, the defining entity should assure that the resulting IndicationIdentifier is not re-used across any IndicationIdentifiers that are produced by this or other providers for the NameSpace of this instance.
	///For DMTF-defined instances, the "preferred" algorithm should be used with the <OrgID> set to CIM.
	IndicationIdentifier string

	// The time and date of creation of the Indication. The property may be set to NULL if the entity creating the Indication is not capable of determining this information. Note that IndicationTime may be the same for two Indications that are generated in rapid succession.
	IndicationTime string

	// Holds the value of the user defined severity value when 'PerceivedSeverity' is 1 ("Other").
	OtherSeverity string

	// An enumerated value that describes the severity of the Indication from the notifier's point of view:
	///1 - Other, by CIM convention, is used to indicate that the Severity's value can be found in the OtherSeverity property.
	///3 - Degraded/Warning should be used when its appropriate to let the user decide if action is needed.
	///4 - Minor should be used to indicate action is needed, but the situation is not serious at this time.
	///5 - Major should be used to indicate action is needed NOW.
	///6 - Critical should be used to indicate action is needed NOW and the scope is broad (perhaps an imminent outage to a critical resource will result).
	///7 - Fatal/NonRecoverable should be used to indicate an error occurred, but it's too late to take remedial action.
	///2 and 0 - Information and Unknown (respectively) follow common usage. Literally, the Indication is purely informational or its severity is simply unknown.
	PerceivedSeverity Indication_PerceivedSeverity

	// The sequence context portion of a sequence identifier for the indication. The sequence number portion of the sequence identifier is provided by the SequenceNumber property. The combination of both property values represents the sequence identifier for the indication.
	///The sequence identifier for the indication enables a CIM listener to identify duplicate indications when the CIM service attempts the delivery retry of indications, to reorder indications that arrive out-of-order, and to detect lost indications.
	///If a CIM service does not support sequence identifiers for indications, this property shall be NULL.
	///If a CIM service supports sequence identifiers for indications, this property shall be maintained by the CIM service for each registered listener destination, and its value shall uniquely identify the CIM service and the indication service within the CIM service such that restarts of the CIM service and deregistration of listener destinations to the CIM service cause the value to change, without reusing earlier values for a sufficiently long time.
	///When retrying the delivery of an indication, this property shall have the same value as in the original delivery.
	///To guarantee this uniqueness, the property value should be constructed using the following format (defined in ABNF): sequence-context = indication-service-name "#" cim-service-start-id "#" listener-destination-creation-time
	///Where: indication-service-name is the value of the Name property of the CIM_IndicationService instance responsible for delivering the indication. cim-service-start-id is an identifier that uniquely identifies the CIM service start, for example via a timestamp of the start time, or via a counter that increases for each start or restart. listener-destination-creation-time is a timestamp of the creation time of the CIM_ListenerDestination instance representing the listener destination.
	///Since this format is only a recommendation, CIM clients shall treat the value as an opaque identifier for the sequence context and shall not rely on this format.
	SequenceContext string

	// The sequence number portion of a sequence identifier for the indication. The sequence context portion of the sequence identifier is provided by the SequenceContext property. The combination of both property values represents the sequence identifier for the indication.
	///The sequence identifier for the indication enables a CIM listener to identify duplicate indications when the CIM service attempts the delivery retry of indications, to reorder indications that arrive out-of-order, and to detect lost indications.
	///If a CIM service does not support sequence identifiers for indications, this property shall be NULL.
	///If a CIM service supports sequence identifiers for indications, this property shall be maintained by the CIM service for each registered listener destination, and its value shall uniquely identify the indication within the sequence context provided by SequenceContext. It shall start at 0 whenever the sequence context string changes. Otherwise, it shall be increased by 1 for every new indication to that listener destination, and it shall wrap to 0 when the value range is exceeded.
	///When retrying the delivery of an indication, this property shall have the same value as in the original delivery.
	SequenceNumber int64
}

func NewCIM_IndicationEx1(instance *cim.WmiInstance) (newInstance *CIM_Indication, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &CIM_Indication{
		WmiInstance: tmp,
	}
	return
}

func NewCIM_IndicationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_Indication, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_Indication{
		WmiInstance: tmp,
	}
	return
}

// SetCorrelatedIndications sets the value of CorrelatedIndications for the instance
func (instance *CIM_Indication) SetPropertyCorrelatedIndications(value []string) (err error) {
	return instance.SetProperty("CorrelatedIndications", (value))
}

// GetCorrelatedIndications gets the value of CorrelatedIndications for the instance
func (instance *CIM_Indication) GetPropertyCorrelatedIndications() (value []string, err error) {
	retValue, err := instance.GetProperty("CorrelatedIndications")
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

// SetIndicationFilterName sets the value of IndicationFilterName for the instance
func (instance *CIM_Indication) SetPropertyIndicationFilterName(value string) (err error) {
	return instance.SetProperty("IndicationFilterName", (value))
}

// GetIndicationFilterName gets the value of IndicationFilterName for the instance
func (instance *CIM_Indication) GetPropertyIndicationFilterName() (value string, err error) {
	retValue, err := instance.GetProperty("IndicationFilterName")
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

// SetIndicationIdentifier sets the value of IndicationIdentifier for the instance
func (instance *CIM_Indication) SetPropertyIndicationIdentifier(value string) (err error) {
	return instance.SetProperty("IndicationIdentifier", (value))
}

// GetIndicationIdentifier gets the value of IndicationIdentifier for the instance
func (instance *CIM_Indication) GetPropertyIndicationIdentifier() (value string, err error) {
	retValue, err := instance.GetProperty("IndicationIdentifier")
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

// SetIndicationTime sets the value of IndicationTime for the instance
func (instance *CIM_Indication) SetPropertyIndicationTime(value string) (err error) {
	return instance.SetProperty("IndicationTime", (value))
}

// GetIndicationTime gets the value of IndicationTime for the instance
func (instance *CIM_Indication) GetPropertyIndicationTime() (value string, err error) {
	retValue, err := instance.GetProperty("IndicationTime")
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

// SetOtherSeverity sets the value of OtherSeverity for the instance
func (instance *CIM_Indication) SetPropertyOtherSeverity(value string) (err error) {
	return instance.SetProperty("OtherSeverity", (value))
}

// GetOtherSeverity gets the value of OtherSeverity for the instance
func (instance *CIM_Indication) GetPropertyOtherSeverity() (value string, err error) {
	retValue, err := instance.GetProperty("OtherSeverity")
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
func (instance *CIM_Indication) SetPropertyPerceivedSeverity(value Indication_PerceivedSeverity) (err error) {
	return instance.SetProperty("PerceivedSeverity", (value))
}

// GetPerceivedSeverity gets the value of PerceivedSeverity for the instance
func (instance *CIM_Indication) GetPropertyPerceivedSeverity() (value Indication_PerceivedSeverity, err error) {
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

	value = Indication_PerceivedSeverity(valuetmp)

	return
}

// SetSequenceContext sets the value of SequenceContext for the instance
func (instance *CIM_Indication) SetPropertySequenceContext(value string) (err error) {
	return instance.SetProperty("SequenceContext", (value))
}

// GetSequenceContext gets the value of SequenceContext for the instance
func (instance *CIM_Indication) GetPropertySequenceContext() (value string, err error) {
	retValue, err := instance.GetProperty("SequenceContext")
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

// SetSequenceNumber sets the value of SequenceNumber for the instance
func (instance *CIM_Indication) SetPropertySequenceNumber(value int64) (err error) {
	return instance.SetProperty("SequenceNumber", (value))
}

// GetSequenceNumber gets the value of SequenceNumber for the instance
func (instance *CIM_Indication) GetPropertySequenceNumber() (value int64, err error) {
	retValue, err := instance.GetProperty("SequenceNumber")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = int64(valuetmp)

	return
}
