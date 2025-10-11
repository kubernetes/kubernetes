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

// MSFT_StorageJob struct
type MSFT_StorageJob struct {
	*MSFT_StorageObject

	//
	BytesProcessed uint64

	//
	BytesTotal uint64

	//
	DeleteOnCompletion bool

	//
	Description string

	//
	ElapsedTime string

	//
	ErrorCode uint16

	//
	ErrorDescription string

	//
	IsBackgroundTask bool

	//
	JobState uint16

	//
	JobStatus string

	//
	LocalOrUtcTime uint16

	//
	Name string

	//
	OperationalStatus []uint16

	//
	OtherRecoveryAction string

	//
	PercentComplete uint16

	//
	RecoveryAction uint16

	//
	StartTime string

	//
	StatusDescriptions []string

	//
	TimeBeforeRemoval string

	//
	TimeOfLastStateChange string

	//
	TimeSubmitted string
}

func NewMSFT_StorageJobEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageJob, err error) {
	tmp, err := NewMSFT_StorageObjectEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageJob{
		MSFT_StorageObject: tmp,
	}
	return
}

func NewMSFT_StorageJobEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageJob, err error) {
	tmp, err := NewMSFT_StorageObjectEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageJob{
		MSFT_StorageObject: tmp,
	}
	return
}

// SetBytesProcessed sets the value of BytesProcessed for the instance
func (instance *MSFT_StorageJob) SetPropertyBytesProcessed(value uint64) (err error) {
	return instance.SetProperty("BytesProcessed", (value))
}

// GetBytesProcessed gets the value of BytesProcessed for the instance
func (instance *MSFT_StorageJob) GetPropertyBytesProcessed() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesProcessed")
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

// SetBytesTotal sets the value of BytesTotal for the instance
func (instance *MSFT_StorageJob) SetPropertyBytesTotal(value uint64) (err error) {
	return instance.SetProperty("BytesTotal", (value))
}

// GetBytesTotal gets the value of BytesTotal for the instance
func (instance *MSFT_StorageJob) GetPropertyBytesTotal() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesTotal")
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

// SetDeleteOnCompletion sets the value of DeleteOnCompletion for the instance
func (instance *MSFT_StorageJob) SetPropertyDeleteOnCompletion(value bool) (err error) {
	return instance.SetProperty("DeleteOnCompletion", (value))
}

// GetDeleteOnCompletion gets the value of DeleteOnCompletion for the instance
func (instance *MSFT_StorageJob) GetPropertyDeleteOnCompletion() (value bool, err error) {
	retValue, err := instance.GetProperty("DeleteOnCompletion")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetDescription sets the value of Description for the instance
func (instance *MSFT_StorageJob) SetPropertyDescription(value string) (err error) {
	return instance.SetProperty("Description", (value))
}

// GetDescription gets the value of Description for the instance
func (instance *MSFT_StorageJob) GetPropertyDescription() (value string, err error) {
	retValue, err := instance.GetProperty("Description")
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

// SetElapsedTime sets the value of ElapsedTime for the instance
func (instance *MSFT_StorageJob) SetPropertyElapsedTime(value string) (err error) {
	return instance.SetProperty("ElapsedTime", (value))
}

// GetElapsedTime gets the value of ElapsedTime for the instance
func (instance *MSFT_StorageJob) GetPropertyElapsedTime() (value string, err error) {
	retValue, err := instance.GetProperty("ElapsedTime")
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

// SetErrorCode sets the value of ErrorCode for the instance
func (instance *MSFT_StorageJob) SetPropertyErrorCode(value uint16) (err error) {
	return instance.SetProperty("ErrorCode", (value))
}

// GetErrorCode gets the value of ErrorCode for the instance
func (instance *MSFT_StorageJob) GetPropertyErrorCode() (value uint16, err error) {
	retValue, err := instance.GetProperty("ErrorCode")
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

// SetErrorDescription sets the value of ErrorDescription for the instance
func (instance *MSFT_StorageJob) SetPropertyErrorDescription(value string) (err error) {
	return instance.SetProperty("ErrorDescription", (value))
}

// GetErrorDescription gets the value of ErrorDescription for the instance
func (instance *MSFT_StorageJob) GetPropertyErrorDescription() (value string, err error) {
	retValue, err := instance.GetProperty("ErrorDescription")
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

// SetIsBackgroundTask sets the value of IsBackgroundTask for the instance
func (instance *MSFT_StorageJob) SetPropertyIsBackgroundTask(value bool) (err error) {
	return instance.SetProperty("IsBackgroundTask", (value))
}

// GetIsBackgroundTask gets the value of IsBackgroundTask for the instance
func (instance *MSFT_StorageJob) GetPropertyIsBackgroundTask() (value bool, err error) {
	retValue, err := instance.GetProperty("IsBackgroundTask")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetJobState sets the value of JobState for the instance
func (instance *MSFT_StorageJob) SetPropertyJobState(value uint16) (err error) {
	return instance.SetProperty("JobState", (value))
}

// GetJobState gets the value of JobState for the instance
func (instance *MSFT_StorageJob) GetPropertyJobState() (value uint16, err error) {
	retValue, err := instance.GetProperty("JobState")
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

// SetJobStatus sets the value of JobStatus for the instance
func (instance *MSFT_StorageJob) SetPropertyJobStatus(value string) (err error) {
	return instance.SetProperty("JobStatus", (value))
}

// GetJobStatus gets the value of JobStatus for the instance
func (instance *MSFT_StorageJob) GetPropertyJobStatus() (value string, err error) {
	retValue, err := instance.GetProperty("JobStatus")
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

// SetLocalOrUtcTime sets the value of LocalOrUtcTime for the instance
func (instance *MSFT_StorageJob) SetPropertyLocalOrUtcTime(value uint16) (err error) {
	return instance.SetProperty("LocalOrUtcTime", (value))
}

// GetLocalOrUtcTime gets the value of LocalOrUtcTime for the instance
func (instance *MSFT_StorageJob) GetPropertyLocalOrUtcTime() (value uint16, err error) {
	retValue, err := instance.GetProperty("LocalOrUtcTime")
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

// SetName sets the value of Name for the instance
func (instance *MSFT_StorageJob) SetPropertyName(value string) (err error) {
	return instance.SetProperty("Name", (value))
}

// GetName gets the value of Name for the instance
func (instance *MSFT_StorageJob) GetPropertyName() (value string, err error) {
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

// SetOperationalStatus sets the value of OperationalStatus for the instance
func (instance *MSFT_StorageJob) SetPropertyOperationalStatus(value []uint16) (err error) {
	return instance.SetProperty("OperationalStatus", (value))
}

// GetOperationalStatus gets the value of OperationalStatus for the instance
func (instance *MSFT_StorageJob) GetPropertyOperationalStatus() (value []uint16, err error) {
	retValue, err := instance.GetProperty("OperationalStatus")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint16)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint16(valuetmp))
	}

	return
}

// SetOtherRecoveryAction sets the value of OtherRecoveryAction for the instance
func (instance *MSFT_StorageJob) SetPropertyOtherRecoveryAction(value string) (err error) {
	return instance.SetProperty("OtherRecoveryAction", (value))
}

// GetOtherRecoveryAction gets the value of OtherRecoveryAction for the instance
func (instance *MSFT_StorageJob) GetPropertyOtherRecoveryAction() (value string, err error) {
	retValue, err := instance.GetProperty("OtherRecoveryAction")
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

// SetPercentComplete sets the value of PercentComplete for the instance
func (instance *MSFT_StorageJob) SetPropertyPercentComplete(value uint16) (err error) {
	return instance.SetProperty("PercentComplete", (value))
}

// GetPercentComplete gets the value of PercentComplete for the instance
func (instance *MSFT_StorageJob) GetPropertyPercentComplete() (value uint16, err error) {
	retValue, err := instance.GetProperty("PercentComplete")
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

// SetRecoveryAction sets the value of RecoveryAction for the instance
func (instance *MSFT_StorageJob) SetPropertyRecoveryAction(value uint16) (err error) {
	return instance.SetProperty("RecoveryAction", (value))
}

// GetRecoveryAction gets the value of RecoveryAction for the instance
func (instance *MSFT_StorageJob) GetPropertyRecoveryAction() (value uint16, err error) {
	retValue, err := instance.GetProperty("RecoveryAction")
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

// SetStartTime sets the value of StartTime for the instance
func (instance *MSFT_StorageJob) SetPropertyStartTime(value string) (err error) {
	return instance.SetProperty("StartTime", (value))
}

// GetStartTime gets the value of StartTime for the instance
func (instance *MSFT_StorageJob) GetPropertyStartTime() (value string, err error) {
	retValue, err := instance.GetProperty("StartTime")
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

// SetStatusDescriptions sets the value of StatusDescriptions for the instance
func (instance *MSFT_StorageJob) SetPropertyStatusDescriptions(value []string) (err error) {
	return instance.SetProperty("StatusDescriptions", (value))
}

// GetStatusDescriptions gets the value of StatusDescriptions for the instance
func (instance *MSFT_StorageJob) GetPropertyStatusDescriptions() (value []string, err error) {
	retValue, err := instance.GetProperty("StatusDescriptions")
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

// SetTimeBeforeRemoval sets the value of TimeBeforeRemoval for the instance
func (instance *MSFT_StorageJob) SetPropertyTimeBeforeRemoval(value string) (err error) {
	return instance.SetProperty("TimeBeforeRemoval", (value))
}

// GetTimeBeforeRemoval gets the value of TimeBeforeRemoval for the instance
func (instance *MSFT_StorageJob) GetPropertyTimeBeforeRemoval() (value string, err error) {
	retValue, err := instance.GetProperty("TimeBeforeRemoval")
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

// SetTimeOfLastStateChange sets the value of TimeOfLastStateChange for the instance
func (instance *MSFT_StorageJob) SetPropertyTimeOfLastStateChange(value string) (err error) {
	return instance.SetProperty("TimeOfLastStateChange", (value))
}

// GetTimeOfLastStateChange gets the value of TimeOfLastStateChange for the instance
func (instance *MSFT_StorageJob) GetPropertyTimeOfLastStateChange() (value string, err error) {
	retValue, err := instance.GetProperty("TimeOfLastStateChange")
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

// SetTimeSubmitted sets the value of TimeSubmitted for the instance
func (instance *MSFT_StorageJob) SetPropertyTimeSubmitted(value string) (err error) {
	return instance.SetProperty("TimeSubmitted", (value))
}

// GetTimeSubmitted gets the value of TimeSubmitted for the instance
func (instance *MSFT_StorageJob) GetPropertyTimeSubmitted() (value string, err error) {
	retValue, err := instance.GetProperty("TimeSubmitted")
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

//

// <param name="RequestedState" type="uint16 "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageJob) RequestStateChange( /* IN */ RequestedState uint16,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("RequestStateChange", RequestedState)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageJob) GetExtendedStatus( /* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetExtendedStatus")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="Channels" type="uint16 []"></param>
// <param name="Messages" type="string []"></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageJob) GetMessages( /* OUT */ Channels []uint16,
	/* OUT */ Messages []string) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetMessages")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="OutParameters" type="MSFT_StorageJobOutParams "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageJob) GetOutParameters( /* OUT */ OutParameters MSFT_StorageJobOutParams) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetOutParameters")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}
