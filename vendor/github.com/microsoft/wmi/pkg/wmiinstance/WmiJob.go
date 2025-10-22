// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// This class implements the swbemproperty class
// Documentation: https://docs.microsoft.com/en-us/windows/win32/wmisdk/swbemmethod

package cim

import (
	"fmt"
	"time"

	"github.com/microsoft/wmi/pkg/errors"
)

// JobState
type JobState int

const (
	// New enum
	JobState_Unknown JobState = 0
	// New enum
	JobState_New JobState = 2
	// Starting enum
	JobState_Starting JobState = 3
	// Running enum
	JobState_Running JobState = 4
	// Suspended enum
	JobState_Suspended JobState = 5
	// Shutting_Down enum
	JobState_Shutting_Down JobState = 6
	// Completed enum
	JobState_Completed JobState = 7
	// Terminated enum
	JobState_Terminated JobState = 8
	// Killed enum
	JobState_Killed JobState = 9
	// Exception enum
	JobState_Exception JobState = 10
	// Service enum
	JobState_Service JobState = 11
	// Query_Pending enum
	JobState_Query_Pending JobState = 12
	// DMTF_Reserved enum
	JobState_DMTF_Reserved JobState = 13
	// Vendor_Reserved enum
	JobState_Vendor_Reserved JobState = 14
)

type WmiJob struct {
	*WmiInstance
}

func NewWmiJob(instance *WmiInstance) (*WmiJob, error) {
	return &WmiJob{instance}, nil
}

func (job *WmiJob) String() string {
	jtype, err := job.JobType()
	if err != nil {
		return ""
	}
	return fmt.Sprintf("Type[%s] State[%s]", jtype, job.GetJobState())
}

// GetJobType gets the value of JobType for the instance
func (job *WmiJob) JobType() (value int32, err error) {
	retValue, err := job.GetProperty("JobType")
	if err != nil {
		return
	}
	value, ok := retValue.(int32)
	if !ok {
		// TODO: Set an error
	}
	return
}

// WaitForPercentComplete waits for the percentComplete or timeout
func (job *WmiJob) WaitForPercentComplete(percentComplete uint16, timeoutSeconds int16) error {
	start := time.Now()

	// Run the loop, only if the job is actually running
	for !job.IsComplete() {
		pComplete, err := job.PercentComplete()
		if err != nil {
			return err
		}
		// Break if have achieved the target
		if pComplete >= percentComplete {
			break
		}
		time.Sleep(100 * time.Millisecond)

		// Infinite Loop
		if timeoutSeconds < 0 {
			continue
		}

		// If we have waited enough time, return with a timeout error
		if time.Since(start) > (time.Duration(timeoutSeconds) * time.Second) {
			state := job.GetJobState()
			exception := job.GetException()
			return errors.Wrapf(errors.Timedout, "WaitForPercentComplete timeout. Current state: [%v], Exception: [%v]", state, exception)
		}
	}

	return job.GetException()
}

// WaitForAction waits for the task based on the action type, percent complete and timeoutSeconds
func (job *WmiJob) WaitForAction(action UserAction, percentComplete uint16, timeoutSeconds int16) error {
	switch action {
	case Wait:
		return job.WaitForPercentComplete(percentComplete, timeoutSeconds)
	case Cancel:
		return job.WaitForPercentComplete(percentComplete, timeoutSeconds)
	case None:
		fallthrough
	case Default:
		fallthrough
	case Async:
		break
	}
	return nil
}

// PercentComplete
func (job *WmiJob) PercentComplete() (uint16, error) {
	err := job.Refresh()
	if err != nil {
		return 0, err
	}
	retValue, err := job.GetProperty("PercentComplete")
	if err != nil {
		return 0, err
	}
	return uint16(retValue.(int32)), nil
}

func (job *WmiJob) GetJobState() (js JobState) {
	state, err := job.GetProperty("JobState")
	if err != nil {
		return
	}
	js = JobState(state.(int32))
	return
}

func (job *WmiJob) IsComplete() bool {
	err := job.Refresh()
	if err != nil {

	}
	state := job.GetJobState()
	switch state {
	case JobState_New:
		fallthrough
	case JobState_Starting:
		fallthrough
	case JobState_Running:
		fallthrough
	case JobState_Suspended:
		fallthrough
	case JobState_Shutting_Down:
		return false
	case JobState_Completed:
		fallthrough
	case JobState_Terminated:
		fallthrough
	case JobState_Killed:
		fallthrough
	case JobState_Exception:
		return true
	}
	return false
}

func (job *WmiJob) GetException() error {
	job.Refresh()
	state := job.GetJobState()
	switch state {
	case JobState_Terminated:
		fallthrough
	case JobState_Killed:
		fallthrough
	case JobState_Exception:
		errorCodeVal, _ := job.GetProperty("ErrorCode")
		errorCode := uint16(errorCodeVal.(int32))
		errorDescriptionVal, _ := job.GetProperty("ErrorDescription")
		errorDescription, _ := errorDescriptionVal.(string)
		errorSummaryDescriptionVal, _ := job.GetProperty("ErrorSummaryDescription")
		errorSummaryDescription, _ := errorSummaryDescriptionVal.(string)
		return errors.Wrapf(errors.NewWMIError(errorCode),
			"ErrorCode[%d] ErrorDescription[%s] ErrorSummaryDescription [%s]",
			errorCode, errorDescription, errorSummaryDescription)
	}
	return nil
}

func (job *WmiJob) WaitForJobCompletion(result int32, timeoutSeconds int16) error {
	if result == 0 {
		return nil
	} else if result == 4096 {
		return job.WaitForAction(Wait, 100, timeoutSeconds)
	} else {
		return errors.Wrapf(errors.Failed, "Unable to Wait for Job on Result[%d] ", result)
	}

}

type WmiJobCollection []*WmiJob

func (c *WmiJobCollection) Close() error {
	var err error
	for _, p := range *c {
		err = p.Close()
	}
	return err
}
