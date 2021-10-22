package crontriggers

import (
	"encoding/json"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

type commonResult struct {
	gophercloud.Result
}

// CreateResult is the response of a Post operations. Call its Extract method to interpret it as a CronTrigger.
type CreateResult struct {
	commonResult
}

// GetResult is the response of Get operations. Call its Extract method to interpret it as a CronTrigger.
type GetResult struct {
	commonResult
}

// DeleteResult is the result from a Delete operation. Call its ExtractErr method to determine the success of the call.
type DeleteResult struct {
	gophercloud.ErrResult
}

// Extract helps to get a CronTrigger struct from a Get or a Create function.
func (r commonResult) Extract() (*CronTrigger, error) {
	var s CronTrigger
	err := r.ExtractInto(&s)
	return &s, err
}

// CronTrigger represents a workflow cron trigger on OpenStack mistral API.
type CronTrigger struct {
	// ID is the cron trigger's unique ID.
	ID string `json:"id"`

	// Name is the name of the cron trigger.
	Name string `json:"name"`

	// Pattern is the cron-like style pattern to execute the workflow.
	// Example of value: "* * * * *"
	Pattern string `json:"pattern"`

	// ProjectID is the project id owner of the cron trigger.
	ProjectID string `json:"project_id"`

	// RemainingExecutions is the number of remaining executions of this trigger.
	RemainingExecutions int `json:"remaining_executions"`

	// Scope is the scope of the trigger.
	// Values can be "private" or "public".
	Scope string `json:"scope"`

	// WorkflowID is the ID of the workflow linked to the trigger.
	WorkflowID string `json:"workflow_id"`

	// WorkflowName is the name of the workflow linked to the trigger.
	WorkflowName string `json:"workflow_name"`

	// WorkflowInput contains the workflow input values.
	WorkflowInput map[string]interface{} `json:"-"`

	// WorkflowParams contains workflow type specific parameters.
	WorkflowParams map[string]interface{} `json:"-"`

	// CreatedAt contains the cron trigger creation date.
	CreatedAt time.Time `json:"-"`

	// FirstExecutionTime is the date of the first execution of the trigger.
	FirstExecutionTime *time.Time `json:"-"`

	// NextExecutionTime is the date of the next execution of the trigger.
	NextExecutionTime *time.Time `json:"-"`
}

// UnmarshalJSON implements unmarshalling custom types
func (r *CronTrigger) UnmarshalJSON(b []byte) error {
	type tmp CronTrigger
	var s struct {
		tmp
		CreatedAt          gophercloud.JSONRFC3339ZNoTNoZ  `json:"created_at"`
		FirstExecutionTime *gophercloud.JSONRFC3339ZNoTNoZ `json:"first_execution_time"`
		NextExecutionTime  *gophercloud.JSONRFC3339ZNoTNoZ `json:"next_execution_time"`
		WorkflowInput      string                          `json:"workflow_input"`
		WorkflowParams     string                          `json:"workflow_params"`
	}

	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}

	*r = CronTrigger(s.tmp)

	r.CreatedAt = time.Time(s.CreatedAt)
	if s.FirstExecutionTime != nil {
		t := time.Time(*s.FirstExecutionTime)
		r.FirstExecutionTime = &t
	}

	if s.NextExecutionTime != nil {
		t := time.Time(*s.NextExecutionTime)
		r.NextExecutionTime = &t
	}

	if s.WorkflowInput != "" {
		if err := json.Unmarshal([]byte(s.WorkflowInput), &r.WorkflowInput); err != nil {
			return err
		}
	}

	if s.WorkflowParams != "" {
		if err := json.Unmarshal([]byte(s.WorkflowParams), &r.WorkflowParams); err != nil {
			return err
		}
	}

	return nil
}

// CronTriggerPage contains a single page of all cron triggers from a List call.
type CronTriggerPage struct {
	pagination.LinkedPageBase
}

// IsEmpty checks if an CronTriggerPage contains any results.
func (r CronTriggerPage) IsEmpty() (bool, error) {
	exec, err := ExtractCronTriggers(r)
	return len(exec) == 0, err
}

// NextPageURL finds the next page URL in a page in order to navigate to the next page of results.
func (r CronTriggerPage) NextPageURL() (string, error) {
	var s struct {
		Next string `json:"next"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return s.Next, nil
}

// ExtractCronTriggers get the list of cron triggers from a page acquired from the List call.
func ExtractCronTriggers(r pagination.Page) ([]CronTrigger, error) {
	var s struct {
		CronTriggers []CronTrigger `json:"cron_triggers"`
	}
	err := (r.(CronTriggerPage)).ExtractInto(&s)
	return s.CronTriggers, err
}
