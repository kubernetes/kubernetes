package executions

import (
	"encoding/json"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

type commonResult struct {
	gophercloud.Result
}

// CreateResult is the response of a Post operations. Call its Extract method to interpret it as an Execution.
type CreateResult struct {
	commonResult
}

// GetResult is the response of Get operations. Call its Extract method to interpret it as an Execution.
type GetResult struct {
	commonResult
}

// Extract helps to get an Execution struct from a Get or a Create function.
func (r commonResult) Extract() (*Execution, error) {
	var s Execution
	err := r.ExtractInto(&s)
	return &s, err
}

// DeleteResult is the result from a Delete operation. Call its ExtractErr method to determine the success of the call.
type DeleteResult struct {
	gophercloud.ErrResult
}

// Execution represents a workflow execution on OpenStack mistral API.
type Execution struct {
	// ID is the execution's unique ID.
	ID string `json:"id"`

	// CreatedAt contains the execution creation date.
	CreatedAt time.Time `json:"-"`

	// UpdatedAt is the last update of the execution.
	UpdatedAt time.Time `json:"-"`

	// RootExecutionID is the parent execution ID.
	RootExecutionID *string `json:"root_execution_id"`

	// TaskExecutionID is the task execution ID.
	TaskExecutionID *string `json:"task_execution_id"`

	// Description is the description of the execution.
	Description string `json:"description"`

	// Input contains the workflow input values.
	Input map[string]interface{} `json:"-"`

	// Ouput contains the workflow output values.
	Output map[string]interface{} `json:"-"`

	// Params contains workflow type specific parameters.
	Params map[string]interface{} `json:"-"`

	// ProjectID is the project id owner of the execution.
	ProjectID string `json:"project_id"`

	// State is the current state of the execution. State can be one of: IDLE, RUNNING, SUCCESS, ERROR, PAUSED, CANCELLED.
	State string `json:"state"`

	// StateInfo contains an optional state information string.
	StateInfo *string `json:"state_info"`

	// WorkflowID is the ID of the workflow linked to the execution.
	WorkflowID string `json:"workflow_id"`

	// WorkflowName is the name of the workflow linked to the execution.
	WorkflowName string `json:"workflow_name"`

	// WorkflowNamespace is the namespace of the workflow linked to the execution.
	WorkflowNamespace string `json:"workflow_namespace"`
}

// UnmarshalJSON implements unmarshalling custom types
func (r *Execution) UnmarshalJSON(b []byte) error {
	type tmp Execution
	var s struct {
		tmp
		CreatedAt gophercloud.JSONRFC3339ZNoTNoZ `json:"created_at"`
		UpdatedAt gophercloud.JSONRFC3339ZNoTNoZ `json:"updated_at"`
		Input     string                         `json:"input"`
		Output    string                         `json:"output"`
		Params    string                         `json:"params"`
	}

	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}

	*r = Execution(s.tmp)

	r.CreatedAt = time.Time(s.CreatedAt)
	r.UpdatedAt = time.Time(s.UpdatedAt)

	if s.Input != "" {
		if err := json.Unmarshal([]byte(s.Input), &r.Input); err != nil {
			return err
		}
	}

	if s.Output != "" {
		if err := json.Unmarshal([]byte(s.Output), &r.Output); err != nil {
			return err
		}
	}

	if s.Params != "" {
		if err := json.Unmarshal([]byte(s.Params), &r.Params); err != nil {
			return err
		}
	}

	return nil
}

// ExecutionPage contains a single page of all executions from a List call.
type ExecutionPage struct {
	pagination.LinkedPageBase
}

// IsEmpty checks if an ExecutionPage contains any results.
func (r ExecutionPage) IsEmpty() (bool, error) {
	exec, err := ExtractExecutions(r)
	return len(exec) == 0, err
}

// NextPageURL finds the next page URL in a page in order to navigate to the next page of results.
func (r ExecutionPage) NextPageURL() (string, error) {
	var s struct {
		Next string `json:"next"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return s.Next, nil
}

// ExtractExecutions get the list of executions from a page acquired from the List call.
func ExtractExecutions(r pagination.Page) ([]Execution, error) {
	var s struct {
		Executions []Execution `json:"executions"`
	}
	err := (r.(ExecutionPage)).ExtractInto(&s)
	return s.Executions, err
}
