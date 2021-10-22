package executions

import (
	"encoding/json"
	"fmt"
	"net/url"
	"reflect"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// CreateOptsBuilder allows extension to add additional parameters to the Create request.
type CreateOptsBuilder interface {
	ToExecutionCreateMap() (map[string]interface{}, error)
}

// CreateOpts specifies parameters used to create an execution.
type CreateOpts struct {
	// ID is the unique ID of the execution.
	ID string `json:"id,omitempty"`

	// SourceExecutionID can be set to create an execution based on another existing execution.
	SourceExecutionID string `json:"source_execution_id,omitempty"`

	// WorkflowID is the unique id of the workflow.
	WorkflowID string `json:"workflow_id,omitempty" or:"WorkflowName"`

	// WorkflowName is the name identifier of the workflow.
	WorkflowName string `json:"workflow_name,omitempty" or:"WorkflowID"`

	// WorkflowNamespace is the namespace of the workflow.
	WorkflowNamespace string `json:"workflow_namespace,omitempty"`

	// Input is a JSON structure containing workflow input values, serialized as string.
	Input map[string]interface{} `json:"input,omitempty"`

	// Params define workflow type specific parameters.
	Params map[string]interface{} `json:"params,omitempty"`

	// Description is the description of the workflow execution.
	Description string `json:"description,omitempty"`
}

// ToExecutionCreateMap constructs a request body from CreateOpts.
func (opts CreateOpts) ToExecutionCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "")
}

// Create requests the creation of a new execution.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToExecutionCreateMap()
	if err != nil {
		r.Err = err
		return
	}

	_, r.Err = client.Post(createURL(client), b, &r.Body, nil)

	return
}

// Get retrieves details of a single execution.
// Use ExtractExecution to convert its result into an Execution.
func Get(client *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = client.Get(getURL(client, id), &r.Body, nil)
	return
}

// Delete deletes the specified execution.
func Delete(client *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = client.Delete(deleteURL(client, id), nil)
	return
}

// ListOptsBuilder allows extension to add additional parameters to the List request.
type ListOptsBuilder interface {
	ToExecutionListQuery() (string, error)
}

// ListOpts filters the result returned by the List() function.
type ListOpts struct {
	// WorkflowName allows to filter by workflow name.
	WorkflowName *ListFilter `q:"-"`
	// WorkflowID allows to filter by workflow id.
	WorkflowID string `q:"workflow_id"`
	// Description allows to filter by execution description.
	Description *ListFilter `q:"-"`
	// Params allows to filter by specific parameters.
	Params map[string]interface{} `q:"-"`
	// TaskExecutionID allows to filter with a specific task execution id.
	TaskExecutionID string `q:"task_execution_id"`
	// RootExecutionID allows to filter with a specific root execution id.
	RootExecutionID string `q:"root_execution_id"`
	// State allows to filter by execution state.
	// Possible values are IDLE, RUNNING, PAUSED, SUCCESS, ERROR, CANCELLED.
	State *ListFilter `q:"-"`
	// StateInfo allows to filter by state info.
	StateInfo *ListFilter `q:"-"`
	// Input allows to filter by specific input.
	Input map[string]interface{} `q:"-"`
	// Output allows to filter by specific output.
	Output map[string]interface{} `q:"-"`
	// CreatedAt allows to filter by execution creation date.
	CreatedAt *ListDateFilter `q:"-"`
	// UpdatedAt allows to filter by last execution update date.
	UpdatedAt *ListDateFilter `q:"-"`
	// IncludeOutput requests to include the output for all executions in the list.
	IncludeOutput bool `q:"-"`
	// ProjectID allows to filter by given project id. Admin required.
	ProjectID string `q:"project_id"`
	// AllProjects requests to get executions of all projects. Admin required.
	AllProjects int `q:"all_projects"`
	// SortDir allows to select sort direction.
	// It can be "asc" or "desc" (default).
	SortDirs string `q:"sort_dirs"`
	// SortKey allows to sort by one of the execution attributes.
	SortKeys string `q:"sort_keys"`
	// Marker and Limit control paging.
	// Marker instructs List where to start listing from.
	Marker string `q:"marker"`
	// Limit instructs List to refrain from sending excessively large lists of
	// executions.
	Limit int `q:"limit"`
}

// ListFilter allows to filter string parameters with different filters.
// Empty value for Filter checks for equality.
type ListFilter struct {
	Filter FilterType
	Value  string
}

func (l ListFilter) String() string {
	if l.Filter != "" {
		return fmt.Sprintf("%s:%s", l.Filter, l.Value)
	}

	return l.Value
}

// ListDateFilter allows to filter date parameters with different filters.
// Empty value for Filter checks for equality.
type ListDateFilter struct {
	Filter FilterType
	Value  time.Time
}

func (l ListDateFilter) String() string {
	v := l.Value.Format(gophercloud.RFC3339ZNoTNoZ)

	if l.Filter != "" {
		return fmt.Sprintf("%s:%s", l.Filter, v)
	}

	return v
}

// FilterType represents a valid filter to use for filtering executions.
type FilterType string

const (
	// FilterEQ checks equality.
	FilterEQ = "eq"
	// FilterNEQ checks non equality.
	FilterNEQ = "neq"
	// FilterIN checks for belonging in a list, comma separated.
	FilterIN = "in"
	// FilterNIN checks for values that does not belong from a list, comma separated.
	FilterNIN = "nin"
	// FilterGT checks for values strictly greater.
	FilterGT = "gt"
	// FilterGTE checks for values greater or equal.
	FilterGTE = "gte"
	// FilterLT checks for values strictly lower.
	FilterLT = "lt"
	// FilterLTE checks for values lower or equal.
	FilterLTE = "lte"
	// FilterHas checks for values that contains the requested parameter.
	FilterHas = "has"
)

// ToExecutionListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToExecutionListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	if err != nil {
		return "", err
	}

	params := q.Query()

	if opts.IncludeOutput {
		params.Add("include_output", "1")
	}

	for queryParam, value := range map[string]map[string]interface{}{"params": opts.Params, "input": opts.Input, "output": opts.Output} {
		if value != nil {
			b, err := json.Marshal(value)
			if err != nil {
				return "", err
			}
			params.Add(queryParam, string(b))
		}
	}

	for queryParam, value := range map[string]fmt.Stringer{
		"created_at":    opts.CreatedAt,
		"updated_at":    opts.UpdatedAt,
		"workflow_name": opts.WorkflowName,
		"description":   opts.Description,
		"state":         opts.State,
		"state_info":    opts.StateInfo,
	} {
		if !reflect.ValueOf(value).IsNil() {
			params.Add(queryParam, value.String())
		}
	}

	q = &url.URL{RawQuery: params.Encode()}
	return q.String(), nil
}

// List performs a call to list executions.
// You may provide options to filter the executions.
func List(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listURL(client)
	if opts != nil {
		query, err := opts.ToExecutionListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}

	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return ExecutionPage{pagination.LinkedPageBase{PageResult: r}}
	})
}
