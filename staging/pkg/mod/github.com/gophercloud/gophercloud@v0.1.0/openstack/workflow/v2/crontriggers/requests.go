package crontriggers

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
	ToCronTriggerCreateMap() (map[string]interface{}, error)
}

// CreateOpts specifies parameters used to create a cron trigger.
type CreateOpts struct {
	// Name is the cron trigger name.
	Name string `json:"name"`

	// Pattern is a Unix crontab patterns format to execute the workflow.
	Pattern string `json:"pattern"`

	// RemainingExecutions sets the number of executions for the trigger.
	RemainingExecutions int `json:"remaining_executions,omitempty"`

	// WorkflowID is the unique id of the workflow.
	WorkflowID string `json:"workflow_id,omitempty" or:"WorkflowName"`

	// WorkflowName is the name of the workflow.
	// It is recommended to refer to workflow by the WorkflowID parameter instead of WorkflowName.
	WorkflowName string `json:"workflow_name,omitempty" or:"WorkflowID"`

	// WorkflowParams defines workflow type specific parameters.
	WorkflowParams map[string]interface{} `json:"workflow_params,omitempty"`

	// WorkflowInput defines workflow input values.
	WorkflowInput map[string]interface{} `json:"workflow_input,omitempty"`

	// FirstExecutionTime defines the first execution time of the trigger.
	FirstExecutionTime *time.Time `json:"-"`
}

// ToCronTriggerCreateMap constructs a request body from CreateOpts.
func (opts CreateOpts) ToCronTriggerCreateMap() (map[string]interface{}, error) {
	b, err := gophercloud.BuildRequestBody(opts, "")
	if err != nil {
		return nil, err
	}

	if opts.FirstExecutionTime != nil {
		b["first_execution_time"] = opts.FirstExecutionTime.Format("2006-01-02 15:04")
	}

	return b, nil
}

// Create requests the creation of a new cron trigger.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToCronTriggerCreateMap()
	if err != nil {
		r.Err = err
		return
	}

	_, r.Err = client.Post(createURL(client), b, &r.Body, nil)

	return
}

// Delete deletes the specified cron trigger.
func Delete(client *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = client.Delete(deleteURL(client, id), nil)
	return
}

// Get retrieves details of a single cron trigger.
// Use Extract to convert its result into an CronTrigger.
func Get(client *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = client.Get(getURL(client, id), &r.Body, nil)
	return
}

// ListOptsBuilder allows extension to add additional parameters to the List request.
type ListOptsBuilder interface {
	ToCronTriggerListQuery() (string, error)
}

// ListOpts filters the result returned by the List() function.
type ListOpts struct {
	// WorkflowName allows to filter by workflow name.
	WorkflowName *ListFilter `q:"-"`
	// WorkflowID allows to filter by workflow id.
	WorkflowID string `q:"workflow_id"`
	// WorkflowInput allows to filter by specific workflow inputs.
	WorkflowInput map[string]interface{} `q:"-"`
	// WorkflowParams allows to filter by specific workflow parameters.
	WorkflowParams map[string]interface{} `q:"-"`
	// Scope filters by the trigger's scope.
	// Values can be "private" or "public".
	Scope string `q:"scope"`
	// Name allows to filter by trigger name.
	Name *ListFilter `q:"-"`
	// Pattern allows to filter by pattern.
	Pattern *ListFilter `q:"-"`
	// RemainingExecutions allows to filter by remaining executions.
	RemainingExecutions *ListIntFilter `q:"-"`
	// FirstExecutionTime allows to filter by first execution time.
	FirstExecutionTime *ListDateFilter `q:"-"`
	// NextExecutionTime allows to filter by next execution time.
	NextExecutionTime *ListDateFilter `q:"-"`
	// CreatedAt allows to filter by trigger creation date.
	CreatedAt *ListDateFilter `q:"-"`
	// UpdatedAt allows to filter by trigger last update date.
	UpdatedAt *ListDateFilter `q:"-"`
	// ProjectID allows to filter by given project id. Admin required.
	ProjectID string `q:"project_id"`
	// AllProjects requests to get executions of all projects. Admin required.
	AllProjects int `q:"all_projects"`
	// SortDirs allows to select sort direction.
	// It can be "asc" or "desc" (default).
	SortDirs string `q:"sort_dirs"`
	// SortKeys allows to sort by one of the cron trigger attributes.
	SortKeys string `q:"sort_key"`
	// Marker and Limit control paging.
	// Marker instructs List where to start listing from.
	Marker string `q:"marker"`
	// Limit instructs List to refrain from sending excessively large lists of
	// cron triggers.
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

// ListIntFilter allows to filter integer parameters with different filters.
// Empty value for Filter checks for equality.
type ListIntFilter struct {
	Filter FilterType
	Value  int
}

func (l ListIntFilter) String() string {
	v := fmt.Sprintf("%d", l.Value)
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

// ToCronTriggerListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToCronTriggerListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	if err != nil {
		return "", err
	}
	params := q.Query()

	for queryParam, value := range map[string]map[string]interface{}{"workflow_params": opts.WorkflowParams, "workflow_input": opts.WorkflowInput} {
		if value != nil {
			b, err := json.Marshal(value)
			if err != nil {
				return "", err
			}
			params.Add(queryParam, string(b))
		}
	}

	for queryParam, value := range map[string]fmt.Stringer{
		"workflow_name":        opts.WorkflowName,
		"name":                 opts.Name,
		"pattern":              opts.Pattern,
		"remaining_executions": opts.RemainingExecutions,
		"first_execution_time": opts.FirstExecutionTime,
		"next_execution_time":  opts.NextExecutionTime,
		"created_at":           opts.CreatedAt,
		"updated_at":           opts.UpdatedAt,
	} {
		if !reflect.ValueOf(value).IsNil() {
			params.Add(queryParam, value.String())
		}
	}
	q = &url.URL{RawQuery: params.Encode()}
	return q.String(), nil
}

// List performs a call to list cron triggers.
// You may provide options to filter the results.
func List(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listURL(client)
	if opts != nil {
		query, err := opts.ToCronTriggerListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return CronTriggerPage{pagination.LinkedPageBase{PageResult: r}}
	})
}
