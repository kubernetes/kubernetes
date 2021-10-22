package workflows

import (
	"fmt"
	"io"
	"net/url"
	"reflect"
	"strings"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// CreateOptsBuilder allows extension to add additional parameters to the Create request.
type CreateOptsBuilder interface {
	ToWorkflowCreateParams() (io.Reader, string, error)
}

// CreateOpts specifies parameters used to create a cron trigger.
type CreateOpts struct {
	// Scope is the scope of the workflow.
	// Allowed values are "private" and "public".
	Scope string `q:"scope"`

	// Namespace will define the namespace of the workflow.
	Namespace string `q:"namespace"`

	// Definition is the workflow definition written in Mistral Workflow Language v2.
	Definition io.Reader
}

// ToWorkflowCreateParams constructs a request query string from CreateOpts.
func (opts CreateOpts) ToWorkflowCreateParams() (io.Reader, string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return opts.Definition, q.String(), err
}

// Create requests the creation of a new execution.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	url := createURL(client)
	var b io.Reader
	if opts != nil {
		tmpB, query, err := opts.ToWorkflowCreateParams()
		if err != nil {
			r.Err = err
			return
		}
		url += query
		b = tmpB
	}

	_, r.Err = client.Post(url, nil, &r.Body, &gophercloud.RequestOpts{
		RawBody: b,
		MoreHeaders: map[string]string{
			"Content-Type": "text/plain",
			"Accept":       "", // Drop default JSON Accept header
		},
	})

	return
}

// Delete deletes the specified execution.
func Delete(client *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = client.Delete(deleteURL(client, id), nil)
	return
}

// Get retrieves details of a single execution.
// Use Extract to convert its result into an Workflow.
func Get(client *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = client.Get(getURL(client, id), &r.Body, nil)
	return
}

// ListOptsBuilder allows extension to add additional parameters to the List request.
type ListOptsBuilder interface {
	ToWorkflowListQuery() (string, error)
}

// ListOpts filters the result returned by the List() function.
type ListOpts struct {
	// Scope filters by the workflow's scope.
	// Values can be "private" or "public".
	Scope string `q:"scope"`
	// CreatedAt allows to filter by workflow creation date.
	CreatedAt *ListDateFilter `q:"-"`
	// UpdatedAt allows to filter by last execution update date.
	UpdatedAt *ListDateFilter `q:"-"`
	// Name allows to filter by workflow name.
	Name *ListFilter `q:"-"`
	// Tags allows to filter by tags.
	Tags []string
	// Definition allows to filter by workflow definition.
	Definition *ListFilter `q:"-"`
	// Namespace allows to filter by workflow namespace.
	Namespace *ListFilter `q:"-"`
	// SortDirs allows to select sort direction.
	// It can be "asc" or "desc" (default).
	SortDirs string `q:"sort_dirs"`
	// SortKeys allows to sort by one of the cron trigger attributes.
	SortKeys string `q:"sort_keys"`
	// Marker and Limit control paging.
	// Marker instructs List where to start listing from.
	Marker string `q:"marker"`
	// Limit instructs List to refrain from sending excessively large lists of
	// cron triggers.
	Limit int `q:"limit"`
	// ProjectID allows to filter by given project id. Admin required.
	ProjectID string `q:"project_id"`
	// AllProjects requests to get executions of all projects. Admin required.
	AllProjects int `q:"all_projects"`
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

// ToWorkflowListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToWorkflowListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	if err != nil {
		return "", err
	}
	params := q.Query()

	if opts.Tags != nil {
		params.Add("tags", strings.Join(opts.Tags, ","))
	}

	for queryParam, value := range map[string]fmt.Stringer{
		"created_at": opts.CreatedAt,
		"updated_at": opts.UpdatedAt,
		"name":       opts.Name,
		"definition": opts.Definition,
		"namespace":  opts.Namespace,
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
		query, err := opts.ToWorkflowListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return WorkflowPage{pagination.LinkedPageBase{PageResult: r}}
	})
}
