package tasks

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// TaskStatus represents valid task status.
// You can use this type to compare the actual status of a task to a one of the
// pre-defined statuses.
type TaskStatus string

const (
	// TaskStatusPending represents status of the pending task.
	TaskStatusPending TaskStatus = "pending"

	// TaskStatusProcessing represents status of the processing task.
	TaskStatusProcessing TaskStatus = "processing"

	// TaskStatusSuccess represents status of the success task.
	TaskStatusSuccess TaskStatus = "success"

	// TaskStatusFailure represents status of the failure task.
	TaskStatusFailure TaskStatus = "failure"
)

// ListOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListOptsBuilder interface {
	ToTaskListQuery() (string, error)
}

// ListOpts allows the filtering and sorting of paginated collections through
// the OpenStack Imageservice tasks API.
type ListOpts struct {
	// Integer value for the limit of values to return.
	Limit int `q:"limit"`

	// ID of the task at which you want to set a marker.
	Marker string `q:"marker"`

	// SortDir allows to select sort direction.
	// It can be "asc" or "desc" (default).
	SortDir string `q:"sort_dir"`

	// SortKey allows to sort by one of the following tTask attributes:
	//  - created_at
	//  - expires_at
	//  - status
	//  - type
	//  - updated_at
	// Default is created_at.
	SortKey string `q:"sort_key"`

	// ID filters on the identifier of the task.
	ID string `json:"id"`

	// Type filters on the type of the task.
	Type string `json:"type"`

	// Status filters on the status of the task.
	Status TaskStatus `q:"status"`
}

// ToTaskListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToTaskListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List returns a Pager which allows you to iterate over a collection of the tasks.
func List(c *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listURL(c)
	if opts != nil {
		query, err := opts.ToTaskListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(c, url, func(r pagination.PageResult) pagination.Page {
		taskPage := TaskPage{
			serviceURL:     c.ServiceURL(),
			LinkedPageBase: pagination.LinkedPageBase{PageResult: r},
		}

		return taskPage
	})
}

// Get retrieves a specific Imageservice task based on its ID.
func Get(c *gophercloud.ServiceClient, taskID string) (r GetResult) {
	_, r.Err = c.Get(getURL(c, taskID), &r.Body, nil)
	return
}

// CreateOptsBuilder allows to add additional parameters to the Create request.
type CreateOptsBuilder interface {
	ToTaskCreateMap() (map[string]interface{}, error)
}

// CreateOpts specifies parameters of a new Imageservice task.
type CreateOpts struct {
	Type  string                 `json:"type" required:"true"`
	Input map[string]interface{} `json:"input"`
}

// ToTaskCreateMap constructs a request body from CreateOpts.
func (opts CreateOpts) ToTaskCreateMap() (map[string]interface{}, error) {
	b, err := gophercloud.BuildRequestBody(opts, "")
	if err != nil {
		return nil, err
	}
	return b, nil
}

// Create requests the creation of a new Imageservice task on the server.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToTaskCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(createURL(client), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{201},
	})
	return
}
