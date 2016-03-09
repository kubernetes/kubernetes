package stackevents

import (
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// Find retrieves stack events for the given stack name.
func Find(c *gophercloud.ServiceClient, stackName string) FindResult {
	var res FindResult

	_, res.Err = c.Request("GET", findURL(c, stackName), gophercloud.RequestOpts{
		JSONResponse: &res.Body,
	})
	return res
}

// SortDir is a type for specifying in which direction to sort a list of events.
type SortDir string

// SortKey is a type for specifying by which key to sort a list of events.
type SortKey string

// ResourceStatus is a type for specifying by which resource status to filter a
// list of events.
type ResourceStatus string

// ResourceAction is a type for specifying by which resource action to filter a
// list of events.
type ResourceAction string

var (
	// ResourceStatusInProgress is used to filter a List request by the 'IN_PROGRESS' status.
	ResourceStatusInProgress ResourceStatus = "IN_PROGRESS"
	// ResourceStatusComplete is used to filter a List request by the 'COMPLETE' status.
	ResourceStatusComplete ResourceStatus = "COMPLETE"
	// ResourceStatusFailed is used to filter a List request by the 'FAILED' status.
	ResourceStatusFailed ResourceStatus = "FAILED"

	// ResourceActionCreate is used to filter a List request by the 'CREATE' action.
	ResourceActionCreate ResourceAction = "CREATE"
	// ResourceActionDelete is used to filter a List request by the 'DELETE' action.
	ResourceActionDelete ResourceAction = "DELETE"
	// ResourceActionUpdate is used to filter a List request by the 'UPDATE' action.
	ResourceActionUpdate ResourceAction = "UPDATE"
	// ResourceActionRollback is used to filter a List request by the 'ROLLBACK' action.
	ResourceActionRollback ResourceAction = "ROLLBACK"
	// ResourceActionSuspend is used to filter a List request by the 'SUSPEND' action.
	ResourceActionSuspend ResourceAction = "SUSPEND"
	// ResourceActionResume is used to filter a List request by the 'RESUME' action.
	ResourceActionResume ResourceAction = "RESUME"
	// ResourceActionAbandon is used to filter a List request by the 'ABANDON' action.
	ResourceActionAbandon ResourceAction = "ABANDON"

	// SortAsc is used to sort a list of stacks in ascending order.
	SortAsc SortDir = "asc"
	// SortDesc is used to sort a list of stacks in descending order.
	SortDesc SortDir = "desc"

	// SortName is used to sort a list of stacks by name.
	SortName SortKey = "name"
	// SortResourceType is used to sort a list of stacks by resource type.
	SortResourceType SortKey = "resource_type"
	// SortCreatedAt is used to sort a list of stacks by date created.
	SortCreatedAt SortKey = "created_at"
)

// ListOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListOptsBuilder interface {
	ToStackEventListQuery() (string, error)
}

// ListOpts allows the filtering and sorting of paginated collections through
// the API. Marker and Limit are used for pagination.
type ListOpts struct {
	// The stack resource ID with which to start the listing.
	Marker string `q:"marker"`
	// Integer value for the limit of values to return.
	Limit int `q:"limit"`
	// Filters the event list by the specified ResourceAction. You can use this
	// filter multiple times to filter by multiple resource actions: CREATE, DELETE,
	// UPDATE, ROLLBACK, SUSPEND, RESUME or ADOPT.
	ResourceActions []ResourceAction `q:"resource_action"`
	// Filters the event list by the specified resource_status. You can use this
	// filter multiple times to filter by multiple resource statuses: IN_PROGRESS,
	// COMPLETE or FAILED.
	ResourceStatuses []ResourceStatus `q:"resource_status"`
	// Filters the event list by the specified resource_name. You can use this
	// filter multiple times to filter by multiple resource names.
	ResourceNames []string `q:"resource_name"`
	// Filters the event list by the specified resource_type. You can use this
	// filter multiple times to filter by multiple resource types: OS::Nova::Server,
	// OS::Cinder::Volume, and so on.
	ResourceTypes []string `q:"resource_type"`
	// Sorts the event list by: resource_type or created_at.
	SortKey SortKey `q:"sort_keys"`
	// The sort direction of the event list. Which is asc (ascending) or desc (descending).
	SortDir SortDir `q:"sort_dir"`
}

// ToStackEventListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToStackEventListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	if err != nil {
		return "", err
	}
	return q.String(), nil
}

// List makes a request against the API to list resources for the given stack.
func List(client *gophercloud.ServiceClient, stackName, stackID string, opts ListOptsBuilder) pagination.Pager {
	url := listURL(client, stackName, stackID)

	if opts != nil {
		query, err := opts.ToStackEventListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}

	createPageFn := func(r pagination.PageResult) pagination.Page {
		p := EventPage{pagination.MarkerPageBase{PageResult: r}}
		p.MarkerPageBase.Owner = p
		return p
	}

	return pagination.NewPager(client, url, createPageFn)
}

// ListResourceEventsOptsBuilder allows extensions to add additional parameters to the
// ListResourceEvents request.
type ListResourceEventsOptsBuilder interface {
	ToResourceEventListQuery() (string, error)
}

// ListResourceEventsOpts allows the filtering and sorting of paginated resource events through
// the API. Marker and Limit are used for pagination.
type ListResourceEventsOpts struct {
	// The stack resource ID with which to start the listing.
	Marker string `q:"marker"`
	// Integer value for the limit of values to return.
	Limit int `q:"limit"`
	// Filters the event list by the specified ResourceAction. You can use this
	// filter multiple times to filter by multiple resource actions: CREATE, DELETE,
	// UPDATE, ROLLBACK, SUSPEND, RESUME or ADOPT.
	ResourceActions []string `q:"resource_action"`
	// Filters the event list by the specified resource_status. You can use this
	// filter multiple times to filter by multiple resource statuses: IN_PROGRESS,
	// COMPLETE or FAILED.
	ResourceStatuses []string `q:"resource_status"`
	// Filters the event list by the specified resource_name. You can use this
	// filter multiple times to filter by multiple resource names.
	ResourceNames []string `q:"resource_name"`
	// Filters the event list by the specified resource_type. You can use this
	// filter multiple times to filter by multiple resource types: OS::Nova::Server,
	// OS::Cinder::Volume, and so on.
	ResourceTypes []string `q:"resource_type"`
	// Sorts the event list by: resource_type or created_at.
	SortKey SortKey `q:"sort_keys"`
	// The sort direction of the event list. Which is asc (ascending) or desc (descending).
	SortDir SortDir `q:"sort_dir"`
}

// ToResourceEventListQuery formats a ListResourceEventsOpts into a query string.
func (opts ListResourceEventsOpts) ToResourceEventListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	if err != nil {
		return "", err
	}
	return q.String(), nil
}

// ListResourceEvents makes a request against the API to list resources for the given stack.
func ListResourceEvents(client *gophercloud.ServiceClient, stackName, stackID, resourceName string, opts ListResourceEventsOptsBuilder) pagination.Pager {
	url := listResourceEventsURL(client, stackName, stackID, resourceName)

	if opts != nil {
		query, err := opts.ToResourceEventListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}

	createPageFn := func(r pagination.PageResult) pagination.Page {
		p := EventPage{pagination.MarkerPageBase{PageResult: r}}
		p.MarkerPageBase.Owner = p
		return p
	}

	return pagination.NewPager(client, url, createPageFn)
}

// Get retreives data for the given stack resource.
func Get(c *gophercloud.ServiceClient, stackName, stackID, resourceName, eventID string) GetResult {
	var res GetResult
	_, res.Err = c.Get(getURL(c, stackName, stackID, resourceName, eventID), &res.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return res
}
