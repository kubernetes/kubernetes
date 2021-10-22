package allocations

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// CreateOptsBuilder allows extensions to add additional parameters to the
// Create request.
type CreateOptsBuilder interface {
	ToAllocationCreateMap() (map[string]interface{}, error)
}

// CreateOpts specifies allocation creation parameters
type CreateOpts struct {
	// The requested resource class for the allocation.
	ResourceClass string `json:"resource_class" required:"true"`

	// The list of nodes (names or UUIDs) that should be considered for this allocation. If not provided, all available nodes will be considered.
	CandidateNodes []string `json:"candidate_nodes,omitempty"`

	// The unique name of the Allocation.
	Name string `json:"name,omitempty"`

	// The list of requested traits for the allocation.
	Traits []string `json:"traits,omitempty"`

	// The UUID for the resource.
	UUID string `json:"uuid,omitempty"`

	// A set of one or more arbitrary metadata key and value pairs.
	Extra map[string]string `json:"extra,omitempty"`
}

// ToAllocationCreateMap assembles a request body based on the contents of a CreateOpts.
func (opts CreateOpts) ToAllocationCreateMap() (map[string]interface{}, error) {
	body, err := gophercloud.BuildRequestBody(opts, "")
	if err != nil {
		return nil, err
	}

	return body, nil
}

// Create requests a node to be created
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	reqBody, err := opts.ToAllocationCreateMap()
	if err != nil {
		r.Err = err
		return
	}

	_, r.Err = client.Post(createURL(client), reqBody, &r.Body, nil)
	return
}

type AllocationState string

var (
	Allocating AllocationState = "allocating"
	Active                     = "active"
	Error                      = "error"
)

// ListOptsBuilder allows extensions to add additional parameters to the List request.
type ListOptsBuilder interface {
	ToAllocationListQuery() (string, error)
}

// ListOpts allows the filtering and sorting of paginated collections through the API.
type ListOpts struct {
	// Filter the list of allocations by the node UUID or name.
	Node string `q:"node"`

	// Filter the list of returned nodes, and only return the ones with the specified resource class.
	ResourceClass string `q:"resource_class"`

	// Filter the list of allocations by the allocation state, one of active, allocating or error.
	State AllocationState `q:"state"`

	// One or more fields to be returned in the response.
	Fields []string `q:"fields"`

	// Requests a page size of items.
	Limit int `q:"limit"`

	// The ID of the last-seen item
	Marker string `q:"marker"`

	// Sorts the response by the requested sort direction.
	// Valid value is asc (ascending) or desc (descending). Default is asc.
	SortDir string `q:"sort_dir"`

	// Sorts the response by the this attribute value. Default is id.
	SortKey string `q:"sort_key"`
}

// ToAllocationListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToAllocationListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List makes a request against the API to list allocations accessible to you.
func List(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listURL(client)
	if opts != nil {
		query, err := opts.ToAllocationListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return AllocationPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// Get requests the details of an allocation by ID.
func Get(client *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = client.Get(getURL(client, id), &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// Delete requests the deletion of an allocation
func Delete(client *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = client.Delete(deleteURL(client, id), nil)
	return
}
