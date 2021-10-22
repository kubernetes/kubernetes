package agents

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// ListOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListOptsBuilder interface {
	ToAgentListQuery() (string, error)
}

// ListOpts allows the filtering and sorting of paginated collections through
// the Neutron API. Filtering is achieved by passing in struct field values
// that map to the agent attributes you want to see returned.
// SortKey allows you to sort by a particular agent attribute.
// SortDir sets the direction, and is either `asc' or `desc'.
// Marker and Limit are used for the pagination.
type ListOpts struct {
	ID               string `q:"id"`
	AgentType        string `q:"agent_type"`
	Alive            *bool  `q:"alive"`
	AvailabilityZone string `q:"availability_zone"`
	Binary           string `q:"binary"`
	Description      string `q:"description"`
	Host             string `q:"host"`
	Topic            string `q:"topic"`
	Limit            int    `q:"limit"`
	Marker           string `q:"marker"`
	SortKey          string `q:"sort_key"`
	SortDir          string `q:"sort_dir"`
}

// ToAgentListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToAgentListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List returns a Pager which allows you to iterate over a collection of
// agents. It accepts a ListOpts struct, which allows you to filter and
// sort the returned collection for greater efficiency.
//
// Default policy settings return only the agents owned by the project
// of the user submitting the request, unless the user has the administrative
// role.
func List(c *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listURL(c)
	if opts != nil {
		query, err := opts.ToAgentListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(c, url, func(r pagination.PageResult) pagination.Page {
		return AgentPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// Get retrieves a specific agent based on its ID.
func Get(c *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = c.Get(getURL(c, id), &r.Body, nil)
	return
}
