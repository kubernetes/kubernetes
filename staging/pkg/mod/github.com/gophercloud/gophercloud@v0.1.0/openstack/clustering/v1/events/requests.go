package events

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// ListOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListOptsBuilder interface {
	ToEventListQuery() (string, error)
}

// ListOpts represents options used to list events.
type ListOpts struct {
	Limit         int    `q:"limit,omitempty"`
	Level         int    `q:"level,omitempty"`
	Marker        string `q:"marker,omitempty"`
	Sort          string `q:"sort,omitempty"`
	GlobalProject *bool  `q:"global_project,omitempty"`
	OID           string `q:"oid,omitempty"`
	OType         string `q:"otype,omitempty"`
	OName         string `q:"oname,omitempty"`
	ClusterID     string `q:"cluster_id,omitempty"`
	Action        string `q:"action"`
}

// ToEventListQuery builds a query string from ListOpts.
func (opts ListOpts) ToEventListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List instructs OpenStack to provide a list of events.
func List(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listURL(client)
	if opts != nil {
		query, err := opts.ToEventListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return EventPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// Get retrieves details of a single event.
func Get(client *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = client.Get(getURL(client, id), &r.Body, &gophercloud.RequestOpts{OkCodes: []int{200}})
	return
}
