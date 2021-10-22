package receivers

import (
	"net/http"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// ReceiverType represents a valid type of receiver
type ReceiverType string

const (
	WebhookReceiver ReceiverType = "webhook"
	MessageReceiver ReceiverType = "message"
)

// CreateOptsBuilder allows extensions to add additional parameters to the
// Create request.
type CreateOptsBuilder interface {
	ToReceiverCreateMap() (map[string]interface{}, error)
}

// CreatOpts represents options used to create a receiver.
type CreateOpts struct {
	Name      string                 `json:"name" required:"true"`
	ClusterID string                 `json:"cluster_id,omitempty"`
	Type      ReceiverType           `json:"type" required:"true"`
	Action    string                 `json:"action,omitempty"`
	Actor     map[string]interface{} `json:"actor,omitempty"`
	Params    map[string]interface{} `json:"params,omitempty"`
}

// ToReceiverCreateMap constructs a request body from CreateOpts.
func (opts CreateOpts) ToReceiverCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "receiver")
}

// Create requests the creation of a new receiver.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToReceiverCreateMap()
	if err != nil {
		r.Err = err
		return
	}

	_, r.Err = client.Post(createURL(client), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 201, 202},
	})
	return
}

// UpdateOptsBuilder allows extensions to add additional parameters to the
// Update request.
type UpdateOptsBuilder interface {
	ToReceiverUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts represents options used to update a receiver.
type UpdateOpts struct {
	Name   string                 `json:"name,omitempty"`
	Action string                 `json:"action,omitempty"`
	Params map[string]interface{} `json:"params,omitempty"`
}

// ToReceiverUpdateMap constructs a request body from UpdateOpts.
func (opts UpdateOpts) ToReceiverUpdateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "receiver")
}

// Update requests the update of a receiver.
func Update(client *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) (r UpdateResult) {
	b, err := opts.ToReceiverUpdateMap()
	if err != nil {
		r.Err = err
		return
	}

	var result *http.Response
	result, r.Err = client.Patch(updateURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 201},
	})

	if r.Err == nil {
		r.Header = result.Header
	}
	return
}

// Get retrieves details of a single receiver. Use Extract to convert its result into a Receiver.
func Get(client *gophercloud.ServiceClient, id string) (r GetResult) {
	var result *http.Response
	result, r.Err = client.Get(getURL(client, id), &r.Body, &gophercloud.RequestOpts{OkCodes: []int{200}})
	if r.Err == nil {
		r.Header = result.Header
	}
	return
}

// ListOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListOptsBuilder interface {
	ToReceiverListQuery() (string, error)
}

// ListOpts represents options used to list recievers.
type ListOpts struct {
	Limit         int    `q:"limit"`
	Marker        string `q:"marker"`
	Sort          string `q:"sort"`
	GlobalProject *bool  `q:"global_project"`
	Name          string `q:"name"`
	Type          string `q:"type"`
	ClusterID     string `q:"cluster_id"`
	Action        string `q:"action"`
	User          string `q:"user"`
}

// ToReceiverListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToReceiverListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List instructs OpenStack to provide a list of cluster.
func List(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listURL(client)
	if opts != nil {
		query, err := opts.ToReceiverListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}

	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return ReceiverPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// Delete deletes the specified receiver ID.
func Delete(client *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = client.Delete(deleteURL(client, id), nil)
	return
}

// Notify Notifies message type receiver
func Notify(client *gophercloud.ServiceClient, id string) (r NotifyResult) {
	result, err := client.Post(notifyURL(client, id), nil, nil, &gophercloud.RequestOpts{
		OkCodes: []int{204},
	})

	if err == nil {
		r.Header = result.Header
	}
	return
}
