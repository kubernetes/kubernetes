package orders

import (
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// OrderType represents the valid types of orders.
type OrderType string

const (
	KeyOrder        OrderType = "key"
	AsymmetricOrder OrderType = "asymmetric"
)

// ListOptsBuilder allows extensions to add additional parameters to
// the List request
type ListOptsBuilder interface {
	ToOrderListQuery() (string, error)
}

// ListOpts provides options to filter the List results.
type ListOpts struct {
	// Limit is the amount of containers to retrieve.
	Limit int `q:"limit"`

	// Offset is the index within the list to retrieve.
	Offset int `q:"offset"`
}

// ToOrderListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToOrderListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List retrieves a list of orders.
func List(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listURL(client)
	if opts != nil {
		query, err := opts.ToOrderListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return OrderPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// Get retrieves details of a orders.
func Get(client *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = client.Get(getURL(client, id), &r.Body, nil)
	return
}

// CreateOptsBuilder allows extensions to add additional parameters to
// the Create request.
type CreateOptsBuilder interface {
	ToOrderCreateMap() (map[string]interface{}, error)
}

// MetaOpts represents options used for creating an order.
type MetaOpts struct {
	// Algorithm is the algorithm of the secret.
	Algorithm string `json:"algorithm"`

	// BitLength is the bit length of the secret.
	BitLength int `json:"bit_length"`

	// Expiration is the expiration date of the order.
	Expiration *time.Time `json:"-"`

	// Mode is the mode of the secret.
	Mode string `json:"mode"`

	// Name is the name of the secret.
	Name string `json:"name,omitempty"`

	// PayloadContentType is the content type of the secret payload.
	PayloadContentType string `json:"payload_content_type,omitempty"`
}

// CreateOpts provides options used to create a orders.
type CreateOpts struct {
	// Type is the type of order to create.
	Type OrderType `json:"type"`

	// Meta contains secrets data to create a secret.
	Meta MetaOpts `json:"meta"`
}

// ToOrderCreateMap formats a CreateOpts into a create request.
func (opts CreateOpts) ToOrderCreateMap() (map[string]interface{}, error) {
	b, err := gophercloud.BuildRequestBody(opts, "")
	if err != nil {
		return nil, err
	}

	if opts.Meta.Expiration != nil {
		meta := b["meta"].(map[string]interface{})
		meta["expiration"] = opts.Meta.Expiration.Format(gophercloud.RFC3339NoZ)
		b["meta"] = meta
	}

	return b, nil
}

// Create creates a new orders.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToOrderCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(createURL(client), &b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{202},
	})
	return
}

// Delete deletes a orders.
func Delete(client *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = client.Delete(deleteURL(client, id), nil)
	return
}
