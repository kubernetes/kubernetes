package aggregates

import (
	"strconv"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// List makes a request against the API to list aggregates.
func List(client *gophercloud.ServiceClient) pagination.Pager {
	return pagination.NewPager(client, aggregatesListURL(client), func(r pagination.PageResult) pagination.Page {
		return AggregatesPage{pagination.SinglePageBase(r)}
	})
}

type CreateOpts struct {
	// The name of the host aggregate.
	Name string `json:"name" required:"true"`

	// The availability zone of the host aggregate.
	// You should use a custom availability zone rather than
	// the default returned by the os-availability-zone API.
	// The availability zone must not include ‘:’ in its name.
	AvailabilityZone string `json:"availability_zone,omitempty"`
}

func (opts CreateOpts) ToAggregatesCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "aggregate")
}

// Create makes a request against the API to create an aggregate.
func Create(client *gophercloud.ServiceClient, opts CreateOpts) (r CreateResult) {
	b, err := opts.ToAggregatesCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(aggregatesCreateURL(client), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// Delete makes a request against the API to delete an aggregate.
func Delete(client *gophercloud.ServiceClient, aggregateID int) (r DeleteResult) {
	v := strconv.Itoa(aggregateID)
	_, r.Err = client.Delete(aggregatesDeleteURL(client, v), &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// Get makes a request against the API to get details for a specific aggregate.
func Get(client *gophercloud.ServiceClient, aggregateID int) (r GetResult) {
	v := strconv.Itoa(aggregateID)
	_, r.Err = client.Get(aggregatesGetURL(client, v), &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

type UpdateOpts struct {
	// The name of the host aggregate.
	Name string `json:"name,omitempty"`

	// The availability zone of the host aggregate.
	// You should use a custom availability zone rather than
	// the default returned by the os-availability-zone API.
	// The availability zone must not include ‘:’ in its name.
	AvailabilityZone string `json:"availability_zone,omitempty"`
}

func (opts UpdateOpts) ToAggregatesUpdateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "aggregate")
}

// Update makes a request against the API to update a specific aggregate.
func Update(client *gophercloud.ServiceClient, aggregateID int, opts UpdateOpts) (r UpdateResult) {
	v := strconv.Itoa(aggregateID)

	b, err := opts.ToAggregatesUpdateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Put(aggregatesUpdateURL(client, v), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

type AddHostOpts struct {
	// The name of the host.
	Host string `json:"host" required:"true"`
}

func (opts AddHostOpts) ToAggregatesAddHostMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "add_host")
}

// AddHost makes a request against the API to add host to a specific aggregate.
func AddHost(client *gophercloud.ServiceClient, aggregateID int, opts AddHostOpts) (r ActionResult) {
	v := strconv.Itoa(aggregateID)

	b, err := opts.ToAggregatesAddHostMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(aggregatesAddHostURL(client, v), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

type RemoveHostOpts struct {
	// The name of the host.
	Host string `json:"host" required:"true"`
}

func (opts RemoveHostOpts) ToAggregatesRemoveHostMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "remove_host")
}

// RemoveHost makes a request against the API to remove host from a specific aggregate.
func RemoveHost(client *gophercloud.ServiceClient, aggregateID int, opts RemoveHostOpts) (r ActionResult) {
	v := strconv.Itoa(aggregateID)

	b, err := opts.ToAggregatesRemoveHostMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(aggregatesRemoveHostURL(client, v), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

type SetMetadataOpts struct {
	Metadata map[string]interface{} `json:"metadata" required:"true"`
}

func (opts SetMetadataOpts) ToSetMetadataMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "set_metadata")
}

// SetMetadata makes a request against the API to set metadata to a specific aggregate.
func SetMetadata(client *gophercloud.ServiceClient, aggregateID int, opts SetMetadataOpts) (r ActionResult) {
	v := strconv.Itoa(aggregateID)

	b, err := opts.ToSetMetadataMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(aggregatesSetMetadataURL(client, v), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}
