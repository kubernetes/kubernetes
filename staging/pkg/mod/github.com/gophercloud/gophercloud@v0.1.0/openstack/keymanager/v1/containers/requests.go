package containers

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// ContainerType represents the valid types of containers.
type ContainerType string

const (
	GenericContainer     ContainerType = "generic"
	RSAContainer         ContainerType = "rsa"
	CertificateContainer ContainerType = "certificate"
)

// ListOptsBuilder allows extensions to add additional parameters to
// the List request
type ListOptsBuilder interface {
	ToContainerListQuery() (string, error)
}

// ListOpts provides options to filter the List results.
type ListOpts struct {
	// Limit is the amount of containers to retrieve.
	Limit int `q:"limit"`

	// Name is the name of the container
	Name string `q:"name"`

	// Offset is the index within the list to retrieve.
	Offset int `q:"offset"`
}

// ToContainerListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToContainerListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List retrieves a list of containers.
func List(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listURL(client)
	if opts != nil {
		query, err := opts.ToContainerListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return ContainerPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// Get retrieves details of a container.
func Get(client *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = client.Get(getURL(client, id), &r.Body, nil)
	return
}

// CreateOptsBuilder allows extensions to add additional parameters to
// the Create request.
type CreateOptsBuilder interface {
	ToContainerCreateMap() (map[string]interface{}, error)
}

// CreateOpts provides options used to create a container.
type CreateOpts struct {
	// Type represents the type of container.
	Type ContainerType `json:"type" required:"true"`

	// Name is the name of the container.
	Name string `json:"name"`

	// SecretRefs is a list of secret refs for the container.
	SecretRefs []SecretRef `json:"secret_refs"`
}

// ToContainerCreateMap formats a CreateOpts into a create request.
func (opts CreateOpts) ToContainerCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "")
}

// Create creates a new container.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToContainerCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(createURL(client), &b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{201},
	})
	return
}

// Delete deletes a container.
func Delete(client *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = client.Delete(deleteURL(client, id), nil)
	return
}

// ListConsumersOptsBuilder allows extensions to add additional parameters to
// the ListConsumers request
type ListConsumersOptsBuilder interface {
	ToContainerListConsumersQuery() (string, error)
}

// ListConsumersOpts provides options to filter the List results.
type ListConsumersOpts struct {
	// Limit is the amount of consumers to retrieve.
	Limit int `q:"limit"`

	// Offset is the index within the list to retrieve.
	Offset int `q:"offset"`
}

// ToContainerListConsumersQuery formats a ListConsumersOpts into a query
// string.
func (opts ListOpts) ToContainerListConsumersQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// ListConsumers retrieves a list of consumers from a container.
func ListConsumers(client *gophercloud.ServiceClient, containerID string, opts ListConsumersOptsBuilder) pagination.Pager {
	url := listConsumersURL(client, containerID)
	if opts != nil {
		query, err := opts.ToContainerListConsumersQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return ConsumerPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// CreateConsumerOptsBuilder allows extensions to add additional parameters to
// the Create request.
type CreateConsumerOptsBuilder interface {
	ToContainerConsumerCreateMap() (map[string]interface{}, error)
}

// CreateConsumerOpts provides options used to create a container.
type CreateConsumerOpts struct {
	// Name is the name of the consumer.
	Name string `json:"name"`

	// URL is the URL to the consumer resource.
	URL string `json:"URL"`
}

// ToContainerConsumerCreateMap formats a CreateConsumerOpts into a create
// request.
func (opts CreateConsumerOpts) ToContainerConsumerCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "")
}

// CreateConsumer creates a new consumer.
func CreateConsumer(client *gophercloud.ServiceClient, containerID string, opts CreateConsumerOptsBuilder) (r CreateConsumerResult) {
	b, err := opts.ToContainerConsumerCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(createConsumerURL(client, containerID), &b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// DeleteConsumerOptsBuilder allows extensions to add additional parameters to
// the Delete request.
type DeleteConsumerOptsBuilder interface {
	ToContainerConsumerDeleteMap() (map[string]interface{}, error)
}

// DeleteConsumerOpts represents options used for deleting a consumer.
type DeleteConsumerOpts struct {
	// Name is the name of the consumer.
	Name string `json:"name"`

	// URL is the URL to the consumer resource.
	URL string `json:"URL"`
}

// ToContainerConsumerDeleteMap formats a DeleteConsumerOpts into a create
// request.
func (opts DeleteConsumerOpts) ToContainerConsumerDeleteMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "")
}

// DeleteConsumer deletes a consumer.
func DeleteConsumer(client *gophercloud.ServiceClient, containerID string, opts DeleteConsumerOptsBuilder) (r DeleteConsumerResult) {
	url := deleteConsumerURL(client, containerID)

	b, err := opts.ToContainerConsumerDeleteMap()
	if err != nil {
		r.Err = err
		return
	}

	_, r.Err = client.Request("DELETE", url, &gophercloud.RequestOpts{
		JSONBody:     b,
		JSONResponse: &r.Body,
		OkCodes:      []int{200},
	})
	return
}
