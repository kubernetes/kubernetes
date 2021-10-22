package containers

import (
	"encoding/json"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Container represents a container in the key manager service.
type Container struct {
	// Consumers are the consumers of the container.
	Consumers []ConsumerRef `json:"consumers"`

	// ContainerRef is the URL to the container
	ContainerRef string `json:"container_ref"`

	// Created is the date the container was created.
	Created time.Time `json:"-"`

	// CreatorID is the creator of the container.
	CreatorID string `json:"creator_id"`

	// Name is the name of the container.
	Name string `json:"name"`

	// SecretRefs are the secret references of the container.
	SecretRefs []SecretRef `json:"secret_refs"`

	// Status is the status of the container.
	Status string `json:"status"`

	// Type is the type of container.
	Type string `json:"type"`

	// Updated is the date the container was updated.
	Updated time.Time `json:"-"`
}

func (r *Container) UnmarshalJSON(b []byte) error {
	type tmp Container
	var s struct {
		tmp
		Created gophercloud.JSONRFC3339NoZ `json:"created"`
		Updated gophercloud.JSONRFC3339NoZ `json:"updated"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = Container(s.tmp)

	r.Created = time.Time(s.Created)
	r.Updated = time.Time(s.Updated)

	return nil
}

// ConsumerRef represents a consumer reference in a container.
type ConsumerRef struct {
	// Name is the name of the consumer.
	Name string `json:"name"`

	// URL is the URL to the consumer resource.
	URL string `json:"url"`
}

// SecretRef is a reference to a secret.
type SecretRef struct {
	SecretRef string `json:"secret_ref"`
	Name      string `json:"name"`
}

type commonResult struct {
	gophercloud.Result
}

// Extract interprets any commonResult as a Container.
func (r commonResult) Extract() (*Container, error) {
	var s *Container
	err := r.ExtractInto(&s)
	return s, err
}

// GetResult is the response from a Get operation. Call its Extract method
// to interpret it as a container.
type GetResult struct {
	commonResult
}

// CreateResult is the response from a Create operation. Call its Extract method
// to interpret it as a container.
type CreateResult struct {
	commonResult
}

// DeleteResult is the response from a Delete operation. Call its ExtractErr to
// determine if the request succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}

// ContainerPage is a single page of container results.
type ContainerPage struct {
	pagination.LinkedPageBase
}

// IsEmpty determines whether or not a page of Container contains any results.
func (r ContainerPage) IsEmpty() (bool, error) {
	containers, err := ExtractContainers(r)
	return len(containers) == 0, err
}

// NextPageURL extracts the "next" link from the links section of the result.
func (r ContainerPage) NextPageURL() (string, error) {
	var s struct {
		Next     string `json:"next"`
		Previous string `json:"previous"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return s.Next, err
}

// ExtractContainers returns a slice of Containers contained in a single page of
// results.
func ExtractContainers(r pagination.Page) ([]Container, error) {
	var s struct {
		Containers []Container `json:"containers"`
	}
	err := (r.(ContainerPage)).ExtractInto(&s)
	return s.Containers, err
}

// Consumer represents a consumer in a container.
type Consumer struct {
	// Created is the date the container was created.
	Created time.Time `json:"-"`

	// Name is the name of the container.
	Name string `json:"name"`

	// Status is the status of the container.
	Status string `json:"status"`

	// Updated is the date the container was updated.
	Updated time.Time `json:"-"`

	// URL is the url to the consumer.
	URL string `json:"url"`
}

func (r *Consumer) UnmarshalJSON(b []byte) error {
	type tmp Consumer
	var s struct {
		tmp
		Created gophercloud.JSONRFC3339NoZ `json:"created"`
		Updated gophercloud.JSONRFC3339NoZ `json:"updated"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = Consumer(s.tmp)

	r.Created = time.Time(s.Created)
	r.Updated = time.Time(s.Updated)

	return nil
}

type consumerResult struct {
	gophercloud.Result
}

// Extract interprets any consumerResult as a Consumer.
func (r consumerResult) Extract() (*Consumer, error) {
	var s *Consumer
	err := r.ExtractInto(&s)
	return s, err
}

// CreateConsumerResult is the response from a CreateConsumer operation.
// Call its Extract method to interpret it as a container.
type CreateConsumerResult struct {
	// This is not a typo.
	commonResult
}

// DeleteConsumerResult is the response from a DeleteConsumer operation.
// Call its Extract to interpret it as a container.
type DeleteConsumerResult struct {
	// This is not a typo.
	commonResult
}

// ConsumerPage is a single page of consumer results.
type ConsumerPage struct {
	pagination.LinkedPageBase
}

// IsEmpty determines whether or not a page of consumers contains any results.
func (r ConsumerPage) IsEmpty() (bool, error) {
	consumers, err := ExtractConsumers(r)
	return len(consumers) == 0, err
}

// NextPageURL extracts the "next" link from the links section of the result.
func (r ConsumerPage) NextPageURL() (string, error) {
	var s struct {
		Next     string `json:"next"`
		Previous string `json:"previous"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return s.Next, err
}

// ExtractConsumers returns a slice of Consumers contained in a single page of
// results.
func ExtractConsumers(r pagination.Page) ([]Consumer, error) {
	var s struct {
		Consumers []Consumer `json:"consumers"`
	}
	err := (r.(ConsumerPage)).ExtractInto(&s)
	return s.Consumers, err
}
