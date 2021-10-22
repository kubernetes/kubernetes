package orders

import (
	"encoding/json"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Order represents an order in the key manager service.
type Order struct {
	// ContainerRef is the container URL.
	ContainerRef string `json:"container_ref"`

	// Created is when the order was created.
	Created time.Time `json:"-"`

	// CreatorID is the creator of the order.
	CreatorID string `json:"creator_id"`

	// ErrorReason is the reason of the error.
	ErrorReason string `json:"error_reason"`

	// ErrorStatusCode is the error status code.
	ErrorStatusCode string `json:"error_status_code"`

	// OrderRef is the order URL.
	OrderRef string `json:"order_ref"`

	// Meta is secret data about the order.
	Meta Meta `json:"meta"`

	// SecretRef is the secret URL.
	SecretRef string `json:"secret_ref"`

	// Status is the status of the order.
	Status string `json:"status"`

	// SubStatus is the status of the order.
	SubStatus string `json:"sub_status"`

	// SubStatusMessage is the message of the sub status.
	SubStatusMessage string `json:"sub_status_message"`

	// Type is the order type.
	Type string `json:"type"`

	// Updated is when the order was updated.
	Updated time.Time `json:"-"`
}

func (r *Order) UnmarshalJSON(b []byte) error {
	type tmp Order
	var s struct {
		tmp
		Created gophercloud.JSONRFC3339NoZ `json:"created"`
		Updated gophercloud.JSONRFC3339NoZ `json:"updated"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = Order(s.tmp)

	r.Created = time.Time(s.Created)
	r.Updated = time.Time(s.Updated)

	return nil
}

type Meta struct {
	// Algorithm is the algorithm of the secret.
	Algorithm string `json:"algorithm"`

	// BitLength is the bit length of the secret.
	BitLength int `json:"bit_length"`

	// Expiration is the expiration date of the order.
	Expiration time.Time `json:"-"`

	// Mode is the mode of the secret.
	Mode string `json:"mode"`

	// Name is the name of the secret.
	Name string `json:"name"`

	// PayloadContentType is the content type of the secret payload.
	PayloadContentType string `json:"payload_content_type"`
}

func (r *Meta) UnmarshalJSON(b []byte) error {
	type tmp Meta
	var s struct {
		tmp
		Expiration gophercloud.JSONRFC3339NoZ `json:"expiration"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = Meta(s.tmp)

	r.Expiration = time.Time(s.Expiration)

	return nil
}

type commonResult struct {
	gophercloud.Result
}

// GetResult is the response from a Get operation. Call its Extract method
// to interpret it as a orders.
type GetResult struct {
	commonResult
}

// CreateResult is the response from a Create operation. Call its Extract method
// to interpret it as a orders.
type CreateResult struct {
	commonResult
}

// DeleteResult is the response from a Delete operation. Call its ExtractErr to
// determine if the request succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}

// OrderPage is a single page of orders results.
type OrderPage struct {
	pagination.LinkedPageBase
}

// IsEmpty determines whether or not a page of ordersS contains any results.
func (r OrderPage) IsEmpty() (bool, error) {
	orders, err := ExtractOrders(r)
	return len(orders) == 0, err
}

// NextPageURL extracts the "next" link from the links section of the result.
func (r OrderPage) NextPageURL() (string, error) {
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

// ExtractOrders returns a slice of Orders contained in a single page of
// results.
func ExtractOrders(r pagination.Page) ([]Order, error) {
	var s struct {
		Orders []Order `json:"orders"`
	}
	err := (r.(OrderPage)).ExtractInto(&s)
	return s.Orders, err
}

// Extract interprets any commonResult as a Order.
func (r commonResult) Extract() (*Order, error) {
	var s *Order
	err := r.ExtractInto(&s)
	return s, err
}
