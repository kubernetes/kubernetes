package aggregates

import (
	"encoding/json"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Aggregate represents a host aggregate in the OpenStack cloud.
type Aggregate struct {
	// The availability zone of the host aggregate.
	AvailabilityZone string `json:"availability_zone"`

	// A list of host ids in this aggregate.
	Hosts []string `json:"hosts"`

	// The ID of the host aggregate.
	ID int `json:"id"`

	// Metadata key and value pairs associate with the aggregate.
	Metadata map[string]string `json:"metadata"`

	// Name of the aggregate.
	Name string `json:"name"`

	// The date and time when the resource was created.
	CreatedAt time.Time `json:"-"`

	// The date and time when the resource was updated,
	// if the resource has not been updated, this field will show as null.
	UpdatedAt time.Time `json:"-"`

	// The date and time when the resource was deleted,
	// if the resource has not been deleted yet, this field will be null.
	DeletedAt time.Time `json:"-"`

	// A boolean indicates whether this aggregate is deleted or not,
	// if it has not been deleted, false will appear.
	Deleted bool `json:"deleted"`
}

// UnmarshalJSON to override default
func (r *Aggregate) UnmarshalJSON(b []byte) error {
	type tmp Aggregate
	var s struct {
		tmp
		CreatedAt gophercloud.JSONRFC3339MilliNoZ `json:"created_at"`
		UpdatedAt gophercloud.JSONRFC3339MilliNoZ `json:"updated_at"`
		DeletedAt gophercloud.JSONRFC3339MilliNoZ `json:"deleted_at"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = Aggregate(s.tmp)

	r.CreatedAt = time.Time(s.CreatedAt)
	r.UpdatedAt = time.Time(s.UpdatedAt)
	r.DeletedAt = time.Time(s.DeletedAt)

	return nil
}

// AggregatesPage represents a single page of all Aggregates from a List
// request.
type AggregatesPage struct {
	pagination.SinglePageBase
}

// IsEmpty determines whether or not a page of Aggregates contains any results.
func (page AggregatesPage) IsEmpty() (bool, error) {
	aggregates, err := ExtractAggregates(page)
	return len(aggregates) == 0, err
}

// ExtractAggregates interprets a page of results as a slice of Aggregates.
func ExtractAggregates(p pagination.Page) ([]Aggregate, error) {
	var a struct {
		Aggregates []Aggregate `json:"aggregates"`
	}
	err := (p.(AggregatesPage)).ExtractInto(&a)
	return a.Aggregates, err
}

type aggregatesResult struct {
	gophercloud.Result
}

func (r aggregatesResult) Extract() (*Aggregate, error) {
	var s struct {
		Aggregate *Aggregate `json:"aggregate"`
	}
	err := r.ExtractInto(&s)
	return s.Aggregate, err
}

type CreateResult struct {
	aggregatesResult
}

type GetResult struct {
	aggregatesResult
}

type DeleteResult struct {
	gophercloud.ErrResult
}

type UpdateResult struct {
	aggregatesResult
}

type ActionResult struct {
	aggregatesResult
}
