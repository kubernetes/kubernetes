package trunks

import (
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

type Subport struct {
	SegmentationID   int    `json:"segmentation_id" required:"true"`
	SegmentationType string `json:"segmentation_type" required:"true"`
	PortID           string `json:"port_id" required:"true"`
}

type commonResult struct {
	gophercloud.Result
}

// CreateResult is the response from a Create operation. Call its Extract method
// to interpret it as a Trunk.
type CreateResult struct {
	commonResult
}

// DeleteResult is the response from a Delete operation. Call its ExtractErr to
// determine if the request succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}

// GetResult is the response from a Get operation. Call its Extract method
// to interpret it as a Trunk.
type GetResult struct {
	commonResult
}

// UpdateResult is the result of an Update request. Call its Extract method to
// interpret it as a Trunk.
type UpdateResult struct {
	commonResult
}

// GetSubportsResult is the result of a Get request on the trunks subports
// resource. Call its Extract method to interpret it as a slice of Subport.
type GetSubportsResult struct {
	commonResult
}

// UpdateSubportsResult is the result of either an AddSubports or a RemoveSubports
// request. Call its Extract method to interpret it as a Trunk.
type UpdateSubportsResult struct {
	commonResult
}

type Trunk struct {
	// Indicates whether the trunk is currently operational. Possible values include
	// `ACTIVE', `DOWN', `BUILD', 'DEGRADED' or `ERROR'.
	Status string `json:"status"`

	// A list of ports associated with the trunk
	Subports []Subport `json:"sub_ports"`

	// Human-readable name for the trunk. Might not be unique.
	Name string `json:"name,omitempty"`

	// The administrative state of the trunk. If false (down), the trunk does not
	// forward packets.
	AdminStateUp bool `json:"admin_state_up,omitempty"`

	// ProjectID is the project owner of the trunk.
	ProjectID string `json:"project_id"`

	// TenantID is the project owner of the trunk.
	TenantID string `json:"tenant_id"`

	// The date and time when the resource was created.
	CreatedAt time.Time `json:"created_at"`

	// The date and time when the resource was updated,
	// if the resource has not been updated, this field will show as null.
	UpdatedAt time.Time `json:"updated_at"`

	RevisionNumber int `json:"revision_number"`

	// UUID of the trunk's parent port
	PortID string `json:"port_id"`

	// UUID for the trunk resource
	ID string `json:"id"`

	// Display description.
	Description string `json:"description"`

	// A list of tags associated with the trunk
	Tags []string `json:"tags,omitempty"`
}

func (r commonResult) Extract() (*Trunk, error) {
	var s struct {
		Trunk *Trunk `json:"trunk"`
	}
	err := r.ExtractInto(&s)
	return s.Trunk, err
}

// TrunkPage is the page returned by a pager when traversing a collection of
// trunk resources.
type TrunkPage struct {
	pagination.LinkedPageBase
}

func (page TrunkPage) IsEmpty() (bool, error) {
	trunks, err := ExtractTrunks(page)
	return len(trunks) == 0, err
}

func ExtractTrunks(page pagination.Page) ([]Trunk, error) {
	var a struct {
		Trunks []Trunk `json:"trunks"`
	}
	err := (page.(TrunkPage)).ExtractInto(&a)
	return a.Trunks, err
}

func (r GetSubportsResult) Extract() ([]Subport, error) {
	var s struct {
		Subports []Subport `json:"sub_ports"`
	}
	err := r.ExtractInto(&s)
	return s.Subports, err
}

func (r UpdateSubportsResult) Extract() (t *Trunk, err error) {
	err = r.ExtractInto(&t)
	return
}
