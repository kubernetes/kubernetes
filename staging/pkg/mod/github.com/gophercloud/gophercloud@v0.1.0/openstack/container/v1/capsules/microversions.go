package capsules

import (
	"encoding/json"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// ExtractV132 is a function that accepts a result and extracts a capsule resource.
func (r commonResult) ExtractV132() (*CapsuleV132, error) {
	var s *CapsuleV132
	err := r.ExtractInto(&s)
	return s, err
}

// Represents a Capsule at microversion vXY or greater.
type CapsuleV132 struct {
	// UUID for the capsule
	UUID string `json:"uuid"`

	// User ID for the capsule
	UserID string `json:"user_id"`

	// Project ID for the capsule
	ProjectID string `json:"project_id"`

	// cpu for the capsule
	CPU float64 `json:"cpu"`

	// Memory for the capsule
	Memory string `json:"memory"`

	// The name of the capsule
	MetaName string `json:"name"`

	// Indicates whether capsule is currently operational.
	Status string `json:"status"`

	// Indicates whether capsule is currently operational.
	StatusReason string `json:"status_reason"`

	// The created time of the capsule.
	CreatedAt time.Time `json:"-"`

	// The updated time of the capsule.
	UpdatedAt time.Time `json:"-"`

	// Links includes HTTP references to the itself, useful for passing along to
	// other APIs that might want a capsule reference.
	Links []interface{} `json:"links"`

	// The capsule restart policy
	RestartPolicy map[string]string `json:"restart_policy"`

	// The capsule metadata labels
	MetaLabels map[string]string `json:"labels"`

	// The capsule IP addresses
	Addresses map[string][]Address `json:"addresses"`

	// The container object inside capsule
	Containers []Container `json:"containers"`

	// The capsule host
	Host string `json:"host"`
}

// ExtractCapsulesV132 accepts a Page struct, specifically a CapsulePage struct,
// and extracts the elements into a slice of CapsuleV132 structs. In other words,
// a generic collection is mapped into a relevant slice.
func ExtractCapsulesV132(r pagination.Page) ([]CapsuleV132, error) {
	var s struct {
		Capsules []CapsuleV132 `json:"capsules"`
	}
	err := (r.(CapsulePage)).ExtractInto(&s)
	return s.Capsules, err
}

func (r *CapsuleV132) UnmarshalJSON(b []byte) error {
	type tmp CapsuleV132

	var s struct {
		tmp
		CreatedAt gophercloud.JSONRFC3339ZNoTNoZ `json:"created_at"`
		UpdatedAt gophercloud.JSONRFC3339ZNoTNoZ `json:"updated_at"`
	}

	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}

	*r = CapsuleV132(s.tmp)

	r.CreatedAt = time.Time(s.CreatedAt)
	r.UpdatedAt = time.Time(s.UpdatedAt)

	return nil
}
