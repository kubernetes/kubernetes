package capsules

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

type commonResult struct {
	gophercloud.Result
}

// ExtractBase is a function that accepts a result and extracts
// a base a capsule resource.
func (r commonResult) ExtractBase() (*Capsule, error) {
	var s *Capsule
	err := r.ExtractInto(&s)
	return s, err
}

// Extract is a function that accepts a result and extracts a capsule result.
// The result will be returned as an interface{} where it should be able to
// be casted as either a Capsule or CapsuleV132.
func (r commonResult) Extract() (interface{}, error) {
	s, err := r.ExtractBase()
	if err == nil {
		return s, nil
	}

	if _, ok := err.(*json.UnmarshalTypeError); !ok {
		return s, err
	}

	return r.ExtractV132()
}

// GetResult represents the result of a get operation.
type GetResult struct {
	commonResult
}

// CreateResult is the response from a Create operation. Call its Extract
// method to interpret it as a Capsule.
type CreateResult struct {
	commonResult
}

// DeleteResult represents the result of a delete operation.
type DeleteResult struct {
	gophercloud.ErrResult
}

type CapsulePage struct {
	pagination.LinkedPageBase
}

// Represents a Capsule
type Capsule struct {
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
	MetaName string `json:"meta_name"`

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

	// The capsule version
	CapsuleVersion string `json:"capsule_version"`

	// The capsule restart policy
	RestartPolicy string `json:"restart_policy"`

	// The capsule metadata labels
	MetaLabels map[string]string `json:"meta_labels"`

	// The list of containers uuids inside capsule.
	ContainersUUIDs []string `json:"containers_uuids"`

	// The capsule IP addresses
	Addresses map[string][]Address `json:"addresses"`

	// The capsule volume attached information
	VolumesInfo map[string][]string `json:"volumes_info"`

	// The container object inside capsule
	Containers []Container `json:"containers"`

	// The capsule host
	Host string `json:"host"`
}

type Container struct {
	// The Container IP addresses
	Addresses map[string][]Address `json:"addresses"`

	// UUID for the container
	UUID string `json:"uuid"`

	// User ID for the container
	UserID string `json:"user_id"`

	// Project ID for the container
	ProjectID string `json:"project_id"`

	// cpu for the container
	CPU float64 `json:"cpu"`

	// Memory for the container
	Memory string `json:"memory"`

	// Image for the container
	Image string `json:"image"`

	// The container container
	Labels map[string]string `json:"labels"`

	// The created time of the container
	CreatedAt time.Time `json:"-"`

	// The updated time of the container
	UpdatedAt time.Time `json:"-"`

	// The started time of the container
	StartedAt time.Time `json:"-"`

	// Name for the container
	Name string `json:"name"`

	// Links includes HTTP references to the itself, useful for passing along to
	// other APIs that might want a capsule reference.
	Links []interface{} `json:"links"`

	// auto remove flag token for the container
	AutoRemove bool `json:"auto_remove"`

	// Host for the container
	Host string `json:"host"`

	// Work directory for the container
	WorkDir string `json:"workdir"`

	// Disk for the container
	Disk int `json:"disk"`

	// Image pull policy for the container
	ImagePullPolicy string `json:"image_pull_policy"`

	// Task state for the container
	TaskState string `json:"task_state"`

	// Host name for the container
	HostName string `json:"hostname"`

	// Environment for the container
	Environment map[string]string `json:"environment"`

	// Status for the container
	Status string `json:"status"`

	// Auto Heal flag for the container
	AutoHeal bool `json:"auto_heal"`

	// Status details for the container
	StatusDetail string `json:"status_detail"`

	// Status reason for the container
	StatusReason string `json:"status_reason"`

	// Image driver for the container
	ImageDriver string `json:"image_driver"`

	// Command for the container
	Command []string `json:"command"`

	// Image for the container
	Runtime string `json:"runtime"`

	// Interactive flag for the container
	Interactive bool `json:"interactive"`

	// Restart Policy for the container
	RestartPolicy map[string]string `json:"restart_policy"`

	// Ports information for the container
	Ports []int `json:"ports"`

	// Security groups for the container
	SecurityGroups []string `json:"security_groups"`
}

type Address struct {
	PreserveOnDelete bool    `json:"preserve_on_delete"`
	Addr             string  `json:"addr"`
	Port             string  `json:"port"`
	Version          float64 `json:"version"`
	SubnetID         string  `json:"subnet_id"`
}

// NextPageURL is invoked when a paginated collection of capsules has reached
// the end of a page and the pager seeks to traverse over a new one. In order
// to do this, it needs to construct the next page's URL.
func (r CapsulePage) NextPageURL() (string, error) {
	var s struct {
		Next string `json:"next"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return s.Next, nil
}

// IsEmpty checks whether a CapsulePage struct is empty.
func (r CapsulePage) IsEmpty() (bool, error) {
	is, err := ExtractCapsules(r)
	if err != nil {
		return false, err
	}

	if v, ok := is.([]Capsule); ok {
		return len(v) == 0, nil
	}

	if v, ok := is.([]CapsuleV132); ok {
		return len(v) == 0, nil
	}

	return false, fmt.Errorf("Unable to determine Capsule type")
}

// ExtractCapsulesBase accepts a Page struct, specifically a CapsulePage struct,
// and extracts the elements into a slice of Capsule structs. In other words,
// a generic collection is mapped into the relevant slice.
func ExtractCapsulesBase(r pagination.Page) ([]Capsule, error) {
	var s struct {
		Capsules []Capsule `json:"capsules"`
	}

	err := (r.(CapsulePage)).ExtractInto(&s)
	return s.Capsules, err
}

// ExtractCapsules accepts a Page struct, specifically a CapsulePage struct,
// and extracts the elements into an interface.
// This interface should be able to be casted as either a Capsule or
// CapsuleV132 struct
func ExtractCapsules(r pagination.Page) (interface{}, error) {
	s, err := ExtractCapsulesBase(r)
	if err == nil {
		return s, nil
	}

	if _, ok := err.(*json.UnmarshalTypeError); !ok {
		return nil, err
	}

	return ExtractCapsulesV132(r)
}

func (r *Capsule) UnmarshalJSON(b []byte) error {
	type tmp Capsule

	// Support for "older" zun time formats.
	var s1 struct {
		tmp
		CreatedAt gophercloud.JSONRFC3339ZNoT `json:"created_at"`
		UpdatedAt gophercloud.JSONRFC3339ZNoT `json:"updated_at"`
	}

	err := json.Unmarshal(b, &s1)
	if err == nil {
		*r = Capsule(s1.tmp)

		r.CreatedAt = time.Time(s1.CreatedAt)
		r.UpdatedAt = time.Time(s1.UpdatedAt)

		return nil
	}

	// Support for "new" zun time formats.
	var s2 struct {
		tmp
		CreatedAt gophercloud.JSONRFC3339ZNoTNoZ `json:"created_at"`
		UpdatedAt gophercloud.JSONRFC3339ZNoTNoZ `json:"updated_at"`
	}

	err = json.Unmarshal(b, &s2)
	if err != nil {
		return err
	}

	*r = Capsule(s2.tmp)

	r.CreatedAt = time.Time(s2.CreatedAt)
	r.UpdatedAt = time.Time(s2.UpdatedAt)

	return nil
}

func (r *Container) UnmarshalJSON(b []byte) error {
	type tmp Container

	// Support for "older" zun time formats.
	var s1 struct {
		tmp
		CreatedAt gophercloud.JSONRFC3339ZNoT `json:"created_at"`
		UpdatedAt gophercloud.JSONRFC3339ZNoT `json:"updated_at"`
		StartedAt gophercloud.JSONRFC3339ZNoT `json:"started_at"`
	}

	err := json.Unmarshal(b, &s1)
	if err == nil {
		*r = Container(s1.tmp)

		r.CreatedAt = time.Time(s1.CreatedAt)
		r.UpdatedAt = time.Time(s1.UpdatedAt)
		r.StartedAt = time.Time(s1.StartedAt)

		return nil
	}

	// Support for "new" zun time formats.
	var s2 struct {
		tmp
		CreatedAt gophercloud.JSONRFC3339ZNoTNoZ `json:"created_at"`
		UpdatedAt gophercloud.JSONRFC3339ZNoTNoZ `json:"updated_at"`
		StartedAt gophercloud.JSONRFC3339ZNoTNoZ `json:"started_at"`
	}

	err = json.Unmarshal(b, &s2)
	if err != nil {
		return err
	}

	*r = Container(s2.tmp)

	r.CreatedAt = time.Time(s2.CreatedAt)
	r.UpdatedAt = time.Time(s2.UpdatedAt)
	r.StartedAt = time.Time(s2.StartedAt)

	return nil
}
