package acls

import (
	"encoding/json"
	"time"

	"github.com/gophercloud/gophercloud"
)

// ACL represents an ACL on a resource.
type ACL map[string]ACLDetails

// ACLDetails represents the details of an ACL.
type ACLDetails struct {
	// Created is when the ACL was created.
	Created time.Time `json:"-"`

	// ProjectAccess denotes project-level access of the resource.
	ProjectAccess bool `json:"project-access"`

	// Updated is when the ACL was updated
	Updated time.Time `json:"-"`

	// Users are the UserIDs who have access to the resource.
	Users []string `json:"users"`
}

func (r *ACLDetails) UnmarshalJSON(b []byte) error {
	type tmp ACLDetails
	var s struct {
		tmp
		Created gophercloud.JSONRFC3339NoZ `json:"created"`
		Updated gophercloud.JSONRFC3339NoZ `json:"updated"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = ACLDetails(s.tmp)

	r.Created = time.Time(s.Created)
	r.Updated = time.Time(s.Updated)

	return nil
}

// ACLRef represents an ACL reference.
type ACLRef string

type commonResult struct {
	gophercloud.Result
}

// Extract interprets any commonResult as an ACL.
func (r commonResult) Extract() (*ACL, error) {
	var s *ACL
	err := r.ExtractInto(&s)
	return s, err
}

// ACLResult is the response from a Get operation. Call its Extract method
// to interpret it as an ACL.
type ACLResult struct {
	commonResult
}

// ACLRefResult is the response from a Set or Update operation. Call its
// Extract method to interpret it as an ACLRef.
type ACLRefResult struct {
	gophercloud.Result
}

func (r ACLRefResult) Extract() (*ACLRef, error) {
	var s struct {
		ACLRef ACLRef `json:"acl_ref"`
	}
	err := r.ExtractInto(&s)
	return &s.ACLRef, err
}

// DeleteResult is the response from a Delete operation. Call its ExtractErr to
// determine if the request succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}
