package members

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Member represents the application running on a backend server.
type Member struct {
	// Status is the status of the member. Indicates whether the member
	// is operational.
	Status string

	// Weight is the weight of member.
	Weight int

	// AdminStateUp is the administrative state of the member, which is up
	// (true) or down (false).
	AdminStateUp bool `json:"admin_state_up"`

	// TenantID is the owner of the member.
	TenantID string `json:"tenant_id"`

	// PoolID is the pool to which the member belongs.
	PoolID string `json:"pool_id"`

	// Address is the IP address of the member.
	Address string

	// ProtocolPort is the port on which the application is hosted.
	ProtocolPort int `json:"protocol_port"`

	// ID is the unique ID for the member.
	ID string
}

// MemberPage is the page returned by a pager when traversing over a
// collection of pool members.
type MemberPage struct {
	pagination.LinkedPageBase
}

// NextPageURL is invoked when a paginated collection of members has reached
// the end of a page and the pager seeks to traverse over a new one. In order
// to do this, it needs to construct the next page's URL.
func (r MemberPage) NextPageURL() (string, error) {
	var s struct {
		Links []gophercloud.Link `json:"members_links"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return gophercloud.ExtractNextURL(s.Links)
}

// IsEmpty checks whether a MemberPage struct is empty.
func (r MemberPage) IsEmpty() (bool, error) {
	is, err := ExtractMembers(r)
	return len(is) == 0, err
}

// ExtractMembers accepts a Page struct, specifically a MemberPage struct,
// and extracts the elements into a slice of Member structs. In other words,
// a generic collection is mapped into a relevant slice.
func ExtractMembers(r pagination.Page) ([]Member, error) {
	var s struct {
		Members []Member `json:"members"`
	}
	err := (r.(MemberPage)).ExtractInto(&s)
	return s.Members, err
}

type commonResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts a member.
func (r commonResult) Extract() (*Member, error) {
	var s struct {
		Member *Member `json:"member"`
	}
	err := r.ExtractInto(&s)
	return s.Member, err
}

// CreateResult represents the result of a create operation. Call its Extract
// method to interpret it as a Member.
type CreateResult struct {
	commonResult
}

// GetResult represents the result of a get operation. Call its Extract
// method to interpret it as a Member.
type GetResult struct {
	commonResult
}

// UpdateResult represents the result of an update operation. Call its Extract
// method to interpret it as a Member.
type UpdateResult struct {
	commonResult
}

// DeleteResult represents the result of a delete operation. Call its
// ExtractErr method to determine if the result succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}
