package members

import (
	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// Member represents the application running on a backend server.
type Member struct {
	// The status of the member. Indicates whether the member is operational.
	Status string

	// Weight of member.
	Weight int

	// The administrative state of the member, which is up (true) or down (false).
	AdminStateUp bool `json:"admin_state_up" mapstructure:"admin_state_up"`

	// Owner of the member. Only an administrative user can specify a tenant ID
	// other than its own.
	TenantID string `json:"tenant_id" mapstructure:"tenant_id"`

	// The pool to which the member belongs.
	PoolID string `json:"pool_id" mapstructure:"pool_id"`

	// The IP address of the member.
	Address string

	// The port on which the application is hosted.
	ProtocolPort int `json:"protocol_port" mapstructure:"protocol_port"`

	// The unique ID for the member.
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
func (p MemberPage) NextPageURL() (string, error) {
	type resp struct {
		Links []gophercloud.Link `mapstructure:"members_links"`
	}

	var r resp
	err := mapstructure.Decode(p.Body, &r)
	if err != nil {
		return "", err
	}

	return gophercloud.ExtractNextURL(r.Links)
}

// IsEmpty checks whether a MemberPage struct is empty.
func (p MemberPage) IsEmpty() (bool, error) {
	is, err := ExtractMembers(p)
	if err != nil {
		return true, nil
	}
	return len(is) == 0, nil
}

// ExtractMembers accepts a Page struct, specifically a MemberPage struct,
// and extracts the elements into a slice of Member structs. In other words,
// a generic collection is mapped into a relevant slice.
func ExtractMembers(page pagination.Page) ([]Member, error) {
	var resp struct {
		Members []Member `mapstructure:"members" json:"members"`
	}

	err := mapstructure.Decode(page.(MemberPage).Body, &resp)
	if err != nil {
		return nil, err
	}

	return resp.Members, nil
}

type commonResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts a router.
func (r commonResult) Extract() (*Member, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res struct {
		Member *Member `json:"member"`
	}

	err := mapstructure.Decode(r.Body, &res)

	return res.Member, err
}

// CreateResult represents the result of a create operation.
type CreateResult struct {
	commonResult
}

// GetResult represents the result of a get operation.
type GetResult struct {
	commonResult
}

// UpdateResult represents the result of an update operation.
type UpdateResult struct {
	commonResult
}

// DeleteResult represents the result of a delete operation.
type DeleteResult struct {
	gophercloud.ErrResult
}
