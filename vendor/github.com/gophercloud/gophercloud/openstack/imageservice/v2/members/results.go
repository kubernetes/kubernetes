package members

import (
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Member represents a member of an Image.
type Member struct {
	CreatedAt time.Time `json:"created_at"`
	ImageID   string    `json:"image_id"`
	MemberID  string    `json:"member_id"`
	Schema    string    `json:"schema"`
	Status    string    `json:"status"`
	UpdatedAt time.Time `json:"updated_at"`
}

// Extract Member model from a request.
func (r commonResult) Extract() (*Member, error) {
	var s *Member
	err := r.ExtractInto(&s)
	return s, err
}

// MemberPage is a single page of Members results.
type MemberPage struct {
	pagination.SinglePageBase
}

// ExtractMembers returns a slice of Members contained in a single page
// of results.
func ExtractMembers(r pagination.Page) ([]Member, error) {
	var s struct {
		Members []Member `json:"members"`
	}
	err := r.(MemberPage).ExtractInto(&s)
	return s.Members, err
}

// IsEmpty determines whether or not a MemberPage contains any results.
func (r MemberPage) IsEmpty() (bool, error) {
	members, err := ExtractMembers(r)
	return len(members) == 0, err
}

type commonResult struct {
	gophercloud.Result
}

// CreateResult represents the result of a Create operation. Call its Extract
// method to interpret it as a Member.
type CreateResult struct {
	commonResult
}

// DetailsResult represents the result of a Get operation. Call its Extract
// method to interpret it as a Member.
type DetailsResult struct {
	commonResult
}

// UpdateResult represents the result of an Update operation. Call its Extract
// method to interpret it as a Member.
type UpdateResult struct {
	commonResult
}

// DeleteResult represents the result of a Delete operation. Call its
// ExtractErr method to determine if the request succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}
