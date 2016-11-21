package servergroups

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// A ServerGroup creates a policy for instance placement in the cloud
type ServerGroup struct {
	// ID is the unique ID of the Server Group.
	ID string `json:"id"`

	// Name is the common name of the server group.
	Name string `json:"name"`

	// Polices are the group policies.
	Policies []string `json:"policies"`

	// Members are the members of the server group.
	Members []string `json:"members"`

	// Metadata includes a list of all user-specified key-value pairs attached to the Server Group.
	Metadata map[string]interface{}
}

// ServerGroupPage stores a single, only page of ServerGroups
// results from a List call.
type ServerGroupPage struct {
	pagination.SinglePageBase
}

// IsEmpty determines whether or not a ServerGroupsPage is empty.
func (page ServerGroupPage) IsEmpty() (bool, error) {
	va, err := ExtractServerGroups(page)
	return len(va) == 0, err
}

// ExtractServerGroups interprets a page of results as a slice of
// ServerGroups.
func ExtractServerGroups(r pagination.Page) ([]ServerGroup, error) {
	var s struct {
		ServerGroups []ServerGroup `json:"server_groups"`
	}
	err := (r.(ServerGroupPage)).ExtractInto(&s)
	return s.ServerGroups, err
}

type ServerGroupResult struct {
	gophercloud.Result
}

// Extract is a method that attempts to interpret any Server Group resource
// response as a ServerGroup struct.
func (r ServerGroupResult) Extract() (*ServerGroup, error) {
	var s struct {
		ServerGroup *ServerGroup `json:"server_group"`
	}
	err := r.ExtractInto(&s)
	return s.ServerGroup, err
}

// CreateResult is the response from a Create operation. Call its Extract method to interpret it
// as a ServerGroup.
type CreateResult struct {
	ServerGroupResult
}

// GetResult is the response from a Get operation. Call its Extract method to interpret it
// as a ServerGroup.
type GetResult struct {
	ServerGroupResult
}

// DeleteResult is the response from a Delete operation. Call its Extract method to determine if
// the call succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}
