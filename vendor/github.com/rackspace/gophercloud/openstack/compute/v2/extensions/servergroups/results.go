package servergroups

import (
	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// A ServerGroup creates a policy for instance placement in the cloud
type ServerGroup struct {
	// ID is the unique ID of the Server Group.
	ID string `mapstructure:"id"`

	// Name is the common name of the server group.
	Name string `mapstructure:"name"`

	// Polices are the group policies.
	Policies []string `mapstructure:"policies"`

	// Members are the members of the server group.
	Members []string `mapstructure:"members"`

	// Metadata includes a list of all user-specified key-value pairs attached to the Server Group.
	Metadata map[string]interface{}
}

// ServerGroupsPage stores a single, only page of ServerGroups
// results from a List call.
type ServerGroupsPage struct {
	pagination.SinglePageBase
}

// IsEmpty determines whether or not a ServerGroupsPage is empty.
func (page ServerGroupsPage) IsEmpty() (bool, error) {
	va, err := ExtractServerGroups(page)
	return len(va) == 0, err
}

// ExtractServerGroups interprets a page of results as a slice of
// ServerGroups.
func ExtractServerGroups(page pagination.Page) ([]ServerGroup, error) {
	casted := page.(ServerGroupsPage).Body
	var response struct {
		ServerGroups []ServerGroup `mapstructure:"server_groups"`
	}

	err := mapstructure.WeakDecode(casted, &response)

	return response.ServerGroups, err
}

type ServerGroupResult struct {
	gophercloud.Result
}

// Extract is a method that attempts to interpret any Server Group resource
// response as a ServerGroup struct.
func (r ServerGroupResult) Extract() (*ServerGroup, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res struct {
		ServerGroup *ServerGroup `json:"server_group" mapstructure:"server_group"`
	}

	err := mapstructure.WeakDecode(r.Body, &res)
	return res.ServerGroup, err
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
