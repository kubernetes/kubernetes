package groups

import (
	"encoding/json"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/internal"
	"github.com/gophercloud/gophercloud/pagination"
)

// Group helps manage related users.
type Group struct {
	// Description describes the group purpose.
	Description string `json:"description"`

	// DomainID is the domain ID the group belongs to.
	DomainID string `json:"domain_id"`

	// ID is the unique ID of the group.
	ID string `json:"id"`

	// Extra is a collection of miscellaneous key/values.
	Extra map[string]interface{} `json:"-"`

	// Links contains referencing links to the group.
	Links map[string]interface{} `json:"links"`

	// Name is the name of the group.
	Name string `json:"name"`
}

func (r *Group) UnmarshalJSON(b []byte) error {
	type tmp Group
	var s struct {
		tmp
		Extra map[string]interface{} `json:"extra"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = Group(s.tmp)

	// Collect other fields and bundle them into Extra
	// but only if a field titled "extra" wasn't sent.
	if s.Extra != nil {
		r.Extra = s.Extra
	} else {
		var result interface{}
		err := json.Unmarshal(b, &result)
		if err != nil {
			return err
		}
		if resultMap, ok := result.(map[string]interface{}); ok {
			r.Extra = internal.RemainingKeys(Group{}, resultMap)
		}
	}

	return err
}

type groupResult struct {
	gophercloud.Result
}

// GetResult is the response from a Get operation. Call its Extract method
// to interpret it as a Group.
type GetResult struct {
	groupResult
}

// CreateResult is the response from a Create operation. Call its Extract method
// to interpret it as a Group.
type CreateResult struct {
	groupResult
}

// UpdateResult is the response from an Update operation. Call its Extract
// method to interpret it as a Group.
type UpdateResult struct {
	groupResult
}

// DeleteResult is the response from a Delete operation. Call its ExtractErr to
// determine if the request succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}

// GroupPage is a single page of Group results.
type GroupPage struct {
	pagination.LinkedPageBase
}

// IsEmpty determines whether or not a page of Groups contains any results.
func (r GroupPage) IsEmpty() (bool, error) {
	groups, err := ExtractGroups(r)
	return len(groups) == 0, err
}

// NextPageURL extracts the "next" link from the links section of the result.
func (r GroupPage) NextPageURL() (string, error) {
	var s struct {
		Links struct {
			Next     string `json:"next"`
			Previous string `json:"previous"`
		} `json:"links"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return s.Links.Next, err
}

// ExtractGroups returns a slice of Groups contained in a single page of results.
func ExtractGroups(r pagination.Page) ([]Group, error) {
	var s struct {
		Groups []Group `json:"groups"`
	}
	err := (r.(GroupPage)).ExtractInto(&s)
	return s.Groups, err
}

// Extract interprets any group results as a Group.
func (r groupResult) Extract() (*Group, error) {
	var s struct {
		Group *Group `json:"group"`
	}
	err := r.ExtractInto(&s)
	return s.Group, err
}
