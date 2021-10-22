package policies

import (
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Policy represents a clustering policy in the Openstack cloud.
type Policy struct {
	CreatedAt time.Time              `json:"-"`
	Data      map[string]interface{} `json:"data"`
	Domain    string                 `json:"domain"`
	ID        string                 `json:"id"`
	Name      string                 `json:"name"`
	Project   string                 `json:"project"`
	Spec      Spec                   `json:"spec"`
	Type      string                 `json:"type"`
	UpdatedAt time.Time              `json:"-"`
	User      string                 `json:"user"`
}

func (r *Policy) UnmarshalJSON(b []byte) error {
	type tmp Policy
	var s struct {
		tmp
		CreatedAt string `json:"created_at,omitempty"`
		UpdatedAt string `json:"updated_at,omitempty"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = Policy(s.tmp)

	if s.CreatedAt != "" {
		r.CreatedAt, err = time.Parse(gophercloud.RFC3339MilliNoZ, s.CreatedAt)
		if err != nil {
			r.CreatedAt, err = time.Parse(time.RFC3339, s.CreatedAt)
			if err != nil {
				return err
			}
		}
	}

	if s.UpdatedAt != "" {
		r.UpdatedAt, err = time.Parse(gophercloud.RFC3339MilliNoZ, s.UpdatedAt)
		if err != nil {
			r.UpdatedAt, err = time.Parse(time.RFC3339, s.UpdatedAt)
			if err != nil {
				return err
			}
		}
	}

	return nil
}

// Spec represents an OpenStack clustering policy spec.
type Spec struct {
	Description string                 `json:"description"`
	Properties  map[string]interface{} `json:"properties"`
	Type        string                 `json:"type"`
	Version     string                 `json:"-"`
}

func (r *Spec) UnmarshalJSON(b []byte) error {
	type tmp Spec
	var s struct {
		tmp
		Version interface{} `json:"version"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = Spec(s.tmp)

	switch t := s.Version.(type) {
	case float64:
		if t == 1 {
			r.Version = fmt.Sprintf("%.1f", t)
		} else {
			r.Version = strconv.FormatFloat(t, 'f', -1, 64)
		}
	case string:
		r.Version = t
	}

	return nil
}

func (r Spec) MarshalJSON() ([]byte, error) {
	spec := struct {
		Type       string                 `json:"type"`
		Version    string                 `json:"version"`
		Properties map[string]interface{} `json:"properties"`
	}{
		Type:       r.Type,
		Version:    r.Version,
		Properties: r.Properties,
	}
	return json.Marshal(spec)
}

// policyResult is the resposne of a base Policy result.
type policyResult struct {
	gophercloud.Result
}

// Extract interpets any policyResult-base result as a Policy.
func (r policyResult) Extract() (*Policy, error) {
	var s struct {
		Policy *Policy `json:"policy"`
	}
	err := r.ExtractInto(&s)

	return s.Policy, err
}

// CreateResult is the result of an Update operation. Call its Extract
// method to interpret it as a Policy.
type CreateResult struct {
	policyResult
}

// GetResult is the result of a Get operation. Call its Extract method to
// interpret it as a Policy.
type GetResult struct {
	policyResult
}

// UpdateResult is the result of an Update operation. Call its Extract
// method to interpret it as a Policy.
type UpdateResult struct {
	policyResult
}

// ValidateResult is the result of a Validate operation. Call its Extract
// method to interpret it as a Policy.
type ValidateResult struct {
	policyResult
}

// DeleteResult is the result of a Delete operation. Call its Extract
// method to interpret it as a DeleteHeader.
type DeleteResult struct {
	gophercloud.ErrResult
}

// PolicyPage contains a list page of all policies from a List call.
type PolicyPage struct {
	pagination.MarkerPageBase
}

// IsEmpty determines if a PolicyPage contains any results.
func (page PolicyPage) IsEmpty() (bool, error) {
	policies, err := ExtractPolicies(page)
	return len(policies) == 0, err
}

// LastMarker returns the last policy ID in a ListResult.
func (r PolicyPage) LastMarker() (string, error) {
	policies, err := ExtractPolicies(r)
	if err != nil {
		return "", err
	}
	if len(policies) == 0 {
		return "", nil
	}
	return policies[len(policies)-1].ID, nil
}

// ExtractPolicies returns a slice of Policies from the List operation.
func ExtractPolicies(r pagination.Page) ([]Policy, error) {
	var s struct {
		Policies []Policy `json:"policies"`
	}
	err := (r.(PolicyPage)).ExtractInto(&s)
	return s.Policies, err
}
