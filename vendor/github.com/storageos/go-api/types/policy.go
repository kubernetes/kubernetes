package types

import (
	"encoding/json"
)

type Policy struct {
	Spec struct {
		User            string `json:"user,omitempty"`
		Group           string `json:"group,omitempty"`
		Readonly        bool   `json:"readonly,omitempty"`
		APIGroup        string `json:"apiGroup,omitempty"`
		Resource        string `json:"resource,omitempty"`
		Namespace       string `json:"namespace,omitempty"`
		NonResourcePath string `json:"nonResourcePath,omitempty"`
	} `json:"spec"`
}

// PolicyWithId is used as an internal type to render table formated versions of the json response
type PolicyWithID struct {
	Policy
	ID string
}

// MarshalJSON returns a marshaled copy of the internal policy object, so it is still valid to use
// with the REST API
func (p *PolicyWithID) MarshalJSON() ([]byte, error) {
	return json.Marshal(p.Policy)
}

// PolicySet is a representation of the data structure returned from the REST API
type PolicySet map[string]Policy

func (p PolicySet) GetPoliciesWithID() []*PolicyWithID {
	rtn := make([]*PolicyWithID, 0, len(p))

	for k, v := range p {
		rtn = append(rtn, &PolicyWithID{
			Policy: v,
			ID:     k,
		})
	}

	return rtn
}
