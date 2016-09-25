//
// Copyright 2016, Sander van Harmelen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package cloudstack

import (
	"encoding/json"
	"net/url"
)

type ListApisParams struct {
	p map[string]interface{}
}

func (p *ListApisParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["name"]; found {
		u.Set("name", v.(string))
	}
	return u
}

func (p *ListApisParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

// You should always use this function to get a new ListApisParams instance,
// as then you are sure you have configured all required params
func (s *APIDiscoveryService) NewListApisParams() *ListApisParams {
	p := &ListApisParams{}
	p.p = make(map[string]interface{})
	return p
}

// lists all available apis on the server, provided by the Api Discovery plugin
func (s *APIDiscoveryService) ListApis(p *ListApisParams) (*ListApisResponse, error) {
	resp, err := s.cs.newRequest("listApis", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListApisResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListApisResponse struct {
	Count int    `json:"count"`
	Apis  []*Api `json:"api"`
}

type Api struct {
	Description string `json:"description,omitempty"`
	Isasync     bool   `json:"isasync,omitempty"`
	Name        string `json:"name,omitempty"`
	Params      []struct {
		Description string `json:"description,omitempty"`
		Length      int    `json:"length,omitempty"`
		Name        string `json:"name,omitempty"`
		Related     string `json:"related,omitempty"`
		Required    bool   `json:"required,omitempty"`
		Since       string `json:"since,omitempty"`
		Type        string `json:"type,omitempty"`
	} `json:"params,omitempty"`
	Related  string `json:"related,omitempty"`
	Response []struct {
		Description string   `json:"description,omitempty"`
		Name        string   `json:"name,omitempty"`
		Response    []string `json:"response,omitempty"`
		Type        string   `json:"type,omitempty"`
	} `json:"response,omitempty"`
	Since string `json:"since,omitempty"`
	Type  string `json:"type,omitempty"`
}
