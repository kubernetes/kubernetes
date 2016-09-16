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
	"fmt"
	"net/url"
	"strconv"
	"strings"
)

type ListHypervisorsParams struct {
	p map[string]interface{}
}

func (p *ListHypervisorsParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["zoneid"]; found {
		u.Set("zoneid", v.(string))
	}
	return u
}

func (p *ListHypervisorsParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new ListHypervisorsParams instance,
// as then you are sure you have configured all required params
func (s *HypervisorService) NewListHypervisorsParams() *ListHypervisorsParams {
	p := &ListHypervisorsParams{}
	p.p = make(map[string]interface{})
	return p
}

// List hypervisors
func (s *HypervisorService) ListHypervisors(p *ListHypervisorsParams) (*ListHypervisorsResponse, error) {
	resp, err := s.cs.newRequest("listHypervisors", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListHypervisorsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListHypervisorsResponse struct {
	Count       int           `json:"count"`
	Hypervisors []*Hypervisor `json:"hypervisor"`
}

type Hypervisor struct {
	Name string `json:"name,omitempty"`
}

type UpdateHypervisorCapabilitiesParams struct {
	p map[string]interface{}
}

func (p *UpdateHypervisorCapabilitiesParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	if v, found := p.p["maxguestslimit"]; found {
		vv := strconv.FormatInt(v.(int64), 10)
		u.Set("maxguestslimit", vv)
	}
	if v, found := p.p["securitygroupenabled"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("securitygroupenabled", vv)
	}
	return u
}

func (p *UpdateHypervisorCapabilitiesParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *UpdateHypervisorCapabilitiesParams) SetMaxguestslimit(v int64) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["maxguestslimit"] = v
	return
}

func (p *UpdateHypervisorCapabilitiesParams) SetSecuritygroupenabled(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["securitygroupenabled"] = v
	return
}

// You should always use this function to get a new UpdateHypervisorCapabilitiesParams instance,
// as then you are sure you have configured all required params
func (s *HypervisorService) NewUpdateHypervisorCapabilitiesParams() *UpdateHypervisorCapabilitiesParams {
	p := &UpdateHypervisorCapabilitiesParams{}
	p.p = make(map[string]interface{})
	return p
}

// Updates a hypervisor capabilities.
func (s *HypervisorService) UpdateHypervisorCapabilities(p *UpdateHypervisorCapabilitiesParams) (*UpdateHypervisorCapabilitiesResponse, error) {
	resp, err := s.cs.newRequest("updateHypervisorCapabilities", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r UpdateHypervisorCapabilitiesResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type UpdateHypervisorCapabilitiesResponse struct {
	Hypervisor           string `json:"hypervisor,omitempty"`
	Hypervisorversion    string `json:"hypervisorversion,omitempty"`
	Id                   string `json:"id,omitempty"`
	Maxdatavolumeslimit  int    `json:"maxdatavolumeslimit,omitempty"`
	Maxguestslimit       int64  `json:"maxguestslimit,omitempty"`
	Maxhostspercluster   int    `json:"maxhostspercluster,omitempty"`
	Securitygroupenabled bool   `json:"securitygroupenabled,omitempty"`
	Storagemotionenabled bool   `json:"storagemotionenabled,omitempty"`
}

type ListHypervisorCapabilitiesParams struct {
	p map[string]interface{}
}

func (p *ListHypervisorCapabilitiesParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["hypervisor"]; found {
		u.Set("hypervisor", v.(string))
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	if v, found := p.p["keyword"]; found {
		u.Set("keyword", v.(string))
	}
	if v, found := p.p["page"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("page", vv)
	}
	if v, found := p.p["pagesize"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("pagesize", vv)
	}
	return u
}

func (p *ListHypervisorCapabilitiesParams) SetHypervisor(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["hypervisor"] = v
	return
}

func (p *ListHypervisorCapabilitiesParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ListHypervisorCapabilitiesParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListHypervisorCapabilitiesParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListHypervisorCapabilitiesParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

// You should always use this function to get a new ListHypervisorCapabilitiesParams instance,
// as then you are sure you have configured all required params
func (s *HypervisorService) NewListHypervisorCapabilitiesParams() *ListHypervisorCapabilitiesParams {
	p := &ListHypervisorCapabilitiesParams{}
	p.p = make(map[string]interface{})
	return p
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *HypervisorService) GetHypervisorCapabilityByID(id string, opts ...OptionFunc) (*HypervisorCapability, int, error) {
	p := &ListHypervisorCapabilitiesParams{}
	p.p = make(map[string]interface{})

	p.p["id"] = id

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return nil, -1, err
		}
	}

	l, err := s.ListHypervisorCapabilities(p)
	if err != nil {
		if strings.Contains(err.Error(), fmt.Sprintf(
			"Invalid parameter id value=%s due to incorrect long value format, "+
				"or entity does not exist", id)) {
			return nil, 0, fmt.Errorf("No match found for %s: %+v", id, l)
		}
		return nil, -1, err
	}

	if l.Count == 0 {
		return nil, l.Count, fmt.Errorf("No match found for %s: %+v", id, l)
	}

	if l.Count == 1 {
		return l.HypervisorCapabilities[0], l.Count, nil
	}
	return nil, l.Count, fmt.Errorf("There is more then one result for HypervisorCapability UUID: %s!", id)
}

// Lists all hypervisor capabilities.
func (s *HypervisorService) ListHypervisorCapabilities(p *ListHypervisorCapabilitiesParams) (*ListHypervisorCapabilitiesResponse, error) {
	resp, err := s.cs.newRequest("listHypervisorCapabilities", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListHypervisorCapabilitiesResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListHypervisorCapabilitiesResponse struct {
	Count                  int                     `json:"count"`
	HypervisorCapabilities []*HypervisorCapability `json:"hypervisorcapability"`
}

type HypervisorCapability struct {
	Hypervisor           string `json:"hypervisor,omitempty"`
	Hypervisorversion    string `json:"hypervisorversion,omitempty"`
	Id                   string `json:"id,omitempty"`
	Maxdatavolumeslimit  int    `json:"maxdatavolumeslimit,omitempty"`
	Maxguestslimit       int64  `json:"maxguestslimit,omitempty"`
	Maxhostspercluster   int    `json:"maxhostspercluster,omitempty"`
	Securitygroupenabled bool   `json:"securitygroupenabled,omitempty"`
	Storagemotionenabled bool   `json:"storagemotionenabled,omitempty"`
}
