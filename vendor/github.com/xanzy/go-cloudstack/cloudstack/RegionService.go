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
	"strconv"
)

type AddRegionParams struct {
	p map[string]interface{}
}

func (p *AddRegionParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["endpoint"]; found {
		u.Set("endpoint", v.(string))
	}
	if v, found := p.p["id"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("id", vv)
	}
	if v, found := p.p["name"]; found {
		u.Set("name", v.(string))
	}
	return u
}

func (p *AddRegionParams) SetEndpoint(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["endpoint"] = v
	return
}

func (p *AddRegionParams) SetId(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *AddRegionParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

// You should always use this function to get a new AddRegionParams instance,
// as then you are sure you have configured all required params
func (s *RegionService) NewAddRegionParams(endpoint string, id int, name string) *AddRegionParams {
	p := &AddRegionParams{}
	p.p = make(map[string]interface{})
	p.p["endpoint"] = endpoint
	p.p["id"] = id
	p.p["name"] = name
	return p
}

// Adds a Region
func (s *RegionService) AddRegion(p *AddRegionParams) (*AddRegionResponse, error) {
	resp, err := s.cs.newRequest("addRegion", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r AddRegionResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type AddRegionResponse struct {
	Endpoint                 string `json:"endpoint,omitempty"`
	Gslbserviceenabled       bool   `json:"gslbserviceenabled,omitempty"`
	Id                       int    `json:"id,omitempty"`
	Name                     string `json:"name,omitempty"`
	Portableipserviceenabled bool   `json:"portableipserviceenabled,omitempty"`
}

type UpdateRegionParams struct {
	p map[string]interface{}
}

func (p *UpdateRegionParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["endpoint"]; found {
		u.Set("endpoint", v.(string))
	}
	if v, found := p.p["id"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("id", vv)
	}
	if v, found := p.p["name"]; found {
		u.Set("name", v.(string))
	}
	return u
}

func (p *UpdateRegionParams) SetEndpoint(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["endpoint"] = v
	return
}

func (p *UpdateRegionParams) SetId(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *UpdateRegionParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

// You should always use this function to get a new UpdateRegionParams instance,
// as then you are sure you have configured all required params
func (s *RegionService) NewUpdateRegionParams(id int) *UpdateRegionParams {
	p := &UpdateRegionParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Updates a region
func (s *RegionService) UpdateRegion(p *UpdateRegionParams) (*UpdateRegionResponse, error) {
	resp, err := s.cs.newRequest("updateRegion", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r UpdateRegionResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type UpdateRegionResponse struct {
	Endpoint                 string `json:"endpoint,omitempty"`
	Gslbserviceenabled       bool   `json:"gslbserviceenabled,omitempty"`
	Id                       int    `json:"id,omitempty"`
	Name                     string `json:"name,omitempty"`
	Portableipserviceenabled bool   `json:"portableipserviceenabled,omitempty"`
}

type RemoveRegionParams struct {
	p map[string]interface{}
}

func (p *RemoveRegionParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("id", vv)
	}
	return u
}

func (p *RemoveRegionParams) SetId(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new RemoveRegionParams instance,
// as then you are sure you have configured all required params
func (s *RegionService) NewRemoveRegionParams(id int) *RemoveRegionParams {
	p := &RemoveRegionParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Removes specified region
func (s *RegionService) RemoveRegion(p *RemoveRegionParams) (*RemoveRegionResponse, error) {
	resp, err := s.cs.newRequest("removeRegion", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r RemoveRegionResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type RemoveRegionResponse struct {
	Displaytext string `json:"displaytext,omitempty"`
	Success     string `json:"success,omitempty"`
}

type ListRegionsParams struct {
	p map[string]interface{}
}

func (p *ListRegionsParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("id", vv)
	}
	if v, found := p.p["keyword"]; found {
		u.Set("keyword", v.(string))
	}
	if v, found := p.p["name"]; found {
		u.Set("name", v.(string))
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

func (p *ListRegionsParams) SetId(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ListRegionsParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListRegionsParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *ListRegionsParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListRegionsParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

// You should always use this function to get a new ListRegionsParams instance,
// as then you are sure you have configured all required params
func (s *RegionService) NewListRegionsParams() *ListRegionsParams {
	p := &ListRegionsParams{}
	p.p = make(map[string]interface{})
	return p
}

// Lists Regions
func (s *RegionService) ListRegions(p *ListRegionsParams) (*ListRegionsResponse, error) {
	resp, err := s.cs.newRequest("listRegions", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListRegionsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListRegionsResponse struct {
	Count   int       `json:"count"`
	Regions []*Region `json:"region"`
}

type Region struct {
	Endpoint                 string `json:"endpoint,omitempty"`
	Gslbserviceenabled       bool   `json:"gslbserviceenabled,omitempty"`
	Id                       int    `json:"id,omitempty"`
	Name                     string `json:"name,omitempty"`
	Portableipserviceenabled bool   `json:"portableipserviceenabled,omitempty"`
}
