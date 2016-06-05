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

type CreateInstanceGroupParams struct {
	p map[string]interface{}
}

func (p *CreateInstanceGroupParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["account"]; found {
		u.Set("account", v.(string))
	}
	if v, found := p.p["domainid"]; found {
		u.Set("domainid", v.(string))
	}
	if v, found := p.p["name"]; found {
		u.Set("name", v.(string))
	}
	if v, found := p.p["projectid"]; found {
		u.Set("projectid", v.(string))
	}
	return u
}

func (p *CreateInstanceGroupParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *CreateInstanceGroupParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *CreateInstanceGroupParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *CreateInstanceGroupParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

// You should always use this function to get a new CreateInstanceGroupParams instance,
// as then you are sure you have configured all required params
func (s *VMGroupService) NewCreateInstanceGroupParams(name string) *CreateInstanceGroupParams {
	p := &CreateInstanceGroupParams{}
	p.p = make(map[string]interface{})
	p.p["name"] = name
	return p
}

// Creates a vm group
func (s *VMGroupService) CreateInstanceGroup(p *CreateInstanceGroupParams) (*CreateInstanceGroupResponse, error) {
	resp, err := s.cs.newRequest("createInstanceGroup", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r CreateInstanceGroupResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type CreateInstanceGroupResponse struct {
	Account   string `json:"account,omitempty"`
	Created   string `json:"created,omitempty"`
	Domain    string `json:"domain,omitempty"`
	Domainid  string `json:"domainid,omitempty"`
	Id        string `json:"id,omitempty"`
	Name      string `json:"name,omitempty"`
	Project   string `json:"project,omitempty"`
	Projectid string `json:"projectid,omitempty"`
}

type DeleteInstanceGroupParams struct {
	p map[string]interface{}
}

func (p *DeleteInstanceGroupParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	return u
}

func (p *DeleteInstanceGroupParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new DeleteInstanceGroupParams instance,
// as then you are sure you have configured all required params
func (s *VMGroupService) NewDeleteInstanceGroupParams(id string) *DeleteInstanceGroupParams {
	p := &DeleteInstanceGroupParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Deletes a vm group
func (s *VMGroupService) DeleteInstanceGroup(p *DeleteInstanceGroupParams) (*DeleteInstanceGroupResponse, error) {
	resp, err := s.cs.newRequest("deleteInstanceGroup", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r DeleteInstanceGroupResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type DeleteInstanceGroupResponse struct {
	Displaytext string `json:"displaytext,omitempty"`
	Success     string `json:"success,omitempty"`
}

type UpdateInstanceGroupParams struct {
	p map[string]interface{}
}

func (p *UpdateInstanceGroupParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	if v, found := p.p["name"]; found {
		u.Set("name", v.(string))
	}
	return u
}

func (p *UpdateInstanceGroupParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *UpdateInstanceGroupParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

// You should always use this function to get a new UpdateInstanceGroupParams instance,
// as then you are sure you have configured all required params
func (s *VMGroupService) NewUpdateInstanceGroupParams(id string) *UpdateInstanceGroupParams {
	p := &UpdateInstanceGroupParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Updates a vm group
func (s *VMGroupService) UpdateInstanceGroup(p *UpdateInstanceGroupParams) (*UpdateInstanceGroupResponse, error) {
	resp, err := s.cs.newRequest("updateInstanceGroup", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r UpdateInstanceGroupResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type UpdateInstanceGroupResponse struct {
	Account   string `json:"account,omitempty"`
	Created   string `json:"created,omitempty"`
	Domain    string `json:"domain,omitempty"`
	Domainid  string `json:"domainid,omitempty"`
	Id        string `json:"id,omitempty"`
	Name      string `json:"name,omitempty"`
	Project   string `json:"project,omitempty"`
	Projectid string `json:"projectid,omitempty"`
}

type ListInstanceGroupsParams struct {
	p map[string]interface{}
}

func (p *ListInstanceGroupsParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["account"]; found {
		u.Set("account", v.(string))
	}
	if v, found := p.p["domainid"]; found {
		u.Set("domainid", v.(string))
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	if v, found := p.p["isrecursive"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("isrecursive", vv)
	}
	if v, found := p.p["keyword"]; found {
		u.Set("keyword", v.(string))
	}
	if v, found := p.p["listall"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("listall", vv)
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
	if v, found := p.p["projectid"]; found {
		u.Set("projectid", v.(string))
	}
	return u
}

func (p *ListInstanceGroupsParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *ListInstanceGroupsParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *ListInstanceGroupsParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ListInstanceGroupsParams) SetIsrecursive(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isrecursive"] = v
	return
}

func (p *ListInstanceGroupsParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListInstanceGroupsParams) SetListall(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["listall"] = v
	return
}

func (p *ListInstanceGroupsParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *ListInstanceGroupsParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListInstanceGroupsParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

func (p *ListInstanceGroupsParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

// You should always use this function to get a new ListInstanceGroupsParams instance,
// as then you are sure you have configured all required params
func (s *VMGroupService) NewListInstanceGroupsParams() *ListInstanceGroupsParams {
	p := &ListInstanceGroupsParams{}
	p.p = make(map[string]interface{})
	return p
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *VMGroupService) GetInstanceGroupID(name string, opts ...OptionFunc) (string, error) {
	p := &ListInstanceGroupsParams{}
	p.p = make(map[string]interface{})

	p.p["name"] = name

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return "", err
		}
	}

	l, err := s.ListInstanceGroups(p)
	if err != nil {
		return "", err
	}

	if l.Count == 0 {
		return "", fmt.Errorf("No match found for %s: %+v", name, l)
	}

	if l.Count == 1 {
		return l.InstanceGroups[0].Id, nil
	}

	if l.Count > 1 {
		for _, v := range l.InstanceGroups {
			if v.Name == name {
				return v.Id, nil
			}
		}
	}
	return "", fmt.Errorf("Could not find an exact match for %s: %+v", name, l)
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *VMGroupService) GetInstanceGroupByName(name string, opts ...OptionFunc) (*InstanceGroup, int, error) {
	id, err := s.GetInstanceGroupID(name, opts...)
	if err != nil {
		return nil, -1, err
	}

	r, count, err := s.GetInstanceGroupByID(id, opts...)
	if err != nil {
		return nil, count, err
	}
	return r, count, nil
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *VMGroupService) GetInstanceGroupByID(id string, opts ...OptionFunc) (*InstanceGroup, int, error) {
	p := &ListInstanceGroupsParams{}
	p.p = make(map[string]interface{})

	p.p["id"] = id

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return nil, -1, err
		}
	}

	l, err := s.ListInstanceGroups(p)
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
		return l.InstanceGroups[0], l.Count, nil
	}
	return nil, l.Count, fmt.Errorf("There is more then one result for InstanceGroup UUID: %s!", id)
}

// Lists vm groups
func (s *VMGroupService) ListInstanceGroups(p *ListInstanceGroupsParams) (*ListInstanceGroupsResponse, error) {
	resp, err := s.cs.newRequest("listInstanceGroups", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListInstanceGroupsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListInstanceGroupsResponse struct {
	Count          int              `json:"count"`
	InstanceGroups []*InstanceGroup `json:"instancegroup"`
}

type InstanceGroup struct {
	Account   string `json:"account,omitempty"`
	Created   string `json:"created,omitempty"`
	Domain    string `json:"domain,omitempty"`
	Domainid  string `json:"domainid,omitempty"`
	Id        string `json:"id,omitempty"`
	Name      string `json:"name,omitempty"`
	Project   string `json:"project,omitempty"`
	Projectid string `json:"projectid,omitempty"`
}
