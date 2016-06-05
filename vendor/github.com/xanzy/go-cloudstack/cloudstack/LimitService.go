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

type UpdateResourceLimitParams struct {
	p map[string]interface{}
}

func (p *UpdateResourceLimitParams) toURLValues() url.Values {
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
	if v, found := p.p["max"]; found {
		vv := strconv.FormatInt(v.(int64), 10)
		u.Set("max", vv)
	}
	if v, found := p.p["projectid"]; found {
		u.Set("projectid", v.(string))
	}
	if v, found := p.p["resourcetype"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("resourcetype", vv)
	}
	return u
}

func (p *UpdateResourceLimitParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *UpdateResourceLimitParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *UpdateResourceLimitParams) SetMax(v int64) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["max"] = v
	return
}

func (p *UpdateResourceLimitParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

func (p *UpdateResourceLimitParams) SetResourcetype(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["resourcetype"] = v
	return
}

// You should always use this function to get a new UpdateResourceLimitParams instance,
// as then you are sure you have configured all required params
func (s *LimitService) NewUpdateResourceLimitParams(resourcetype int) *UpdateResourceLimitParams {
	p := &UpdateResourceLimitParams{}
	p.p = make(map[string]interface{})
	p.p["resourcetype"] = resourcetype
	return p
}

// Updates resource limits for an account or domain.
func (s *LimitService) UpdateResourceLimit(p *UpdateResourceLimitParams) (*UpdateResourceLimitResponse, error) {
	resp, err := s.cs.newRequest("updateResourceLimit", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r UpdateResourceLimitResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type UpdateResourceLimitResponse struct {
	Account      string `json:"account,omitempty"`
	Domain       string `json:"domain,omitempty"`
	Domainid     string `json:"domainid,omitempty"`
	Max          int64  `json:"max,omitempty"`
	Project      string `json:"project,omitempty"`
	Projectid    string `json:"projectid,omitempty"`
	Resourcetype string `json:"resourcetype,omitempty"`
}

type UpdateResourceCountParams struct {
	p map[string]interface{}
}

func (p *UpdateResourceCountParams) toURLValues() url.Values {
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
	if v, found := p.p["projectid"]; found {
		u.Set("projectid", v.(string))
	}
	if v, found := p.p["resourcetype"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("resourcetype", vv)
	}
	return u
}

func (p *UpdateResourceCountParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *UpdateResourceCountParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *UpdateResourceCountParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

func (p *UpdateResourceCountParams) SetResourcetype(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["resourcetype"] = v
	return
}

// You should always use this function to get a new UpdateResourceCountParams instance,
// as then you are sure you have configured all required params
func (s *LimitService) NewUpdateResourceCountParams(domainid string) *UpdateResourceCountParams {
	p := &UpdateResourceCountParams{}
	p.p = make(map[string]interface{})
	p.p["domainid"] = domainid
	return p
}

// Recalculate and update resource count for an account or domain.
func (s *LimitService) UpdateResourceCount(p *UpdateResourceCountParams) (*UpdateResourceCountResponse, error) {
	resp, err := s.cs.newRequest("updateResourceCount", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r UpdateResourceCountResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type UpdateResourceCountResponse struct {
	Account       string `json:"account,omitempty"`
	Domain        string `json:"domain,omitempty"`
	Domainid      string `json:"domainid,omitempty"`
	Project       string `json:"project,omitempty"`
	Projectid     string `json:"projectid,omitempty"`
	Resourcecount int64  `json:"resourcecount,omitempty"`
	Resourcetype  string `json:"resourcetype,omitempty"`
}

type ListResourceLimitsParams struct {
	p map[string]interface{}
}

func (p *ListResourceLimitsParams) toURLValues() url.Values {
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
		vv := strconv.FormatInt(v.(int64), 10)
		u.Set("id", vv)
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
	if v, found := p.p["resourcetype"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("resourcetype", vv)
	}
	return u
}

func (p *ListResourceLimitsParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *ListResourceLimitsParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *ListResourceLimitsParams) SetId(v int64) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ListResourceLimitsParams) SetIsrecursive(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isrecursive"] = v
	return
}

func (p *ListResourceLimitsParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListResourceLimitsParams) SetListall(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["listall"] = v
	return
}

func (p *ListResourceLimitsParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListResourceLimitsParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

func (p *ListResourceLimitsParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

func (p *ListResourceLimitsParams) SetResourcetype(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["resourcetype"] = v
	return
}

// You should always use this function to get a new ListResourceLimitsParams instance,
// as then you are sure you have configured all required params
func (s *LimitService) NewListResourceLimitsParams() *ListResourceLimitsParams {
	p := &ListResourceLimitsParams{}
	p.p = make(map[string]interface{})
	return p
}

// Lists resource limits.
func (s *LimitService) ListResourceLimits(p *ListResourceLimitsParams) (*ListResourceLimitsResponse, error) {
	resp, err := s.cs.newRequest("listResourceLimits", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListResourceLimitsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListResourceLimitsResponse struct {
	Count          int              `json:"count"`
	ResourceLimits []*ResourceLimit `json:"resourcelimit"`
}

type ResourceLimit struct {
	Account      string `json:"account,omitempty"`
	Domain       string `json:"domain,omitempty"`
	Domainid     string `json:"domainid,omitempty"`
	Max          int64  `json:"max,omitempty"`
	Project      string `json:"project,omitempty"`
	Projectid    string `json:"projectid,omitempty"`
	Resourcetype string `json:"resourcetype,omitempty"`
}

type GetApiLimitParams struct {
	p map[string]interface{}
}

func (p *GetApiLimitParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	return u
}

// You should always use this function to get a new GetApiLimitParams instance,
// as then you are sure you have configured all required params
func (s *LimitService) NewGetApiLimitParams() *GetApiLimitParams {
	p := &GetApiLimitParams{}
	p.p = make(map[string]interface{})
	return p
}

// Get API limit count for the caller
func (s *LimitService) GetApiLimit(p *GetApiLimitParams) (*GetApiLimitResponse, error) {
	resp, err := s.cs.newRequest("getApiLimit", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r GetApiLimitResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type GetApiLimitResponse struct {
	Account     string `json:"account,omitempty"`
	Accountid   string `json:"accountid,omitempty"`
	ApiAllowed  int    `json:"apiAllowed,omitempty"`
	ApiIssued   int    `json:"apiIssued,omitempty"`
	ExpireAfter int64  `json:"expireAfter,omitempty"`
}

type ResetApiLimitParams struct {
	p map[string]interface{}
}

func (p *ResetApiLimitParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["account"]; found {
		u.Set("account", v.(string))
	}
	return u
}

func (p *ResetApiLimitParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

// You should always use this function to get a new ResetApiLimitParams instance,
// as then you are sure you have configured all required params
func (s *LimitService) NewResetApiLimitParams() *ResetApiLimitParams {
	p := &ResetApiLimitParams{}
	p.p = make(map[string]interface{})
	return p
}

// Reset api count
func (s *LimitService) ResetApiLimit(p *ResetApiLimitParams) (*ResetApiLimitResponse, error) {
	resp, err := s.cs.newRequest("resetApiLimit", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ResetApiLimitResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ResetApiLimitResponse struct {
	Account     string `json:"account,omitempty"`
	Accountid   string `json:"accountid,omitempty"`
	ApiAllowed  int    `json:"apiAllowed,omitempty"`
	ApiIssued   int    `json:"apiIssued,omitempty"`
	ExpireAfter int64  `json:"expireAfter,omitempty"`
}
