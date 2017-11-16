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

type ListOsTypesParams struct {
	p map[string]interface{}
}

func (p *ListOsTypesParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["description"]; found {
		u.Set("description", v.(string))
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	if v, found := p.p["keyword"]; found {
		u.Set("keyword", v.(string))
	}
	if v, found := p.p["oscategoryid"]; found {
		u.Set("oscategoryid", v.(string))
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

func (p *ListOsTypesParams) SetDescription(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["description"] = v
	return
}

func (p *ListOsTypesParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ListOsTypesParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListOsTypesParams) SetOscategoryid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["oscategoryid"] = v
	return
}

func (p *ListOsTypesParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListOsTypesParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

// You should always use this function to get a new ListOsTypesParams instance,
// as then you are sure you have configured all required params
func (s *GuestOSService) NewListOsTypesParams() *ListOsTypesParams {
	p := &ListOsTypesParams{}
	p.p = make(map[string]interface{})
	return p
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *GuestOSService) GetOsTypeByID(id string, opts ...OptionFunc) (*OsType, int, error) {
	p := &ListOsTypesParams{}
	p.p = make(map[string]interface{})

	p.p["id"] = id

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return nil, -1, err
		}
	}

	l, err := s.ListOsTypes(p)
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
		return l.OsTypes[0], l.Count, nil
	}
	return nil, l.Count, fmt.Errorf("There is more then one result for OsType UUID: %s!", id)
}

// Lists all supported OS types for this cloud.
func (s *GuestOSService) ListOsTypes(p *ListOsTypesParams) (*ListOsTypesResponse, error) {
	resp, err := s.cs.newRequest("listOsTypes", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListOsTypesResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListOsTypesResponse struct {
	Count   int       `json:"count"`
	OsTypes []*OsType `json:"ostype"`
}

type OsType struct {
	Description   string `json:"description,omitempty"`
	Id            string `json:"id,omitempty"`
	Isuserdefined string `json:"isuserdefined,omitempty"`
	Oscategoryid  string `json:"oscategoryid,omitempty"`
}

type ListOsCategoriesParams struct {
	p map[string]interface{}
}

func (p *ListOsCategoriesParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
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

func (p *ListOsCategoriesParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ListOsCategoriesParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListOsCategoriesParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *ListOsCategoriesParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListOsCategoriesParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

// You should always use this function to get a new ListOsCategoriesParams instance,
// as then you are sure you have configured all required params
func (s *GuestOSService) NewListOsCategoriesParams() *ListOsCategoriesParams {
	p := &ListOsCategoriesParams{}
	p.p = make(map[string]interface{})
	return p
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *GuestOSService) GetOsCategoryID(name string, opts ...OptionFunc) (string, int, error) {
	p := &ListOsCategoriesParams{}
	p.p = make(map[string]interface{})

	p.p["name"] = name

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return "", -1, err
		}
	}

	l, err := s.ListOsCategories(p)
	if err != nil {
		return "", -1, err
	}

	if l.Count == 0 {
		return "", l.Count, fmt.Errorf("No match found for %s: %+v", name, l)
	}

	if l.Count == 1 {
		return l.OsCategories[0].Id, l.Count, nil
	}

	if l.Count > 1 {
		for _, v := range l.OsCategories {
			if v.Name == name {
				return v.Id, l.Count, nil
			}
		}
	}
	return "", l.Count, fmt.Errorf("Could not find an exact match for %s: %+v", name, l)
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *GuestOSService) GetOsCategoryByName(name string, opts ...OptionFunc) (*OsCategory, int, error) {
	id, count, err := s.GetOsCategoryID(name, opts...)
	if err != nil {
		return nil, count, err
	}

	r, count, err := s.GetOsCategoryByID(id, opts...)
	if err != nil {
		return nil, count, err
	}
	return r, count, nil
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *GuestOSService) GetOsCategoryByID(id string, opts ...OptionFunc) (*OsCategory, int, error) {
	p := &ListOsCategoriesParams{}
	p.p = make(map[string]interface{})

	p.p["id"] = id

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return nil, -1, err
		}
	}

	l, err := s.ListOsCategories(p)
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
		return l.OsCategories[0], l.Count, nil
	}
	return nil, l.Count, fmt.Errorf("There is more then one result for OsCategory UUID: %s!", id)
}

// Lists all supported OS categories for this cloud.
func (s *GuestOSService) ListOsCategories(p *ListOsCategoriesParams) (*ListOsCategoriesResponse, error) {
	resp, err := s.cs.newRequest("listOsCategories", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListOsCategoriesResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListOsCategoriesResponse struct {
	Count        int           `json:"count"`
	OsCategories []*OsCategory `json:"oscategory"`
}

type OsCategory struct {
	Id   string `json:"id,omitempty"`
	Name string `json:"name,omitempty"`
}

type AddGuestOsParams struct {
	p map[string]interface{}
}

func (p *AddGuestOsParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["name"]; found {
		u.Set("name", v.(string))
	}
	if v, found := p.p["oscategoryid"]; found {
		u.Set("oscategoryid", v.(string))
	}
	if v, found := p.p["osdisplayname"]; found {
		u.Set("osdisplayname", v.(string))
	}
	return u
}

func (p *AddGuestOsParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *AddGuestOsParams) SetOscategoryid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["oscategoryid"] = v
	return
}

func (p *AddGuestOsParams) SetOsdisplayname(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["osdisplayname"] = v
	return
}

// You should always use this function to get a new AddGuestOsParams instance,
// as then you are sure you have configured all required params
func (s *GuestOSService) NewAddGuestOsParams(oscategoryid string, osdisplayname string) *AddGuestOsParams {
	p := &AddGuestOsParams{}
	p.p = make(map[string]interface{})
	p.p["oscategoryid"] = oscategoryid
	p.p["osdisplayname"] = osdisplayname
	return p
}

// Add a new guest OS type
func (s *GuestOSService) AddGuestOs(p *AddGuestOsParams) (*AddGuestOsResponse, error) {
	resp, err := s.cs.newRequest("addGuestOs", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r AddGuestOsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}

	// If we have a async client, we need to wait for the async result
	if s.cs.async {
		b, err := s.cs.GetAsyncJobResult(r.JobID, s.cs.timeout)
		if err != nil {
			if err == AsyncTimeoutErr {
				return &r, err
			}
			return nil, err
		}

		b, err = getRawValue(b)
		if err != nil {
			return nil, err
		}

		if err := json.Unmarshal(b, &r); err != nil {
			return nil, err
		}
	}
	return &r, nil
}

type AddGuestOsResponse struct {
	JobID         string `json:"jobid,omitempty"`
	Description   string `json:"description,omitempty"`
	Id            string `json:"id,omitempty"`
	Isuserdefined string `json:"isuserdefined,omitempty"`
	Oscategoryid  string `json:"oscategoryid,omitempty"`
}

type UpdateGuestOsParams struct {
	p map[string]interface{}
}

func (p *UpdateGuestOsParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	if v, found := p.p["osdisplayname"]; found {
		u.Set("osdisplayname", v.(string))
	}
	return u
}

func (p *UpdateGuestOsParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *UpdateGuestOsParams) SetOsdisplayname(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["osdisplayname"] = v
	return
}

// You should always use this function to get a new UpdateGuestOsParams instance,
// as then you are sure you have configured all required params
func (s *GuestOSService) NewUpdateGuestOsParams(id string, osdisplayname string) *UpdateGuestOsParams {
	p := &UpdateGuestOsParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	p.p["osdisplayname"] = osdisplayname
	return p
}

// Updates the information about Guest OS
func (s *GuestOSService) UpdateGuestOs(p *UpdateGuestOsParams) (*UpdateGuestOsResponse, error) {
	resp, err := s.cs.newRequest("updateGuestOs", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r UpdateGuestOsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}

	// If we have a async client, we need to wait for the async result
	if s.cs.async {
		b, err := s.cs.GetAsyncJobResult(r.JobID, s.cs.timeout)
		if err != nil {
			if err == AsyncTimeoutErr {
				return &r, err
			}
			return nil, err
		}

		b, err = getRawValue(b)
		if err != nil {
			return nil, err
		}

		if err := json.Unmarshal(b, &r); err != nil {
			return nil, err
		}
	}
	return &r, nil
}

type UpdateGuestOsResponse struct {
	JobID         string `json:"jobid,omitempty"`
	Description   string `json:"description,omitempty"`
	Id            string `json:"id,omitempty"`
	Isuserdefined string `json:"isuserdefined,omitempty"`
	Oscategoryid  string `json:"oscategoryid,omitempty"`
}

type RemoveGuestOsParams struct {
	p map[string]interface{}
}

func (p *RemoveGuestOsParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	return u
}

func (p *RemoveGuestOsParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new RemoveGuestOsParams instance,
// as then you are sure you have configured all required params
func (s *GuestOSService) NewRemoveGuestOsParams(id string) *RemoveGuestOsParams {
	p := &RemoveGuestOsParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Removes a Guest OS from listing.
func (s *GuestOSService) RemoveGuestOs(p *RemoveGuestOsParams) (*RemoveGuestOsResponse, error) {
	resp, err := s.cs.newRequest("removeGuestOs", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r RemoveGuestOsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}

	// If we have a async client, we need to wait for the async result
	if s.cs.async {
		b, err := s.cs.GetAsyncJobResult(r.JobID, s.cs.timeout)
		if err != nil {
			if err == AsyncTimeoutErr {
				return &r, err
			}
			return nil, err
		}

		if err := json.Unmarshal(b, &r); err != nil {
			return nil, err
		}
	}
	return &r, nil
}

type RemoveGuestOsResponse struct {
	JobID       string `json:"jobid,omitempty"`
	Displaytext string `json:"displaytext,omitempty"`
	Success     bool   `json:"success,omitempty"`
}

type ListGuestOsMappingParams struct {
	p map[string]interface{}
}

func (p *ListGuestOsMappingParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["hypervisor"]; found {
		u.Set("hypervisor", v.(string))
	}
	if v, found := p.p["hypervisorversion"]; found {
		u.Set("hypervisorversion", v.(string))
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	if v, found := p.p["keyword"]; found {
		u.Set("keyword", v.(string))
	}
	if v, found := p.p["ostypeid"]; found {
		u.Set("ostypeid", v.(string))
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

func (p *ListGuestOsMappingParams) SetHypervisor(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["hypervisor"] = v
	return
}

func (p *ListGuestOsMappingParams) SetHypervisorversion(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["hypervisorversion"] = v
	return
}

func (p *ListGuestOsMappingParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ListGuestOsMappingParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListGuestOsMappingParams) SetOstypeid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["ostypeid"] = v
	return
}

func (p *ListGuestOsMappingParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListGuestOsMappingParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

// You should always use this function to get a new ListGuestOsMappingParams instance,
// as then you are sure you have configured all required params
func (s *GuestOSService) NewListGuestOsMappingParams() *ListGuestOsMappingParams {
	p := &ListGuestOsMappingParams{}
	p.p = make(map[string]interface{})
	return p
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *GuestOSService) GetGuestOsMappingByID(id string, opts ...OptionFunc) (*GuestOsMapping, int, error) {
	p := &ListGuestOsMappingParams{}
	p.p = make(map[string]interface{})

	p.p["id"] = id

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return nil, -1, err
		}
	}

	l, err := s.ListGuestOsMapping(p)
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
		return l.GuestOsMapping[0], l.Count, nil
	}
	return nil, l.Count, fmt.Errorf("There is more then one result for GuestOsMapping UUID: %s!", id)
}

// Lists all available OS mappings for given hypervisor
func (s *GuestOSService) ListGuestOsMapping(p *ListGuestOsMappingParams) (*ListGuestOsMappingResponse, error) {
	resp, err := s.cs.newRequest("listGuestOsMapping", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListGuestOsMappingResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListGuestOsMappingResponse struct {
	Count          int               `json:"count"`
	GuestOsMapping []*GuestOsMapping `json:"guestosmapping"`
}

type GuestOsMapping struct {
	Hypervisor          string `json:"hypervisor,omitempty"`
	Hypervisorversion   string `json:"hypervisorversion,omitempty"`
	Id                  string `json:"id,omitempty"`
	Isuserdefined       string `json:"isuserdefined,omitempty"`
	Osdisplayname       string `json:"osdisplayname,omitempty"`
	Osnameforhypervisor string `json:"osnameforhypervisor,omitempty"`
	Ostypeid            string `json:"ostypeid,omitempty"`
}

type AddGuestOsMappingParams struct {
	p map[string]interface{}
}

func (p *AddGuestOsMappingParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["hypervisor"]; found {
		u.Set("hypervisor", v.(string))
	}
	if v, found := p.p["hypervisorversion"]; found {
		u.Set("hypervisorversion", v.(string))
	}
	if v, found := p.p["osdisplayname"]; found {
		u.Set("osdisplayname", v.(string))
	}
	if v, found := p.p["osnameforhypervisor"]; found {
		u.Set("osnameforhypervisor", v.(string))
	}
	if v, found := p.p["ostypeid"]; found {
		u.Set("ostypeid", v.(string))
	}
	return u
}

func (p *AddGuestOsMappingParams) SetHypervisor(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["hypervisor"] = v
	return
}

func (p *AddGuestOsMappingParams) SetHypervisorversion(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["hypervisorversion"] = v
	return
}

func (p *AddGuestOsMappingParams) SetOsdisplayname(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["osdisplayname"] = v
	return
}

func (p *AddGuestOsMappingParams) SetOsnameforhypervisor(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["osnameforhypervisor"] = v
	return
}

func (p *AddGuestOsMappingParams) SetOstypeid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["ostypeid"] = v
	return
}

// You should always use this function to get a new AddGuestOsMappingParams instance,
// as then you are sure you have configured all required params
func (s *GuestOSService) NewAddGuestOsMappingParams(hypervisor string, hypervisorversion string, osnameforhypervisor string) *AddGuestOsMappingParams {
	p := &AddGuestOsMappingParams{}
	p.p = make(map[string]interface{})
	p.p["hypervisor"] = hypervisor
	p.p["hypervisorversion"] = hypervisorversion
	p.p["osnameforhypervisor"] = osnameforhypervisor
	return p
}

// Adds a guest OS name to hypervisor OS name mapping
func (s *GuestOSService) AddGuestOsMapping(p *AddGuestOsMappingParams) (*AddGuestOsMappingResponse, error) {
	resp, err := s.cs.newRequest("addGuestOsMapping", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r AddGuestOsMappingResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}

	// If we have a async client, we need to wait for the async result
	if s.cs.async {
		b, err := s.cs.GetAsyncJobResult(r.JobID, s.cs.timeout)
		if err != nil {
			if err == AsyncTimeoutErr {
				return &r, err
			}
			return nil, err
		}

		b, err = getRawValue(b)
		if err != nil {
			return nil, err
		}

		if err := json.Unmarshal(b, &r); err != nil {
			return nil, err
		}
	}
	return &r, nil
}

type AddGuestOsMappingResponse struct {
	JobID               string `json:"jobid,omitempty"`
	Hypervisor          string `json:"hypervisor,omitempty"`
	Hypervisorversion   string `json:"hypervisorversion,omitempty"`
	Id                  string `json:"id,omitempty"`
	Isuserdefined       string `json:"isuserdefined,omitempty"`
	Osdisplayname       string `json:"osdisplayname,omitempty"`
	Osnameforhypervisor string `json:"osnameforhypervisor,omitempty"`
	Ostypeid            string `json:"ostypeid,omitempty"`
}

type UpdateGuestOsMappingParams struct {
	p map[string]interface{}
}

func (p *UpdateGuestOsMappingParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	if v, found := p.p["osnameforhypervisor"]; found {
		u.Set("osnameforhypervisor", v.(string))
	}
	return u
}

func (p *UpdateGuestOsMappingParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *UpdateGuestOsMappingParams) SetOsnameforhypervisor(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["osnameforhypervisor"] = v
	return
}

// You should always use this function to get a new UpdateGuestOsMappingParams instance,
// as then you are sure you have configured all required params
func (s *GuestOSService) NewUpdateGuestOsMappingParams(id string, osnameforhypervisor string) *UpdateGuestOsMappingParams {
	p := &UpdateGuestOsMappingParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	p.p["osnameforhypervisor"] = osnameforhypervisor
	return p
}

// Updates the information about Guest OS to Hypervisor specific name mapping
func (s *GuestOSService) UpdateGuestOsMapping(p *UpdateGuestOsMappingParams) (*UpdateGuestOsMappingResponse, error) {
	resp, err := s.cs.newRequest("updateGuestOsMapping", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r UpdateGuestOsMappingResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}

	// If we have a async client, we need to wait for the async result
	if s.cs.async {
		b, err := s.cs.GetAsyncJobResult(r.JobID, s.cs.timeout)
		if err != nil {
			if err == AsyncTimeoutErr {
				return &r, err
			}
			return nil, err
		}

		b, err = getRawValue(b)
		if err != nil {
			return nil, err
		}

		if err := json.Unmarshal(b, &r); err != nil {
			return nil, err
		}
	}
	return &r, nil
}

type UpdateGuestOsMappingResponse struct {
	JobID               string `json:"jobid,omitempty"`
	Hypervisor          string `json:"hypervisor,omitempty"`
	Hypervisorversion   string `json:"hypervisorversion,omitempty"`
	Id                  string `json:"id,omitempty"`
	Isuserdefined       string `json:"isuserdefined,omitempty"`
	Osdisplayname       string `json:"osdisplayname,omitempty"`
	Osnameforhypervisor string `json:"osnameforhypervisor,omitempty"`
	Ostypeid            string `json:"ostypeid,omitempty"`
}

type RemoveGuestOsMappingParams struct {
	p map[string]interface{}
}

func (p *RemoveGuestOsMappingParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	return u
}

func (p *RemoveGuestOsMappingParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new RemoveGuestOsMappingParams instance,
// as then you are sure you have configured all required params
func (s *GuestOSService) NewRemoveGuestOsMappingParams(id string) *RemoveGuestOsMappingParams {
	p := &RemoveGuestOsMappingParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Removes a Guest OS Mapping.
func (s *GuestOSService) RemoveGuestOsMapping(p *RemoveGuestOsMappingParams) (*RemoveGuestOsMappingResponse, error) {
	resp, err := s.cs.newRequest("removeGuestOsMapping", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r RemoveGuestOsMappingResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}

	// If we have a async client, we need to wait for the async result
	if s.cs.async {
		b, err := s.cs.GetAsyncJobResult(r.JobID, s.cs.timeout)
		if err != nil {
			if err == AsyncTimeoutErr {
				return &r, err
			}
			return nil, err
		}

		if err := json.Unmarshal(b, &r); err != nil {
			return nil, err
		}
	}
	return &r, nil
}

type RemoveGuestOsMappingResponse struct {
	JobID       string `json:"jobid,omitempty"`
	Displaytext string `json:"displaytext,omitempty"`
	Success     bool   `json:"success,omitempty"`
}
