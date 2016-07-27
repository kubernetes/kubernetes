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

type AddUcsManagerParams struct {
	p map[string]interface{}
}

func (p *AddUcsManagerParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["name"]; found {
		u.Set("name", v.(string))
	}
	if v, found := p.p["password"]; found {
		u.Set("password", v.(string))
	}
	if v, found := p.p["url"]; found {
		u.Set("url", v.(string))
	}
	if v, found := p.p["username"]; found {
		u.Set("username", v.(string))
	}
	if v, found := p.p["zoneid"]; found {
		u.Set("zoneid", v.(string))
	}
	return u
}

func (p *AddUcsManagerParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *AddUcsManagerParams) SetPassword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["password"] = v
	return
}

func (p *AddUcsManagerParams) SetUrl(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["url"] = v
	return
}

func (p *AddUcsManagerParams) SetUsername(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["username"] = v
	return
}

func (p *AddUcsManagerParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new AddUcsManagerParams instance,
// as then you are sure you have configured all required params
func (s *UCSService) NewAddUcsManagerParams(password string, url string, username string, zoneid string) *AddUcsManagerParams {
	p := &AddUcsManagerParams{}
	p.p = make(map[string]interface{})
	p.p["password"] = password
	p.p["url"] = url
	p.p["username"] = username
	p.p["zoneid"] = zoneid
	return p
}

// Adds a Ucs manager
func (s *UCSService) AddUcsManager(p *AddUcsManagerParams) (*AddUcsManagerResponse, error) {
	resp, err := s.cs.newRequest("addUcsManager", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r AddUcsManagerResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type AddUcsManagerResponse struct {
	Id     string `json:"id,omitempty"`
	Name   string `json:"name,omitempty"`
	Url    string `json:"url,omitempty"`
	Zoneid string `json:"zoneid,omitempty"`
}

type ListUcsManagersParams struct {
	p map[string]interface{}
}

func (p *ListUcsManagersParams) toURLValues() url.Values {
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
	if v, found := p.p["page"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("page", vv)
	}
	if v, found := p.p["pagesize"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("pagesize", vv)
	}
	if v, found := p.p["zoneid"]; found {
		u.Set("zoneid", v.(string))
	}
	return u
}

func (p *ListUcsManagersParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ListUcsManagersParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListUcsManagersParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListUcsManagersParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

func (p *ListUcsManagersParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new ListUcsManagersParams instance,
// as then you are sure you have configured all required params
func (s *UCSService) NewListUcsManagersParams() *ListUcsManagersParams {
	p := &ListUcsManagersParams{}
	p.p = make(map[string]interface{})
	return p
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *UCSService) GetUcsManagerID(keyword string, opts ...OptionFunc) (string, error) {
	p := &ListUcsManagersParams{}
	p.p = make(map[string]interface{})

	p.p["keyword"] = keyword

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return "", err
		}
	}

	l, err := s.ListUcsManagers(p)
	if err != nil {
		return "", err
	}

	if l.Count == 0 {
		return "", fmt.Errorf("No match found for %s: %+v", keyword, l)
	}

	if l.Count == 1 {
		return l.UcsManagers[0].Id, nil
	}

	if l.Count > 1 {
		for _, v := range l.UcsManagers {
			if v.Name == keyword {
				return v.Id, nil
			}
		}
	}
	return "", fmt.Errorf("Could not find an exact match for %s: %+v", keyword, l)
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *UCSService) GetUcsManagerByName(name string, opts ...OptionFunc) (*UcsManager, int, error) {
	id, err := s.GetUcsManagerID(name, opts...)
	if err != nil {
		return nil, -1, err
	}

	r, count, err := s.GetUcsManagerByID(id, opts...)
	if err != nil {
		return nil, count, err
	}
	return r, count, nil
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *UCSService) GetUcsManagerByID(id string, opts ...OptionFunc) (*UcsManager, int, error) {
	p := &ListUcsManagersParams{}
	p.p = make(map[string]interface{})

	p.p["id"] = id

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return nil, -1, err
		}
	}

	l, err := s.ListUcsManagers(p)
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
		return l.UcsManagers[0], l.Count, nil
	}
	return nil, l.Count, fmt.Errorf("There is more then one result for UcsManager UUID: %s!", id)
}

// List ucs manager
func (s *UCSService) ListUcsManagers(p *ListUcsManagersParams) (*ListUcsManagersResponse, error) {
	resp, err := s.cs.newRequest("listUcsManagers", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListUcsManagersResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListUcsManagersResponse struct {
	Count       int           `json:"count"`
	UcsManagers []*UcsManager `json:"ucsmanager"`
}

type UcsManager struct {
	Id     string `json:"id,omitempty"`
	Name   string `json:"name,omitempty"`
	Url    string `json:"url,omitempty"`
	Zoneid string `json:"zoneid,omitempty"`
}

type ListUcsProfilesParams struct {
	p map[string]interface{}
}

func (p *ListUcsProfilesParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
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
	if v, found := p.p["ucsmanagerid"]; found {
		u.Set("ucsmanagerid", v.(string))
	}
	return u
}

func (p *ListUcsProfilesParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListUcsProfilesParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListUcsProfilesParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

func (p *ListUcsProfilesParams) SetUcsmanagerid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["ucsmanagerid"] = v
	return
}

// You should always use this function to get a new ListUcsProfilesParams instance,
// as then you are sure you have configured all required params
func (s *UCSService) NewListUcsProfilesParams(ucsmanagerid string) *ListUcsProfilesParams {
	p := &ListUcsProfilesParams{}
	p.p = make(map[string]interface{})
	p.p["ucsmanagerid"] = ucsmanagerid
	return p
}

// List profile in ucs manager
func (s *UCSService) ListUcsProfiles(p *ListUcsProfilesParams) (*ListUcsProfilesResponse, error) {
	resp, err := s.cs.newRequest("listUcsProfiles", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListUcsProfilesResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListUcsProfilesResponse struct {
	Count       int           `json:"count"`
	UcsProfiles []*UcsProfile `json:"ucsprofile"`
}

type UcsProfile struct {
	Ucsdn string `json:"ucsdn,omitempty"`
}

type ListUcsBladesParams struct {
	p map[string]interface{}
}

func (p *ListUcsBladesParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
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
	if v, found := p.p["ucsmanagerid"]; found {
		u.Set("ucsmanagerid", v.(string))
	}
	return u
}

func (p *ListUcsBladesParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListUcsBladesParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListUcsBladesParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

func (p *ListUcsBladesParams) SetUcsmanagerid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["ucsmanagerid"] = v
	return
}

// You should always use this function to get a new ListUcsBladesParams instance,
// as then you are sure you have configured all required params
func (s *UCSService) NewListUcsBladesParams(ucsmanagerid string) *ListUcsBladesParams {
	p := &ListUcsBladesParams{}
	p.p = make(map[string]interface{})
	p.p["ucsmanagerid"] = ucsmanagerid
	return p
}

// List ucs blades
func (s *UCSService) ListUcsBlades(p *ListUcsBladesParams) (*ListUcsBladesResponse, error) {
	resp, err := s.cs.newRequest("listUcsBlades", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListUcsBladesResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListUcsBladesResponse struct {
	Count     int         `json:"count"`
	UcsBlades []*UcsBlade `json:"ucsblade"`
}

type UcsBlade struct {
	Bladedn      string `json:"bladedn,omitempty"`
	Hostid       string `json:"hostid,omitempty"`
	Id           string `json:"id,omitempty"`
	Profiledn    string `json:"profiledn,omitempty"`
	Ucsmanagerid string `json:"ucsmanagerid,omitempty"`
}

type AssociateUcsProfileToBladeParams struct {
	p map[string]interface{}
}

func (p *AssociateUcsProfileToBladeParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["bladeid"]; found {
		u.Set("bladeid", v.(string))
	}
	if v, found := p.p["profiledn"]; found {
		u.Set("profiledn", v.(string))
	}
	if v, found := p.p["ucsmanagerid"]; found {
		u.Set("ucsmanagerid", v.(string))
	}
	return u
}

func (p *AssociateUcsProfileToBladeParams) SetBladeid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["bladeid"] = v
	return
}

func (p *AssociateUcsProfileToBladeParams) SetProfiledn(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["profiledn"] = v
	return
}

func (p *AssociateUcsProfileToBladeParams) SetUcsmanagerid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["ucsmanagerid"] = v
	return
}

// You should always use this function to get a new AssociateUcsProfileToBladeParams instance,
// as then you are sure you have configured all required params
func (s *UCSService) NewAssociateUcsProfileToBladeParams(bladeid string, profiledn string, ucsmanagerid string) *AssociateUcsProfileToBladeParams {
	p := &AssociateUcsProfileToBladeParams{}
	p.p = make(map[string]interface{})
	p.p["bladeid"] = bladeid
	p.p["profiledn"] = profiledn
	p.p["ucsmanagerid"] = ucsmanagerid
	return p
}

// associate a profile to a blade
func (s *UCSService) AssociateUcsProfileToBlade(p *AssociateUcsProfileToBladeParams) (*AssociateUcsProfileToBladeResponse, error) {
	resp, err := s.cs.newRequest("associateUcsProfileToBlade", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r AssociateUcsProfileToBladeResponse
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

type AssociateUcsProfileToBladeResponse struct {
	JobID        string `json:"jobid,omitempty"`
	Bladedn      string `json:"bladedn,omitempty"`
	Hostid       string `json:"hostid,omitempty"`
	Id           string `json:"id,omitempty"`
	Profiledn    string `json:"profiledn,omitempty"`
	Ucsmanagerid string `json:"ucsmanagerid,omitempty"`
}
