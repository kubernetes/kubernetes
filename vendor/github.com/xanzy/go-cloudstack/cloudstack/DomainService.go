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

type CreateDomainParams struct {
	p map[string]interface{}
}

func (p *CreateDomainParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["domainid"]; found {
		u.Set("domainid", v.(string))
	}
	if v, found := p.p["name"]; found {
		u.Set("name", v.(string))
	}
	if v, found := p.p["networkdomain"]; found {
		u.Set("networkdomain", v.(string))
	}
	if v, found := p.p["parentdomainid"]; found {
		u.Set("parentdomainid", v.(string))
	}
	return u
}

func (p *CreateDomainParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *CreateDomainParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *CreateDomainParams) SetNetworkdomain(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["networkdomain"] = v
	return
}

func (p *CreateDomainParams) SetParentdomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["parentdomainid"] = v
	return
}

// You should always use this function to get a new CreateDomainParams instance,
// as then you are sure you have configured all required params
func (s *DomainService) NewCreateDomainParams(name string) *CreateDomainParams {
	p := &CreateDomainParams{}
	p.p = make(map[string]interface{})
	p.p["name"] = name
	return p
}

// Creates a domain
func (s *DomainService) CreateDomain(p *CreateDomainParams) (*CreateDomainResponse, error) {
	resp, err := s.cs.newRequest("createDomain", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r CreateDomainResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type CreateDomainResponse struct {
	Cpuavailable              string `json:"cpuavailable,omitempty"`
	Cpulimit                  string `json:"cpulimit,omitempty"`
	Cputotal                  int64  `json:"cputotal,omitempty"`
	Haschild                  bool   `json:"haschild,omitempty"`
	Id                        string `json:"id,omitempty"`
	Ipavailable               string `json:"ipavailable,omitempty"`
	Iplimit                   string `json:"iplimit,omitempty"`
	Iptotal                   int64  `json:"iptotal,omitempty"`
	Level                     int    `json:"level,omitempty"`
	Memoryavailable           string `json:"memoryavailable,omitempty"`
	Memorylimit               string `json:"memorylimit,omitempty"`
	Memorytotal               int64  `json:"memorytotal,omitempty"`
	Name                      string `json:"name,omitempty"`
	Networkavailable          string `json:"networkavailable,omitempty"`
	Networkdomain             string `json:"networkdomain,omitempty"`
	Networklimit              string `json:"networklimit,omitempty"`
	Networktotal              int64  `json:"networktotal,omitempty"`
	Parentdomainid            string `json:"parentdomainid,omitempty"`
	Parentdomainname          string `json:"parentdomainname,omitempty"`
	Path                      string `json:"path,omitempty"`
	Primarystorageavailable   string `json:"primarystorageavailable,omitempty"`
	Primarystoragelimit       string `json:"primarystoragelimit,omitempty"`
	Primarystoragetotal       int64  `json:"primarystoragetotal,omitempty"`
	Projectavailable          string `json:"projectavailable,omitempty"`
	Projectlimit              string `json:"projectlimit,omitempty"`
	Projecttotal              int64  `json:"projecttotal,omitempty"`
	Secondarystorageavailable string `json:"secondarystorageavailable,omitempty"`
	Secondarystoragelimit     string `json:"secondarystoragelimit,omitempty"`
	Secondarystoragetotal     int64  `json:"secondarystoragetotal,omitempty"`
	Snapshotavailable         string `json:"snapshotavailable,omitempty"`
	Snapshotlimit             string `json:"snapshotlimit,omitempty"`
	Snapshottotal             int64  `json:"snapshottotal,omitempty"`
	State                     string `json:"state,omitempty"`
	Templateavailable         string `json:"templateavailable,omitempty"`
	Templatelimit             string `json:"templatelimit,omitempty"`
	Templatetotal             int64  `json:"templatetotal,omitempty"`
	Vmavailable               string `json:"vmavailable,omitempty"`
	Vmlimit                   string `json:"vmlimit,omitempty"`
	Vmtotal                   int64  `json:"vmtotal,omitempty"`
	Volumeavailable           string `json:"volumeavailable,omitempty"`
	Volumelimit               string `json:"volumelimit,omitempty"`
	Volumetotal               int64  `json:"volumetotal,omitempty"`
	Vpcavailable              string `json:"vpcavailable,omitempty"`
	Vpclimit                  string `json:"vpclimit,omitempty"`
	Vpctotal                  int64  `json:"vpctotal,omitempty"`
}

type UpdateDomainParams struct {
	p map[string]interface{}
}

func (p *UpdateDomainParams) toURLValues() url.Values {
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
	if v, found := p.p["networkdomain"]; found {
		u.Set("networkdomain", v.(string))
	}
	return u
}

func (p *UpdateDomainParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *UpdateDomainParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *UpdateDomainParams) SetNetworkdomain(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["networkdomain"] = v
	return
}

// You should always use this function to get a new UpdateDomainParams instance,
// as then you are sure you have configured all required params
func (s *DomainService) NewUpdateDomainParams(id string) *UpdateDomainParams {
	p := &UpdateDomainParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Updates a domain with a new name
func (s *DomainService) UpdateDomain(p *UpdateDomainParams) (*UpdateDomainResponse, error) {
	resp, err := s.cs.newRequest("updateDomain", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r UpdateDomainResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type UpdateDomainResponse struct {
	Cpuavailable              string `json:"cpuavailable,omitempty"`
	Cpulimit                  string `json:"cpulimit,omitempty"`
	Cputotal                  int64  `json:"cputotal,omitempty"`
	Haschild                  bool   `json:"haschild,omitempty"`
	Id                        string `json:"id,omitempty"`
	Ipavailable               string `json:"ipavailable,omitempty"`
	Iplimit                   string `json:"iplimit,omitempty"`
	Iptotal                   int64  `json:"iptotal,omitempty"`
	Level                     int    `json:"level,omitempty"`
	Memoryavailable           string `json:"memoryavailable,omitempty"`
	Memorylimit               string `json:"memorylimit,omitempty"`
	Memorytotal               int64  `json:"memorytotal,omitempty"`
	Name                      string `json:"name,omitempty"`
	Networkavailable          string `json:"networkavailable,omitempty"`
	Networkdomain             string `json:"networkdomain,omitempty"`
	Networklimit              string `json:"networklimit,omitempty"`
	Networktotal              int64  `json:"networktotal,omitempty"`
	Parentdomainid            string `json:"parentdomainid,omitempty"`
	Parentdomainname          string `json:"parentdomainname,omitempty"`
	Path                      string `json:"path,omitempty"`
	Primarystorageavailable   string `json:"primarystorageavailable,omitempty"`
	Primarystoragelimit       string `json:"primarystoragelimit,omitempty"`
	Primarystoragetotal       int64  `json:"primarystoragetotal,omitempty"`
	Projectavailable          string `json:"projectavailable,omitempty"`
	Projectlimit              string `json:"projectlimit,omitempty"`
	Projecttotal              int64  `json:"projecttotal,omitempty"`
	Secondarystorageavailable string `json:"secondarystorageavailable,omitempty"`
	Secondarystoragelimit     string `json:"secondarystoragelimit,omitempty"`
	Secondarystoragetotal     int64  `json:"secondarystoragetotal,omitempty"`
	Snapshotavailable         string `json:"snapshotavailable,omitempty"`
	Snapshotlimit             string `json:"snapshotlimit,omitempty"`
	Snapshottotal             int64  `json:"snapshottotal,omitempty"`
	State                     string `json:"state,omitempty"`
	Templateavailable         string `json:"templateavailable,omitempty"`
	Templatelimit             string `json:"templatelimit,omitempty"`
	Templatetotal             int64  `json:"templatetotal,omitempty"`
	Vmavailable               string `json:"vmavailable,omitempty"`
	Vmlimit                   string `json:"vmlimit,omitempty"`
	Vmtotal                   int64  `json:"vmtotal,omitempty"`
	Volumeavailable           string `json:"volumeavailable,omitempty"`
	Volumelimit               string `json:"volumelimit,omitempty"`
	Volumetotal               int64  `json:"volumetotal,omitempty"`
	Vpcavailable              string `json:"vpcavailable,omitempty"`
	Vpclimit                  string `json:"vpclimit,omitempty"`
	Vpctotal                  int64  `json:"vpctotal,omitempty"`
}

type DeleteDomainParams struct {
	p map[string]interface{}
}

func (p *DeleteDomainParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["cleanup"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("cleanup", vv)
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	return u
}

func (p *DeleteDomainParams) SetCleanup(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["cleanup"] = v
	return
}

func (p *DeleteDomainParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new DeleteDomainParams instance,
// as then you are sure you have configured all required params
func (s *DomainService) NewDeleteDomainParams(id string) *DeleteDomainParams {
	p := &DeleteDomainParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Deletes a specified domain
func (s *DomainService) DeleteDomain(p *DeleteDomainParams) (*DeleteDomainResponse, error) {
	resp, err := s.cs.newRequest("deleteDomain", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r DeleteDomainResponse
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

type DeleteDomainResponse struct {
	JobID       string `json:"jobid,omitempty"`
	Displaytext string `json:"displaytext,omitempty"`
	Success     bool   `json:"success,omitempty"`
}

type ListDomainsParams struct {
	p map[string]interface{}
}

func (p *ListDomainsParams) toURLValues() url.Values {
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
	if v, found := p.p["level"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("level", vv)
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
	return u
}

func (p *ListDomainsParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ListDomainsParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListDomainsParams) SetLevel(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["level"] = v
	return
}

func (p *ListDomainsParams) SetListall(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["listall"] = v
	return
}

func (p *ListDomainsParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *ListDomainsParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListDomainsParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

// You should always use this function to get a new ListDomainsParams instance,
// as then you are sure you have configured all required params
func (s *DomainService) NewListDomainsParams() *ListDomainsParams {
	p := &ListDomainsParams{}
	p.p = make(map[string]interface{})
	return p
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *DomainService) GetDomainID(name string, opts ...OptionFunc) (string, int, error) {
	p := &ListDomainsParams{}
	p.p = make(map[string]interface{})

	p.p["name"] = name

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return "", -1, err
		}
	}

	l, err := s.ListDomains(p)
	if err != nil {
		return "", -1, err
	}

	if l.Count == 0 {
		return "", l.Count, fmt.Errorf("No match found for %s: %+v", name, l)
	}

	if l.Count == 1 {
		return l.Domains[0].Id, l.Count, nil
	}

	if l.Count > 1 {
		for _, v := range l.Domains {
			if v.Name == name {
				return v.Id, l.Count, nil
			}
		}
	}
	return "", l.Count, fmt.Errorf("Could not find an exact match for %s: %+v", name, l)
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *DomainService) GetDomainByName(name string, opts ...OptionFunc) (*Domain, int, error) {
	id, count, err := s.GetDomainID(name, opts...)
	if err != nil {
		return nil, count, err
	}

	r, count, err := s.GetDomainByID(id, opts...)
	if err != nil {
		return nil, count, err
	}
	return r, count, nil
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *DomainService) GetDomainByID(id string, opts ...OptionFunc) (*Domain, int, error) {
	p := &ListDomainsParams{}
	p.p = make(map[string]interface{})

	p.p["id"] = id

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return nil, -1, err
		}
	}

	l, err := s.ListDomains(p)
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
		return l.Domains[0], l.Count, nil
	}
	return nil, l.Count, fmt.Errorf("There is more then one result for Domain UUID: %s!", id)
}

// Lists domains and provides detailed information for listed domains
func (s *DomainService) ListDomains(p *ListDomainsParams) (*ListDomainsResponse, error) {
	resp, err := s.cs.newRequest("listDomains", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListDomainsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListDomainsResponse struct {
	Count   int       `json:"count"`
	Domains []*Domain `json:"domain"`
}

type Domain struct {
	Cpuavailable              string `json:"cpuavailable,omitempty"`
	Cpulimit                  string `json:"cpulimit,omitempty"`
	Cputotal                  int64  `json:"cputotal,omitempty"`
	Haschild                  bool   `json:"haschild,omitempty"`
	Id                        string `json:"id,omitempty"`
	Ipavailable               string `json:"ipavailable,omitempty"`
	Iplimit                   string `json:"iplimit,omitempty"`
	Iptotal                   int64  `json:"iptotal,omitempty"`
	Level                     int    `json:"level,omitempty"`
	Memoryavailable           string `json:"memoryavailable,omitempty"`
	Memorylimit               string `json:"memorylimit,omitempty"`
	Memorytotal               int64  `json:"memorytotal,omitempty"`
	Name                      string `json:"name,omitempty"`
	Networkavailable          string `json:"networkavailable,omitempty"`
	Networkdomain             string `json:"networkdomain,omitempty"`
	Networklimit              string `json:"networklimit,omitempty"`
	Networktotal              int64  `json:"networktotal,omitempty"`
	Parentdomainid            string `json:"parentdomainid,omitempty"`
	Parentdomainname          string `json:"parentdomainname,omitempty"`
	Path                      string `json:"path,omitempty"`
	Primarystorageavailable   string `json:"primarystorageavailable,omitempty"`
	Primarystoragelimit       string `json:"primarystoragelimit,omitempty"`
	Primarystoragetotal       int64  `json:"primarystoragetotal,omitempty"`
	Projectavailable          string `json:"projectavailable,omitempty"`
	Projectlimit              string `json:"projectlimit,omitempty"`
	Projecttotal              int64  `json:"projecttotal,omitempty"`
	Secondarystorageavailable string `json:"secondarystorageavailable,omitempty"`
	Secondarystoragelimit     string `json:"secondarystoragelimit,omitempty"`
	Secondarystoragetotal     int64  `json:"secondarystoragetotal,omitempty"`
	Snapshotavailable         string `json:"snapshotavailable,omitempty"`
	Snapshotlimit             string `json:"snapshotlimit,omitempty"`
	Snapshottotal             int64  `json:"snapshottotal,omitempty"`
	State                     string `json:"state,omitempty"`
	Templateavailable         string `json:"templateavailable,omitempty"`
	Templatelimit             string `json:"templatelimit,omitempty"`
	Templatetotal             int64  `json:"templatetotal,omitempty"`
	Vmavailable               string `json:"vmavailable,omitempty"`
	Vmlimit                   string `json:"vmlimit,omitempty"`
	Vmtotal                   int64  `json:"vmtotal,omitempty"`
	Volumeavailable           string `json:"volumeavailable,omitempty"`
	Volumelimit               string `json:"volumelimit,omitempty"`
	Volumetotal               int64  `json:"volumetotal,omitempty"`
	Vpcavailable              string `json:"vpcavailable,omitempty"`
	Vpclimit                  string `json:"vpclimit,omitempty"`
	Vpctotal                  int64  `json:"vpctotal,omitempty"`
}

type ListDomainChildrenParams struct {
	p map[string]interface{}
}

func (p *ListDomainChildrenParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
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
	return u
}

func (p *ListDomainChildrenParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ListDomainChildrenParams) SetIsrecursive(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isrecursive"] = v
	return
}

func (p *ListDomainChildrenParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListDomainChildrenParams) SetListall(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["listall"] = v
	return
}

func (p *ListDomainChildrenParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *ListDomainChildrenParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListDomainChildrenParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

// You should always use this function to get a new ListDomainChildrenParams instance,
// as then you are sure you have configured all required params
func (s *DomainService) NewListDomainChildrenParams() *ListDomainChildrenParams {
	p := &ListDomainChildrenParams{}
	p.p = make(map[string]interface{})
	return p
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *DomainService) GetDomainChildrenID(name string, opts ...OptionFunc) (string, int, error) {
	p := &ListDomainChildrenParams{}
	p.p = make(map[string]interface{})

	p.p["name"] = name

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return "", -1, err
		}
	}

	l, err := s.ListDomainChildren(p)
	if err != nil {
		return "", -1, err
	}

	if l.Count == 0 {
		return "", l.Count, fmt.Errorf("No match found for %s: %+v", name, l)
	}

	if l.Count == 1 {
		return l.DomainChildren[0].Id, l.Count, nil
	}

	if l.Count > 1 {
		for _, v := range l.DomainChildren {
			if v.Name == name {
				return v.Id, l.Count, nil
			}
		}
	}
	return "", l.Count, fmt.Errorf("Could not find an exact match for %s: %+v", name, l)
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *DomainService) GetDomainChildrenByName(name string, opts ...OptionFunc) (*DomainChildren, int, error) {
	id, count, err := s.GetDomainChildrenID(name, opts...)
	if err != nil {
		return nil, count, err
	}

	r, count, err := s.GetDomainChildrenByID(id, opts...)
	if err != nil {
		return nil, count, err
	}
	return r, count, nil
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *DomainService) GetDomainChildrenByID(id string, opts ...OptionFunc) (*DomainChildren, int, error) {
	p := &ListDomainChildrenParams{}
	p.p = make(map[string]interface{})

	p.p["id"] = id

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return nil, -1, err
		}
	}

	l, err := s.ListDomainChildren(p)
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
		return l.DomainChildren[0], l.Count, nil
	}
	return nil, l.Count, fmt.Errorf("There is more then one result for DomainChildren UUID: %s!", id)
}

// Lists all children domains belonging to a specified domain
func (s *DomainService) ListDomainChildren(p *ListDomainChildrenParams) (*ListDomainChildrenResponse, error) {
	resp, err := s.cs.newRequest("listDomainChildren", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListDomainChildrenResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListDomainChildrenResponse struct {
	Count          int               `json:"count"`
	DomainChildren []*DomainChildren `json:"domainchildren"`
}

type DomainChildren struct {
	Cpuavailable              string `json:"cpuavailable,omitempty"`
	Cpulimit                  string `json:"cpulimit,omitempty"`
	Cputotal                  int64  `json:"cputotal,omitempty"`
	Haschild                  bool   `json:"haschild,omitempty"`
	Id                        string `json:"id,omitempty"`
	Ipavailable               string `json:"ipavailable,omitempty"`
	Iplimit                   string `json:"iplimit,omitempty"`
	Iptotal                   int64  `json:"iptotal,omitempty"`
	Level                     int    `json:"level,omitempty"`
	Memoryavailable           string `json:"memoryavailable,omitempty"`
	Memorylimit               string `json:"memorylimit,omitempty"`
	Memorytotal               int64  `json:"memorytotal,omitempty"`
	Name                      string `json:"name,omitempty"`
	Networkavailable          string `json:"networkavailable,omitempty"`
	Networkdomain             string `json:"networkdomain,omitempty"`
	Networklimit              string `json:"networklimit,omitempty"`
	Networktotal              int64  `json:"networktotal,omitempty"`
	Parentdomainid            string `json:"parentdomainid,omitempty"`
	Parentdomainname          string `json:"parentdomainname,omitempty"`
	Path                      string `json:"path,omitempty"`
	Primarystorageavailable   string `json:"primarystorageavailable,omitempty"`
	Primarystoragelimit       string `json:"primarystoragelimit,omitempty"`
	Primarystoragetotal       int64  `json:"primarystoragetotal,omitempty"`
	Projectavailable          string `json:"projectavailable,omitempty"`
	Projectlimit              string `json:"projectlimit,omitempty"`
	Projecttotal              int64  `json:"projecttotal,omitempty"`
	Secondarystorageavailable string `json:"secondarystorageavailable,omitempty"`
	Secondarystoragelimit     string `json:"secondarystoragelimit,omitempty"`
	Secondarystoragetotal     int64  `json:"secondarystoragetotal,omitempty"`
	Snapshotavailable         string `json:"snapshotavailable,omitempty"`
	Snapshotlimit             string `json:"snapshotlimit,omitempty"`
	Snapshottotal             int64  `json:"snapshottotal,omitempty"`
	State                     string `json:"state,omitempty"`
	Templateavailable         string `json:"templateavailable,omitempty"`
	Templatelimit             string `json:"templatelimit,omitempty"`
	Templatetotal             int64  `json:"templatetotal,omitempty"`
	Vmavailable               string `json:"vmavailable,omitempty"`
	Vmlimit                   string `json:"vmlimit,omitempty"`
	Vmtotal                   int64  `json:"vmtotal,omitempty"`
	Volumeavailable           string `json:"volumeavailable,omitempty"`
	Volumelimit               string `json:"volumelimit,omitempty"`
	Volumetotal               int64  `json:"volumetotal,omitempty"`
	Vpcavailable              string `json:"vpcavailable,omitempty"`
	Vpclimit                  string `json:"vpclimit,omitempty"`
	Vpctotal                  int64  `json:"vpctotal,omitempty"`
}

type LinkDomainToLdapParams struct {
	p map[string]interface{}
}

func (p *LinkDomainToLdapParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["accounttype"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("accounttype", vv)
	}
	if v, found := p.p["admin"]; found {
		u.Set("admin", v.(string))
	}
	if v, found := p.p["domainid"]; found {
		u.Set("domainid", v.(string))
	}
	if v, found := p.p["name"]; found {
		u.Set("name", v.(string))
	}
	if v, found := p.p["type"]; found {
		u.Set("type", v.(string))
	}
	return u
}

func (p *LinkDomainToLdapParams) SetAccounttype(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["accounttype"] = v
	return
}

func (p *LinkDomainToLdapParams) SetAdmin(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["admin"] = v
	return
}

func (p *LinkDomainToLdapParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *LinkDomainToLdapParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *LinkDomainToLdapParams) SetType(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainType"] = v
	return
}

// You should always use this function to get a new LinkDomainToLdapParams instance,
// as then you are sure you have configured all required params
func (s *DomainService) NewLinkDomainToLdapParams(accounttype int, domainid string, name string, domainType string) *LinkDomainToLdapParams {
	p := &LinkDomainToLdapParams{}
	p.p = make(map[string]interface{})
	p.p["accounttype"] = accounttype
	p.p["domainid"] = domainid
	p.p["name"] = name
	p.p["domainType"] = domainType
	return p
}

// link an existing cloudstack domain to group or OU in ldap
func (s *DomainService) LinkDomainToLdap(p *LinkDomainToLdapParams) (*LinkDomainToLdapResponse, error) {
	resp, err := s.cs.newRequest("linkDomainToLdap", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r LinkDomainToLdapResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type LinkDomainToLdapResponse struct {
	Accountid   string `json:"accountid,omitempty"`
	Accounttype int    `json:"accounttype,omitempty"`
	Domainid    int64  `json:"domainid,omitempty"`
	Name        string `json:"name,omitempty"`
	Type        string `json:"type,omitempty"`
}
