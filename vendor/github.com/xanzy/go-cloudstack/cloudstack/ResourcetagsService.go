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

type ListStorageTagsParams struct {
	p map[string]interface{}
}

func (p *ListStorageTagsParams) toURLValues() url.Values {
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
	return u
}

func (p *ListStorageTagsParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListStorageTagsParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListStorageTagsParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

// You should always use this function to get a new ListStorageTagsParams instance,
// as then you are sure you have configured all required params
func (s *ResourcetagsService) NewListStorageTagsParams() *ListStorageTagsParams {
	p := &ListStorageTagsParams{}
	p.p = make(map[string]interface{})
	return p
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *ResourcetagsService) GetStorageTagID(keyword string, opts ...OptionFunc) (string, int, error) {
	p := &ListStorageTagsParams{}
	p.p = make(map[string]interface{})

	p.p["keyword"] = keyword

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return "", -1, err
		}
	}

	l, err := s.ListStorageTags(p)
	if err != nil {
		return "", -1, err
	}

	if l.Count == 0 {
		return "", l.Count, fmt.Errorf("No match found for %s: %+v", keyword, l)
	}

	if l.Count == 1 {
		return l.StorageTags[0].Id, l.Count, nil
	}

	if l.Count > 1 {
		for _, v := range l.StorageTags {
			if v.Name == keyword {
				return v.Id, l.Count, nil
			}
		}
	}
	return "", l.Count, fmt.Errorf("Could not find an exact match for %s: %+v", keyword, l)
}

// Lists storage tags
func (s *ResourcetagsService) ListStorageTags(p *ListStorageTagsParams) (*ListStorageTagsResponse, error) {
	resp, err := s.cs.newRequest("listStorageTags", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListStorageTagsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListStorageTagsResponse struct {
	Count       int           `json:"count"`
	StorageTags []*StorageTag `json:"storagetag"`
}

type StorageTag struct {
	Id     string `json:"id,omitempty"`
	Name   string `json:"name,omitempty"`
	Poolid int64  `json:"poolid,omitempty"`
}

type CreateTagsParams struct {
	p map[string]interface{}
}

func (p *CreateTagsParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["customer"]; found {
		u.Set("customer", v.(string))
	}
	if v, found := p.p["resourceids"]; found {
		vv := strings.Join(v.([]string), ",")
		u.Set("resourceids", vv)
	}
	if v, found := p.p["resourcetype"]; found {
		u.Set("resourcetype", v.(string))
	}
	if v, found := p.p["tags"]; found {
		i := 0
		for k, vv := range v.(map[string]string) {
			u.Set(fmt.Sprintf("tags[%d].key", i), k)
			u.Set(fmt.Sprintf("tags[%d].value", i), vv)
			i++
		}
	}
	return u
}

func (p *CreateTagsParams) SetCustomer(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["customer"] = v
	return
}

func (p *CreateTagsParams) SetResourceids(v []string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["resourceids"] = v
	return
}

func (p *CreateTagsParams) SetResourcetype(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["resourcetype"] = v
	return
}

func (p *CreateTagsParams) SetTags(v map[string]string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["tags"] = v
	return
}

// You should always use this function to get a new CreateTagsParams instance,
// as then you are sure you have configured all required params
func (s *ResourcetagsService) NewCreateTagsParams(resourceids []string, resourcetype string, tags map[string]string) *CreateTagsParams {
	p := &CreateTagsParams{}
	p.p = make(map[string]interface{})
	p.p["resourceids"] = resourceids
	p.p["resourcetype"] = resourcetype
	p.p["tags"] = tags
	return p
}

// Creates resource tag(s)
func (s *ResourcetagsService) CreateTags(p *CreateTagsParams) (*CreateTagsResponse, error) {
	resp, err := s.cs.newRequest("createTags", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r CreateTagsResponse
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

type CreateTagsResponse struct {
	JobID       string `json:"jobid,omitempty"`
	Displaytext string `json:"displaytext,omitempty"`
	Success     bool   `json:"success,omitempty"`
}

type DeleteTagsParams struct {
	p map[string]interface{}
}

func (p *DeleteTagsParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["resourceids"]; found {
		vv := strings.Join(v.([]string), ",")
		u.Set("resourceids", vv)
	}
	if v, found := p.p["resourcetype"]; found {
		u.Set("resourcetype", v.(string))
	}
	if v, found := p.p["tags"]; found {
		i := 0
		for k, vv := range v.(map[string]string) {
			u.Set(fmt.Sprintf("tags[%d].key", i), k)
			u.Set(fmt.Sprintf("tags[%d].value", i), vv)
			i++
		}
	}
	return u
}

func (p *DeleteTagsParams) SetResourceids(v []string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["resourceids"] = v
	return
}

func (p *DeleteTagsParams) SetResourcetype(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["resourcetype"] = v
	return
}

func (p *DeleteTagsParams) SetTags(v map[string]string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["tags"] = v
	return
}

// You should always use this function to get a new DeleteTagsParams instance,
// as then you are sure you have configured all required params
func (s *ResourcetagsService) NewDeleteTagsParams(resourceids []string, resourcetype string) *DeleteTagsParams {
	p := &DeleteTagsParams{}
	p.p = make(map[string]interface{})
	p.p["resourceids"] = resourceids
	p.p["resourcetype"] = resourcetype
	return p
}

// Deleting resource tag(s)
func (s *ResourcetagsService) DeleteTags(p *DeleteTagsParams) (*DeleteTagsResponse, error) {
	resp, err := s.cs.newRequest("deleteTags", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r DeleteTagsResponse
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

type DeleteTagsResponse struct {
	JobID       string `json:"jobid,omitempty"`
	Displaytext string `json:"displaytext,omitempty"`
	Success     bool   `json:"success,omitempty"`
}

type ListTagsParams struct {
	p map[string]interface{}
}

func (p *ListTagsParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["account"]; found {
		u.Set("account", v.(string))
	}
	if v, found := p.p["customer"]; found {
		u.Set("customer", v.(string))
	}
	if v, found := p.p["domainid"]; found {
		u.Set("domainid", v.(string))
	}
	if v, found := p.p["isrecursive"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("isrecursive", vv)
	}
	if v, found := p.p["key"]; found {
		u.Set("key", v.(string))
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
	if v, found := p.p["resourceid"]; found {
		u.Set("resourceid", v.(string))
	}
	if v, found := p.p["resourcetype"]; found {
		u.Set("resourcetype", v.(string))
	}
	if v, found := p.p["value"]; found {
		u.Set("value", v.(string))
	}
	return u
}

func (p *ListTagsParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *ListTagsParams) SetCustomer(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["customer"] = v
	return
}

func (p *ListTagsParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *ListTagsParams) SetIsrecursive(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isrecursive"] = v
	return
}

func (p *ListTagsParams) SetKey(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["key"] = v
	return
}

func (p *ListTagsParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListTagsParams) SetListall(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["listall"] = v
	return
}

func (p *ListTagsParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListTagsParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

func (p *ListTagsParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

func (p *ListTagsParams) SetResourceid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["resourceid"] = v
	return
}

func (p *ListTagsParams) SetResourcetype(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["resourcetype"] = v
	return
}

func (p *ListTagsParams) SetValue(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["value"] = v
	return
}

// You should always use this function to get a new ListTagsParams instance,
// as then you are sure you have configured all required params
func (s *ResourcetagsService) NewListTagsParams() *ListTagsParams {
	p := &ListTagsParams{}
	p.p = make(map[string]interface{})
	return p
}

// List resource tag(s)
func (s *ResourcetagsService) ListTags(p *ListTagsParams) (*ListTagsResponse, error) {
	resp, err := s.cs.newRequest("listTags", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListTagsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListTagsResponse struct {
	Count int    `json:"count"`
	Tags  []*Tag `json:"tag"`
}

type Tag struct {
	Account      string `json:"account,omitempty"`
	Customer     string `json:"customer,omitempty"`
	Domain       string `json:"domain,omitempty"`
	Domainid     string `json:"domainid,omitempty"`
	Key          string `json:"key,omitempty"`
	Project      string `json:"project,omitempty"`
	Projectid    string `json:"projectid,omitempty"`
	Resourceid   string `json:"resourceid,omitempty"`
	Resourcetype string `json:"resourcetype,omitempty"`
	Value        string `json:"value,omitempty"`
}
