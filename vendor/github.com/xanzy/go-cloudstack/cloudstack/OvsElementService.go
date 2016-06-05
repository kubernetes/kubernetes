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

type ConfigureOvsElementParams struct {
	p map[string]interface{}
}

func (p *ConfigureOvsElementParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["enabled"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("enabled", vv)
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	return u
}

func (p *ConfigureOvsElementParams) SetEnabled(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["enabled"] = v
	return
}

func (p *ConfigureOvsElementParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new ConfigureOvsElementParams instance,
// as then you are sure you have configured all required params
func (s *OvsElementService) NewConfigureOvsElementParams(enabled bool, id string) *ConfigureOvsElementParams {
	p := &ConfigureOvsElementParams{}
	p.p = make(map[string]interface{})
	p.p["enabled"] = enabled
	p.p["id"] = id
	return p
}

// Configures an ovs element.
func (s *OvsElementService) ConfigureOvsElement(p *ConfigureOvsElementParams) (*ConfigureOvsElementResponse, error) {
	resp, err := s.cs.newRequest("configureOvsElement", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ConfigureOvsElementResponse
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

type ConfigureOvsElementResponse struct {
	JobID     string `json:"jobid,omitempty"`
	Account   string `json:"account,omitempty"`
	Domain    string `json:"domain,omitempty"`
	Domainid  string `json:"domainid,omitempty"`
	Enabled   bool   `json:"enabled,omitempty"`
	Id        string `json:"id,omitempty"`
	Nspid     string `json:"nspid,omitempty"`
	Project   string `json:"project,omitempty"`
	Projectid string `json:"projectid,omitempty"`
}

type ListOvsElementsParams struct {
	p map[string]interface{}
}

func (p *ListOvsElementsParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["enabled"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("enabled", vv)
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	if v, found := p.p["keyword"]; found {
		u.Set("keyword", v.(string))
	}
	if v, found := p.p["nspid"]; found {
		u.Set("nspid", v.(string))
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

func (p *ListOvsElementsParams) SetEnabled(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["enabled"] = v
	return
}

func (p *ListOvsElementsParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ListOvsElementsParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListOvsElementsParams) SetNspid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["nspid"] = v
	return
}

func (p *ListOvsElementsParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListOvsElementsParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

// You should always use this function to get a new ListOvsElementsParams instance,
// as then you are sure you have configured all required params
func (s *OvsElementService) NewListOvsElementsParams() *ListOvsElementsParams {
	p := &ListOvsElementsParams{}
	p.p = make(map[string]interface{})
	return p
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *OvsElementService) GetOvsElementByID(id string, opts ...OptionFunc) (*OvsElement, int, error) {
	p := &ListOvsElementsParams{}
	p.p = make(map[string]interface{})

	p.p["id"] = id

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return nil, -1, err
		}
	}

	l, err := s.ListOvsElements(p)
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
		return l.OvsElements[0], l.Count, nil
	}
	return nil, l.Count, fmt.Errorf("There is more then one result for OvsElement UUID: %s!", id)
}

// Lists all available ovs elements.
func (s *OvsElementService) ListOvsElements(p *ListOvsElementsParams) (*ListOvsElementsResponse, error) {
	resp, err := s.cs.newRequest("listOvsElements", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListOvsElementsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListOvsElementsResponse struct {
	Count       int           `json:"count"`
	OvsElements []*OvsElement `json:"ovselement"`
}

type OvsElement struct {
	Account   string `json:"account,omitempty"`
	Domain    string `json:"domain,omitempty"`
	Domainid  string `json:"domainid,omitempty"`
	Enabled   bool   `json:"enabled,omitempty"`
	Id        string `json:"id,omitempty"`
	Nspid     string `json:"nspid,omitempty"`
	Project   string `json:"project,omitempty"`
	Projectid string `json:"projectid,omitempty"`
}
