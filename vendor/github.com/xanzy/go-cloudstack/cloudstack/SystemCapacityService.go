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

type ListCapacityParams struct {
	p map[string]interface{}
}

func (p *ListCapacityParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["clusterid"]; found {
		u.Set("clusterid", v.(string))
	}
	if v, found := p.p["fetchlatest"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("fetchlatest", vv)
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
	if v, found := p.p["podid"]; found {
		u.Set("podid", v.(string))
	}
	if v, found := p.p["sortby"]; found {
		u.Set("sortby", v.(string))
	}
	if v, found := p.p["type"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("type", vv)
	}
	if v, found := p.p["zoneid"]; found {
		u.Set("zoneid", v.(string))
	}
	return u
}

func (p *ListCapacityParams) SetClusterid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["clusterid"] = v
	return
}

func (p *ListCapacityParams) SetFetchlatest(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["fetchlatest"] = v
	return
}

func (p *ListCapacityParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListCapacityParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListCapacityParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

func (p *ListCapacityParams) SetPodid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["podid"] = v
	return
}

func (p *ListCapacityParams) SetSortby(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["sortby"] = v
	return
}

func (p *ListCapacityParams) SetType(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["systemCapacityType"] = v
	return
}

func (p *ListCapacityParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new ListCapacityParams instance,
// as then you are sure you have configured all required params
func (s *SystemCapacityService) NewListCapacityParams() *ListCapacityParams {
	p := &ListCapacityParams{}
	p.p = make(map[string]interface{})
	return p
}

// Lists all the system wide capacities.
func (s *SystemCapacityService) ListCapacity(p *ListCapacityParams) (*ListCapacityResponse, error) {
	resp, err := s.cs.newRequest("listCapacity", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListCapacityResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListCapacityResponse struct {
	Count    int         `json:"count"`
	Capacity []*Capacity `json:"capacity"`
}

type Capacity struct {
	Capacitytotal int64  `json:"capacitytotal,omitempty"`
	Capacityused  int64  `json:"capacityused,omitempty"`
	Clusterid     string `json:"clusterid,omitempty"`
	Clustername   string `json:"clustername,omitempty"`
	Percentused   string `json:"percentused,omitempty"`
	Podid         string `json:"podid,omitempty"`
	Podname       string `json:"podname,omitempty"`
	Type          int    `json:"type,omitempty"`
	Zoneid        string `json:"zoneid,omitempty"`
	Zonename      string `json:"zonename,omitempty"`
}
