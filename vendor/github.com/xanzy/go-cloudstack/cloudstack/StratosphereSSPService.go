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
)

type AddStratosphereSspParams struct {
	p map[string]interface{}
}

func (p *AddStratosphereSspParams) toURLValues() url.Values {
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
	if v, found := p.p["tenantuuid"]; found {
		u.Set("tenantuuid", v.(string))
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

func (p *AddStratosphereSspParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *AddStratosphereSspParams) SetPassword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["password"] = v
	return
}

func (p *AddStratosphereSspParams) SetTenantuuid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["tenantuuid"] = v
	return
}

func (p *AddStratosphereSspParams) SetUrl(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["url"] = v
	return
}

func (p *AddStratosphereSspParams) SetUsername(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["username"] = v
	return
}

func (p *AddStratosphereSspParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new AddStratosphereSspParams instance,
// as then you are sure you have configured all required params
func (s *StratosphereSSPService) NewAddStratosphereSspParams(name string, url string, zoneid string) *AddStratosphereSspParams {
	p := &AddStratosphereSspParams{}
	p.p = make(map[string]interface{})
	p.p["name"] = name
	p.p["url"] = url
	p.p["zoneid"] = zoneid
	return p
}

// Adds stratosphere ssp server
func (s *StratosphereSSPService) AddStratosphereSsp(p *AddStratosphereSspParams) (*AddStratosphereSspResponse, error) {
	resp, err := s.cs.newRequest("addStratosphereSsp", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r AddStratosphereSspResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type AddStratosphereSspResponse struct {
	Hostid string `json:"hostid,omitempty"`
	Name   string `json:"name,omitempty"`
	Url    string `json:"url,omitempty"`
	Zoneid string `json:"zoneid,omitempty"`
}
