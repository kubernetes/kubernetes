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
)

type AddNetworkDeviceParams struct {
	p map[string]interface{}
}

func (p *AddNetworkDeviceParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["networkdeviceparameterlist"]; found {
		i := 0
		for k, vv := range v.(map[string]string) {
			u.Set(fmt.Sprintf("networkdeviceparameterlist[%d].key", i), k)
			u.Set(fmt.Sprintf("networkdeviceparameterlist[%d].value", i), vv)
			i++
		}
	}
	if v, found := p.p["networkdevicetype"]; found {
		u.Set("networkdevicetype", v.(string))
	}
	return u
}

func (p *AddNetworkDeviceParams) SetNetworkdeviceparameterlist(v map[string]string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["networkdeviceparameterlist"] = v
	return
}

func (p *AddNetworkDeviceParams) SetNetworkdevicetype(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["networkdevicetype"] = v
	return
}

// You should always use this function to get a new AddNetworkDeviceParams instance,
// as then you are sure you have configured all required params
func (s *NetworkDeviceService) NewAddNetworkDeviceParams() *AddNetworkDeviceParams {
	p := &AddNetworkDeviceParams{}
	p.p = make(map[string]interface{})
	return p
}

// Adds a network device of one of the following types: ExternalDhcp, ExternalFirewall, ExternalLoadBalancer, PxeServer
func (s *NetworkDeviceService) AddNetworkDevice(p *AddNetworkDeviceParams) (*AddNetworkDeviceResponse, error) {
	resp, err := s.cs.newRequest("addNetworkDevice", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r AddNetworkDeviceResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type AddNetworkDeviceResponse struct {
	Id string `json:"id,omitempty"`
}

type ListNetworkDeviceParams struct {
	p map[string]interface{}
}

func (p *ListNetworkDeviceParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["keyword"]; found {
		u.Set("keyword", v.(string))
	}
	if v, found := p.p["networkdeviceparameterlist"]; found {
		i := 0
		for k, vv := range v.(map[string]string) {
			u.Set(fmt.Sprintf("networkdeviceparameterlist[%d].key", i), k)
			u.Set(fmt.Sprintf("networkdeviceparameterlist[%d].value", i), vv)
			i++
		}
	}
	if v, found := p.p["networkdevicetype"]; found {
		u.Set("networkdevicetype", v.(string))
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

func (p *ListNetworkDeviceParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListNetworkDeviceParams) SetNetworkdeviceparameterlist(v map[string]string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["networkdeviceparameterlist"] = v
	return
}

func (p *ListNetworkDeviceParams) SetNetworkdevicetype(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["networkdevicetype"] = v
	return
}

func (p *ListNetworkDeviceParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListNetworkDeviceParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

// You should always use this function to get a new ListNetworkDeviceParams instance,
// as then you are sure you have configured all required params
func (s *NetworkDeviceService) NewListNetworkDeviceParams() *ListNetworkDeviceParams {
	p := &ListNetworkDeviceParams{}
	p.p = make(map[string]interface{})
	return p
}

// List network devices
func (s *NetworkDeviceService) ListNetworkDevice(p *ListNetworkDeviceParams) (*ListNetworkDeviceResponse, error) {
	resp, err := s.cs.newRequest("listNetworkDevice", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListNetworkDeviceResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListNetworkDeviceResponse struct {
	Count         int              `json:"count"`
	NetworkDevice []*NetworkDevice `json:"networkdevice"`
}

type NetworkDevice struct {
	Id string `json:"id,omitempty"`
}

type DeleteNetworkDeviceParams struct {
	p map[string]interface{}
}

func (p *DeleteNetworkDeviceParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	return u
}

func (p *DeleteNetworkDeviceParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new DeleteNetworkDeviceParams instance,
// as then you are sure you have configured all required params
func (s *NetworkDeviceService) NewDeleteNetworkDeviceParams(id string) *DeleteNetworkDeviceParams {
	p := &DeleteNetworkDeviceParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Deletes network device.
func (s *NetworkDeviceService) DeleteNetworkDevice(p *DeleteNetworkDeviceParams) (*DeleteNetworkDeviceResponse, error) {
	resp, err := s.cs.newRequest("deleteNetworkDevice", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r DeleteNetworkDeviceResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type DeleteNetworkDeviceResponse struct {
	Displaytext string `json:"displaytext,omitempty"`
	Success     string `json:"success,omitempty"`
}
