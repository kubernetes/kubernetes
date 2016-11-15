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

type CreatePortableIpRangeParams struct {
	p map[string]interface{}
}

func (p *CreatePortableIpRangeParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["endip"]; found {
		u.Set("endip", v.(string))
	}
	if v, found := p.p["gateway"]; found {
		u.Set("gateway", v.(string))
	}
	if v, found := p.p["netmask"]; found {
		u.Set("netmask", v.(string))
	}
	if v, found := p.p["regionid"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("regionid", vv)
	}
	if v, found := p.p["startip"]; found {
		u.Set("startip", v.(string))
	}
	if v, found := p.p["vlan"]; found {
		u.Set("vlan", v.(string))
	}
	return u
}

func (p *CreatePortableIpRangeParams) SetEndip(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["endip"] = v
	return
}

func (p *CreatePortableIpRangeParams) SetGateway(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["gateway"] = v
	return
}

func (p *CreatePortableIpRangeParams) SetNetmask(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["netmask"] = v
	return
}

func (p *CreatePortableIpRangeParams) SetRegionid(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["regionid"] = v
	return
}

func (p *CreatePortableIpRangeParams) SetStartip(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["startip"] = v
	return
}

func (p *CreatePortableIpRangeParams) SetVlan(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["vlan"] = v
	return
}

// You should always use this function to get a new CreatePortableIpRangeParams instance,
// as then you are sure you have configured all required params
func (s *PortableIPService) NewCreatePortableIpRangeParams(endip string, gateway string, netmask string, regionid int, startip string) *CreatePortableIpRangeParams {
	p := &CreatePortableIpRangeParams{}
	p.p = make(map[string]interface{})
	p.p["endip"] = endip
	p.p["gateway"] = gateway
	p.p["netmask"] = netmask
	p.p["regionid"] = regionid
	p.p["startip"] = startip
	return p
}

// adds a range of portable public IP's to a region
func (s *PortableIPService) CreatePortableIpRange(p *CreatePortableIpRangeParams) (*CreatePortableIpRangeResponse, error) {
	resp, err := s.cs.newRequest("createPortableIpRange", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r CreatePortableIpRangeResponse
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

type CreatePortableIpRangeResponse struct {
	JobID             string `json:"jobid,omitempty"`
	Endip             string `json:"endip,omitempty"`
	Gateway           string `json:"gateway,omitempty"`
	Id                string `json:"id,omitempty"`
	Netmask           string `json:"netmask,omitempty"`
	Portableipaddress []struct {
		Accountid         string `json:"accountid,omitempty"`
		Allocated         string `json:"allocated,omitempty"`
		Domainid          string `json:"domainid,omitempty"`
		Ipaddress         string `json:"ipaddress,omitempty"`
		Networkid         string `json:"networkid,omitempty"`
		Physicalnetworkid string `json:"physicalnetworkid,omitempty"`
		Regionid          int    `json:"regionid,omitempty"`
		State             string `json:"state,omitempty"`
		Vpcid             string `json:"vpcid,omitempty"`
		Zoneid            string `json:"zoneid,omitempty"`
	} `json:"portableipaddress,omitempty"`
	Regionid int    `json:"regionid,omitempty"`
	Startip  string `json:"startip,omitempty"`
	Vlan     string `json:"vlan,omitempty"`
}

type DeletePortableIpRangeParams struct {
	p map[string]interface{}
}

func (p *DeletePortableIpRangeParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	return u
}

func (p *DeletePortableIpRangeParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new DeletePortableIpRangeParams instance,
// as then you are sure you have configured all required params
func (s *PortableIPService) NewDeletePortableIpRangeParams(id string) *DeletePortableIpRangeParams {
	p := &DeletePortableIpRangeParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// deletes a range of portable public IP's associated with a region
func (s *PortableIPService) DeletePortableIpRange(p *DeletePortableIpRangeParams) (*DeletePortableIpRangeResponse, error) {
	resp, err := s.cs.newRequest("deletePortableIpRange", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r DeletePortableIpRangeResponse
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

type DeletePortableIpRangeResponse struct {
	JobID       string `json:"jobid,omitempty"`
	Displaytext string `json:"displaytext,omitempty"`
	Success     bool   `json:"success,omitempty"`
}

type ListPortableIpRangesParams struct {
	p map[string]interface{}
}

func (p *ListPortableIpRangesParams) toURLValues() url.Values {
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
	if v, found := p.p["regionid"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("regionid", vv)
	}
	return u
}

func (p *ListPortableIpRangesParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ListPortableIpRangesParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListPortableIpRangesParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListPortableIpRangesParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

func (p *ListPortableIpRangesParams) SetRegionid(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["regionid"] = v
	return
}

// You should always use this function to get a new ListPortableIpRangesParams instance,
// as then you are sure you have configured all required params
func (s *PortableIPService) NewListPortableIpRangesParams() *ListPortableIpRangesParams {
	p := &ListPortableIpRangesParams{}
	p.p = make(map[string]interface{})
	return p
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *PortableIPService) GetPortableIpRangeByID(id string, opts ...OptionFunc) (*PortableIpRange, int, error) {
	p := &ListPortableIpRangesParams{}
	p.p = make(map[string]interface{})

	p.p["id"] = id

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return nil, -1, err
		}
	}

	l, err := s.ListPortableIpRanges(p)
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
		return l.PortableIpRanges[0], l.Count, nil
	}
	return nil, l.Count, fmt.Errorf("There is more then one result for PortableIpRange UUID: %s!", id)
}

// list portable IP ranges
func (s *PortableIPService) ListPortableIpRanges(p *ListPortableIpRangesParams) (*ListPortableIpRangesResponse, error) {
	resp, err := s.cs.newRequest("listPortableIpRanges", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListPortableIpRangesResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListPortableIpRangesResponse struct {
	Count            int                `json:"count"`
	PortableIpRanges []*PortableIpRange `json:"portableiprange"`
}

type PortableIpRange struct {
	Endip             string `json:"endip,omitempty"`
	Gateway           string `json:"gateway,omitempty"`
	Id                string `json:"id,omitempty"`
	Netmask           string `json:"netmask,omitempty"`
	Portableipaddress []struct {
		Accountid         string `json:"accountid,omitempty"`
		Allocated         string `json:"allocated,omitempty"`
		Domainid          string `json:"domainid,omitempty"`
		Ipaddress         string `json:"ipaddress,omitempty"`
		Networkid         string `json:"networkid,omitempty"`
		Physicalnetworkid string `json:"physicalnetworkid,omitempty"`
		Regionid          int    `json:"regionid,omitempty"`
		State             string `json:"state,omitempty"`
		Vpcid             string `json:"vpcid,omitempty"`
		Zoneid            string `json:"zoneid,omitempty"`
	} `json:"portableipaddress,omitempty"`
	Regionid int    `json:"regionid,omitempty"`
	Startip  string `json:"startip,omitempty"`
	Vlan     string `json:"vlan,omitempty"`
}
