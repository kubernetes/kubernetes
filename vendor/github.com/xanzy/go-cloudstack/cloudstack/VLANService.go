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

type CreateVlanIpRangeParams struct {
	p map[string]interface{}
}

func (p *CreateVlanIpRangeParams) toURLValues() url.Values {
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
	if v, found := p.p["endip"]; found {
		u.Set("endip", v.(string))
	}
	if v, found := p.p["endipv6"]; found {
		u.Set("endipv6", v.(string))
	}
	if v, found := p.p["forvirtualnetwork"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("forvirtualnetwork", vv)
	}
	if v, found := p.p["gateway"]; found {
		u.Set("gateway", v.(string))
	}
	if v, found := p.p["ip6cidr"]; found {
		u.Set("ip6cidr", v.(string))
	}
	if v, found := p.p["ip6gateway"]; found {
		u.Set("ip6gateway", v.(string))
	}
	if v, found := p.p["netmask"]; found {
		u.Set("netmask", v.(string))
	}
	if v, found := p.p["networkid"]; found {
		u.Set("networkid", v.(string))
	}
	if v, found := p.p["physicalnetworkid"]; found {
		u.Set("physicalnetworkid", v.(string))
	}
	if v, found := p.p["podid"]; found {
		u.Set("podid", v.(string))
	}
	if v, found := p.p["projectid"]; found {
		u.Set("projectid", v.(string))
	}
	if v, found := p.p["startip"]; found {
		u.Set("startip", v.(string))
	}
	if v, found := p.p["startipv6"]; found {
		u.Set("startipv6", v.(string))
	}
	if v, found := p.p["vlan"]; found {
		u.Set("vlan", v.(string))
	}
	if v, found := p.p["zoneid"]; found {
		u.Set("zoneid", v.(string))
	}
	return u
}

func (p *CreateVlanIpRangeParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *CreateVlanIpRangeParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *CreateVlanIpRangeParams) SetEndip(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["endip"] = v
	return
}

func (p *CreateVlanIpRangeParams) SetEndipv6(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["endipv6"] = v
	return
}

func (p *CreateVlanIpRangeParams) SetForvirtualnetwork(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["forvirtualnetwork"] = v
	return
}

func (p *CreateVlanIpRangeParams) SetGateway(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["gateway"] = v
	return
}

func (p *CreateVlanIpRangeParams) SetIp6cidr(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["ip6cidr"] = v
	return
}

func (p *CreateVlanIpRangeParams) SetIp6gateway(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["ip6gateway"] = v
	return
}

func (p *CreateVlanIpRangeParams) SetNetmask(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["netmask"] = v
	return
}

func (p *CreateVlanIpRangeParams) SetNetworkid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["networkid"] = v
	return
}

func (p *CreateVlanIpRangeParams) SetPhysicalnetworkid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["physicalnetworkid"] = v
	return
}

func (p *CreateVlanIpRangeParams) SetPodid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["podid"] = v
	return
}

func (p *CreateVlanIpRangeParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

func (p *CreateVlanIpRangeParams) SetStartip(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["startip"] = v
	return
}

func (p *CreateVlanIpRangeParams) SetStartipv6(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["startipv6"] = v
	return
}

func (p *CreateVlanIpRangeParams) SetVlan(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["vlan"] = v
	return
}

func (p *CreateVlanIpRangeParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new CreateVlanIpRangeParams instance,
// as then you are sure you have configured all required params
func (s *VLANService) NewCreateVlanIpRangeParams() *CreateVlanIpRangeParams {
	p := &CreateVlanIpRangeParams{}
	p.p = make(map[string]interface{})
	return p
}

// Creates a VLAN IP range.
func (s *VLANService) CreateVlanIpRange(p *CreateVlanIpRangeParams) (*CreateVlanIpRangeResponse, error) {
	resp, err := s.cs.newRequest("createVlanIpRange", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r CreateVlanIpRangeResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type CreateVlanIpRangeResponse struct {
	Account           string `json:"account,omitempty"`
	Description       string `json:"description,omitempty"`
	Domain            string `json:"domain,omitempty"`
	Domainid          string `json:"domainid,omitempty"`
	Endip             string `json:"endip,omitempty"`
	Endipv6           string `json:"endipv6,omitempty"`
	Forvirtualnetwork bool   `json:"forvirtualnetwork,omitempty"`
	Gateway           string `json:"gateway,omitempty"`
	Id                string `json:"id,omitempty"`
	Ip6cidr           string `json:"ip6cidr,omitempty"`
	Ip6gateway        string `json:"ip6gateway,omitempty"`
	Netmask           string `json:"netmask,omitempty"`
	Networkid         string `json:"networkid,omitempty"`
	Physicalnetworkid string `json:"physicalnetworkid,omitempty"`
	Podid             string `json:"podid,omitempty"`
	Podname           string `json:"podname,omitempty"`
	Project           string `json:"project,omitempty"`
	Projectid         string `json:"projectid,omitempty"`
	Startip           string `json:"startip,omitempty"`
	Startipv6         string `json:"startipv6,omitempty"`
	Vlan              string `json:"vlan,omitempty"`
	Zoneid            string `json:"zoneid,omitempty"`
}

type DeleteVlanIpRangeParams struct {
	p map[string]interface{}
}

func (p *DeleteVlanIpRangeParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	return u
}

func (p *DeleteVlanIpRangeParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new DeleteVlanIpRangeParams instance,
// as then you are sure you have configured all required params
func (s *VLANService) NewDeleteVlanIpRangeParams(id string) *DeleteVlanIpRangeParams {
	p := &DeleteVlanIpRangeParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Creates a VLAN IP range.
func (s *VLANService) DeleteVlanIpRange(p *DeleteVlanIpRangeParams) (*DeleteVlanIpRangeResponse, error) {
	resp, err := s.cs.newRequest("deleteVlanIpRange", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r DeleteVlanIpRangeResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type DeleteVlanIpRangeResponse struct {
	Displaytext string `json:"displaytext,omitempty"`
	Success     string `json:"success,omitempty"`
}

type ListVlanIpRangesParams struct {
	p map[string]interface{}
}

func (p *ListVlanIpRangesParams) toURLValues() url.Values {
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
	if v, found := p.p["forvirtualnetwork"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("forvirtualnetwork", vv)
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	if v, found := p.p["keyword"]; found {
		u.Set("keyword", v.(string))
	}
	if v, found := p.p["networkid"]; found {
		u.Set("networkid", v.(string))
	}
	if v, found := p.p["page"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("page", vv)
	}
	if v, found := p.p["pagesize"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("pagesize", vv)
	}
	if v, found := p.p["physicalnetworkid"]; found {
		u.Set("physicalnetworkid", v.(string))
	}
	if v, found := p.p["podid"]; found {
		u.Set("podid", v.(string))
	}
	if v, found := p.p["projectid"]; found {
		u.Set("projectid", v.(string))
	}
	if v, found := p.p["vlan"]; found {
		u.Set("vlan", v.(string))
	}
	if v, found := p.p["zoneid"]; found {
		u.Set("zoneid", v.(string))
	}
	return u
}

func (p *ListVlanIpRangesParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *ListVlanIpRangesParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *ListVlanIpRangesParams) SetForvirtualnetwork(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["forvirtualnetwork"] = v
	return
}

func (p *ListVlanIpRangesParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ListVlanIpRangesParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListVlanIpRangesParams) SetNetworkid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["networkid"] = v
	return
}

func (p *ListVlanIpRangesParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListVlanIpRangesParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

func (p *ListVlanIpRangesParams) SetPhysicalnetworkid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["physicalnetworkid"] = v
	return
}

func (p *ListVlanIpRangesParams) SetPodid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["podid"] = v
	return
}

func (p *ListVlanIpRangesParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

func (p *ListVlanIpRangesParams) SetVlan(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["vlan"] = v
	return
}

func (p *ListVlanIpRangesParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new ListVlanIpRangesParams instance,
// as then you are sure you have configured all required params
func (s *VLANService) NewListVlanIpRangesParams() *ListVlanIpRangesParams {
	p := &ListVlanIpRangesParams{}
	p.p = make(map[string]interface{})
	return p
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *VLANService) GetVlanIpRangeByID(id string, opts ...OptionFunc) (*VlanIpRange, int, error) {
	p := &ListVlanIpRangesParams{}
	p.p = make(map[string]interface{})

	p.p["id"] = id

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return nil, -1, err
		}
	}

	l, err := s.ListVlanIpRanges(p)
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
		return l.VlanIpRanges[0], l.Count, nil
	}
	return nil, l.Count, fmt.Errorf("There is more then one result for VlanIpRange UUID: %s!", id)
}

// Lists all VLAN IP ranges.
func (s *VLANService) ListVlanIpRanges(p *ListVlanIpRangesParams) (*ListVlanIpRangesResponse, error) {
	resp, err := s.cs.newRequest("listVlanIpRanges", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListVlanIpRangesResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListVlanIpRangesResponse struct {
	Count        int            `json:"count"`
	VlanIpRanges []*VlanIpRange `json:"vlaniprange"`
}

type VlanIpRange struct {
	Account           string `json:"account,omitempty"`
	Description       string `json:"description,omitempty"`
	Domain            string `json:"domain,omitempty"`
	Domainid          string `json:"domainid,omitempty"`
	Endip             string `json:"endip,omitempty"`
	Endipv6           string `json:"endipv6,omitempty"`
	Forvirtualnetwork bool   `json:"forvirtualnetwork,omitempty"`
	Gateway           string `json:"gateway,omitempty"`
	Id                string `json:"id,omitempty"`
	Ip6cidr           string `json:"ip6cidr,omitempty"`
	Ip6gateway        string `json:"ip6gateway,omitempty"`
	Netmask           string `json:"netmask,omitempty"`
	Networkid         string `json:"networkid,omitempty"`
	Physicalnetworkid string `json:"physicalnetworkid,omitempty"`
	Podid             string `json:"podid,omitempty"`
	Podname           string `json:"podname,omitempty"`
	Project           string `json:"project,omitempty"`
	Projectid         string `json:"projectid,omitempty"`
	Startip           string `json:"startip,omitempty"`
	Startipv6         string `json:"startipv6,omitempty"`
	Vlan              string `json:"vlan,omitempty"`
	Zoneid            string `json:"zoneid,omitempty"`
}

type DedicateGuestVlanRangeParams struct {
	p map[string]interface{}
}

func (p *DedicateGuestVlanRangeParams) toURLValues() url.Values {
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
	if v, found := p.p["physicalnetworkid"]; found {
		u.Set("physicalnetworkid", v.(string))
	}
	if v, found := p.p["projectid"]; found {
		u.Set("projectid", v.(string))
	}
	if v, found := p.p["vlanrange"]; found {
		u.Set("vlanrange", v.(string))
	}
	return u
}

func (p *DedicateGuestVlanRangeParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *DedicateGuestVlanRangeParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *DedicateGuestVlanRangeParams) SetPhysicalnetworkid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["physicalnetworkid"] = v
	return
}

func (p *DedicateGuestVlanRangeParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

func (p *DedicateGuestVlanRangeParams) SetVlanrange(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["vlanrange"] = v
	return
}

// You should always use this function to get a new DedicateGuestVlanRangeParams instance,
// as then you are sure you have configured all required params
func (s *VLANService) NewDedicateGuestVlanRangeParams(account string, domainid string, physicalnetworkid string, vlanrange string) *DedicateGuestVlanRangeParams {
	p := &DedicateGuestVlanRangeParams{}
	p.p = make(map[string]interface{})
	p.p["account"] = account
	p.p["domainid"] = domainid
	p.p["physicalnetworkid"] = physicalnetworkid
	p.p["vlanrange"] = vlanrange
	return p
}

// Dedicates a guest vlan range to an account
func (s *VLANService) DedicateGuestVlanRange(p *DedicateGuestVlanRangeParams) (*DedicateGuestVlanRangeResponse, error) {
	resp, err := s.cs.newRequest("dedicateGuestVlanRange", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r DedicateGuestVlanRangeResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type DedicateGuestVlanRangeResponse struct {
	Account           string `json:"account,omitempty"`
	Domain            string `json:"domain,omitempty"`
	Domainid          string `json:"domainid,omitempty"`
	Guestvlanrange    string `json:"guestvlanrange,omitempty"`
	Id                string `json:"id,omitempty"`
	Physicalnetworkid int64  `json:"physicalnetworkid,omitempty"`
	Project           string `json:"project,omitempty"`
	Projectid         string `json:"projectid,omitempty"`
	Zoneid            int64  `json:"zoneid,omitempty"`
}

type ReleaseDedicatedGuestVlanRangeParams struct {
	p map[string]interface{}
}

func (p *ReleaseDedicatedGuestVlanRangeParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	return u
}

func (p *ReleaseDedicatedGuestVlanRangeParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new ReleaseDedicatedGuestVlanRangeParams instance,
// as then you are sure you have configured all required params
func (s *VLANService) NewReleaseDedicatedGuestVlanRangeParams(id string) *ReleaseDedicatedGuestVlanRangeParams {
	p := &ReleaseDedicatedGuestVlanRangeParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Releases a dedicated guest vlan range to the system
func (s *VLANService) ReleaseDedicatedGuestVlanRange(p *ReleaseDedicatedGuestVlanRangeParams) (*ReleaseDedicatedGuestVlanRangeResponse, error) {
	resp, err := s.cs.newRequest("releaseDedicatedGuestVlanRange", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ReleaseDedicatedGuestVlanRangeResponse
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

type ReleaseDedicatedGuestVlanRangeResponse struct {
	JobID       string `json:"jobid,omitempty"`
	Displaytext string `json:"displaytext,omitempty"`
	Success     bool   `json:"success,omitempty"`
}

type ListDedicatedGuestVlanRangesParams struct {
	p map[string]interface{}
}

func (p *ListDedicatedGuestVlanRangesParams) toURLValues() url.Values {
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
	if v, found := p.p["guestvlanrange"]; found {
		u.Set("guestvlanrange", v.(string))
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
	if v, found := p.p["physicalnetworkid"]; found {
		u.Set("physicalnetworkid", v.(string))
	}
	if v, found := p.p["projectid"]; found {
		u.Set("projectid", v.(string))
	}
	if v, found := p.p["zoneid"]; found {
		u.Set("zoneid", v.(string))
	}
	return u
}

func (p *ListDedicatedGuestVlanRangesParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *ListDedicatedGuestVlanRangesParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *ListDedicatedGuestVlanRangesParams) SetGuestvlanrange(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["guestvlanrange"] = v
	return
}

func (p *ListDedicatedGuestVlanRangesParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ListDedicatedGuestVlanRangesParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListDedicatedGuestVlanRangesParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListDedicatedGuestVlanRangesParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

func (p *ListDedicatedGuestVlanRangesParams) SetPhysicalnetworkid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["physicalnetworkid"] = v
	return
}

func (p *ListDedicatedGuestVlanRangesParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

func (p *ListDedicatedGuestVlanRangesParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new ListDedicatedGuestVlanRangesParams instance,
// as then you are sure you have configured all required params
func (s *VLANService) NewListDedicatedGuestVlanRangesParams() *ListDedicatedGuestVlanRangesParams {
	p := &ListDedicatedGuestVlanRangesParams{}
	p.p = make(map[string]interface{})
	return p
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *VLANService) GetDedicatedGuestVlanRangeByID(id string, opts ...OptionFunc) (*DedicatedGuestVlanRange, int, error) {
	p := &ListDedicatedGuestVlanRangesParams{}
	p.p = make(map[string]interface{})

	p.p["id"] = id

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return nil, -1, err
		}
	}

	l, err := s.ListDedicatedGuestVlanRanges(p)
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
		return l.DedicatedGuestVlanRanges[0], l.Count, nil
	}
	return nil, l.Count, fmt.Errorf("There is more then one result for DedicatedGuestVlanRange UUID: %s!", id)
}

// Lists dedicated guest vlan ranges
func (s *VLANService) ListDedicatedGuestVlanRanges(p *ListDedicatedGuestVlanRangesParams) (*ListDedicatedGuestVlanRangesResponse, error) {
	resp, err := s.cs.newRequest("listDedicatedGuestVlanRanges", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListDedicatedGuestVlanRangesResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListDedicatedGuestVlanRangesResponse struct {
	Count                    int                        `json:"count"`
	DedicatedGuestVlanRanges []*DedicatedGuestVlanRange `json:"dedicatedguestvlanrange"`
}

type DedicatedGuestVlanRange struct {
	Account           string `json:"account,omitempty"`
	Domain            string `json:"domain,omitempty"`
	Domainid          string `json:"domainid,omitempty"`
	Guestvlanrange    string `json:"guestvlanrange,omitempty"`
	Id                string `json:"id,omitempty"`
	Physicalnetworkid int64  `json:"physicalnetworkid,omitempty"`
	Project           string `json:"project,omitempty"`
	Projectid         string `json:"projectid,omitempty"`
	Zoneid            int64  `json:"zoneid,omitempty"`
}
