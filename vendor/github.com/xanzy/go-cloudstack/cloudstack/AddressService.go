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

type AssociateIpAddressParams struct {
	p map[string]interface{}
}

func (p *AssociateIpAddressParams) toURLValues() url.Values {
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
	if v, found := p.p["fordisplay"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("fordisplay", vv)
	}
	if v, found := p.p["isportable"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("isportable", vv)
	}
	if v, found := p.p["networkid"]; found {
		u.Set("networkid", v.(string))
	}
	if v, found := p.p["projectid"]; found {
		u.Set("projectid", v.(string))
	}
	if v, found := p.p["regionid"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("regionid", vv)
	}
	if v, found := p.p["vpcid"]; found {
		u.Set("vpcid", v.(string))
	}
	if v, found := p.p["zoneid"]; found {
		u.Set("zoneid", v.(string))
	}
	return u
}

func (p *AssociateIpAddressParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *AssociateIpAddressParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *AssociateIpAddressParams) SetFordisplay(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["fordisplay"] = v
	return
}

func (p *AssociateIpAddressParams) SetIsportable(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isportable"] = v
	return
}

func (p *AssociateIpAddressParams) SetNetworkid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["networkid"] = v
	return
}

func (p *AssociateIpAddressParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

func (p *AssociateIpAddressParams) SetRegionid(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["regionid"] = v
	return
}

func (p *AssociateIpAddressParams) SetVpcid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["vpcid"] = v
	return
}

func (p *AssociateIpAddressParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new AssociateIpAddressParams instance,
// as then you are sure you have configured all required params
func (s *AddressService) NewAssociateIpAddressParams() *AssociateIpAddressParams {
	p := &AssociateIpAddressParams{}
	p.p = make(map[string]interface{})
	return p
}

// Acquires and associates a public IP to an account.
func (s *AddressService) AssociateIpAddress(p *AssociateIpAddressParams) (*AssociateIpAddressResponse, error) {
	resp, err := s.cs.newRequest("associateIpAddress", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r AssociateIpAddressResponse
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

type AssociateIpAddressResponse struct {
	JobID                 string `json:"jobid,omitempty"`
	Account               string `json:"account,omitempty"`
	Allocated             string `json:"allocated,omitempty"`
	Associatednetworkid   string `json:"associatednetworkid,omitempty"`
	Associatednetworkname string `json:"associatednetworkname,omitempty"`
	Domain                string `json:"domain,omitempty"`
	Domainid              string `json:"domainid,omitempty"`
	Fordisplay            bool   `json:"fordisplay,omitempty"`
	Forvirtualnetwork     bool   `json:"forvirtualnetwork,omitempty"`
	Id                    string `json:"id,omitempty"`
	Ipaddress             string `json:"ipaddress,omitempty"`
	Isportable            bool   `json:"isportable,omitempty"`
	Issourcenat           bool   `json:"issourcenat,omitempty"`
	Isstaticnat           bool   `json:"isstaticnat,omitempty"`
	Issystem              bool   `json:"issystem,omitempty"`
	Networkid             string `json:"networkid,omitempty"`
	Physicalnetworkid     string `json:"physicalnetworkid,omitempty"`
	Project               string `json:"project,omitempty"`
	Projectid             string `json:"projectid,omitempty"`
	Purpose               string `json:"purpose,omitempty"`
	State                 string `json:"state,omitempty"`
	Tags                  []struct {
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
	} `json:"tags,omitempty"`
	Virtualmachinedisplayname string `json:"virtualmachinedisplayname,omitempty"`
	Virtualmachineid          string `json:"virtualmachineid,omitempty"`
	Virtualmachinename        string `json:"virtualmachinename,omitempty"`
	Vlanid                    string `json:"vlanid,omitempty"`
	Vlanname                  string `json:"vlanname,omitempty"`
	Vmipaddress               string `json:"vmipaddress,omitempty"`
	Vpcid                     string `json:"vpcid,omitempty"`
	Zoneid                    string `json:"zoneid,omitempty"`
	Zonename                  string `json:"zonename,omitempty"`
}

type DisassociateIpAddressParams struct {
	p map[string]interface{}
}

func (p *DisassociateIpAddressParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	return u
}

func (p *DisassociateIpAddressParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new DisassociateIpAddressParams instance,
// as then you are sure you have configured all required params
func (s *AddressService) NewDisassociateIpAddressParams(id string) *DisassociateIpAddressParams {
	p := &DisassociateIpAddressParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Disassociates an IP address from the account.
func (s *AddressService) DisassociateIpAddress(p *DisassociateIpAddressParams) (*DisassociateIpAddressResponse, error) {
	resp, err := s.cs.newRequest("disassociateIpAddress", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r DisassociateIpAddressResponse
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

type DisassociateIpAddressResponse struct {
	JobID       string `json:"jobid,omitempty"`
	Displaytext string `json:"displaytext,omitempty"`
	Success     bool   `json:"success,omitempty"`
}

type ListPublicIpAddressesParams struct {
	p map[string]interface{}
}

func (p *ListPublicIpAddressesParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["account"]; found {
		u.Set("account", v.(string))
	}
	if v, found := p.p["allocatedonly"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("allocatedonly", vv)
	}
	if v, found := p.p["associatednetworkid"]; found {
		u.Set("associatednetworkid", v.(string))
	}
	if v, found := p.p["domainid"]; found {
		u.Set("domainid", v.(string))
	}
	if v, found := p.p["fordisplay"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("fordisplay", vv)
	}
	if v, found := p.p["forloadbalancing"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("forloadbalancing", vv)
	}
	if v, found := p.p["forvirtualnetwork"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("forvirtualnetwork", vv)
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	if v, found := p.p["ipaddress"]; found {
		u.Set("ipaddress", v.(string))
	}
	if v, found := p.p["isrecursive"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("isrecursive", vv)
	}
	if v, found := p.p["issourcenat"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("issourcenat", vv)
	}
	if v, found := p.p["isstaticnat"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("isstaticnat", vv)
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
	if v, found := p.p["physicalnetworkid"]; found {
		u.Set("physicalnetworkid", v.(string))
	}
	if v, found := p.p["projectid"]; found {
		u.Set("projectid", v.(string))
	}
	if v, found := p.p["state"]; found {
		u.Set("state", v.(string))
	}
	if v, found := p.p["tags"]; found {
		i := 0
		for k, vv := range v.(map[string]string) {
			u.Set(fmt.Sprintf("tags[%d].key", i), k)
			u.Set(fmt.Sprintf("tags[%d].value", i), vv)
			i++
		}
	}
	if v, found := p.p["vlanid"]; found {
		u.Set("vlanid", v.(string))
	}
	if v, found := p.p["vpcid"]; found {
		u.Set("vpcid", v.(string))
	}
	if v, found := p.p["zoneid"]; found {
		u.Set("zoneid", v.(string))
	}
	return u
}

func (p *ListPublicIpAddressesParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *ListPublicIpAddressesParams) SetAllocatedonly(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["allocatedonly"] = v
	return
}

func (p *ListPublicIpAddressesParams) SetAssociatednetworkid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["associatednetworkid"] = v
	return
}

func (p *ListPublicIpAddressesParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *ListPublicIpAddressesParams) SetFordisplay(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["fordisplay"] = v
	return
}

func (p *ListPublicIpAddressesParams) SetForloadbalancing(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["forloadbalancing"] = v
	return
}

func (p *ListPublicIpAddressesParams) SetForvirtualnetwork(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["forvirtualnetwork"] = v
	return
}

func (p *ListPublicIpAddressesParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ListPublicIpAddressesParams) SetIpaddress(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["ipaddress"] = v
	return
}

func (p *ListPublicIpAddressesParams) SetIsrecursive(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isrecursive"] = v
	return
}

func (p *ListPublicIpAddressesParams) SetIssourcenat(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["issourcenat"] = v
	return
}

func (p *ListPublicIpAddressesParams) SetIsstaticnat(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isstaticnat"] = v
	return
}

func (p *ListPublicIpAddressesParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListPublicIpAddressesParams) SetListall(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["listall"] = v
	return
}

func (p *ListPublicIpAddressesParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListPublicIpAddressesParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

func (p *ListPublicIpAddressesParams) SetPhysicalnetworkid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["physicalnetworkid"] = v
	return
}

func (p *ListPublicIpAddressesParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

func (p *ListPublicIpAddressesParams) SetState(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["state"] = v
	return
}

func (p *ListPublicIpAddressesParams) SetTags(v map[string]string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["tags"] = v
	return
}

func (p *ListPublicIpAddressesParams) SetVlanid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["vlanid"] = v
	return
}

func (p *ListPublicIpAddressesParams) SetVpcid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["vpcid"] = v
	return
}

func (p *ListPublicIpAddressesParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new ListPublicIpAddressesParams instance,
// as then you are sure you have configured all required params
func (s *AddressService) NewListPublicIpAddressesParams() *ListPublicIpAddressesParams {
	p := &ListPublicIpAddressesParams{}
	p.p = make(map[string]interface{})
	return p
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *AddressService) GetPublicIpAddressByID(id string, opts ...OptionFunc) (*PublicIpAddress, int, error) {
	p := &ListPublicIpAddressesParams{}
	p.p = make(map[string]interface{})

	p.p["id"] = id

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return nil, -1, err
		}
	}

	l, err := s.ListPublicIpAddresses(p)
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
		return l.PublicIpAddresses[0], l.Count, nil
	}
	return nil, l.Count, fmt.Errorf("There is more then one result for PublicIpAddress UUID: %s!", id)
}

// Lists all public ip addresses
func (s *AddressService) ListPublicIpAddresses(p *ListPublicIpAddressesParams) (*ListPublicIpAddressesResponse, error) {
	resp, err := s.cs.newRequest("listPublicIpAddresses", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListPublicIpAddressesResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListPublicIpAddressesResponse struct {
	Count             int                `json:"count"`
	PublicIpAddresses []*PublicIpAddress `json:"publicipaddress"`
}

type PublicIpAddress struct {
	Account               string `json:"account,omitempty"`
	Allocated             string `json:"allocated,omitempty"`
	Associatednetworkid   string `json:"associatednetworkid,omitempty"`
	Associatednetworkname string `json:"associatednetworkname,omitempty"`
	Domain                string `json:"domain,omitempty"`
	Domainid              string `json:"domainid,omitempty"`
	Fordisplay            bool   `json:"fordisplay,omitempty"`
	Forvirtualnetwork     bool   `json:"forvirtualnetwork,omitempty"`
	Id                    string `json:"id,omitempty"`
	Ipaddress             string `json:"ipaddress,omitempty"`
	Isportable            bool   `json:"isportable,omitempty"`
	Issourcenat           bool   `json:"issourcenat,omitempty"`
	Isstaticnat           bool   `json:"isstaticnat,omitempty"`
	Issystem              bool   `json:"issystem,omitempty"`
	Networkid             string `json:"networkid,omitempty"`
	Physicalnetworkid     string `json:"physicalnetworkid,omitempty"`
	Project               string `json:"project,omitempty"`
	Projectid             string `json:"projectid,omitempty"`
	Purpose               string `json:"purpose,omitempty"`
	State                 string `json:"state,omitempty"`
	Tags                  []struct {
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
	} `json:"tags,omitempty"`
	Virtualmachinedisplayname string `json:"virtualmachinedisplayname,omitempty"`
	Virtualmachineid          string `json:"virtualmachineid,omitempty"`
	Virtualmachinename        string `json:"virtualmachinename,omitempty"`
	Vlanid                    string `json:"vlanid,omitempty"`
	Vlanname                  string `json:"vlanname,omitempty"`
	Vmipaddress               string `json:"vmipaddress,omitempty"`
	Vpcid                     string `json:"vpcid,omitempty"`
	Zoneid                    string `json:"zoneid,omitempty"`
	Zonename                  string `json:"zonename,omitempty"`
}

type UpdateIpAddressParams struct {
	p map[string]interface{}
}

func (p *UpdateIpAddressParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["customid"]; found {
		u.Set("customid", v.(string))
	}
	if v, found := p.p["fordisplay"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("fordisplay", vv)
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	return u
}

func (p *UpdateIpAddressParams) SetCustomid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["customid"] = v
	return
}

func (p *UpdateIpAddressParams) SetFordisplay(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["fordisplay"] = v
	return
}

func (p *UpdateIpAddressParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new UpdateIpAddressParams instance,
// as then you are sure you have configured all required params
func (s *AddressService) NewUpdateIpAddressParams(id string) *UpdateIpAddressParams {
	p := &UpdateIpAddressParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Updates an IP address
func (s *AddressService) UpdateIpAddress(p *UpdateIpAddressParams) (*UpdateIpAddressResponse, error) {
	resp, err := s.cs.newRequest("updateIpAddress", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r UpdateIpAddressResponse
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

type UpdateIpAddressResponse struct {
	JobID                 string `json:"jobid,omitempty"`
	Account               string `json:"account,omitempty"`
	Allocated             string `json:"allocated,omitempty"`
	Associatednetworkid   string `json:"associatednetworkid,omitempty"`
	Associatednetworkname string `json:"associatednetworkname,omitempty"`
	Domain                string `json:"domain,omitempty"`
	Domainid              string `json:"domainid,omitempty"`
	Fordisplay            bool   `json:"fordisplay,omitempty"`
	Forvirtualnetwork     bool   `json:"forvirtualnetwork,omitempty"`
	Id                    string `json:"id,omitempty"`
	Ipaddress             string `json:"ipaddress,omitempty"`
	Isportable            bool   `json:"isportable,omitempty"`
	Issourcenat           bool   `json:"issourcenat,omitempty"`
	Isstaticnat           bool   `json:"isstaticnat,omitempty"`
	Issystem              bool   `json:"issystem,omitempty"`
	Networkid             string `json:"networkid,omitempty"`
	Physicalnetworkid     string `json:"physicalnetworkid,omitempty"`
	Project               string `json:"project,omitempty"`
	Projectid             string `json:"projectid,omitempty"`
	Purpose               string `json:"purpose,omitempty"`
	State                 string `json:"state,omitempty"`
	Tags                  []struct {
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
	} `json:"tags,omitempty"`
	Virtualmachinedisplayname string `json:"virtualmachinedisplayname,omitempty"`
	Virtualmachineid          string `json:"virtualmachineid,omitempty"`
	Virtualmachinename        string `json:"virtualmachinename,omitempty"`
	Vlanid                    string `json:"vlanid,omitempty"`
	Vlanname                  string `json:"vlanname,omitempty"`
	Vmipaddress               string `json:"vmipaddress,omitempty"`
	Vpcid                     string `json:"vpcid,omitempty"`
	Zoneid                    string `json:"zoneid,omitempty"`
	Zonename                  string `json:"zonename,omitempty"`
}
