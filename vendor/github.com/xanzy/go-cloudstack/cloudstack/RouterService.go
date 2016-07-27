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

type StartRouterParams struct {
	p map[string]interface{}
}

func (p *StartRouterParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	return u
}

func (p *StartRouterParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new StartRouterParams instance,
// as then you are sure you have configured all required params
func (s *RouterService) NewStartRouterParams(id string) *StartRouterParams {
	p := &StartRouterParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Starts a router.
func (s *RouterService) StartRouter(p *StartRouterParams) (*StartRouterResponse, error) {
	resp, err := s.cs.newRequest("startRouter", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r StartRouterResponse
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

type StartRouterResponse struct {
	JobID               string `json:"jobid,omitempty"`
	Account             string `json:"account,omitempty"`
	Created             string `json:"created,omitempty"`
	Dns1                string `json:"dns1,omitempty"`
	Dns2                string `json:"dns2,omitempty"`
	Domain              string `json:"domain,omitempty"`
	Domainid            string `json:"domainid,omitempty"`
	Gateway             string `json:"gateway,omitempty"`
	Guestipaddress      string `json:"guestipaddress,omitempty"`
	Guestmacaddress     string `json:"guestmacaddress,omitempty"`
	Guestnetmask        string `json:"guestnetmask,omitempty"`
	Guestnetworkid      string `json:"guestnetworkid,omitempty"`
	Guestnetworkname    string `json:"guestnetworkname,omitempty"`
	Hostid              string `json:"hostid,omitempty"`
	Hostname            string `json:"hostname,omitempty"`
	Hypervisor          string `json:"hypervisor,omitempty"`
	Id                  string `json:"id,omitempty"`
	Ip6dns1             string `json:"ip6dns1,omitempty"`
	Ip6dns2             string `json:"ip6dns2,omitempty"`
	Isredundantrouter   bool   `json:"isredundantrouter,omitempty"`
	Linklocalip         string `json:"linklocalip,omitempty"`
	Linklocalmacaddress string `json:"linklocalmacaddress,omitempty"`
	Linklocalnetmask    string `json:"linklocalnetmask,omitempty"`
	Linklocalnetworkid  string `json:"linklocalnetworkid,omitempty"`
	Name                string `json:"name,omitempty"`
	Networkdomain       string `json:"networkdomain,omitempty"`
	Nic                 []struct {
		Broadcasturi string `json:"broadcasturi,omitempty"`
		Deviceid     string `json:"deviceid,omitempty"`
		Gateway      string `json:"gateway,omitempty"`
		Id           string `json:"id,omitempty"`
		Ip6address   string `json:"ip6address,omitempty"`
		Ip6cidr      string `json:"ip6cidr,omitempty"`
		Ip6gateway   string `json:"ip6gateway,omitempty"`
		Ipaddress    string `json:"ipaddress,omitempty"`
		Isdefault    bool   `json:"isdefault,omitempty"`
		Isolationuri string `json:"isolationuri,omitempty"`
		Macaddress   string `json:"macaddress,omitempty"`
		Netmask      string `json:"netmask,omitempty"`
		Networkid    string `json:"networkid,omitempty"`
		Networkname  string `json:"networkname,omitempty"`
		Secondaryip  []struct {
			Id        string `json:"id,omitempty"`
			Ipaddress string `json:"ipaddress,omitempty"`
		} `json:"secondaryip,omitempty"`
		Traffictype      string `json:"traffictype,omitempty"`
		Type             string `json:"type,omitempty"`
		Virtualmachineid string `json:"virtualmachineid,omitempty"`
	} `json:"nic,omitempty"`
	Podid               string `json:"podid,omitempty"`
	Project             string `json:"project,omitempty"`
	Projectid           string `json:"projectid,omitempty"`
	Publicip            string `json:"publicip,omitempty"`
	Publicmacaddress    string `json:"publicmacaddress,omitempty"`
	Publicnetmask       string `json:"publicnetmask,omitempty"`
	Publicnetworkid     string `json:"publicnetworkid,omitempty"`
	Redundantstate      string `json:"redundantstate,omitempty"`
	Requiresupgrade     bool   `json:"requiresupgrade,omitempty"`
	Role                string `json:"role,omitempty"`
	Scriptsversion      string `json:"scriptsversion,omitempty"`
	Serviceofferingid   string `json:"serviceofferingid,omitempty"`
	Serviceofferingname string `json:"serviceofferingname,omitempty"`
	State               string `json:"state,omitempty"`
	Templateid          string `json:"templateid,omitempty"`
	Version             string `json:"version,omitempty"`
	Vpcid               string `json:"vpcid,omitempty"`
	Vpcname             string `json:"vpcname,omitempty"`
	Zoneid              string `json:"zoneid,omitempty"`
	Zonename            string `json:"zonename,omitempty"`
}

type RebootRouterParams struct {
	p map[string]interface{}
}

func (p *RebootRouterParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	return u
}

func (p *RebootRouterParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new RebootRouterParams instance,
// as then you are sure you have configured all required params
func (s *RouterService) NewRebootRouterParams(id string) *RebootRouterParams {
	p := &RebootRouterParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Starts a router.
func (s *RouterService) RebootRouter(p *RebootRouterParams) (*RebootRouterResponse, error) {
	resp, err := s.cs.newRequest("rebootRouter", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r RebootRouterResponse
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

type RebootRouterResponse struct {
	JobID               string `json:"jobid,omitempty"`
	Account             string `json:"account,omitempty"`
	Created             string `json:"created,omitempty"`
	Dns1                string `json:"dns1,omitempty"`
	Dns2                string `json:"dns2,omitempty"`
	Domain              string `json:"domain,omitempty"`
	Domainid            string `json:"domainid,omitempty"`
	Gateway             string `json:"gateway,omitempty"`
	Guestipaddress      string `json:"guestipaddress,omitempty"`
	Guestmacaddress     string `json:"guestmacaddress,omitempty"`
	Guestnetmask        string `json:"guestnetmask,omitempty"`
	Guestnetworkid      string `json:"guestnetworkid,omitempty"`
	Guestnetworkname    string `json:"guestnetworkname,omitempty"`
	Hostid              string `json:"hostid,omitempty"`
	Hostname            string `json:"hostname,omitempty"`
	Hypervisor          string `json:"hypervisor,omitempty"`
	Id                  string `json:"id,omitempty"`
	Ip6dns1             string `json:"ip6dns1,omitempty"`
	Ip6dns2             string `json:"ip6dns2,omitempty"`
	Isredundantrouter   bool   `json:"isredundantrouter,omitempty"`
	Linklocalip         string `json:"linklocalip,omitempty"`
	Linklocalmacaddress string `json:"linklocalmacaddress,omitempty"`
	Linklocalnetmask    string `json:"linklocalnetmask,omitempty"`
	Linklocalnetworkid  string `json:"linklocalnetworkid,omitempty"`
	Name                string `json:"name,omitempty"`
	Networkdomain       string `json:"networkdomain,omitempty"`
	Nic                 []struct {
		Broadcasturi string `json:"broadcasturi,omitempty"`
		Deviceid     string `json:"deviceid,omitempty"`
		Gateway      string `json:"gateway,omitempty"`
		Id           string `json:"id,omitempty"`
		Ip6address   string `json:"ip6address,omitempty"`
		Ip6cidr      string `json:"ip6cidr,omitempty"`
		Ip6gateway   string `json:"ip6gateway,omitempty"`
		Ipaddress    string `json:"ipaddress,omitempty"`
		Isdefault    bool   `json:"isdefault,omitempty"`
		Isolationuri string `json:"isolationuri,omitempty"`
		Macaddress   string `json:"macaddress,omitempty"`
		Netmask      string `json:"netmask,omitempty"`
		Networkid    string `json:"networkid,omitempty"`
		Networkname  string `json:"networkname,omitempty"`
		Secondaryip  []struct {
			Id        string `json:"id,omitempty"`
			Ipaddress string `json:"ipaddress,omitempty"`
		} `json:"secondaryip,omitempty"`
		Traffictype      string `json:"traffictype,omitempty"`
		Type             string `json:"type,omitempty"`
		Virtualmachineid string `json:"virtualmachineid,omitempty"`
	} `json:"nic,omitempty"`
	Podid               string `json:"podid,omitempty"`
	Project             string `json:"project,omitempty"`
	Projectid           string `json:"projectid,omitempty"`
	Publicip            string `json:"publicip,omitempty"`
	Publicmacaddress    string `json:"publicmacaddress,omitempty"`
	Publicnetmask       string `json:"publicnetmask,omitempty"`
	Publicnetworkid     string `json:"publicnetworkid,omitempty"`
	Redundantstate      string `json:"redundantstate,omitempty"`
	Requiresupgrade     bool   `json:"requiresupgrade,omitempty"`
	Role                string `json:"role,omitempty"`
	Scriptsversion      string `json:"scriptsversion,omitempty"`
	Serviceofferingid   string `json:"serviceofferingid,omitempty"`
	Serviceofferingname string `json:"serviceofferingname,omitempty"`
	State               string `json:"state,omitempty"`
	Templateid          string `json:"templateid,omitempty"`
	Version             string `json:"version,omitempty"`
	Vpcid               string `json:"vpcid,omitempty"`
	Vpcname             string `json:"vpcname,omitempty"`
	Zoneid              string `json:"zoneid,omitempty"`
	Zonename            string `json:"zonename,omitempty"`
}

type StopRouterParams struct {
	p map[string]interface{}
}

func (p *StopRouterParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["forced"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("forced", vv)
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	return u
}

func (p *StopRouterParams) SetForced(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["forced"] = v
	return
}

func (p *StopRouterParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new StopRouterParams instance,
// as then you are sure you have configured all required params
func (s *RouterService) NewStopRouterParams(id string) *StopRouterParams {
	p := &StopRouterParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Stops a router.
func (s *RouterService) StopRouter(p *StopRouterParams) (*StopRouterResponse, error) {
	resp, err := s.cs.newRequest("stopRouter", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r StopRouterResponse
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

type StopRouterResponse struct {
	JobID               string `json:"jobid,omitempty"`
	Account             string `json:"account,omitempty"`
	Created             string `json:"created,omitempty"`
	Dns1                string `json:"dns1,omitempty"`
	Dns2                string `json:"dns2,omitempty"`
	Domain              string `json:"domain,omitempty"`
	Domainid            string `json:"domainid,omitempty"`
	Gateway             string `json:"gateway,omitempty"`
	Guestipaddress      string `json:"guestipaddress,omitempty"`
	Guestmacaddress     string `json:"guestmacaddress,omitempty"`
	Guestnetmask        string `json:"guestnetmask,omitempty"`
	Guestnetworkid      string `json:"guestnetworkid,omitempty"`
	Guestnetworkname    string `json:"guestnetworkname,omitempty"`
	Hostid              string `json:"hostid,omitempty"`
	Hostname            string `json:"hostname,omitempty"`
	Hypervisor          string `json:"hypervisor,omitempty"`
	Id                  string `json:"id,omitempty"`
	Ip6dns1             string `json:"ip6dns1,omitempty"`
	Ip6dns2             string `json:"ip6dns2,omitempty"`
	Isredundantrouter   bool   `json:"isredundantrouter,omitempty"`
	Linklocalip         string `json:"linklocalip,omitempty"`
	Linklocalmacaddress string `json:"linklocalmacaddress,omitempty"`
	Linklocalnetmask    string `json:"linklocalnetmask,omitempty"`
	Linklocalnetworkid  string `json:"linklocalnetworkid,omitempty"`
	Name                string `json:"name,omitempty"`
	Networkdomain       string `json:"networkdomain,omitempty"`
	Nic                 []struct {
		Broadcasturi string `json:"broadcasturi,omitempty"`
		Deviceid     string `json:"deviceid,omitempty"`
		Gateway      string `json:"gateway,omitempty"`
		Id           string `json:"id,omitempty"`
		Ip6address   string `json:"ip6address,omitempty"`
		Ip6cidr      string `json:"ip6cidr,omitempty"`
		Ip6gateway   string `json:"ip6gateway,omitempty"`
		Ipaddress    string `json:"ipaddress,omitempty"`
		Isdefault    bool   `json:"isdefault,omitempty"`
		Isolationuri string `json:"isolationuri,omitempty"`
		Macaddress   string `json:"macaddress,omitempty"`
		Netmask      string `json:"netmask,omitempty"`
		Networkid    string `json:"networkid,omitempty"`
		Networkname  string `json:"networkname,omitempty"`
		Secondaryip  []struct {
			Id        string `json:"id,omitempty"`
			Ipaddress string `json:"ipaddress,omitempty"`
		} `json:"secondaryip,omitempty"`
		Traffictype      string `json:"traffictype,omitempty"`
		Type             string `json:"type,omitempty"`
		Virtualmachineid string `json:"virtualmachineid,omitempty"`
	} `json:"nic,omitempty"`
	Podid               string `json:"podid,omitempty"`
	Project             string `json:"project,omitempty"`
	Projectid           string `json:"projectid,omitempty"`
	Publicip            string `json:"publicip,omitempty"`
	Publicmacaddress    string `json:"publicmacaddress,omitempty"`
	Publicnetmask       string `json:"publicnetmask,omitempty"`
	Publicnetworkid     string `json:"publicnetworkid,omitempty"`
	Redundantstate      string `json:"redundantstate,omitempty"`
	Requiresupgrade     bool   `json:"requiresupgrade,omitempty"`
	Role                string `json:"role,omitempty"`
	Scriptsversion      string `json:"scriptsversion,omitempty"`
	Serviceofferingid   string `json:"serviceofferingid,omitempty"`
	Serviceofferingname string `json:"serviceofferingname,omitempty"`
	State               string `json:"state,omitempty"`
	Templateid          string `json:"templateid,omitempty"`
	Version             string `json:"version,omitempty"`
	Vpcid               string `json:"vpcid,omitempty"`
	Vpcname             string `json:"vpcname,omitempty"`
	Zoneid              string `json:"zoneid,omitempty"`
	Zonename            string `json:"zonename,omitempty"`
}

type DestroyRouterParams struct {
	p map[string]interface{}
}

func (p *DestroyRouterParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	return u
}

func (p *DestroyRouterParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new DestroyRouterParams instance,
// as then you are sure you have configured all required params
func (s *RouterService) NewDestroyRouterParams(id string) *DestroyRouterParams {
	p := &DestroyRouterParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Destroys a router.
func (s *RouterService) DestroyRouter(p *DestroyRouterParams) (*DestroyRouterResponse, error) {
	resp, err := s.cs.newRequest("destroyRouter", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r DestroyRouterResponse
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

type DestroyRouterResponse struct {
	JobID               string `json:"jobid,omitempty"`
	Account             string `json:"account,omitempty"`
	Created             string `json:"created,omitempty"`
	Dns1                string `json:"dns1,omitempty"`
	Dns2                string `json:"dns2,omitempty"`
	Domain              string `json:"domain,omitempty"`
	Domainid            string `json:"domainid,omitempty"`
	Gateway             string `json:"gateway,omitempty"`
	Guestipaddress      string `json:"guestipaddress,omitempty"`
	Guestmacaddress     string `json:"guestmacaddress,omitempty"`
	Guestnetmask        string `json:"guestnetmask,omitempty"`
	Guestnetworkid      string `json:"guestnetworkid,omitempty"`
	Guestnetworkname    string `json:"guestnetworkname,omitempty"`
	Hostid              string `json:"hostid,omitempty"`
	Hostname            string `json:"hostname,omitempty"`
	Hypervisor          string `json:"hypervisor,omitempty"`
	Id                  string `json:"id,omitempty"`
	Ip6dns1             string `json:"ip6dns1,omitempty"`
	Ip6dns2             string `json:"ip6dns2,omitempty"`
	Isredundantrouter   bool   `json:"isredundantrouter,omitempty"`
	Linklocalip         string `json:"linklocalip,omitempty"`
	Linklocalmacaddress string `json:"linklocalmacaddress,omitempty"`
	Linklocalnetmask    string `json:"linklocalnetmask,omitempty"`
	Linklocalnetworkid  string `json:"linklocalnetworkid,omitempty"`
	Name                string `json:"name,omitempty"`
	Networkdomain       string `json:"networkdomain,omitempty"`
	Nic                 []struct {
		Broadcasturi string `json:"broadcasturi,omitempty"`
		Deviceid     string `json:"deviceid,omitempty"`
		Gateway      string `json:"gateway,omitempty"`
		Id           string `json:"id,omitempty"`
		Ip6address   string `json:"ip6address,omitempty"`
		Ip6cidr      string `json:"ip6cidr,omitempty"`
		Ip6gateway   string `json:"ip6gateway,omitempty"`
		Ipaddress    string `json:"ipaddress,omitempty"`
		Isdefault    bool   `json:"isdefault,omitempty"`
		Isolationuri string `json:"isolationuri,omitempty"`
		Macaddress   string `json:"macaddress,omitempty"`
		Netmask      string `json:"netmask,omitempty"`
		Networkid    string `json:"networkid,omitempty"`
		Networkname  string `json:"networkname,omitempty"`
		Secondaryip  []struct {
			Id        string `json:"id,omitempty"`
			Ipaddress string `json:"ipaddress,omitempty"`
		} `json:"secondaryip,omitempty"`
		Traffictype      string `json:"traffictype,omitempty"`
		Type             string `json:"type,omitempty"`
		Virtualmachineid string `json:"virtualmachineid,omitempty"`
	} `json:"nic,omitempty"`
	Podid               string `json:"podid,omitempty"`
	Project             string `json:"project,omitempty"`
	Projectid           string `json:"projectid,omitempty"`
	Publicip            string `json:"publicip,omitempty"`
	Publicmacaddress    string `json:"publicmacaddress,omitempty"`
	Publicnetmask       string `json:"publicnetmask,omitempty"`
	Publicnetworkid     string `json:"publicnetworkid,omitempty"`
	Redundantstate      string `json:"redundantstate,omitempty"`
	Requiresupgrade     bool   `json:"requiresupgrade,omitempty"`
	Role                string `json:"role,omitempty"`
	Scriptsversion      string `json:"scriptsversion,omitempty"`
	Serviceofferingid   string `json:"serviceofferingid,omitempty"`
	Serviceofferingname string `json:"serviceofferingname,omitempty"`
	State               string `json:"state,omitempty"`
	Templateid          string `json:"templateid,omitempty"`
	Version             string `json:"version,omitempty"`
	Vpcid               string `json:"vpcid,omitempty"`
	Vpcname             string `json:"vpcname,omitempty"`
	Zoneid              string `json:"zoneid,omitempty"`
	Zonename            string `json:"zonename,omitempty"`
}

type ChangeServiceForRouterParams struct {
	p map[string]interface{}
}

func (p *ChangeServiceForRouterParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	if v, found := p.p["serviceofferingid"]; found {
		u.Set("serviceofferingid", v.(string))
	}
	return u
}

func (p *ChangeServiceForRouterParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ChangeServiceForRouterParams) SetServiceofferingid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["serviceofferingid"] = v
	return
}

// You should always use this function to get a new ChangeServiceForRouterParams instance,
// as then you are sure you have configured all required params
func (s *RouterService) NewChangeServiceForRouterParams(id string, serviceofferingid string) *ChangeServiceForRouterParams {
	p := &ChangeServiceForRouterParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	p.p["serviceofferingid"] = serviceofferingid
	return p
}

// Upgrades domain router to a new service offering
func (s *RouterService) ChangeServiceForRouter(p *ChangeServiceForRouterParams) (*ChangeServiceForRouterResponse, error) {
	resp, err := s.cs.newRequest("changeServiceForRouter", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ChangeServiceForRouterResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ChangeServiceForRouterResponse struct {
	Account             string `json:"account,omitempty"`
	Created             string `json:"created,omitempty"`
	Dns1                string `json:"dns1,omitempty"`
	Dns2                string `json:"dns2,omitempty"`
	Domain              string `json:"domain,omitempty"`
	Domainid            string `json:"domainid,omitempty"`
	Gateway             string `json:"gateway,omitempty"`
	Guestipaddress      string `json:"guestipaddress,omitempty"`
	Guestmacaddress     string `json:"guestmacaddress,omitempty"`
	Guestnetmask        string `json:"guestnetmask,omitempty"`
	Guestnetworkid      string `json:"guestnetworkid,omitempty"`
	Guestnetworkname    string `json:"guestnetworkname,omitempty"`
	Hostid              string `json:"hostid,omitempty"`
	Hostname            string `json:"hostname,omitempty"`
	Hypervisor          string `json:"hypervisor,omitempty"`
	Id                  string `json:"id,omitempty"`
	Ip6dns1             string `json:"ip6dns1,omitempty"`
	Ip6dns2             string `json:"ip6dns2,omitempty"`
	Isredundantrouter   bool   `json:"isredundantrouter,omitempty"`
	Linklocalip         string `json:"linklocalip,omitempty"`
	Linklocalmacaddress string `json:"linklocalmacaddress,omitempty"`
	Linklocalnetmask    string `json:"linklocalnetmask,omitempty"`
	Linklocalnetworkid  string `json:"linklocalnetworkid,omitempty"`
	Name                string `json:"name,omitempty"`
	Networkdomain       string `json:"networkdomain,omitempty"`
	Nic                 []struct {
		Broadcasturi string `json:"broadcasturi,omitempty"`
		Deviceid     string `json:"deviceid,omitempty"`
		Gateway      string `json:"gateway,omitempty"`
		Id           string `json:"id,omitempty"`
		Ip6address   string `json:"ip6address,omitempty"`
		Ip6cidr      string `json:"ip6cidr,omitempty"`
		Ip6gateway   string `json:"ip6gateway,omitempty"`
		Ipaddress    string `json:"ipaddress,omitempty"`
		Isdefault    bool   `json:"isdefault,omitempty"`
		Isolationuri string `json:"isolationuri,omitempty"`
		Macaddress   string `json:"macaddress,omitempty"`
		Netmask      string `json:"netmask,omitempty"`
		Networkid    string `json:"networkid,omitempty"`
		Networkname  string `json:"networkname,omitempty"`
		Secondaryip  []struct {
			Id        string `json:"id,omitempty"`
			Ipaddress string `json:"ipaddress,omitempty"`
		} `json:"secondaryip,omitempty"`
		Traffictype      string `json:"traffictype,omitempty"`
		Type             string `json:"type,omitempty"`
		Virtualmachineid string `json:"virtualmachineid,omitempty"`
	} `json:"nic,omitempty"`
	Podid               string `json:"podid,omitempty"`
	Project             string `json:"project,omitempty"`
	Projectid           string `json:"projectid,omitempty"`
	Publicip            string `json:"publicip,omitempty"`
	Publicmacaddress    string `json:"publicmacaddress,omitempty"`
	Publicnetmask       string `json:"publicnetmask,omitempty"`
	Publicnetworkid     string `json:"publicnetworkid,omitempty"`
	Redundantstate      string `json:"redundantstate,omitempty"`
	Requiresupgrade     bool   `json:"requiresupgrade,omitempty"`
	Role                string `json:"role,omitempty"`
	Scriptsversion      string `json:"scriptsversion,omitempty"`
	Serviceofferingid   string `json:"serviceofferingid,omitempty"`
	Serviceofferingname string `json:"serviceofferingname,omitempty"`
	State               string `json:"state,omitempty"`
	Templateid          string `json:"templateid,omitempty"`
	Version             string `json:"version,omitempty"`
	Vpcid               string `json:"vpcid,omitempty"`
	Vpcname             string `json:"vpcname,omitempty"`
	Zoneid              string `json:"zoneid,omitempty"`
	Zonename            string `json:"zonename,omitempty"`
}

type ListRoutersParams struct {
	p map[string]interface{}
}

func (p *ListRoutersParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["account"]; found {
		u.Set("account", v.(string))
	}
	if v, found := p.p["clusterid"]; found {
		u.Set("clusterid", v.(string))
	}
	if v, found := p.p["domainid"]; found {
		u.Set("domainid", v.(string))
	}
	if v, found := p.p["forvpc"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("forvpc", vv)
	}
	if v, found := p.p["hostid"]; found {
		u.Set("hostid", v.(string))
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
	if v, found := p.p["podid"]; found {
		u.Set("podid", v.(string))
	}
	if v, found := p.p["projectid"]; found {
		u.Set("projectid", v.(string))
	}
	if v, found := p.p["state"]; found {
		u.Set("state", v.(string))
	}
	if v, found := p.p["version"]; found {
		u.Set("version", v.(string))
	}
	if v, found := p.p["vpcid"]; found {
		u.Set("vpcid", v.(string))
	}
	if v, found := p.p["zoneid"]; found {
		u.Set("zoneid", v.(string))
	}
	return u
}

func (p *ListRoutersParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *ListRoutersParams) SetClusterid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["clusterid"] = v
	return
}

func (p *ListRoutersParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *ListRoutersParams) SetForvpc(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["forvpc"] = v
	return
}

func (p *ListRoutersParams) SetHostid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["hostid"] = v
	return
}

func (p *ListRoutersParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ListRoutersParams) SetIsrecursive(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isrecursive"] = v
	return
}

func (p *ListRoutersParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListRoutersParams) SetListall(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["listall"] = v
	return
}

func (p *ListRoutersParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *ListRoutersParams) SetNetworkid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["networkid"] = v
	return
}

func (p *ListRoutersParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListRoutersParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

func (p *ListRoutersParams) SetPodid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["podid"] = v
	return
}

func (p *ListRoutersParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

func (p *ListRoutersParams) SetState(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["state"] = v
	return
}

func (p *ListRoutersParams) SetVersion(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["version"] = v
	return
}

func (p *ListRoutersParams) SetVpcid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["vpcid"] = v
	return
}

func (p *ListRoutersParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new ListRoutersParams instance,
// as then you are sure you have configured all required params
func (s *RouterService) NewListRoutersParams() *ListRoutersParams {
	p := &ListRoutersParams{}
	p.p = make(map[string]interface{})
	return p
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *RouterService) GetRouterID(name string, opts ...OptionFunc) (string, error) {
	p := &ListRoutersParams{}
	p.p = make(map[string]interface{})

	p.p["name"] = name

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return "", err
		}
	}

	l, err := s.ListRouters(p)
	if err != nil {
		return "", err
	}

	if l.Count == 0 {
		return "", fmt.Errorf("No match found for %s: %+v", name, l)
	}

	if l.Count == 1 {
		return l.Routers[0].Id, nil
	}

	if l.Count > 1 {
		for _, v := range l.Routers {
			if v.Name == name {
				return v.Id, nil
			}
		}
	}
	return "", fmt.Errorf("Could not find an exact match for %s: %+v", name, l)
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *RouterService) GetRouterByName(name string, opts ...OptionFunc) (*Router, int, error) {
	id, err := s.GetRouterID(name, opts...)
	if err != nil {
		return nil, -1, err
	}

	r, count, err := s.GetRouterByID(id, opts...)
	if err != nil {
		return nil, count, err
	}
	return r, count, nil
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *RouterService) GetRouterByID(id string, opts ...OptionFunc) (*Router, int, error) {
	p := &ListRoutersParams{}
	p.p = make(map[string]interface{})

	p.p["id"] = id

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return nil, -1, err
		}
	}

	l, err := s.ListRouters(p)
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
		return l.Routers[0], l.Count, nil
	}
	return nil, l.Count, fmt.Errorf("There is more then one result for Router UUID: %s!", id)
}

// List routers.
func (s *RouterService) ListRouters(p *ListRoutersParams) (*ListRoutersResponse, error) {
	resp, err := s.cs.newRequest("listRouters", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListRoutersResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListRoutersResponse struct {
	Count   int       `json:"count"`
	Routers []*Router `json:"router"`
}

type Router struct {
	Account             string `json:"account,omitempty"`
	Created             string `json:"created,omitempty"`
	Dns1                string `json:"dns1,omitempty"`
	Dns2                string `json:"dns2,omitempty"`
	Domain              string `json:"domain,omitempty"`
	Domainid            string `json:"domainid,omitempty"`
	Gateway             string `json:"gateway,omitempty"`
	Guestipaddress      string `json:"guestipaddress,omitempty"`
	Guestmacaddress     string `json:"guestmacaddress,omitempty"`
	Guestnetmask        string `json:"guestnetmask,omitempty"`
	Guestnetworkid      string `json:"guestnetworkid,omitempty"`
	Guestnetworkname    string `json:"guestnetworkname,omitempty"`
	Hostid              string `json:"hostid,omitempty"`
	Hostname            string `json:"hostname,omitempty"`
	Hypervisor          string `json:"hypervisor,omitempty"`
	Id                  string `json:"id,omitempty"`
	Ip6dns1             string `json:"ip6dns1,omitempty"`
	Ip6dns2             string `json:"ip6dns2,omitempty"`
	Isredundantrouter   bool   `json:"isredundantrouter,omitempty"`
	Linklocalip         string `json:"linklocalip,omitempty"`
	Linklocalmacaddress string `json:"linklocalmacaddress,omitempty"`
	Linklocalnetmask    string `json:"linklocalnetmask,omitempty"`
	Linklocalnetworkid  string `json:"linklocalnetworkid,omitempty"`
	Name                string `json:"name,omitempty"`
	Networkdomain       string `json:"networkdomain,omitempty"`
	Nic                 []struct {
		Broadcasturi string `json:"broadcasturi,omitempty"`
		Deviceid     string `json:"deviceid,omitempty"`
		Gateway      string `json:"gateway,omitempty"`
		Id           string `json:"id,omitempty"`
		Ip6address   string `json:"ip6address,omitempty"`
		Ip6cidr      string `json:"ip6cidr,omitempty"`
		Ip6gateway   string `json:"ip6gateway,omitempty"`
		Ipaddress    string `json:"ipaddress,omitempty"`
		Isdefault    bool   `json:"isdefault,omitempty"`
		Isolationuri string `json:"isolationuri,omitempty"`
		Macaddress   string `json:"macaddress,omitempty"`
		Netmask      string `json:"netmask,omitempty"`
		Networkid    string `json:"networkid,omitempty"`
		Networkname  string `json:"networkname,omitempty"`
		Secondaryip  []struct {
			Id        string `json:"id,omitempty"`
			Ipaddress string `json:"ipaddress,omitempty"`
		} `json:"secondaryip,omitempty"`
		Traffictype      string `json:"traffictype,omitempty"`
		Type             string `json:"type,omitempty"`
		Virtualmachineid string `json:"virtualmachineid,omitempty"`
	} `json:"nic,omitempty"`
	Podid               string `json:"podid,omitempty"`
	Project             string `json:"project,omitempty"`
	Projectid           string `json:"projectid,omitempty"`
	Publicip            string `json:"publicip,omitempty"`
	Publicmacaddress    string `json:"publicmacaddress,omitempty"`
	Publicnetmask       string `json:"publicnetmask,omitempty"`
	Publicnetworkid     string `json:"publicnetworkid,omitempty"`
	Redundantstate      string `json:"redundantstate,omitempty"`
	Requiresupgrade     bool   `json:"requiresupgrade,omitempty"`
	Role                string `json:"role,omitempty"`
	Scriptsversion      string `json:"scriptsversion,omitempty"`
	Serviceofferingid   string `json:"serviceofferingid,omitempty"`
	Serviceofferingname string `json:"serviceofferingname,omitempty"`
	State               string `json:"state,omitempty"`
	Templateid          string `json:"templateid,omitempty"`
	Version             string `json:"version,omitempty"`
	Vpcid               string `json:"vpcid,omitempty"`
	Vpcname             string `json:"vpcname,omitempty"`
	Zoneid              string `json:"zoneid,omitempty"`
	Zonename            string `json:"zonename,omitempty"`
}

type ListVirtualRouterElementsParams struct {
	p map[string]interface{}
}

func (p *ListVirtualRouterElementsParams) toURLValues() url.Values {
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

func (p *ListVirtualRouterElementsParams) SetEnabled(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["enabled"] = v
	return
}

func (p *ListVirtualRouterElementsParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ListVirtualRouterElementsParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListVirtualRouterElementsParams) SetNspid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["nspid"] = v
	return
}

func (p *ListVirtualRouterElementsParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListVirtualRouterElementsParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

// You should always use this function to get a new ListVirtualRouterElementsParams instance,
// as then you are sure you have configured all required params
func (s *RouterService) NewListVirtualRouterElementsParams() *ListVirtualRouterElementsParams {
	p := &ListVirtualRouterElementsParams{}
	p.p = make(map[string]interface{})
	return p
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *RouterService) GetVirtualRouterElementByID(id string, opts ...OptionFunc) (*VirtualRouterElement, int, error) {
	p := &ListVirtualRouterElementsParams{}
	p.p = make(map[string]interface{})

	p.p["id"] = id

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return nil, -1, err
		}
	}

	l, err := s.ListVirtualRouterElements(p)
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
		return l.VirtualRouterElements[0], l.Count, nil
	}
	return nil, l.Count, fmt.Errorf("There is more then one result for VirtualRouterElement UUID: %s!", id)
}

// Lists all available virtual router elements.
func (s *RouterService) ListVirtualRouterElements(p *ListVirtualRouterElementsParams) (*ListVirtualRouterElementsResponse, error) {
	resp, err := s.cs.newRequest("listVirtualRouterElements", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListVirtualRouterElementsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListVirtualRouterElementsResponse struct {
	Count                 int                     `json:"count"`
	VirtualRouterElements []*VirtualRouterElement `json:"virtualrouterelement"`
}

type VirtualRouterElement struct {
	Account   string `json:"account,omitempty"`
	Domain    string `json:"domain,omitempty"`
	Domainid  string `json:"domainid,omitempty"`
	Enabled   bool   `json:"enabled,omitempty"`
	Id        string `json:"id,omitempty"`
	Nspid     string `json:"nspid,omitempty"`
	Project   string `json:"project,omitempty"`
	Projectid string `json:"projectid,omitempty"`
}

type ConfigureVirtualRouterElementParams struct {
	p map[string]interface{}
}

func (p *ConfigureVirtualRouterElementParams) toURLValues() url.Values {
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

func (p *ConfigureVirtualRouterElementParams) SetEnabled(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["enabled"] = v
	return
}

func (p *ConfigureVirtualRouterElementParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new ConfigureVirtualRouterElementParams instance,
// as then you are sure you have configured all required params
func (s *RouterService) NewConfigureVirtualRouterElementParams(enabled bool, id string) *ConfigureVirtualRouterElementParams {
	p := &ConfigureVirtualRouterElementParams{}
	p.p = make(map[string]interface{})
	p.p["enabled"] = enabled
	p.p["id"] = id
	return p
}

// Configures a virtual router element.
func (s *RouterService) ConfigureVirtualRouterElement(p *ConfigureVirtualRouterElementParams) (*ConfigureVirtualRouterElementResponse, error) {
	resp, err := s.cs.newRequest("configureVirtualRouterElement", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ConfigureVirtualRouterElementResponse
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

type ConfigureVirtualRouterElementResponse struct {
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

type CreateVirtualRouterElementParams struct {
	p map[string]interface{}
}

func (p *CreateVirtualRouterElementParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["nspid"]; found {
		u.Set("nspid", v.(string))
	}
	if v, found := p.p["providertype"]; found {
		u.Set("providertype", v.(string))
	}
	return u
}

func (p *CreateVirtualRouterElementParams) SetNspid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["nspid"] = v
	return
}

func (p *CreateVirtualRouterElementParams) SetProvidertype(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["providertype"] = v
	return
}

// You should always use this function to get a new CreateVirtualRouterElementParams instance,
// as then you are sure you have configured all required params
func (s *RouterService) NewCreateVirtualRouterElementParams(nspid string) *CreateVirtualRouterElementParams {
	p := &CreateVirtualRouterElementParams{}
	p.p = make(map[string]interface{})
	p.p["nspid"] = nspid
	return p
}

// Create a virtual router element.
func (s *RouterService) CreateVirtualRouterElement(p *CreateVirtualRouterElementParams) (*CreateVirtualRouterElementResponse, error) {
	resp, err := s.cs.newRequest("createVirtualRouterElement", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r CreateVirtualRouterElementResponse
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

type CreateVirtualRouterElementResponse struct {
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
