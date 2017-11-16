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

type AddBaremetalPxeKickStartServerParams struct {
	p map[string]interface{}
}

func (p *AddBaremetalPxeKickStartServerParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["password"]; found {
		u.Set("password", v.(string))
	}
	if v, found := p.p["physicalnetworkid"]; found {
		u.Set("physicalnetworkid", v.(string))
	}
	if v, found := p.p["podid"]; found {
		u.Set("podid", v.(string))
	}
	if v, found := p.p["pxeservertype"]; found {
		u.Set("pxeservertype", v.(string))
	}
	if v, found := p.p["tftpdir"]; found {
		u.Set("tftpdir", v.(string))
	}
	if v, found := p.p["url"]; found {
		u.Set("url", v.(string))
	}
	if v, found := p.p["username"]; found {
		u.Set("username", v.(string))
	}
	return u
}

func (p *AddBaremetalPxeKickStartServerParams) SetPassword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["password"] = v
	return
}

func (p *AddBaremetalPxeKickStartServerParams) SetPhysicalnetworkid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["physicalnetworkid"] = v
	return
}

func (p *AddBaremetalPxeKickStartServerParams) SetPodid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["podid"] = v
	return
}

func (p *AddBaremetalPxeKickStartServerParams) SetPxeservertype(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pxeservertype"] = v
	return
}

func (p *AddBaremetalPxeKickStartServerParams) SetTftpdir(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["tftpdir"] = v
	return
}

func (p *AddBaremetalPxeKickStartServerParams) SetUrl(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["url"] = v
	return
}

func (p *AddBaremetalPxeKickStartServerParams) SetUsername(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["username"] = v
	return
}

// You should always use this function to get a new AddBaremetalPxeKickStartServerParams instance,
// as then you are sure you have configured all required params
func (s *BaremetalService) NewAddBaremetalPxeKickStartServerParams(password string, physicalnetworkid string, pxeservertype string, tftpdir string, url string, username string) *AddBaremetalPxeKickStartServerParams {
	p := &AddBaremetalPxeKickStartServerParams{}
	p.p = make(map[string]interface{})
	p.p["password"] = password
	p.p["physicalnetworkid"] = physicalnetworkid
	p.p["pxeservertype"] = pxeservertype
	p.p["tftpdir"] = tftpdir
	p.p["url"] = url
	p.p["username"] = username
	return p
}

// add a baremetal pxe server
func (s *BaremetalService) AddBaremetalPxeKickStartServer(p *AddBaremetalPxeKickStartServerParams) (*AddBaremetalPxeKickStartServerResponse, error) {
	resp, err := s.cs.newRequest("addBaremetalPxeKickStartServer", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r AddBaremetalPxeKickStartServerResponse
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

type AddBaremetalPxeKickStartServerResponse struct {
	JobID   string `json:"jobid,omitempty"`
	Tftpdir string `json:"tftpdir,omitempty"`
}

type AddBaremetalPxePingServerParams struct {
	p map[string]interface{}
}

func (p *AddBaremetalPxePingServerParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["password"]; found {
		u.Set("password", v.(string))
	}
	if v, found := p.p["physicalnetworkid"]; found {
		u.Set("physicalnetworkid", v.(string))
	}
	if v, found := p.p["pingcifspassword"]; found {
		u.Set("pingcifspassword", v.(string))
	}
	if v, found := p.p["pingcifsusername"]; found {
		u.Set("pingcifsusername", v.(string))
	}
	if v, found := p.p["pingdir"]; found {
		u.Set("pingdir", v.(string))
	}
	if v, found := p.p["pingstorageserverip"]; found {
		u.Set("pingstorageserverip", v.(string))
	}
	if v, found := p.p["podid"]; found {
		u.Set("podid", v.(string))
	}
	if v, found := p.p["pxeservertype"]; found {
		u.Set("pxeservertype", v.(string))
	}
	if v, found := p.p["tftpdir"]; found {
		u.Set("tftpdir", v.(string))
	}
	if v, found := p.p["url"]; found {
		u.Set("url", v.(string))
	}
	if v, found := p.p["username"]; found {
		u.Set("username", v.(string))
	}
	return u
}

func (p *AddBaremetalPxePingServerParams) SetPassword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["password"] = v
	return
}

func (p *AddBaremetalPxePingServerParams) SetPhysicalnetworkid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["physicalnetworkid"] = v
	return
}

func (p *AddBaremetalPxePingServerParams) SetPingcifspassword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pingcifspassword"] = v
	return
}

func (p *AddBaremetalPxePingServerParams) SetPingcifsusername(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pingcifsusername"] = v
	return
}

func (p *AddBaremetalPxePingServerParams) SetPingdir(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pingdir"] = v
	return
}

func (p *AddBaremetalPxePingServerParams) SetPingstorageserverip(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pingstorageserverip"] = v
	return
}

func (p *AddBaremetalPxePingServerParams) SetPodid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["podid"] = v
	return
}

func (p *AddBaremetalPxePingServerParams) SetPxeservertype(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pxeservertype"] = v
	return
}

func (p *AddBaremetalPxePingServerParams) SetTftpdir(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["tftpdir"] = v
	return
}

func (p *AddBaremetalPxePingServerParams) SetUrl(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["url"] = v
	return
}

func (p *AddBaremetalPxePingServerParams) SetUsername(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["username"] = v
	return
}

// You should always use this function to get a new AddBaremetalPxePingServerParams instance,
// as then you are sure you have configured all required params
func (s *BaremetalService) NewAddBaremetalPxePingServerParams(password string, physicalnetworkid string, pingdir string, pingstorageserverip string, pxeservertype string, tftpdir string, url string, username string) *AddBaremetalPxePingServerParams {
	p := &AddBaremetalPxePingServerParams{}
	p.p = make(map[string]interface{})
	p.p["password"] = password
	p.p["physicalnetworkid"] = physicalnetworkid
	p.p["pingdir"] = pingdir
	p.p["pingstorageserverip"] = pingstorageserverip
	p.p["pxeservertype"] = pxeservertype
	p.p["tftpdir"] = tftpdir
	p.p["url"] = url
	p.p["username"] = username
	return p
}

// add a baremetal ping pxe server
func (s *BaremetalService) AddBaremetalPxePingServer(p *AddBaremetalPxePingServerParams) (*AddBaremetalPxePingServerResponse, error) {
	resp, err := s.cs.newRequest("addBaremetalPxePingServer", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r AddBaremetalPxePingServerResponse
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

type AddBaremetalPxePingServerResponse struct {
	JobID               string `json:"jobid,omitempty"`
	Pingdir             string `json:"pingdir,omitempty"`
	Pingstorageserverip string `json:"pingstorageserverip,omitempty"`
	Tftpdir             string `json:"tftpdir,omitempty"`
}

type AddBaremetalDhcpParams struct {
	p map[string]interface{}
}

func (p *AddBaremetalDhcpParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["dhcpservertype"]; found {
		u.Set("dhcpservertype", v.(string))
	}
	if v, found := p.p["password"]; found {
		u.Set("password", v.(string))
	}
	if v, found := p.p["physicalnetworkid"]; found {
		u.Set("physicalnetworkid", v.(string))
	}
	if v, found := p.p["url"]; found {
		u.Set("url", v.(string))
	}
	if v, found := p.p["username"]; found {
		u.Set("username", v.(string))
	}
	return u
}

func (p *AddBaremetalDhcpParams) SetDhcpservertype(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["dhcpservertype"] = v
	return
}

func (p *AddBaremetalDhcpParams) SetPassword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["password"] = v
	return
}

func (p *AddBaremetalDhcpParams) SetPhysicalnetworkid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["physicalnetworkid"] = v
	return
}

func (p *AddBaremetalDhcpParams) SetUrl(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["url"] = v
	return
}

func (p *AddBaremetalDhcpParams) SetUsername(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["username"] = v
	return
}

// You should always use this function to get a new AddBaremetalDhcpParams instance,
// as then you are sure you have configured all required params
func (s *BaremetalService) NewAddBaremetalDhcpParams(dhcpservertype string, password string, physicalnetworkid string, url string, username string) *AddBaremetalDhcpParams {
	p := &AddBaremetalDhcpParams{}
	p.p = make(map[string]interface{})
	p.p["dhcpservertype"] = dhcpservertype
	p.p["password"] = password
	p.p["physicalnetworkid"] = physicalnetworkid
	p.p["url"] = url
	p.p["username"] = username
	return p
}

// adds a baremetal dhcp server
func (s *BaremetalService) AddBaremetalDhcp(p *AddBaremetalDhcpParams) (*AddBaremetalDhcpResponse, error) {
	resp, err := s.cs.newRequest("addBaremetalDhcp", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r AddBaremetalDhcpResponse
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

type AddBaremetalDhcpResponse struct {
	JobID             string `json:"jobid,omitempty"`
	Dhcpservertype    string `json:"dhcpservertype,omitempty"`
	Id                string `json:"id,omitempty"`
	Physicalnetworkid string `json:"physicalnetworkid,omitempty"`
	Provider          string `json:"provider,omitempty"`
	Url               string `json:"url,omitempty"`
}

type ListBaremetalDhcpParams struct {
	p map[string]interface{}
}

func (p *ListBaremetalDhcpParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["dhcpservertype"]; found {
		u.Set("dhcpservertype", v.(string))
	}
	if v, found := p.p["id"]; found {
		vv := strconv.FormatInt(v.(int64), 10)
		u.Set("id", vv)
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
	return u
}

func (p *ListBaremetalDhcpParams) SetDhcpservertype(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["dhcpservertype"] = v
	return
}

func (p *ListBaremetalDhcpParams) SetId(v int64) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ListBaremetalDhcpParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListBaremetalDhcpParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListBaremetalDhcpParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

func (p *ListBaremetalDhcpParams) SetPhysicalnetworkid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["physicalnetworkid"] = v
	return
}

// You should always use this function to get a new ListBaremetalDhcpParams instance,
// as then you are sure you have configured all required params
func (s *BaremetalService) NewListBaremetalDhcpParams(physicalnetworkid string) *ListBaremetalDhcpParams {
	p := &ListBaremetalDhcpParams{}
	p.p = make(map[string]interface{})
	p.p["physicalnetworkid"] = physicalnetworkid
	return p
}

// list baremetal dhcp servers
func (s *BaremetalService) ListBaremetalDhcp(p *ListBaremetalDhcpParams) (*ListBaremetalDhcpResponse, error) {
	resp, err := s.cs.newRequest("listBaremetalDhcp", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListBaremetalDhcpResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListBaremetalDhcpResponse struct {
	Count         int              `json:"count"`
	BaremetalDhcp []*BaremetalDhcp `json:"baremetaldhcp"`
}

type BaremetalDhcp struct {
	Dhcpservertype    string `json:"dhcpservertype,omitempty"`
	Id                string `json:"id,omitempty"`
	Physicalnetworkid string `json:"physicalnetworkid,omitempty"`
	Provider          string `json:"provider,omitempty"`
	Url               string `json:"url,omitempty"`
}

type ListBaremetalPxeServersParams struct {
	p map[string]interface{}
}

func (p *ListBaremetalPxeServersParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		vv := strconv.FormatInt(v.(int64), 10)
		u.Set("id", vv)
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
	return u
}

func (p *ListBaremetalPxeServersParams) SetId(v int64) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ListBaremetalPxeServersParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListBaremetalPxeServersParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListBaremetalPxeServersParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

func (p *ListBaremetalPxeServersParams) SetPhysicalnetworkid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["physicalnetworkid"] = v
	return
}

// You should always use this function to get a new ListBaremetalPxeServersParams instance,
// as then you are sure you have configured all required params
func (s *BaremetalService) NewListBaremetalPxeServersParams(physicalnetworkid string) *ListBaremetalPxeServersParams {
	p := &ListBaremetalPxeServersParams{}
	p.p = make(map[string]interface{})
	p.p["physicalnetworkid"] = physicalnetworkid
	return p
}

// list baremetal pxe server
func (s *BaremetalService) ListBaremetalPxeServers(p *ListBaremetalPxeServersParams) (*ListBaremetalPxeServersResponse, error) {
	resp, err := s.cs.newRequest("listBaremetalPxeServers", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListBaremetalPxeServersResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListBaremetalPxeServersResponse struct {
	Count               int                   `json:"count"`
	BaremetalPxeServers []*BaremetalPxeServer `json:"baremetalpxeserver"`
}

type BaremetalPxeServer struct {
	Id                string `json:"id,omitempty"`
	Physicalnetworkid string `json:"physicalnetworkid,omitempty"`
	Provider          string `json:"provider,omitempty"`
	Url               string `json:"url,omitempty"`
}

type AddBaremetalRctParams struct {
	p map[string]interface{}
}

func (p *AddBaremetalRctParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["baremetalrcturl"]; found {
		u.Set("baremetalrcturl", v.(string))
	}
	return u
}

func (p *AddBaremetalRctParams) SetBaremetalrcturl(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["baremetalrcturl"] = v
	return
}

// You should always use this function to get a new AddBaremetalRctParams instance,
// as then you are sure you have configured all required params
func (s *BaremetalService) NewAddBaremetalRctParams(baremetalrcturl string) *AddBaremetalRctParams {
	p := &AddBaremetalRctParams{}
	p.p = make(map[string]interface{})
	p.p["baremetalrcturl"] = baremetalrcturl
	return p
}

// adds baremetal rack configuration text
func (s *BaremetalService) AddBaremetalRct(p *AddBaremetalRctParams) (*AddBaremetalRctResponse, error) {
	resp, err := s.cs.newRequest("addBaremetalRct", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r AddBaremetalRctResponse
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

type AddBaremetalRctResponse struct {
	JobID string `json:"jobid,omitempty"`
	Id    string `json:"id,omitempty"`
	Url   string `json:"url,omitempty"`
}

type DeleteBaremetalRctParams struct {
	p map[string]interface{}
}

func (p *DeleteBaremetalRctParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	return u
}

func (p *DeleteBaremetalRctParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new DeleteBaremetalRctParams instance,
// as then you are sure you have configured all required params
func (s *BaremetalService) NewDeleteBaremetalRctParams(id string) *DeleteBaremetalRctParams {
	p := &DeleteBaremetalRctParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// deletes baremetal rack configuration text
func (s *BaremetalService) DeleteBaremetalRct(p *DeleteBaremetalRctParams) (*DeleteBaremetalRctResponse, error) {
	resp, err := s.cs.newRequest("deleteBaremetalRct", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r DeleteBaremetalRctResponse
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

type DeleteBaremetalRctResponse struct {
	JobID       string `json:"jobid,omitempty"`
	Displaytext string `json:"displaytext,omitempty"`
	Success     bool   `json:"success,omitempty"`
}

type ListBaremetalRctParams struct {
	p map[string]interface{}
}

func (p *ListBaremetalRctParams) toURLValues() url.Values {
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

func (p *ListBaremetalRctParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListBaremetalRctParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListBaremetalRctParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

// You should always use this function to get a new ListBaremetalRctParams instance,
// as then you are sure you have configured all required params
func (s *BaremetalService) NewListBaremetalRctParams() *ListBaremetalRctParams {
	p := &ListBaremetalRctParams{}
	p.p = make(map[string]interface{})
	return p
}

// list baremetal rack configuration
func (s *BaremetalService) ListBaremetalRct(p *ListBaremetalRctParams) (*ListBaremetalRctResponse, error) {
	resp, err := s.cs.newRequest("listBaremetalRct", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListBaremetalRctResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListBaremetalRctResponse struct {
	Count        int             `json:"count"`
	BaremetalRct []*BaremetalRct `json:"baremetalrct"`
}

type BaremetalRct struct {
	Id  string `json:"id,omitempty"`
	Url string `json:"url,omitempty"`
}
