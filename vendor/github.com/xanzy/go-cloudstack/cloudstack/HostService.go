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

type AddHostParams struct {
	p map[string]interface{}
}

func (p *AddHostParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["allocationstate"]; found {
		u.Set("allocationstate", v.(string))
	}
	if v, found := p.p["clusterid"]; found {
		u.Set("clusterid", v.(string))
	}
	if v, found := p.p["clustername"]; found {
		u.Set("clustername", v.(string))
	}
	if v, found := p.p["hosttags"]; found {
		vv := strings.Join(v.([]string), ",")
		u.Set("hosttags", vv)
	}
	if v, found := p.p["hypervisor"]; found {
		u.Set("hypervisor", v.(string))
	}
	if v, found := p.p["password"]; found {
		u.Set("password", v.(string))
	}
	if v, found := p.p["podid"]; found {
		u.Set("podid", v.(string))
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

func (p *AddHostParams) SetAllocationstate(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["allocationstate"] = v
	return
}

func (p *AddHostParams) SetClusterid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["clusterid"] = v
	return
}

func (p *AddHostParams) SetClustername(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["clustername"] = v
	return
}

func (p *AddHostParams) SetHosttags(v []string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["hosttags"] = v
	return
}

func (p *AddHostParams) SetHypervisor(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["hypervisor"] = v
	return
}

func (p *AddHostParams) SetPassword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["password"] = v
	return
}

func (p *AddHostParams) SetPodid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["podid"] = v
	return
}

func (p *AddHostParams) SetUrl(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["url"] = v
	return
}

func (p *AddHostParams) SetUsername(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["username"] = v
	return
}

func (p *AddHostParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new AddHostParams instance,
// as then you are sure you have configured all required params
func (s *HostService) NewAddHostParams(hypervisor string, password string, podid string, url string, username string, zoneid string) *AddHostParams {
	p := &AddHostParams{}
	p.p = make(map[string]interface{})
	p.p["hypervisor"] = hypervisor
	p.p["password"] = password
	p.p["podid"] = podid
	p.p["url"] = url
	p.p["username"] = username
	p.p["zoneid"] = zoneid
	return p
}

// Adds a new host.
func (s *HostService) AddHost(p *AddHostParams) (*AddHostResponse, error) {
	resp, err := s.cs.newRequest("addHost", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r AddHostResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type AddHostResponse struct {
	Averageload             int64             `json:"averageload,omitempty"`
	Capabilities            string            `json:"capabilities,omitempty"`
	Clusterid               string            `json:"clusterid,omitempty"`
	Clustername             string            `json:"clustername,omitempty"`
	Clustertype             string            `json:"clustertype,omitempty"`
	Cpuallocated            string            `json:"cpuallocated,omitempty"`
	Cpunumber               int               `json:"cpunumber,omitempty"`
	Cpusockets              int               `json:"cpusockets,omitempty"`
	Cpuspeed                int64             `json:"cpuspeed,omitempty"`
	Cpuused                 string            `json:"cpuused,omitempty"`
	Cpuwithoverprovisioning string            `json:"cpuwithoverprovisioning,omitempty"`
	Created                 string            `json:"created,omitempty"`
	Details                 map[string]string `json:"details,omitempty"`
	Disconnected            string            `json:"disconnected,omitempty"`
	Disksizeallocated       int64             `json:"disksizeallocated,omitempty"`
	Disksizetotal           int64             `json:"disksizetotal,omitempty"`
	Events                  string            `json:"events,omitempty"`
	Gpugroup                []struct {
		Gpugroupname string `json:"gpugroupname,omitempty"`
		Vgpu         []struct {
			Maxcapacity       int64  `json:"maxcapacity,omitempty"`
			Maxheads          int64  `json:"maxheads,omitempty"`
			Maxresolutionx    int64  `json:"maxresolutionx,omitempty"`
			Maxresolutiony    int64  `json:"maxresolutiony,omitempty"`
			Maxvgpuperpgpu    int64  `json:"maxvgpuperpgpu,omitempty"`
			Remainingcapacity int64  `json:"remainingcapacity,omitempty"`
			Vgputype          string `json:"vgputype,omitempty"`
			Videoram          int64  `json:"videoram,omitempty"`
		} `json:"vgpu,omitempty"`
	} `json:"gpugroup,omitempty"`
	Hahost               bool   `json:"hahost,omitempty"`
	Hasenoughcapacity    bool   `json:"hasenoughcapacity,omitempty"`
	Hosttags             string `json:"hosttags,omitempty"`
	Hypervisor           string `json:"hypervisor,omitempty"`
	Hypervisorversion    string `json:"hypervisorversion,omitempty"`
	Id                   string `json:"id,omitempty"`
	Ipaddress            string `json:"ipaddress,omitempty"`
	Islocalstorageactive bool   `json:"islocalstorageactive,omitempty"`
	Lastpinged           string `json:"lastpinged,omitempty"`
	Managementserverid   int64  `json:"managementserverid,omitempty"`
	Memoryallocated      int64  `json:"memoryallocated,omitempty"`
	Memorytotal          int64  `json:"memorytotal,omitempty"`
	Memoryused           int64  `json:"memoryused,omitempty"`
	Name                 string `json:"name,omitempty"`
	Networkkbsread       int64  `json:"networkkbsread,omitempty"`
	Networkkbswrite      int64  `json:"networkkbswrite,omitempty"`
	Oscategoryid         string `json:"oscategoryid,omitempty"`
	Oscategoryname       string `json:"oscategoryname,omitempty"`
	Podid                string `json:"podid,omitempty"`
	Podname              string `json:"podname,omitempty"`
	Removed              string `json:"removed,omitempty"`
	Resourcestate        string `json:"resourcestate,omitempty"`
	State                string `json:"state,omitempty"`
	Suitableformigration bool   `json:"suitableformigration,omitempty"`
	Type                 string `json:"type,omitempty"`
	Version              string `json:"version,omitempty"`
	Zoneid               string `json:"zoneid,omitempty"`
	Zonename             string `json:"zonename,omitempty"`
}

type ReconnectHostParams struct {
	p map[string]interface{}
}

func (p *ReconnectHostParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	return u
}

func (p *ReconnectHostParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new ReconnectHostParams instance,
// as then you are sure you have configured all required params
func (s *HostService) NewReconnectHostParams(id string) *ReconnectHostParams {
	p := &ReconnectHostParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Reconnects a host.
func (s *HostService) ReconnectHost(p *ReconnectHostParams) (*ReconnectHostResponse, error) {
	resp, err := s.cs.newRequest("reconnectHost", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ReconnectHostResponse
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

type ReconnectHostResponse struct {
	JobID                   string            `json:"jobid,omitempty"`
	Averageload             int64             `json:"averageload,omitempty"`
	Capabilities            string            `json:"capabilities,omitempty"`
	Clusterid               string            `json:"clusterid,omitempty"`
	Clustername             string            `json:"clustername,omitempty"`
	Clustertype             string            `json:"clustertype,omitempty"`
	Cpuallocated            string            `json:"cpuallocated,omitempty"`
	Cpunumber               int               `json:"cpunumber,omitempty"`
	Cpusockets              int               `json:"cpusockets,omitempty"`
	Cpuspeed                int64             `json:"cpuspeed,omitempty"`
	Cpuused                 string            `json:"cpuused,omitempty"`
	Cpuwithoverprovisioning string            `json:"cpuwithoverprovisioning,omitempty"`
	Created                 string            `json:"created,omitempty"`
	Details                 map[string]string `json:"details,omitempty"`
	Disconnected            string            `json:"disconnected,omitempty"`
	Disksizeallocated       int64             `json:"disksizeallocated,omitempty"`
	Disksizetotal           int64             `json:"disksizetotal,omitempty"`
	Events                  string            `json:"events,omitempty"`
	Gpugroup                []struct {
		Gpugroupname string `json:"gpugroupname,omitempty"`
		Vgpu         []struct {
			Maxcapacity       int64  `json:"maxcapacity,omitempty"`
			Maxheads          int64  `json:"maxheads,omitempty"`
			Maxresolutionx    int64  `json:"maxresolutionx,omitempty"`
			Maxresolutiony    int64  `json:"maxresolutiony,omitempty"`
			Maxvgpuperpgpu    int64  `json:"maxvgpuperpgpu,omitempty"`
			Remainingcapacity int64  `json:"remainingcapacity,omitempty"`
			Vgputype          string `json:"vgputype,omitempty"`
			Videoram          int64  `json:"videoram,omitempty"`
		} `json:"vgpu,omitempty"`
	} `json:"gpugroup,omitempty"`
	Hahost               bool   `json:"hahost,omitempty"`
	Hasenoughcapacity    bool   `json:"hasenoughcapacity,omitempty"`
	Hosttags             string `json:"hosttags,omitempty"`
	Hypervisor           string `json:"hypervisor,omitempty"`
	Hypervisorversion    string `json:"hypervisorversion,omitempty"`
	Id                   string `json:"id,omitempty"`
	Ipaddress            string `json:"ipaddress,omitempty"`
	Islocalstorageactive bool   `json:"islocalstorageactive,omitempty"`
	Lastpinged           string `json:"lastpinged,omitempty"`
	Managementserverid   int64  `json:"managementserverid,omitempty"`
	Memoryallocated      int64  `json:"memoryallocated,omitempty"`
	Memorytotal          int64  `json:"memorytotal,omitempty"`
	Memoryused           int64  `json:"memoryused,omitempty"`
	Name                 string `json:"name,omitempty"`
	Networkkbsread       int64  `json:"networkkbsread,omitempty"`
	Networkkbswrite      int64  `json:"networkkbswrite,omitempty"`
	Oscategoryid         string `json:"oscategoryid,omitempty"`
	Oscategoryname       string `json:"oscategoryname,omitempty"`
	Podid                string `json:"podid,omitempty"`
	Podname              string `json:"podname,omitempty"`
	Removed              string `json:"removed,omitempty"`
	Resourcestate        string `json:"resourcestate,omitempty"`
	State                string `json:"state,omitempty"`
	Suitableformigration bool   `json:"suitableformigration,omitempty"`
	Type                 string `json:"type,omitempty"`
	Version              string `json:"version,omitempty"`
	Zoneid               string `json:"zoneid,omitempty"`
	Zonename             string `json:"zonename,omitempty"`
}

type UpdateHostParams struct {
	p map[string]interface{}
}

func (p *UpdateHostParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["allocationstate"]; found {
		u.Set("allocationstate", v.(string))
	}
	if v, found := p.p["hosttags"]; found {
		vv := strings.Join(v.([]string), ",")
		u.Set("hosttags", vv)
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	if v, found := p.p["oscategoryid"]; found {
		u.Set("oscategoryid", v.(string))
	}
	if v, found := p.p["url"]; found {
		u.Set("url", v.(string))
	}
	return u
}

func (p *UpdateHostParams) SetAllocationstate(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["allocationstate"] = v
	return
}

func (p *UpdateHostParams) SetHosttags(v []string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["hosttags"] = v
	return
}

func (p *UpdateHostParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *UpdateHostParams) SetOscategoryid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["oscategoryid"] = v
	return
}

func (p *UpdateHostParams) SetUrl(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["url"] = v
	return
}

// You should always use this function to get a new UpdateHostParams instance,
// as then you are sure you have configured all required params
func (s *HostService) NewUpdateHostParams(id string) *UpdateHostParams {
	p := &UpdateHostParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Updates a host.
func (s *HostService) UpdateHost(p *UpdateHostParams) (*UpdateHostResponse, error) {
	resp, err := s.cs.newRequest("updateHost", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r UpdateHostResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type UpdateHostResponse struct {
	Averageload             int64             `json:"averageload,omitempty"`
	Capabilities            string            `json:"capabilities,omitempty"`
	Clusterid               string            `json:"clusterid,omitempty"`
	Clustername             string            `json:"clustername,omitempty"`
	Clustertype             string            `json:"clustertype,omitempty"`
	Cpuallocated            string            `json:"cpuallocated,omitempty"`
	Cpunumber               int               `json:"cpunumber,omitempty"`
	Cpusockets              int               `json:"cpusockets,omitempty"`
	Cpuspeed                int64             `json:"cpuspeed,omitempty"`
	Cpuused                 string            `json:"cpuused,omitempty"`
	Cpuwithoverprovisioning string            `json:"cpuwithoverprovisioning,omitempty"`
	Created                 string            `json:"created,omitempty"`
	Details                 map[string]string `json:"details,omitempty"`
	Disconnected            string            `json:"disconnected,omitempty"`
	Disksizeallocated       int64             `json:"disksizeallocated,omitempty"`
	Disksizetotal           int64             `json:"disksizetotal,omitempty"`
	Events                  string            `json:"events,omitempty"`
	Gpugroup                []struct {
		Gpugroupname string `json:"gpugroupname,omitempty"`
		Vgpu         []struct {
			Maxcapacity       int64  `json:"maxcapacity,omitempty"`
			Maxheads          int64  `json:"maxheads,omitempty"`
			Maxresolutionx    int64  `json:"maxresolutionx,omitempty"`
			Maxresolutiony    int64  `json:"maxresolutiony,omitempty"`
			Maxvgpuperpgpu    int64  `json:"maxvgpuperpgpu,omitempty"`
			Remainingcapacity int64  `json:"remainingcapacity,omitempty"`
			Vgputype          string `json:"vgputype,omitempty"`
			Videoram          int64  `json:"videoram,omitempty"`
		} `json:"vgpu,omitempty"`
	} `json:"gpugroup,omitempty"`
	Hahost               bool   `json:"hahost,omitempty"`
	Hasenoughcapacity    bool   `json:"hasenoughcapacity,omitempty"`
	Hosttags             string `json:"hosttags,omitempty"`
	Hypervisor           string `json:"hypervisor,omitempty"`
	Hypervisorversion    string `json:"hypervisorversion,omitempty"`
	Id                   string `json:"id,omitempty"`
	Ipaddress            string `json:"ipaddress,omitempty"`
	Islocalstorageactive bool   `json:"islocalstorageactive,omitempty"`
	Lastpinged           string `json:"lastpinged,omitempty"`
	Managementserverid   int64  `json:"managementserverid,omitempty"`
	Memoryallocated      int64  `json:"memoryallocated,omitempty"`
	Memorytotal          int64  `json:"memorytotal,omitempty"`
	Memoryused           int64  `json:"memoryused,omitempty"`
	Name                 string `json:"name,omitempty"`
	Networkkbsread       int64  `json:"networkkbsread,omitempty"`
	Networkkbswrite      int64  `json:"networkkbswrite,omitempty"`
	Oscategoryid         string `json:"oscategoryid,omitempty"`
	Oscategoryname       string `json:"oscategoryname,omitempty"`
	Podid                string `json:"podid,omitempty"`
	Podname              string `json:"podname,omitempty"`
	Removed              string `json:"removed,omitempty"`
	Resourcestate        string `json:"resourcestate,omitempty"`
	State                string `json:"state,omitempty"`
	Suitableformigration bool   `json:"suitableformigration,omitempty"`
	Type                 string `json:"type,omitempty"`
	Version              string `json:"version,omitempty"`
	Zoneid               string `json:"zoneid,omitempty"`
	Zonename             string `json:"zonename,omitempty"`
}

type DeleteHostParams struct {
	p map[string]interface{}
}

func (p *DeleteHostParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["forced"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("forced", vv)
	}
	if v, found := p.p["forcedestroylocalstorage"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("forcedestroylocalstorage", vv)
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	return u
}

func (p *DeleteHostParams) SetForced(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["forced"] = v
	return
}

func (p *DeleteHostParams) SetForcedestroylocalstorage(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["forcedestroylocalstorage"] = v
	return
}

func (p *DeleteHostParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new DeleteHostParams instance,
// as then you are sure you have configured all required params
func (s *HostService) NewDeleteHostParams(id string) *DeleteHostParams {
	p := &DeleteHostParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Deletes a host.
func (s *HostService) DeleteHost(p *DeleteHostParams) (*DeleteHostResponse, error) {
	resp, err := s.cs.newRequest("deleteHost", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r DeleteHostResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type DeleteHostResponse struct {
	Displaytext string `json:"displaytext,omitempty"`
	Success     string `json:"success,omitempty"`
}

type PrepareHostForMaintenanceParams struct {
	p map[string]interface{}
}

func (p *PrepareHostForMaintenanceParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	return u
}

func (p *PrepareHostForMaintenanceParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new PrepareHostForMaintenanceParams instance,
// as then you are sure you have configured all required params
func (s *HostService) NewPrepareHostForMaintenanceParams(id string) *PrepareHostForMaintenanceParams {
	p := &PrepareHostForMaintenanceParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Prepares a host for maintenance.
func (s *HostService) PrepareHostForMaintenance(p *PrepareHostForMaintenanceParams) (*PrepareHostForMaintenanceResponse, error) {
	resp, err := s.cs.newRequest("prepareHostForMaintenance", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r PrepareHostForMaintenanceResponse
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

type PrepareHostForMaintenanceResponse struct {
	JobID                   string            `json:"jobid,omitempty"`
	Averageload             int64             `json:"averageload,omitempty"`
	Capabilities            string            `json:"capabilities,omitempty"`
	Clusterid               string            `json:"clusterid,omitempty"`
	Clustername             string            `json:"clustername,omitempty"`
	Clustertype             string            `json:"clustertype,omitempty"`
	Cpuallocated            string            `json:"cpuallocated,omitempty"`
	Cpunumber               int               `json:"cpunumber,omitempty"`
	Cpusockets              int               `json:"cpusockets,omitempty"`
	Cpuspeed                int64             `json:"cpuspeed,omitempty"`
	Cpuused                 string            `json:"cpuused,omitempty"`
	Cpuwithoverprovisioning string            `json:"cpuwithoverprovisioning,omitempty"`
	Created                 string            `json:"created,omitempty"`
	Details                 map[string]string `json:"details,omitempty"`
	Disconnected            string            `json:"disconnected,omitempty"`
	Disksizeallocated       int64             `json:"disksizeallocated,omitempty"`
	Disksizetotal           int64             `json:"disksizetotal,omitempty"`
	Events                  string            `json:"events,omitempty"`
	Gpugroup                []struct {
		Gpugroupname string `json:"gpugroupname,omitempty"`
		Vgpu         []struct {
			Maxcapacity       int64  `json:"maxcapacity,omitempty"`
			Maxheads          int64  `json:"maxheads,omitempty"`
			Maxresolutionx    int64  `json:"maxresolutionx,omitempty"`
			Maxresolutiony    int64  `json:"maxresolutiony,omitempty"`
			Maxvgpuperpgpu    int64  `json:"maxvgpuperpgpu,omitempty"`
			Remainingcapacity int64  `json:"remainingcapacity,omitempty"`
			Vgputype          string `json:"vgputype,omitempty"`
			Videoram          int64  `json:"videoram,omitempty"`
		} `json:"vgpu,omitempty"`
	} `json:"gpugroup,omitempty"`
	Hahost               bool   `json:"hahost,omitempty"`
	Hasenoughcapacity    bool   `json:"hasenoughcapacity,omitempty"`
	Hosttags             string `json:"hosttags,omitempty"`
	Hypervisor           string `json:"hypervisor,omitempty"`
	Hypervisorversion    string `json:"hypervisorversion,omitempty"`
	Id                   string `json:"id,omitempty"`
	Ipaddress            string `json:"ipaddress,omitempty"`
	Islocalstorageactive bool   `json:"islocalstorageactive,omitempty"`
	Lastpinged           string `json:"lastpinged,omitempty"`
	Managementserverid   int64  `json:"managementserverid,omitempty"`
	Memoryallocated      int64  `json:"memoryallocated,omitempty"`
	Memorytotal          int64  `json:"memorytotal,omitempty"`
	Memoryused           int64  `json:"memoryused,omitempty"`
	Name                 string `json:"name,omitempty"`
	Networkkbsread       int64  `json:"networkkbsread,omitempty"`
	Networkkbswrite      int64  `json:"networkkbswrite,omitempty"`
	Oscategoryid         string `json:"oscategoryid,omitempty"`
	Oscategoryname       string `json:"oscategoryname,omitempty"`
	Podid                string `json:"podid,omitempty"`
	Podname              string `json:"podname,omitempty"`
	Removed              string `json:"removed,omitempty"`
	Resourcestate        string `json:"resourcestate,omitempty"`
	State                string `json:"state,omitempty"`
	Suitableformigration bool   `json:"suitableformigration,omitempty"`
	Type                 string `json:"type,omitempty"`
	Version              string `json:"version,omitempty"`
	Zoneid               string `json:"zoneid,omitempty"`
	Zonename             string `json:"zonename,omitempty"`
}

type CancelHostMaintenanceParams struct {
	p map[string]interface{}
}

func (p *CancelHostMaintenanceParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	return u
}

func (p *CancelHostMaintenanceParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new CancelHostMaintenanceParams instance,
// as then you are sure you have configured all required params
func (s *HostService) NewCancelHostMaintenanceParams(id string) *CancelHostMaintenanceParams {
	p := &CancelHostMaintenanceParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Cancels host maintenance.
func (s *HostService) CancelHostMaintenance(p *CancelHostMaintenanceParams) (*CancelHostMaintenanceResponse, error) {
	resp, err := s.cs.newRequest("cancelHostMaintenance", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r CancelHostMaintenanceResponse
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

type CancelHostMaintenanceResponse struct {
	JobID                   string            `json:"jobid,omitempty"`
	Averageload             int64             `json:"averageload,omitempty"`
	Capabilities            string            `json:"capabilities,omitempty"`
	Clusterid               string            `json:"clusterid,omitempty"`
	Clustername             string            `json:"clustername,omitempty"`
	Clustertype             string            `json:"clustertype,omitempty"`
	Cpuallocated            string            `json:"cpuallocated,omitempty"`
	Cpunumber               int               `json:"cpunumber,omitempty"`
	Cpusockets              int               `json:"cpusockets,omitempty"`
	Cpuspeed                int64             `json:"cpuspeed,omitempty"`
	Cpuused                 string            `json:"cpuused,omitempty"`
	Cpuwithoverprovisioning string            `json:"cpuwithoverprovisioning,omitempty"`
	Created                 string            `json:"created,omitempty"`
	Details                 map[string]string `json:"details,omitempty"`
	Disconnected            string            `json:"disconnected,omitempty"`
	Disksizeallocated       int64             `json:"disksizeallocated,omitempty"`
	Disksizetotal           int64             `json:"disksizetotal,omitempty"`
	Events                  string            `json:"events,omitempty"`
	Gpugroup                []struct {
		Gpugroupname string `json:"gpugroupname,omitempty"`
		Vgpu         []struct {
			Maxcapacity       int64  `json:"maxcapacity,omitempty"`
			Maxheads          int64  `json:"maxheads,omitempty"`
			Maxresolutionx    int64  `json:"maxresolutionx,omitempty"`
			Maxresolutiony    int64  `json:"maxresolutiony,omitempty"`
			Maxvgpuperpgpu    int64  `json:"maxvgpuperpgpu,omitempty"`
			Remainingcapacity int64  `json:"remainingcapacity,omitempty"`
			Vgputype          string `json:"vgputype,omitempty"`
			Videoram          int64  `json:"videoram,omitempty"`
		} `json:"vgpu,omitempty"`
	} `json:"gpugroup,omitempty"`
	Hahost               bool   `json:"hahost,omitempty"`
	Hasenoughcapacity    bool   `json:"hasenoughcapacity,omitempty"`
	Hosttags             string `json:"hosttags,omitempty"`
	Hypervisor           string `json:"hypervisor,omitempty"`
	Hypervisorversion    string `json:"hypervisorversion,omitempty"`
	Id                   string `json:"id,omitempty"`
	Ipaddress            string `json:"ipaddress,omitempty"`
	Islocalstorageactive bool   `json:"islocalstorageactive,omitempty"`
	Lastpinged           string `json:"lastpinged,omitempty"`
	Managementserverid   int64  `json:"managementserverid,omitempty"`
	Memoryallocated      int64  `json:"memoryallocated,omitempty"`
	Memorytotal          int64  `json:"memorytotal,omitempty"`
	Memoryused           int64  `json:"memoryused,omitempty"`
	Name                 string `json:"name,omitempty"`
	Networkkbsread       int64  `json:"networkkbsread,omitempty"`
	Networkkbswrite      int64  `json:"networkkbswrite,omitempty"`
	Oscategoryid         string `json:"oscategoryid,omitempty"`
	Oscategoryname       string `json:"oscategoryname,omitempty"`
	Podid                string `json:"podid,omitempty"`
	Podname              string `json:"podname,omitempty"`
	Removed              string `json:"removed,omitempty"`
	Resourcestate        string `json:"resourcestate,omitempty"`
	State                string `json:"state,omitempty"`
	Suitableformigration bool   `json:"suitableformigration,omitempty"`
	Type                 string `json:"type,omitempty"`
	Version              string `json:"version,omitempty"`
	Zoneid               string `json:"zoneid,omitempty"`
	Zonename             string `json:"zonename,omitempty"`
}

type ListHostsParams struct {
	p map[string]interface{}
}

func (p *ListHostsParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["clusterid"]; found {
		u.Set("clusterid", v.(string))
	}
	if v, found := p.p["details"]; found {
		vv := strings.Join(v.([]string), ",")
		u.Set("details", vv)
	}
	if v, found := p.p["hahost"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("hahost", vv)
	}
	if v, found := p.p["hypervisor"]; found {
		u.Set("hypervisor", v.(string))
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	if v, found := p.p["keyword"]; found {
		u.Set("keyword", v.(string))
	}
	if v, found := p.p["name"]; found {
		u.Set("name", v.(string))
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
	if v, found := p.p["resourcestate"]; found {
		u.Set("resourcestate", v.(string))
	}
	if v, found := p.p["state"]; found {
		u.Set("state", v.(string))
	}
	if v, found := p.p["type"]; found {
		u.Set("type", v.(string))
	}
	if v, found := p.p["virtualmachineid"]; found {
		u.Set("virtualmachineid", v.(string))
	}
	if v, found := p.p["zoneid"]; found {
		u.Set("zoneid", v.(string))
	}
	return u
}

func (p *ListHostsParams) SetClusterid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["clusterid"] = v
	return
}

func (p *ListHostsParams) SetDetails(v []string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["details"] = v
	return
}

func (p *ListHostsParams) SetHahost(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["hahost"] = v
	return
}

func (p *ListHostsParams) SetHypervisor(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["hypervisor"] = v
	return
}

func (p *ListHostsParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ListHostsParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListHostsParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *ListHostsParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListHostsParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

func (p *ListHostsParams) SetPodid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["podid"] = v
	return
}

func (p *ListHostsParams) SetResourcestate(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["resourcestate"] = v
	return
}

func (p *ListHostsParams) SetState(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["state"] = v
	return
}

func (p *ListHostsParams) SetType(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["hostType"] = v
	return
}

func (p *ListHostsParams) SetVirtualmachineid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["virtualmachineid"] = v
	return
}

func (p *ListHostsParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new ListHostsParams instance,
// as then you are sure you have configured all required params
func (s *HostService) NewListHostsParams() *ListHostsParams {
	p := &ListHostsParams{}
	p.p = make(map[string]interface{})
	return p
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *HostService) GetHostID(name string, opts ...OptionFunc) (string, int, error) {
	p := &ListHostsParams{}
	p.p = make(map[string]interface{})

	p.p["name"] = name

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return "", -1, err
		}
	}

	l, err := s.ListHosts(p)
	if err != nil {
		return "", -1, err
	}

	if l.Count == 0 {
		return "", l.Count, fmt.Errorf("No match found for %s: %+v", name, l)
	}

	if l.Count == 1 {
		return l.Hosts[0].Id, l.Count, nil
	}

	if l.Count > 1 {
		for _, v := range l.Hosts {
			if v.Name == name {
				return v.Id, l.Count, nil
			}
		}
	}
	return "", l.Count, fmt.Errorf("Could not find an exact match for %s: %+v", name, l)
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *HostService) GetHostByName(name string, opts ...OptionFunc) (*Host, int, error) {
	id, count, err := s.GetHostID(name, opts...)
	if err != nil {
		return nil, count, err
	}

	r, count, err := s.GetHostByID(id, opts...)
	if err != nil {
		return nil, count, err
	}
	return r, count, nil
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *HostService) GetHostByID(id string, opts ...OptionFunc) (*Host, int, error) {
	p := &ListHostsParams{}
	p.p = make(map[string]interface{})

	p.p["id"] = id

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return nil, -1, err
		}
	}

	l, err := s.ListHosts(p)
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
		return l.Hosts[0], l.Count, nil
	}
	return nil, l.Count, fmt.Errorf("There is more then one result for Host UUID: %s!", id)
}

// Lists hosts.
func (s *HostService) ListHosts(p *ListHostsParams) (*ListHostsResponse, error) {
	resp, err := s.cs.newRequest("listHosts", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListHostsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListHostsResponse struct {
	Count int     `json:"count"`
	Hosts []*Host `json:"host"`
}

type Host struct {
	Averageload             int64             `json:"averageload,omitempty"`
	Capabilities            string            `json:"capabilities,omitempty"`
	Clusterid               string            `json:"clusterid,omitempty"`
	Clustername             string            `json:"clustername,omitempty"`
	Clustertype             string            `json:"clustertype,omitempty"`
	Cpuallocated            string            `json:"cpuallocated,omitempty"`
	Cpunumber               int               `json:"cpunumber,omitempty"`
	Cpusockets              int               `json:"cpusockets,omitempty"`
	Cpuspeed                int64             `json:"cpuspeed,omitempty"`
	Cpuused                 string            `json:"cpuused,omitempty"`
	Cpuwithoverprovisioning string            `json:"cpuwithoverprovisioning,omitempty"`
	Created                 string            `json:"created,omitempty"`
	Details                 map[string]string `json:"details,omitempty"`
	Disconnected            string            `json:"disconnected,omitempty"`
	Disksizeallocated       int64             `json:"disksizeallocated,omitempty"`
	Disksizetotal           int64             `json:"disksizetotal,omitempty"`
	Events                  string            `json:"events,omitempty"`
	Gpugroup                []struct {
		Gpugroupname string `json:"gpugroupname,omitempty"`
		Vgpu         []struct {
			Maxcapacity       int64  `json:"maxcapacity,omitempty"`
			Maxheads          int64  `json:"maxheads,omitempty"`
			Maxresolutionx    int64  `json:"maxresolutionx,omitempty"`
			Maxresolutiony    int64  `json:"maxresolutiony,omitempty"`
			Maxvgpuperpgpu    int64  `json:"maxvgpuperpgpu,omitempty"`
			Remainingcapacity int64  `json:"remainingcapacity,omitempty"`
			Vgputype          string `json:"vgputype,omitempty"`
			Videoram          int64  `json:"videoram,omitempty"`
		} `json:"vgpu,omitempty"`
	} `json:"gpugroup,omitempty"`
	Hahost               bool   `json:"hahost,omitempty"`
	Hasenoughcapacity    bool   `json:"hasenoughcapacity,omitempty"`
	Hosttags             string `json:"hosttags,omitempty"`
	Hypervisor           string `json:"hypervisor,omitempty"`
	Hypervisorversion    string `json:"hypervisorversion,omitempty"`
	Id                   string `json:"id,omitempty"`
	Ipaddress            string `json:"ipaddress,omitempty"`
	Islocalstorageactive bool   `json:"islocalstorageactive,omitempty"`
	Lastpinged           string `json:"lastpinged,omitempty"`
	Managementserverid   int64  `json:"managementserverid,omitempty"`
	Memoryallocated      int64  `json:"memoryallocated,omitempty"`
	Memorytotal          int64  `json:"memorytotal,omitempty"`
	Memoryused           int64  `json:"memoryused,omitempty"`
	Name                 string `json:"name,omitempty"`
	Networkkbsread       int64  `json:"networkkbsread,omitempty"`
	Networkkbswrite      int64  `json:"networkkbswrite,omitempty"`
	Oscategoryid         string `json:"oscategoryid,omitempty"`
	Oscategoryname       string `json:"oscategoryname,omitempty"`
	Podid                string `json:"podid,omitempty"`
	Podname              string `json:"podname,omitempty"`
	Removed              string `json:"removed,omitempty"`
	Resourcestate        string `json:"resourcestate,omitempty"`
	State                string `json:"state,omitempty"`
	Suitableformigration bool   `json:"suitableformigration,omitempty"`
	Type                 string `json:"type,omitempty"`
	Version              string `json:"version,omitempty"`
	Zoneid               string `json:"zoneid,omitempty"`
	Zonename             string `json:"zonename,omitempty"`
}

type ListHostTagsParams struct {
	p map[string]interface{}
}

func (p *ListHostTagsParams) toURLValues() url.Values {
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

func (p *ListHostTagsParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListHostTagsParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListHostTagsParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

// You should always use this function to get a new ListHostTagsParams instance,
// as then you are sure you have configured all required params
func (s *HostService) NewListHostTagsParams() *ListHostTagsParams {
	p := &ListHostTagsParams{}
	p.p = make(map[string]interface{})
	return p
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *HostService) GetHostTagID(keyword string, opts ...OptionFunc) (string, int, error) {
	p := &ListHostTagsParams{}
	p.p = make(map[string]interface{})

	p.p["keyword"] = keyword

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return "", -1, err
		}
	}

	l, err := s.ListHostTags(p)
	if err != nil {
		return "", -1, err
	}

	if l.Count == 0 {
		return "", l.Count, fmt.Errorf("No match found for %s: %+v", keyword, l)
	}

	if l.Count == 1 {
		return l.HostTags[0].Id, l.Count, nil
	}

	if l.Count > 1 {
		for _, v := range l.HostTags {
			if v.Name == keyword {
				return v.Id, l.Count, nil
			}
		}
	}
	return "", l.Count, fmt.Errorf("Could not find an exact match for %s: %+v", keyword, l)
}

// Lists host tags
func (s *HostService) ListHostTags(p *ListHostTagsParams) (*ListHostTagsResponse, error) {
	resp, err := s.cs.newRequest("listHostTags", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListHostTagsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListHostTagsResponse struct {
	Count    int        `json:"count"`
	HostTags []*HostTag `json:"hosttag"`
}

type HostTag struct {
	Hostid int64  `json:"hostid,omitempty"`
	Id     string `json:"id,omitempty"`
	Name   string `json:"name,omitempty"`
}

type FindHostsForMigrationParams struct {
	p map[string]interface{}
}

func (p *FindHostsForMigrationParams) toURLValues() url.Values {
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
	if v, found := p.p["virtualmachineid"]; found {
		u.Set("virtualmachineid", v.(string))
	}
	return u
}

func (p *FindHostsForMigrationParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *FindHostsForMigrationParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *FindHostsForMigrationParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

func (p *FindHostsForMigrationParams) SetVirtualmachineid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["virtualmachineid"] = v
	return
}

// You should always use this function to get a new FindHostsForMigrationParams instance,
// as then you are sure you have configured all required params
func (s *HostService) NewFindHostsForMigrationParams(virtualmachineid string) *FindHostsForMigrationParams {
	p := &FindHostsForMigrationParams{}
	p.p = make(map[string]interface{})
	p.p["virtualmachineid"] = virtualmachineid
	return p
}

// Find hosts suitable for migrating a virtual machine.
func (s *HostService) FindHostsForMigration(p *FindHostsForMigrationParams) (*FindHostsForMigrationResponse, error) {
	resp, err := s.cs.newRequest("findHostsForMigration", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r FindHostsForMigrationResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type FindHostsForMigrationResponse struct {
	Averageload             int64  `json:"averageload,omitempty"`
	Capabilities            string `json:"capabilities,omitempty"`
	Clusterid               string `json:"clusterid,omitempty"`
	Clustername             string `json:"clustername,omitempty"`
	Clustertype             string `json:"clustertype,omitempty"`
	Cpuallocated            string `json:"cpuallocated,omitempty"`
	Cpunumber               int    `json:"cpunumber,omitempty"`
	Cpuspeed                int64  `json:"cpuspeed,omitempty"`
	Cpuused                 string `json:"cpuused,omitempty"`
	Cpuwithoverprovisioning string `json:"cpuwithoverprovisioning,omitempty"`
	Created                 string `json:"created,omitempty"`
	Disconnected            string `json:"disconnected,omitempty"`
	Disksizeallocated       int64  `json:"disksizeallocated,omitempty"`
	Disksizetotal           int64  `json:"disksizetotal,omitempty"`
	Events                  string `json:"events,omitempty"`
	Hahost                  bool   `json:"hahost,omitempty"`
	Hasenoughcapacity       bool   `json:"hasenoughcapacity,omitempty"`
	Hosttags                string `json:"hosttags,omitempty"`
	Hypervisor              string `json:"hypervisor,omitempty"`
	Hypervisorversion       string `json:"hypervisorversion,omitempty"`
	Id                      string `json:"id,omitempty"`
	Ipaddress               string `json:"ipaddress,omitempty"`
	Islocalstorageactive    bool   `json:"islocalstorageactive,omitempty"`
	Lastpinged              string `json:"lastpinged,omitempty"`
	Managementserverid      int64  `json:"managementserverid,omitempty"`
	Memoryallocated         int64  `json:"memoryallocated,omitempty"`
	Memorytotal             int64  `json:"memorytotal,omitempty"`
	Memoryused              int64  `json:"memoryused,omitempty"`
	Name                    string `json:"name,omitempty"`
	Networkkbsread          int64  `json:"networkkbsread,omitempty"`
	Networkkbswrite         int64  `json:"networkkbswrite,omitempty"`
	Oscategoryid            string `json:"oscategoryid,omitempty"`
	Oscategoryname          string `json:"oscategoryname,omitempty"`
	Podid                   string `json:"podid,omitempty"`
	Podname                 string `json:"podname,omitempty"`
	Removed                 string `json:"removed,omitempty"`
	RequiresStorageMotion   bool   `json:"requiresStorageMotion,omitempty"`
	Resourcestate           string `json:"resourcestate,omitempty"`
	State                   string `json:"state,omitempty"`
	Suitableformigration    bool   `json:"suitableformigration,omitempty"`
	Type                    string `json:"type,omitempty"`
	Version                 string `json:"version,omitempty"`
	Zoneid                  string `json:"zoneid,omitempty"`
	Zonename                string `json:"zonename,omitempty"`
}

type AddSecondaryStorageParams struct {
	p map[string]interface{}
}

func (p *AddSecondaryStorageParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["url"]; found {
		u.Set("url", v.(string))
	}
	if v, found := p.p["zoneid"]; found {
		u.Set("zoneid", v.(string))
	}
	return u
}

func (p *AddSecondaryStorageParams) SetUrl(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["url"] = v
	return
}

func (p *AddSecondaryStorageParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new AddSecondaryStorageParams instance,
// as then you are sure you have configured all required params
func (s *HostService) NewAddSecondaryStorageParams(url string) *AddSecondaryStorageParams {
	p := &AddSecondaryStorageParams{}
	p.p = make(map[string]interface{})
	p.p["url"] = url
	return p
}

// Adds secondary storage.
func (s *HostService) AddSecondaryStorage(p *AddSecondaryStorageParams) (*AddSecondaryStorageResponse, error) {
	resp, err := s.cs.newRequest("addSecondaryStorage", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r AddSecondaryStorageResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type AddSecondaryStorageResponse struct {
	Details      []string `json:"details,omitempty"`
	Id           string   `json:"id,omitempty"`
	Name         string   `json:"name,omitempty"`
	Protocol     string   `json:"protocol,omitempty"`
	Providername string   `json:"providername,omitempty"`
	Scope        string   `json:"scope,omitempty"`
	Url          string   `json:"url,omitempty"`
	Zoneid       string   `json:"zoneid,omitempty"`
	Zonename     string   `json:"zonename,omitempty"`
}

type UpdateHostPasswordParams struct {
	p map[string]interface{}
}

func (p *UpdateHostPasswordParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["clusterid"]; found {
		u.Set("clusterid", v.(string))
	}
	if v, found := p.p["hostid"]; found {
		u.Set("hostid", v.(string))
	}
	if v, found := p.p["password"]; found {
		u.Set("password", v.(string))
	}
	if v, found := p.p["update_passwd_on_host"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("update_passwd_on_host", vv)
	}
	if v, found := p.p["username"]; found {
		u.Set("username", v.(string))
	}
	return u
}

func (p *UpdateHostPasswordParams) SetClusterid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["clusterid"] = v
	return
}

func (p *UpdateHostPasswordParams) SetHostid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["hostid"] = v
	return
}

func (p *UpdateHostPasswordParams) SetPassword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["password"] = v
	return
}

func (p *UpdateHostPasswordParams) SetUpdate_passwd_on_host(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["update_passwd_on_host"] = v
	return
}

func (p *UpdateHostPasswordParams) SetUsername(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["username"] = v
	return
}

// You should always use this function to get a new UpdateHostPasswordParams instance,
// as then you are sure you have configured all required params
func (s *HostService) NewUpdateHostPasswordParams(password string, username string) *UpdateHostPasswordParams {
	p := &UpdateHostPasswordParams{}
	p.p = make(map[string]interface{})
	p.p["password"] = password
	p.p["username"] = username
	return p
}

// Update password of a host/pool on management server.
func (s *HostService) UpdateHostPassword(p *UpdateHostPasswordParams) (*UpdateHostPasswordResponse, error) {
	resp, err := s.cs.newRequest("updateHostPassword", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r UpdateHostPasswordResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type UpdateHostPasswordResponse struct {
	Displaytext string `json:"displaytext,omitempty"`
	Success     string `json:"success,omitempty"`
}

type ReleaseHostReservationParams struct {
	p map[string]interface{}
}

func (p *ReleaseHostReservationParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	return u
}

func (p *ReleaseHostReservationParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new ReleaseHostReservationParams instance,
// as then you are sure you have configured all required params
func (s *HostService) NewReleaseHostReservationParams(id string) *ReleaseHostReservationParams {
	p := &ReleaseHostReservationParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Releases host reservation.
func (s *HostService) ReleaseHostReservation(p *ReleaseHostReservationParams) (*ReleaseHostReservationResponse, error) {
	resp, err := s.cs.newRequest("releaseHostReservation", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ReleaseHostReservationResponse
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

type ReleaseHostReservationResponse struct {
	JobID       string `json:"jobid,omitempty"`
	Displaytext string `json:"displaytext,omitempty"`
	Success     bool   `json:"success,omitempty"`
}

type AddBaremetalHostParams struct {
	p map[string]interface{}
}

func (p *AddBaremetalHostParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["allocationstate"]; found {
		u.Set("allocationstate", v.(string))
	}
	if v, found := p.p["clusterid"]; found {
		u.Set("clusterid", v.(string))
	}
	if v, found := p.p["clustername"]; found {
		u.Set("clustername", v.(string))
	}
	if v, found := p.p["hosttags"]; found {
		vv := strings.Join(v.([]string), ",")
		u.Set("hosttags", vv)
	}
	if v, found := p.p["hypervisor"]; found {
		u.Set("hypervisor", v.(string))
	}
	if v, found := p.p["ipaddress"]; found {
		u.Set("ipaddress", v.(string))
	}
	if v, found := p.p["password"]; found {
		u.Set("password", v.(string))
	}
	if v, found := p.p["podid"]; found {
		u.Set("podid", v.(string))
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

func (p *AddBaremetalHostParams) SetAllocationstate(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["allocationstate"] = v
	return
}

func (p *AddBaremetalHostParams) SetClusterid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["clusterid"] = v
	return
}

func (p *AddBaremetalHostParams) SetClustername(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["clustername"] = v
	return
}

func (p *AddBaremetalHostParams) SetHosttags(v []string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["hosttags"] = v
	return
}

func (p *AddBaremetalHostParams) SetHypervisor(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["hypervisor"] = v
	return
}

func (p *AddBaremetalHostParams) SetIpaddress(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["ipaddress"] = v
	return
}

func (p *AddBaremetalHostParams) SetPassword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["password"] = v
	return
}

func (p *AddBaremetalHostParams) SetPodid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["podid"] = v
	return
}

func (p *AddBaremetalHostParams) SetUrl(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["url"] = v
	return
}

func (p *AddBaremetalHostParams) SetUsername(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["username"] = v
	return
}

func (p *AddBaremetalHostParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new AddBaremetalHostParams instance,
// as then you are sure you have configured all required params
func (s *HostService) NewAddBaremetalHostParams(hypervisor string, password string, podid string, url string, username string, zoneid string) *AddBaremetalHostParams {
	p := &AddBaremetalHostParams{}
	p.p = make(map[string]interface{})
	p.p["hypervisor"] = hypervisor
	p.p["password"] = password
	p.p["podid"] = podid
	p.p["url"] = url
	p.p["username"] = username
	p.p["zoneid"] = zoneid
	return p
}

// add a baremetal host
func (s *HostService) AddBaremetalHost(p *AddBaremetalHostParams) (*AddBaremetalHostResponse, error) {
	resp, err := s.cs.newRequest("addBaremetalHost", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r AddBaremetalHostResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type AddBaremetalHostResponse struct {
	Averageload             int64             `json:"averageload,omitempty"`
	Capabilities            string            `json:"capabilities,omitempty"`
	Clusterid               string            `json:"clusterid,omitempty"`
	Clustername             string            `json:"clustername,omitempty"`
	Clustertype             string            `json:"clustertype,omitempty"`
	Cpuallocated            string            `json:"cpuallocated,omitempty"`
	Cpunumber               int               `json:"cpunumber,omitempty"`
	Cpusockets              int               `json:"cpusockets,omitempty"`
	Cpuspeed                int64             `json:"cpuspeed,omitempty"`
	Cpuused                 string            `json:"cpuused,omitempty"`
	Cpuwithoverprovisioning string            `json:"cpuwithoverprovisioning,omitempty"`
	Created                 string            `json:"created,omitempty"`
	Details                 map[string]string `json:"details,omitempty"`
	Disconnected            string            `json:"disconnected,omitempty"`
	Disksizeallocated       int64             `json:"disksizeallocated,omitempty"`
	Disksizetotal           int64             `json:"disksizetotal,omitempty"`
	Events                  string            `json:"events,omitempty"`
	Gpugroup                []struct {
		Gpugroupname string `json:"gpugroupname,omitempty"`
		Vgpu         []struct {
			Maxcapacity       int64  `json:"maxcapacity,omitempty"`
			Maxheads          int64  `json:"maxheads,omitempty"`
			Maxresolutionx    int64  `json:"maxresolutionx,omitempty"`
			Maxresolutiony    int64  `json:"maxresolutiony,omitempty"`
			Maxvgpuperpgpu    int64  `json:"maxvgpuperpgpu,omitempty"`
			Remainingcapacity int64  `json:"remainingcapacity,omitempty"`
			Vgputype          string `json:"vgputype,omitempty"`
			Videoram          int64  `json:"videoram,omitempty"`
		} `json:"vgpu,omitempty"`
	} `json:"gpugroup,omitempty"`
	Hahost               bool   `json:"hahost,omitempty"`
	Hasenoughcapacity    bool   `json:"hasenoughcapacity,omitempty"`
	Hosttags             string `json:"hosttags,omitempty"`
	Hypervisor           string `json:"hypervisor,omitempty"`
	Hypervisorversion    string `json:"hypervisorversion,omitempty"`
	Id                   string `json:"id,omitempty"`
	Ipaddress            string `json:"ipaddress,omitempty"`
	Islocalstorageactive bool   `json:"islocalstorageactive,omitempty"`
	Lastpinged           string `json:"lastpinged,omitempty"`
	Managementserverid   int64  `json:"managementserverid,omitempty"`
	Memoryallocated      int64  `json:"memoryallocated,omitempty"`
	Memorytotal          int64  `json:"memorytotal,omitempty"`
	Memoryused           int64  `json:"memoryused,omitempty"`
	Name                 string `json:"name,omitempty"`
	Networkkbsread       int64  `json:"networkkbsread,omitempty"`
	Networkkbswrite      int64  `json:"networkkbswrite,omitempty"`
	Oscategoryid         string `json:"oscategoryid,omitempty"`
	Oscategoryname       string `json:"oscategoryname,omitempty"`
	Podid                string `json:"podid,omitempty"`
	Podname              string `json:"podname,omitempty"`
	Removed              string `json:"removed,omitempty"`
	Resourcestate        string `json:"resourcestate,omitempty"`
	State                string `json:"state,omitempty"`
	Suitableformigration bool   `json:"suitableformigration,omitempty"`
	Type                 string `json:"type,omitempty"`
	Version              string `json:"version,omitempty"`
	Zoneid               string `json:"zoneid,omitempty"`
	Zonename             string `json:"zonename,omitempty"`
}

type DedicateHostParams struct {
	p map[string]interface{}
}

func (p *DedicateHostParams) toURLValues() url.Values {
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
	if v, found := p.p["hostid"]; found {
		u.Set("hostid", v.(string))
	}
	return u
}

func (p *DedicateHostParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *DedicateHostParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *DedicateHostParams) SetHostid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["hostid"] = v
	return
}

// You should always use this function to get a new DedicateHostParams instance,
// as then you are sure you have configured all required params
func (s *HostService) NewDedicateHostParams(domainid string, hostid string) *DedicateHostParams {
	p := &DedicateHostParams{}
	p.p = make(map[string]interface{})
	p.p["domainid"] = domainid
	p.p["hostid"] = hostid
	return p
}

// Dedicates a host.
func (s *HostService) DedicateHost(p *DedicateHostParams) (*DedicateHostResponse, error) {
	resp, err := s.cs.newRequest("dedicateHost", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r DedicateHostResponse
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

type DedicateHostResponse struct {
	JobID           string `json:"jobid,omitempty"`
	Accountid       string `json:"accountid,omitempty"`
	Affinitygroupid string `json:"affinitygroupid,omitempty"`
	Domainid        string `json:"domainid,omitempty"`
	Hostid          string `json:"hostid,omitempty"`
	Hostname        string `json:"hostname,omitempty"`
	Id              string `json:"id,omitempty"`
}

type ReleaseDedicatedHostParams struct {
	p map[string]interface{}
}

func (p *ReleaseDedicatedHostParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["hostid"]; found {
		u.Set("hostid", v.(string))
	}
	return u
}

func (p *ReleaseDedicatedHostParams) SetHostid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["hostid"] = v
	return
}

// You should always use this function to get a new ReleaseDedicatedHostParams instance,
// as then you are sure you have configured all required params
func (s *HostService) NewReleaseDedicatedHostParams(hostid string) *ReleaseDedicatedHostParams {
	p := &ReleaseDedicatedHostParams{}
	p.p = make(map[string]interface{})
	p.p["hostid"] = hostid
	return p
}

// Release the dedication for host
func (s *HostService) ReleaseDedicatedHost(p *ReleaseDedicatedHostParams) (*ReleaseDedicatedHostResponse, error) {
	resp, err := s.cs.newRequest("releaseDedicatedHost", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ReleaseDedicatedHostResponse
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

type ReleaseDedicatedHostResponse struct {
	JobID       string `json:"jobid,omitempty"`
	Displaytext string `json:"displaytext,omitempty"`
	Success     bool   `json:"success,omitempty"`
}

type ListDedicatedHostsParams struct {
	p map[string]interface{}
}

func (p *ListDedicatedHostsParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["account"]; found {
		u.Set("account", v.(string))
	}
	if v, found := p.p["affinitygroupid"]; found {
		u.Set("affinitygroupid", v.(string))
	}
	if v, found := p.p["domainid"]; found {
		u.Set("domainid", v.(string))
	}
	if v, found := p.p["hostid"]; found {
		u.Set("hostid", v.(string))
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

func (p *ListDedicatedHostsParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *ListDedicatedHostsParams) SetAffinitygroupid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["affinitygroupid"] = v
	return
}

func (p *ListDedicatedHostsParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *ListDedicatedHostsParams) SetHostid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["hostid"] = v
	return
}

func (p *ListDedicatedHostsParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListDedicatedHostsParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListDedicatedHostsParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

// You should always use this function to get a new ListDedicatedHostsParams instance,
// as then you are sure you have configured all required params
func (s *HostService) NewListDedicatedHostsParams() *ListDedicatedHostsParams {
	p := &ListDedicatedHostsParams{}
	p.p = make(map[string]interface{})
	return p
}

// Lists dedicated hosts.
func (s *HostService) ListDedicatedHosts(p *ListDedicatedHostsParams) (*ListDedicatedHostsResponse, error) {
	resp, err := s.cs.newRequest("listDedicatedHosts", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListDedicatedHostsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListDedicatedHostsResponse struct {
	Count          int              `json:"count"`
	DedicatedHosts []*DedicatedHost `json:"dedicatedhost"`
}

type DedicatedHost struct {
	Accountid       string `json:"accountid,omitempty"`
	Affinitygroupid string `json:"affinitygroupid,omitempty"`
	Domainid        string `json:"domainid,omitempty"`
	Hostid          string `json:"hostid,omitempty"`
	Hostname        string `json:"hostname,omitempty"`
	Id              string `json:"id,omitempty"`
}

type AddGloboDnsHostParams struct {
	p map[string]interface{}
}

func (p *AddGloboDnsHostParams) toURLValues() url.Values {
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
	if v, found := p.p["url"]; found {
		u.Set("url", v.(string))
	}
	if v, found := p.p["username"]; found {
		u.Set("username", v.(string))
	}
	return u
}

func (p *AddGloboDnsHostParams) SetPassword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["password"] = v
	return
}

func (p *AddGloboDnsHostParams) SetPhysicalnetworkid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["physicalnetworkid"] = v
	return
}

func (p *AddGloboDnsHostParams) SetUrl(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["url"] = v
	return
}

func (p *AddGloboDnsHostParams) SetUsername(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["username"] = v
	return
}

// You should always use this function to get a new AddGloboDnsHostParams instance,
// as then you are sure you have configured all required params
func (s *HostService) NewAddGloboDnsHostParams(password string, physicalnetworkid string, url string, username string) *AddGloboDnsHostParams {
	p := &AddGloboDnsHostParams{}
	p.p = make(map[string]interface{})
	p.p["password"] = password
	p.p["physicalnetworkid"] = physicalnetworkid
	p.p["url"] = url
	p.p["username"] = username
	return p
}

// Adds the GloboDNS external host
func (s *HostService) AddGloboDnsHost(p *AddGloboDnsHostParams) (*AddGloboDnsHostResponse, error) {
	resp, err := s.cs.newRequest("addGloboDnsHost", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r AddGloboDnsHostResponse
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

type AddGloboDnsHostResponse struct {
	JobID       string `json:"jobid,omitempty"`
	Displaytext string `json:"displaytext,omitempty"`
	Success     bool   `json:"success,omitempty"`
}
