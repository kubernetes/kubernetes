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

type CreateAffinityGroupParams struct {
	p map[string]interface{}
}

func (p *CreateAffinityGroupParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["account"]; found {
		u.Set("account", v.(string))
	}
	if v, found := p.p["description"]; found {
		u.Set("description", v.(string))
	}
	if v, found := p.p["domainid"]; found {
		u.Set("domainid", v.(string))
	}
	if v, found := p.p["name"]; found {
		u.Set("name", v.(string))
	}
	if v, found := p.p["projectid"]; found {
		u.Set("projectid", v.(string))
	}
	if v, found := p.p["type"]; found {
		u.Set("type", v.(string))
	}
	return u
}

func (p *CreateAffinityGroupParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *CreateAffinityGroupParams) SetDescription(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["description"] = v
	return
}

func (p *CreateAffinityGroupParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *CreateAffinityGroupParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *CreateAffinityGroupParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

func (p *CreateAffinityGroupParams) SetType(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["affinityGroupType"] = v
	return
}

// You should always use this function to get a new CreateAffinityGroupParams instance,
// as then you are sure you have configured all required params
func (s *AffinityGroupService) NewCreateAffinityGroupParams(name string, affinityGroupType string) *CreateAffinityGroupParams {
	p := &CreateAffinityGroupParams{}
	p.p = make(map[string]interface{})
	p.p["name"] = name
	p.p["affinityGroupType"] = affinityGroupType
	return p
}

// Creates an affinity/anti-affinity group
func (s *AffinityGroupService) CreateAffinityGroup(p *CreateAffinityGroupParams) (*CreateAffinityGroupResponse, error) {
	resp, err := s.cs.newRequest("createAffinityGroup", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r CreateAffinityGroupResponse
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

type CreateAffinityGroupResponse struct {
	JobID             string   `json:"jobid,omitempty"`
	Account           string   `json:"account,omitempty"`
	Description       string   `json:"description,omitempty"`
	Domain            string   `json:"domain,omitempty"`
	Domainid          string   `json:"domainid,omitempty"`
	Id                string   `json:"id,omitempty"`
	Name              string   `json:"name,omitempty"`
	Project           string   `json:"project,omitempty"`
	Projectid         string   `json:"projectid,omitempty"`
	Type              string   `json:"type,omitempty"`
	VirtualmachineIds []string `json:"virtualmachineIds,omitempty"`
}

type DeleteAffinityGroupParams struct {
	p map[string]interface{}
}

func (p *DeleteAffinityGroupParams) toURLValues() url.Values {
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
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	if v, found := p.p["name"]; found {
		u.Set("name", v.(string))
	}
	if v, found := p.p["projectid"]; found {
		u.Set("projectid", v.(string))
	}
	return u
}

func (p *DeleteAffinityGroupParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *DeleteAffinityGroupParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *DeleteAffinityGroupParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *DeleteAffinityGroupParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *DeleteAffinityGroupParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

// You should always use this function to get a new DeleteAffinityGroupParams instance,
// as then you are sure you have configured all required params
func (s *AffinityGroupService) NewDeleteAffinityGroupParams() *DeleteAffinityGroupParams {
	p := &DeleteAffinityGroupParams{}
	p.p = make(map[string]interface{})
	return p
}

// Deletes affinity group
func (s *AffinityGroupService) DeleteAffinityGroup(p *DeleteAffinityGroupParams) (*DeleteAffinityGroupResponse, error) {
	resp, err := s.cs.newRequest("deleteAffinityGroup", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r DeleteAffinityGroupResponse
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

type DeleteAffinityGroupResponse struct {
	JobID       string `json:"jobid,omitempty"`
	Displaytext string `json:"displaytext,omitempty"`
	Success     bool   `json:"success,omitempty"`
}

type ListAffinityGroupsParams struct {
	p map[string]interface{}
}

func (p *ListAffinityGroupsParams) toURLValues() url.Values {
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
	if v, found := p.p["page"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("page", vv)
	}
	if v, found := p.p["pagesize"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("pagesize", vv)
	}
	if v, found := p.p["projectid"]; found {
		u.Set("projectid", v.(string))
	}
	if v, found := p.p["type"]; found {
		u.Set("type", v.(string))
	}
	if v, found := p.p["virtualmachineid"]; found {
		u.Set("virtualmachineid", v.(string))
	}
	return u
}

func (p *ListAffinityGroupsParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *ListAffinityGroupsParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *ListAffinityGroupsParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ListAffinityGroupsParams) SetIsrecursive(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isrecursive"] = v
	return
}

func (p *ListAffinityGroupsParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListAffinityGroupsParams) SetListall(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["listall"] = v
	return
}

func (p *ListAffinityGroupsParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *ListAffinityGroupsParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListAffinityGroupsParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

func (p *ListAffinityGroupsParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

func (p *ListAffinityGroupsParams) SetType(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["affinityGroupType"] = v
	return
}

func (p *ListAffinityGroupsParams) SetVirtualmachineid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["virtualmachineid"] = v
	return
}

// You should always use this function to get a new ListAffinityGroupsParams instance,
// as then you are sure you have configured all required params
func (s *AffinityGroupService) NewListAffinityGroupsParams() *ListAffinityGroupsParams {
	p := &ListAffinityGroupsParams{}
	p.p = make(map[string]interface{})
	return p
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *AffinityGroupService) GetAffinityGroupID(name string, opts ...OptionFunc) (string, int, error) {
	p := &ListAffinityGroupsParams{}
	p.p = make(map[string]interface{})

	p.p["name"] = name

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return "", -1, err
		}
	}

	l, err := s.ListAffinityGroups(p)
	if err != nil {
		return "", -1, err
	}

	if l.Count == 0 {
		return "", l.Count, fmt.Errorf("No match found for %s: %+v", name, l)
	}

	if l.Count == 1 {
		return l.AffinityGroups[0].Id, l.Count, nil
	}

	if l.Count > 1 {
		for _, v := range l.AffinityGroups {
			if v.Name == name {
				return v.Id, l.Count, nil
			}
		}
	}
	return "", l.Count, fmt.Errorf("Could not find an exact match for %s: %+v", name, l)
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *AffinityGroupService) GetAffinityGroupByName(name string, opts ...OptionFunc) (*AffinityGroup, int, error) {
	id, count, err := s.GetAffinityGroupID(name, opts...)
	if err != nil {
		return nil, count, err
	}

	r, count, err := s.GetAffinityGroupByID(id, opts...)
	if err != nil {
		return nil, count, err
	}
	return r, count, nil
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *AffinityGroupService) GetAffinityGroupByID(id string, opts ...OptionFunc) (*AffinityGroup, int, error) {
	p := &ListAffinityGroupsParams{}
	p.p = make(map[string]interface{})

	p.p["id"] = id

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return nil, -1, err
		}
	}

	l, err := s.ListAffinityGroups(p)
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
		return l.AffinityGroups[0], l.Count, nil
	}
	return nil, l.Count, fmt.Errorf("There is more then one result for AffinityGroup UUID: %s!", id)
}

// Lists affinity groups
func (s *AffinityGroupService) ListAffinityGroups(p *ListAffinityGroupsParams) (*ListAffinityGroupsResponse, error) {
	resp, err := s.cs.newRequest("listAffinityGroups", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListAffinityGroupsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListAffinityGroupsResponse struct {
	Count          int              `json:"count"`
	AffinityGroups []*AffinityGroup `json:"affinitygroup"`
}

type AffinityGroup struct {
	Account           string   `json:"account,omitempty"`
	Description       string   `json:"description,omitempty"`
	Domain            string   `json:"domain,omitempty"`
	Domainid          string   `json:"domainid,omitempty"`
	Id                string   `json:"id,omitempty"`
	Name              string   `json:"name,omitempty"`
	Project           string   `json:"project,omitempty"`
	Projectid         string   `json:"projectid,omitempty"`
	Type              string   `json:"type,omitempty"`
	VirtualmachineIds []string `json:"virtualmachineIds,omitempty"`
}

type UpdateVMAffinityGroupParams struct {
	p map[string]interface{}
}

func (p *UpdateVMAffinityGroupParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["affinitygroupids"]; found {
		vv := strings.Join(v.([]string), ",")
		u.Set("affinitygroupids", vv)
	}
	if v, found := p.p["affinitygroupnames"]; found {
		vv := strings.Join(v.([]string), ",")
		u.Set("affinitygroupnames", vv)
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	return u
}

func (p *UpdateVMAffinityGroupParams) SetAffinitygroupids(v []string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["affinitygroupids"] = v
	return
}

func (p *UpdateVMAffinityGroupParams) SetAffinitygroupnames(v []string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["affinitygroupnames"] = v
	return
}

func (p *UpdateVMAffinityGroupParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new UpdateVMAffinityGroupParams instance,
// as then you are sure you have configured all required params
func (s *AffinityGroupService) NewUpdateVMAffinityGroupParams(id string) *UpdateVMAffinityGroupParams {
	p := &UpdateVMAffinityGroupParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Updates the affinity/anti-affinity group associations of a virtual machine. The VM has to be stopped and restarted for the new properties to take effect.
func (s *AffinityGroupService) UpdateVMAffinityGroup(p *UpdateVMAffinityGroupParams) (*UpdateVMAffinityGroupResponse, error) {
	resp, err := s.cs.newRequest("updateVMAffinityGroup", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r UpdateVMAffinityGroupResponse
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

type UpdateVMAffinityGroupResponse struct {
	JobID         string `json:"jobid,omitempty"`
	Account       string `json:"account,omitempty"`
	Affinitygroup []struct {
		Account           string   `json:"account,omitempty"`
		Description       string   `json:"description,omitempty"`
		Domain            string   `json:"domain,omitempty"`
		Domainid          string   `json:"domainid,omitempty"`
		Id                string   `json:"id,omitempty"`
		Name              string   `json:"name,omitempty"`
		Project           string   `json:"project,omitempty"`
		Projectid         string   `json:"projectid,omitempty"`
		Type              string   `json:"type,omitempty"`
		VirtualmachineIds []string `json:"virtualmachineIds,omitempty"`
	} `json:"affinitygroup,omitempty"`
	Cpunumber             int               `json:"cpunumber,omitempty"`
	Cpuspeed              int               `json:"cpuspeed,omitempty"`
	Cpuused               string            `json:"cpuused,omitempty"`
	Created               string            `json:"created,omitempty"`
	Details               map[string]string `json:"details,omitempty"`
	Diskioread            int64             `json:"diskioread,omitempty"`
	Diskiowrite           int64             `json:"diskiowrite,omitempty"`
	Diskkbsread           int64             `json:"diskkbsread,omitempty"`
	Diskkbswrite          int64             `json:"diskkbswrite,omitempty"`
	Diskofferingid        string            `json:"diskofferingid,omitempty"`
	Diskofferingname      string            `json:"diskofferingname,omitempty"`
	Displayname           string            `json:"displayname,omitempty"`
	Displayvm             bool              `json:"displayvm,omitempty"`
	Domain                string            `json:"domain,omitempty"`
	Domainid              string            `json:"domainid,omitempty"`
	Forvirtualnetwork     bool              `json:"forvirtualnetwork,omitempty"`
	Group                 string            `json:"group,omitempty"`
	Groupid               string            `json:"groupid,omitempty"`
	Guestosid             string            `json:"guestosid,omitempty"`
	Haenable              bool              `json:"haenable,omitempty"`
	Hostid                string            `json:"hostid,omitempty"`
	Hostname              string            `json:"hostname,omitempty"`
	Hypervisor            string            `json:"hypervisor,omitempty"`
	Id                    string            `json:"id,omitempty"`
	Instancename          string            `json:"instancename,omitempty"`
	Isdynamicallyscalable bool              `json:"isdynamicallyscalable,omitempty"`
	Isodisplaytext        string            `json:"isodisplaytext,omitempty"`
	Isoid                 string            `json:"isoid,omitempty"`
	Isoname               string            `json:"isoname,omitempty"`
	Keypair               string            `json:"keypair,omitempty"`
	Memory                int               `json:"memory,omitempty"`
	Name                  string            `json:"name,omitempty"`
	Networkkbsread        int64             `json:"networkkbsread,omitempty"`
	Networkkbswrite       int64             `json:"networkkbswrite,omitempty"`
	Nic                   []struct {
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
	Ostypeid        int64  `json:"ostypeid,omitempty"`
	Password        string `json:"password,omitempty"`
	Passwordenabled bool   `json:"passwordenabled,omitempty"`
	Project         string `json:"project,omitempty"`
	Projectid       string `json:"projectid,omitempty"`
	Publicip        string `json:"publicip,omitempty"`
	Publicipid      string `json:"publicipid,omitempty"`
	Rootdeviceid    int64  `json:"rootdeviceid,omitempty"`
	Rootdevicetype  string `json:"rootdevicetype,omitempty"`
	Securitygroup   []struct {
		Account     string `json:"account,omitempty"`
		Description string `json:"description,omitempty"`
		Domain      string `json:"domain,omitempty"`
		Domainid    string `json:"domainid,omitempty"`
		Egressrule  []struct {
			Account           string `json:"account,omitempty"`
			Cidr              string `json:"cidr,omitempty"`
			Endport           int    `json:"endport,omitempty"`
			Icmpcode          int    `json:"icmpcode,omitempty"`
			Icmptype          int    `json:"icmptype,omitempty"`
			Protocol          string `json:"protocol,omitempty"`
			Ruleid            string `json:"ruleid,omitempty"`
			Securitygroupname string `json:"securitygroupname,omitempty"`
			Startport         int    `json:"startport,omitempty"`
			Tags              []struct {
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
		} `json:"egressrule,omitempty"`
		Id          string `json:"id,omitempty"`
		Ingressrule []struct {
			Account           string `json:"account,omitempty"`
			Cidr              string `json:"cidr,omitempty"`
			Endport           int    `json:"endport,omitempty"`
			Icmpcode          int    `json:"icmpcode,omitempty"`
			Icmptype          int    `json:"icmptype,omitempty"`
			Protocol          string `json:"protocol,omitempty"`
			Ruleid            string `json:"ruleid,omitempty"`
			Securitygroupname string `json:"securitygroupname,omitempty"`
			Startport         int    `json:"startport,omitempty"`
			Tags              []struct {
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
		} `json:"ingressrule,omitempty"`
		Name      string `json:"name,omitempty"`
		Project   string `json:"project,omitempty"`
		Projectid string `json:"projectid,omitempty"`
		Tags      []struct {
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
		Virtualmachinecount int      `json:"virtualmachinecount,omitempty"`
		Virtualmachineids   []string `json:"virtualmachineids,omitempty"`
	} `json:"securitygroup,omitempty"`
	Serviceofferingid   string `json:"serviceofferingid,omitempty"`
	Serviceofferingname string `json:"serviceofferingname,omitempty"`
	Servicestate        string `json:"servicestate,omitempty"`
	State               string `json:"state,omitempty"`
	Tags                []struct {
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
	Templatedisplaytext string `json:"templatedisplaytext,omitempty"`
	Templateid          string `json:"templateid,omitempty"`
	Templatename        string `json:"templatename,omitempty"`
	Userid              string `json:"userid,omitempty"`
	Username            string `json:"username,omitempty"`
	Vgpu                string `json:"vgpu,omitempty"`
	Zoneid              string `json:"zoneid,omitempty"`
	Zonename            string `json:"zonename,omitempty"`
}

type ListAffinityGroupTypesParams struct {
	p map[string]interface{}
}

func (p *ListAffinityGroupTypesParams) toURLValues() url.Values {
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

func (p *ListAffinityGroupTypesParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListAffinityGroupTypesParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListAffinityGroupTypesParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

// You should always use this function to get a new ListAffinityGroupTypesParams instance,
// as then you are sure you have configured all required params
func (s *AffinityGroupService) NewListAffinityGroupTypesParams() *ListAffinityGroupTypesParams {
	p := &ListAffinityGroupTypesParams{}
	p.p = make(map[string]interface{})
	return p
}

// Lists affinity group types available
func (s *AffinityGroupService) ListAffinityGroupTypes(p *ListAffinityGroupTypesParams) (*ListAffinityGroupTypesResponse, error) {
	resp, err := s.cs.newRequest("listAffinityGroupTypes", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListAffinityGroupTypesResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListAffinityGroupTypesResponse struct {
	Count              int                  `json:"count"`
	AffinityGroupTypes []*AffinityGroupType `json:"affinitygrouptype"`
}

type AffinityGroupType struct {
	Type string `json:"type,omitempty"`
}
