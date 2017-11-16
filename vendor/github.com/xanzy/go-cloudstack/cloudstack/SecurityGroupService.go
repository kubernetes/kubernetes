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

type CreateSecurityGroupParams struct {
	p map[string]interface{}
}

func (p *CreateSecurityGroupParams) toURLValues() url.Values {
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
	return u
}

func (p *CreateSecurityGroupParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *CreateSecurityGroupParams) SetDescription(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["description"] = v
	return
}

func (p *CreateSecurityGroupParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *CreateSecurityGroupParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *CreateSecurityGroupParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

// You should always use this function to get a new CreateSecurityGroupParams instance,
// as then you are sure you have configured all required params
func (s *SecurityGroupService) NewCreateSecurityGroupParams(name string) *CreateSecurityGroupParams {
	p := &CreateSecurityGroupParams{}
	p.p = make(map[string]interface{})
	p.p["name"] = name
	return p
}

// Creates a security group
func (s *SecurityGroupService) CreateSecurityGroup(p *CreateSecurityGroupParams) (*CreateSecurityGroupResponse, error) {
	resp, err := s.cs.newRequest("createSecurityGroup", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r CreateSecurityGroupResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type CreateSecurityGroupResponse struct {
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
}

type DeleteSecurityGroupParams struct {
	p map[string]interface{}
}

func (p *DeleteSecurityGroupParams) toURLValues() url.Values {
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

func (p *DeleteSecurityGroupParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *DeleteSecurityGroupParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *DeleteSecurityGroupParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *DeleteSecurityGroupParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *DeleteSecurityGroupParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

// You should always use this function to get a new DeleteSecurityGroupParams instance,
// as then you are sure you have configured all required params
func (s *SecurityGroupService) NewDeleteSecurityGroupParams() *DeleteSecurityGroupParams {
	p := &DeleteSecurityGroupParams{}
	p.p = make(map[string]interface{})
	return p
}

// Deletes security group
func (s *SecurityGroupService) DeleteSecurityGroup(p *DeleteSecurityGroupParams) (*DeleteSecurityGroupResponse, error) {
	resp, err := s.cs.newRequest("deleteSecurityGroup", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r DeleteSecurityGroupResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type DeleteSecurityGroupResponse struct {
	Displaytext string `json:"displaytext,omitempty"`
	Success     string `json:"success,omitempty"`
}

type AuthorizeSecurityGroupIngressParams struct {
	p map[string]interface{}
}

func (p *AuthorizeSecurityGroupIngressParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["account"]; found {
		u.Set("account", v.(string))
	}
	if v, found := p.p["cidrlist"]; found {
		vv := strings.Join(v.([]string), ",")
		u.Set("cidrlist", vv)
	}
	if v, found := p.p["domainid"]; found {
		u.Set("domainid", v.(string))
	}
	if v, found := p.p["endport"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("endport", vv)
	}
	if v, found := p.p["icmpcode"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("icmpcode", vv)
	}
	if v, found := p.p["icmptype"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("icmptype", vv)
	}
	if v, found := p.p["projectid"]; found {
		u.Set("projectid", v.(string))
	}
	if v, found := p.p["protocol"]; found {
		u.Set("protocol", v.(string))
	}
	if v, found := p.p["securitygroupid"]; found {
		u.Set("securitygroupid", v.(string))
	}
	if v, found := p.p["securitygroupname"]; found {
		u.Set("securitygroupname", v.(string))
	}
	if v, found := p.p["startport"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("startport", vv)
	}
	if v, found := p.p["usersecuritygrouplist"]; found {
		i := 0
		for k, vv := range v.(map[string]string) {
			u.Set(fmt.Sprintf("usersecuritygrouplist[%d].key", i), k)
			u.Set(fmt.Sprintf("usersecuritygrouplist[%d].value", i), vv)
			i++
		}
	}
	return u
}

func (p *AuthorizeSecurityGroupIngressParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *AuthorizeSecurityGroupIngressParams) SetCidrlist(v []string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["cidrlist"] = v
	return
}

func (p *AuthorizeSecurityGroupIngressParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *AuthorizeSecurityGroupIngressParams) SetEndport(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["endport"] = v
	return
}

func (p *AuthorizeSecurityGroupIngressParams) SetIcmpcode(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["icmpcode"] = v
	return
}

func (p *AuthorizeSecurityGroupIngressParams) SetIcmptype(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["icmptype"] = v
	return
}

func (p *AuthorizeSecurityGroupIngressParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

func (p *AuthorizeSecurityGroupIngressParams) SetProtocol(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["protocol"] = v
	return
}

func (p *AuthorizeSecurityGroupIngressParams) SetSecuritygroupid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["securitygroupid"] = v
	return
}

func (p *AuthorizeSecurityGroupIngressParams) SetSecuritygroupname(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["securitygroupname"] = v
	return
}

func (p *AuthorizeSecurityGroupIngressParams) SetStartport(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["startport"] = v
	return
}

func (p *AuthorizeSecurityGroupIngressParams) SetUsersecuritygrouplist(v map[string]string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["usersecuritygrouplist"] = v
	return
}

// You should always use this function to get a new AuthorizeSecurityGroupIngressParams instance,
// as then you are sure you have configured all required params
func (s *SecurityGroupService) NewAuthorizeSecurityGroupIngressParams() *AuthorizeSecurityGroupIngressParams {
	p := &AuthorizeSecurityGroupIngressParams{}
	p.p = make(map[string]interface{})
	return p
}

// Authorizes a particular ingress rule for this security group
func (s *SecurityGroupService) AuthorizeSecurityGroupIngress(p *AuthorizeSecurityGroupIngressParams) (*AuthorizeSecurityGroupIngressResponse, error) {
	resp, err := s.cs.newRequest("authorizeSecurityGroupIngress", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r AuthorizeSecurityGroupIngressResponse
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

type AuthorizeSecurityGroupIngressResponse struct {
	JobID             string `json:"jobid,omitempty"`
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
}

type RevokeSecurityGroupIngressParams struct {
	p map[string]interface{}
}

func (p *RevokeSecurityGroupIngressParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	return u
}

func (p *RevokeSecurityGroupIngressParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new RevokeSecurityGroupIngressParams instance,
// as then you are sure you have configured all required params
func (s *SecurityGroupService) NewRevokeSecurityGroupIngressParams(id string) *RevokeSecurityGroupIngressParams {
	p := &RevokeSecurityGroupIngressParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Deletes a particular ingress rule from this security group
func (s *SecurityGroupService) RevokeSecurityGroupIngress(p *RevokeSecurityGroupIngressParams) (*RevokeSecurityGroupIngressResponse, error) {
	resp, err := s.cs.newRequest("revokeSecurityGroupIngress", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r RevokeSecurityGroupIngressResponse
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

type RevokeSecurityGroupIngressResponse struct {
	JobID       string `json:"jobid,omitempty"`
	Displaytext string `json:"displaytext,omitempty"`
	Success     bool   `json:"success,omitempty"`
}

type AuthorizeSecurityGroupEgressParams struct {
	p map[string]interface{}
}

func (p *AuthorizeSecurityGroupEgressParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["account"]; found {
		u.Set("account", v.(string))
	}
	if v, found := p.p["cidrlist"]; found {
		vv := strings.Join(v.([]string), ",")
		u.Set("cidrlist", vv)
	}
	if v, found := p.p["domainid"]; found {
		u.Set("domainid", v.(string))
	}
	if v, found := p.p["endport"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("endport", vv)
	}
	if v, found := p.p["icmpcode"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("icmpcode", vv)
	}
	if v, found := p.p["icmptype"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("icmptype", vv)
	}
	if v, found := p.p["projectid"]; found {
		u.Set("projectid", v.(string))
	}
	if v, found := p.p["protocol"]; found {
		u.Set("protocol", v.(string))
	}
	if v, found := p.p["securitygroupid"]; found {
		u.Set("securitygroupid", v.(string))
	}
	if v, found := p.p["securitygroupname"]; found {
		u.Set("securitygroupname", v.(string))
	}
	if v, found := p.p["startport"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("startport", vv)
	}
	if v, found := p.p["usersecuritygrouplist"]; found {
		i := 0
		for k, vv := range v.(map[string]string) {
			u.Set(fmt.Sprintf("usersecuritygrouplist[%d].key", i), k)
			u.Set(fmt.Sprintf("usersecuritygrouplist[%d].value", i), vv)
			i++
		}
	}
	return u
}

func (p *AuthorizeSecurityGroupEgressParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *AuthorizeSecurityGroupEgressParams) SetCidrlist(v []string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["cidrlist"] = v
	return
}

func (p *AuthorizeSecurityGroupEgressParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *AuthorizeSecurityGroupEgressParams) SetEndport(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["endport"] = v
	return
}

func (p *AuthorizeSecurityGroupEgressParams) SetIcmpcode(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["icmpcode"] = v
	return
}

func (p *AuthorizeSecurityGroupEgressParams) SetIcmptype(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["icmptype"] = v
	return
}

func (p *AuthorizeSecurityGroupEgressParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

func (p *AuthorizeSecurityGroupEgressParams) SetProtocol(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["protocol"] = v
	return
}

func (p *AuthorizeSecurityGroupEgressParams) SetSecuritygroupid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["securitygroupid"] = v
	return
}

func (p *AuthorizeSecurityGroupEgressParams) SetSecuritygroupname(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["securitygroupname"] = v
	return
}

func (p *AuthorizeSecurityGroupEgressParams) SetStartport(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["startport"] = v
	return
}

func (p *AuthorizeSecurityGroupEgressParams) SetUsersecuritygrouplist(v map[string]string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["usersecuritygrouplist"] = v
	return
}

// You should always use this function to get a new AuthorizeSecurityGroupEgressParams instance,
// as then you are sure you have configured all required params
func (s *SecurityGroupService) NewAuthorizeSecurityGroupEgressParams() *AuthorizeSecurityGroupEgressParams {
	p := &AuthorizeSecurityGroupEgressParams{}
	p.p = make(map[string]interface{})
	return p
}

// Authorizes a particular egress rule for this security group
func (s *SecurityGroupService) AuthorizeSecurityGroupEgress(p *AuthorizeSecurityGroupEgressParams) (*AuthorizeSecurityGroupEgressResponse, error) {
	resp, err := s.cs.newRequest("authorizeSecurityGroupEgress", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r AuthorizeSecurityGroupEgressResponse
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

type AuthorizeSecurityGroupEgressResponse struct {
	JobID             string `json:"jobid,omitempty"`
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
}

type RevokeSecurityGroupEgressParams struct {
	p map[string]interface{}
}

func (p *RevokeSecurityGroupEgressParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	return u
}

func (p *RevokeSecurityGroupEgressParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new RevokeSecurityGroupEgressParams instance,
// as then you are sure you have configured all required params
func (s *SecurityGroupService) NewRevokeSecurityGroupEgressParams(id string) *RevokeSecurityGroupEgressParams {
	p := &RevokeSecurityGroupEgressParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Deletes a particular egress rule from this security group
func (s *SecurityGroupService) RevokeSecurityGroupEgress(p *RevokeSecurityGroupEgressParams) (*RevokeSecurityGroupEgressResponse, error) {
	resp, err := s.cs.newRequest("revokeSecurityGroupEgress", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r RevokeSecurityGroupEgressResponse
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

type RevokeSecurityGroupEgressResponse struct {
	JobID       string `json:"jobid,omitempty"`
	Displaytext string `json:"displaytext,omitempty"`
	Success     bool   `json:"success,omitempty"`
}

type ListSecurityGroupsParams struct {
	p map[string]interface{}
}

func (p *ListSecurityGroupsParams) toURLValues() url.Values {
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
	if v, found := p.p["securitygroupname"]; found {
		u.Set("securitygroupname", v.(string))
	}
	if v, found := p.p["tags"]; found {
		i := 0
		for k, vv := range v.(map[string]string) {
			u.Set(fmt.Sprintf("tags[%d].key", i), k)
			u.Set(fmt.Sprintf("tags[%d].value", i), vv)
			i++
		}
	}
	if v, found := p.p["virtualmachineid"]; found {
		u.Set("virtualmachineid", v.(string))
	}
	return u
}

func (p *ListSecurityGroupsParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *ListSecurityGroupsParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *ListSecurityGroupsParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ListSecurityGroupsParams) SetIsrecursive(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isrecursive"] = v
	return
}

func (p *ListSecurityGroupsParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListSecurityGroupsParams) SetListall(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["listall"] = v
	return
}

func (p *ListSecurityGroupsParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListSecurityGroupsParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

func (p *ListSecurityGroupsParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

func (p *ListSecurityGroupsParams) SetSecuritygroupname(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["securitygroupname"] = v
	return
}

func (p *ListSecurityGroupsParams) SetTags(v map[string]string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["tags"] = v
	return
}

func (p *ListSecurityGroupsParams) SetVirtualmachineid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["virtualmachineid"] = v
	return
}

// You should always use this function to get a new ListSecurityGroupsParams instance,
// as then you are sure you have configured all required params
func (s *SecurityGroupService) NewListSecurityGroupsParams() *ListSecurityGroupsParams {
	p := &ListSecurityGroupsParams{}
	p.p = make(map[string]interface{})
	return p
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *SecurityGroupService) GetSecurityGroupID(keyword string, opts ...OptionFunc) (string, int, error) {
	p := &ListSecurityGroupsParams{}
	p.p = make(map[string]interface{})

	p.p["keyword"] = keyword

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return "", -1, err
		}
	}

	l, err := s.ListSecurityGroups(p)
	if err != nil {
		return "", -1, err
	}

	if l.Count == 0 {
		return "", l.Count, fmt.Errorf("No match found for %s: %+v", keyword, l)
	}

	if l.Count == 1 {
		return l.SecurityGroups[0].Id, l.Count, nil
	}

	if l.Count > 1 {
		for _, v := range l.SecurityGroups {
			if v.Name == keyword {
				return v.Id, l.Count, nil
			}
		}
	}
	return "", l.Count, fmt.Errorf("Could not find an exact match for %s: %+v", keyword, l)
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *SecurityGroupService) GetSecurityGroupByName(name string, opts ...OptionFunc) (*SecurityGroup, int, error) {
	id, count, err := s.GetSecurityGroupID(name, opts...)
	if err != nil {
		return nil, count, err
	}

	r, count, err := s.GetSecurityGroupByID(id, opts...)
	if err != nil {
		return nil, count, err
	}
	return r, count, nil
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *SecurityGroupService) GetSecurityGroupByID(id string, opts ...OptionFunc) (*SecurityGroup, int, error) {
	p := &ListSecurityGroupsParams{}
	p.p = make(map[string]interface{})

	p.p["id"] = id

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return nil, -1, err
		}
	}

	l, err := s.ListSecurityGroups(p)
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
		return l.SecurityGroups[0], l.Count, nil
	}
	return nil, l.Count, fmt.Errorf("There is more then one result for SecurityGroup UUID: %s!", id)
}

// Lists security groups
func (s *SecurityGroupService) ListSecurityGroups(p *ListSecurityGroupsParams) (*ListSecurityGroupsResponse, error) {
	resp, err := s.cs.newRequest("listSecurityGroups", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListSecurityGroupsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListSecurityGroupsResponse struct {
	Count          int              `json:"count"`
	SecurityGroups []*SecurityGroup `json:"securitygroup"`
}

type SecurityGroup struct {
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
}
