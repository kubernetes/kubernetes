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

type UpdateConfigurationParams struct {
	p map[string]interface{}
}

func (p *UpdateConfigurationParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["accountid"]; found {
		u.Set("accountid", v.(string))
	}
	if v, found := p.p["clusterid"]; found {
		u.Set("clusterid", v.(string))
	}
	if v, found := p.p["name"]; found {
		u.Set("name", v.(string))
	}
	if v, found := p.p["storageid"]; found {
		u.Set("storageid", v.(string))
	}
	if v, found := p.p["value"]; found {
		u.Set("value", v.(string))
	}
	if v, found := p.p["zoneid"]; found {
		u.Set("zoneid", v.(string))
	}
	return u
}

func (p *UpdateConfigurationParams) SetAccountid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["accountid"] = v
	return
}

func (p *UpdateConfigurationParams) SetClusterid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["clusterid"] = v
	return
}

func (p *UpdateConfigurationParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *UpdateConfigurationParams) SetStorageid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["storageid"] = v
	return
}

func (p *UpdateConfigurationParams) SetValue(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["value"] = v
	return
}

func (p *UpdateConfigurationParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new UpdateConfigurationParams instance,
// as then you are sure you have configured all required params
func (s *ConfigurationService) NewUpdateConfigurationParams(name string) *UpdateConfigurationParams {
	p := &UpdateConfigurationParams{}
	p.p = make(map[string]interface{})
	p.p["name"] = name
	return p
}

// Updates a configuration.
func (s *ConfigurationService) UpdateConfiguration(p *UpdateConfigurationParams) (*UpdateConfigurationResponse, error) {
	resp, err := s.cs.newRequest("updateConfiguration", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r UpdateConfigurationResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type UpdateConfigurationResponse struct {
	Category    string `json:"category,omitempty"`
	Description string `json:"description,omitempty"`
	Id          int64  `json:"id,omitempty"`
	Name        string `json:"name,omitempty"`
	Scope       string `json:"scope,omitempty"`
	Value       string `json:"value,omitempty"`
}

type ListConfigurationsParams struct {
	p map[string]interface{}
}

func (p *ListConfigurationsParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["accountid"]; found {
		u.Set("accountid", v.(string))
	}
	if v, found := p.p["category"]; found {
		u.Set("category", v.(string))
	}
	if v, found := p.p["clusterid"]; found {
		u.Set("clusterid", v.(string))
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
	if v, found := p.p["storageid"]; found {
		u.Set("storageid", v.(string))
	}
	if v, found := p.p["zoneid"]; found {
		u.Set("zoneid", v.(string))
	}
	return u
}

func (p *ListConfigurationsParams) SetAccountid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["accountid"] = v
	return
}

func (p *ListConfigurationsParams) SetCategory(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["category"] = v
	return
}

func (p *ListConfigurationsParams) SetClusterid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["clusterid"] = v
	return
}

func (p *ListConfigurationsParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListConfigurationsParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *ListConfigurationsParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListConfigurationsParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

func (p *ListConfigurationsParams) SetStorageid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["storageid"] = v
	return
}

func (p *ListConfigurationsParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new ListConfigurationsParams instance,
// as then you are sure you have configured all required params
func (s *ConfigurationService) NewListConfigurationsParams() *ListConfigurationsParams {
	p := &ListConfigurationsParams{}
	p.p = make(map[string]interface{})
	return p
}

// Lists all configurations.
func (s *ConfigurationService) ListConfigurations(p *ListConfigurationsParams) (*ListConfigurationsResponse, error) {
	resp, err := s.cs.newRequest("listConfigurations", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListConfigurationsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListConfigurationsResponse struct {
	Count          int              `json:"count"`
	Configurations []*Configuration `json:"configuration"`
}

type Configuration struct {
	Category    string `json:"category,omitempty"`
	Description string `json:"description,omitempty"`
	Id          int64  `json:"id,omitempty"`
	Name        string `json:"name,omitempty"`
	Scope       string `json:"scope,omitempty"`
	Value       string `json:"value,omitempty"`
}

type ListCapabilitiesParams struct {
	p map[string]interface{}
}

func (p *ListCapabilitiesParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	return u
}

// You should always use this function to get a new ListCapabilitiesParams instance,
// as then you are sure you have configured all required params
func (s *ConfigurationService) NewListCapabilitiesParams() *ListCapabilitiesParams {
	p := &ListCapabilitiesParams{}
	p.p = make(map[string]interface{})
	return p
}

// Lists capabilities
func (s *ConfigurationService) ListCapabilities(p *ListCapabilitiesParams) (*ListCapabilitiesResponse, error) {
	resp, err := s.cs.newRequest("listCapabilities", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListCapabilitiesResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListCapabilitiesResponse struct {
	Count        int           `json:"count"`
	Capabilities []*Capability `json:"capability"`
}

type Capability struct {
	Allowusercreateprojects   bool   `json:"allowusercreateprojects,omitempty"`
	Allowuserexpungerecovervm bool   `json:"allowuserexpungerecovervm,omitempty"`
	Allowuserviewdestroyedvm  bool   `json:"allowuserviewdestroyedvm,omitempty"`
	Apilimitinterval          int    `json:"apilimitinterval,omitempty"`
	Apilimitmax               int    `json:"apilimitmax,omitempty"`
	Cloudstackversion         string `json:"cloudstackversion,omitempty"`
	Customdiskofferingmaxsize int64  `json:"customdiskofferingmaxsize,omitempty"`
	Customdiskofferingminsize int64  `json:"customdiskofferingminsize,omitempty"`
	Kvmsnapshotenabled        bool   `json:"kvmsnapshotenabled,omitempty"`
	Projectinviterequired     bool   `json:"projectinviterequired,omitempty"`
	Regionsecondaryenabled    bool   `json:"regionsecondaryenabled,omitempty"`
	Securitygroupsenabled     bool   `json:"securitygroupsenabled,omitempty"`
	SupportELB                string `json:"supportELB,omitempty"`
	Userpublictemplateenabled bool   `json:"userpublictemplateenabled,omitempty"`
}

type ListDeploymentPlannersParams struct {
	p map[string]interface{}
}

func (p *ListDeploymentPlannersParams) toURLValues() url.Values {
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

func (p *ListDeploymentPlannersParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListDeploymentPlannersParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListDeploymentPlannersParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

// You should always use this function to get a new ListDeploymentPlannersParams instance,
// as then you are sure you have configured all required params
func (s *ConfigurationService) NewListDeploymentPlannersParams() *ListDeploymentPlannersParams {
	p := &ListDeploymentPlannersParams{}
	p.p = make(map[string]interface{})
	return p
}

// Lists all DeploymentPlanners available.
func (s *ConfigurationService) ListDeploymentPlanners(p *ListDeploymentPlannersParams) (*ListDeploymentPlannersResponse, error) {
	resp, err := s.cs.newRequest("listDeploymentPlanners", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListDeploymentPlannersResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListDeploymentPlannersResponse struct {
	Count              int                  `json:"count"`
	DeploymentPlanners []*DeploymentPlanner `json:"deploymentplanner"`
}

type DeploymentPlanner struct {
	Name string `json:"name,omitempty"`
}

type ListLdapConfigurationsParams struct {
	p map[string]interface{}
}

func (p *ListLdapConfigurationsParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["hostname"]; found {
		u.Set("hostname", v.(string))
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
	if v, found := p.p["port"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("port", vv)
	}
	return u
}

func (p *ListLdapConfigurationsParams) SetHostname(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["hostname"] = v
	return
}

func (p *ListLdapConfigurationsParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListLdapConfigurationsParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListLdapConfigurationsParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

func (p *ListLdapConfigurationsParams) SetPort(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["port"] = v
	return
}

// You should always use this function to get a new ListLdapConfigurationsParams instance,
// as then you are sure you have configured all required params
func (s *ConfigurationService) NewListLdapConfigurationsParams() *ListLdapConfigurationsParams {
	p := &ListLdapConfigurationsParams{}
	p.p = make(map[string]interface{})
	return p
}

// Lists all LDAP configurations
func (s *ConfigurationService) ListLdapConfigurations(p *ListLdapConfigurationsParams) (*ListLdapConfigurationsResponse, error) {
	resp, err := s.cs.newRequest("listLdapConfigurations", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListLdapConfigurationsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListLdapConfigurationsResponse struct {
	Count              int                  `json:"count"`
	LdapConfigurations []*LdapConfiguration `json:"ldapconfiguration"`
}

type LdapConfiguration struct {
	Hostname string `json:"hostname,omitempty"`
	Port     int    `json:"port,omitempty"`
}

type AddLdapConfigurationParams struct {
	p map[string]interface{}
}

func (p *AddLdapConfigurationParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["hostname"]; found {
		u.Set("hostname", v.(string))
	}
	if v, found := p.p["port"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("port", vv)
	}
	return u
}

func (p *AddLdapConfigurationParams) SetHostname(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["hostname"] = v
	return
}

func (p *AddLdapConfigurationParams) SetPort(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["port"] = v
	return
}

// You should always use this function to get a new AddLdapConfigurationParams instance,
// as then you are sure you have configured all required params
func (s *ConfigurationService) NewAddLdapConfigurationParams(hostname string, port int) *AddLdapConfigurationParams {
	p := &AddLdapConfigurationParams{}
	p.p = make(map[string]interface{})
	p.p["hostname"] = hostname
	p.p["port"] = port
	return p
}

// Add a new Ldap Configuration
func (s *ConfigurationService) AddLdapConfiguration(p *AddLdapConfigurationParams) (*AddLdapConfigurationResponse, error) {
	resp, err := s.cs.newRequest("addLdapConfiguration", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r AddLdapConfigurationResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type AddLdapConfigurationResponse struct {
	Hostname string `json:"hostname,omitempty"`
	Port     int    `json:"port,omitempty"`
}

type DeleteLdapConfigurationParams struct {
	p map[string]interface{}
}

func (p *DeleteLdapConfigurationParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["hostname"]; found {
		u.Set("hostname", v.(string))
	}
	return u
}

func (p *DeleteLdapConfigurationParams) SetHostname(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["hostname"] = v
	return
}

// You should always use this function to get a new DeleteLdapConfigurationParams instance,
// as then you are sure you have configured all required params
func (s *ConfigurationService) NewDeleteLdapConfigurationParams(hostname string) *DeleteLdapConfigurationParams {
	p := &DeleteLdapConfigurationParams{}
	p.p = make(map[string]interface{})
	p.p["hostname"] = hostname
	return p
}

// Remove an Ldap Configuration
func (s *ConfigurationService) DeleteLdapConfiguration(p *DeleteLdapConfigurationParams) (*DeleteLdapConfigurationResponse, error) {
	resp, err := s.cs.newRequest("deleteLdapConfiguration", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r DeleteLdapConfigurationResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type DeleteLdapConfigurationResponse struct {
	Hostname string `json:"hostname,omitempty"`
	Port     int    `json:"port,omitempty"`
}
