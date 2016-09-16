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

type CreateTemplateParams struct {
	p map[string]interface{}
}

func (p *CreateTemplateParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["bits"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("bits", vv)
	}
	if v, found := p.p["details"]; found {
		i := 0
		for k, vv := range v.(map[string]string) {
			u.Set(fmt.Sprintf("details[%d].key", i), k)
			u.Set(fmt.Sprintf("details[%d].value", i), vv)
			i++
		}
	}
	if v, found := p.p["displaytext"]; found {
		u.Set("displaytext", v.(string))
	}
	if v, found := p.p["isdynamicallyscalable"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("isdynamicallyscalable", vv)
	}
	if v, found := p.p["isfeatured"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("isfeatured", vv)
	}
	if v, found := p.p["ispublic"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("ispublic", vv)
	}
	if v, found := p.p["name"]; found {
		u.Set("name", v.(string))
	}
	if v, found := p.p["ostypeid"]; found {
		u.Set("ostypeid", v.(string))
	}
	if v, found := p.p["passwordenabled"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("passwordenabled", vv)
	}
	if v, found := p.p["projectid"]; found {
		u.Set("projectid", v.(string))
	}
	if v, found := p.p["requireshvm"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("requireshvm", vv)
	}
	if v, found := p.p["snapshotid"]; found {
		u.Set("snapshotid", v.(string))
	}
	if v, found := p.p["templatetag"]; found {
		u.Set("templatetag", v.(string))
	}
	if v, found := p.p["url"]; found {
		u.Set("url", v.(string))
	}
	if v, found := p.p["virtualmachineid"]; found {
		u.Set("virtualmachineid", v.(string))
	}
	if v, found := p.p["volumeid"]; found {
		u.Set("volumeid", v.(string))
	}
	return u
}

func (p *CreateTemplateParams) SetBits(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["bits"] = v
	return
}

func (p *CreateTemplateParams) SetDetails(v map[string]string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["details"] = v
	return
}

func (p *CreateTemplateParams) SetDisplaytext(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["displaytext"] = v
	return
}

func (p *CreateTemplateParams) SetIsdynamicallyscalable(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isdynamicallyscalable"] = v
	return
}

func (p *CreateTemplateParams) SetIsfeatured(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isfeatured"] = v
	return
}

func (p *CreateTemplateParams) SetIspublic(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["ispublic"] = v
	return
}

func (p *CreateTemplateParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *CreateTemplateParams) SetOstypeid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["ostypeid"] = v
	return
}

func (p *CreateTemplateParams) SetPasswordenabled(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["passwordenabled"] = v
	return
}

func (p *CreateTemplateParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

func (p *CreateTemplateParams) SetRequireshvm(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["requireshvm"] = v
	return
}

func (p *CreateTemplateParams) SetSnapshotid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["snapshotid"] = v
	return
}

func (p *CreateTemplateParams) SetTemplatetag(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["templatetag"] = v
	return
}

func (p *CreateTemplateParams) SetUrl(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["url"] = v
	return
}

func (p *CreateTemplateParams) SetVirtualmachineid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["virtualmachineid"] = v
	return
}

func (p *CreateTemplateParams) SetVolumeid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["volumeid"] = v
	return
}

// You should always use this function to get a new CreateTemplateParams instance,
// as then you are sure you have configured all required params
func (s *TemplateService) NewCreateTemplateParams(displaytext string, name string, ostypeid string) *CreateTemplateParams {
	p := &CreateTemplateParams{}
	p.p = make(map[string]interface{})
	p.p["displaytext"] = displaytext
	p.p["name"] = name
	p.p["ostypeid"] = ostypeid
	return p
}

// Creates a template of a virtual machine. The virtual machine must be in a STOPPED state. A template created from this command is automatically designated as a private template visible to the account that created it.
func (s *TemplateService) CreateTemplate(p *CreateTemplateParams) (*CreateTemplateResponse, error) {
	resp, err := s.cs.newRequest("createTemplate", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r CreateTemplateResponse
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

type CreateTemplateResponse struct {
	JobID                 string            `json:"jobid,omitempty"`
	Account               string            `json:"account,omitempty"`
	Accountid             string            `json:"accountid,omitempty"`
	Bootable              bool              `json:"bootable,omitempty"`
	Checksum              string            `json:"checksum,omitempty"`
	Created               string            `json:"created,omitempty"`
	CrossZones            bool              `json:"crossZones,omitempty"`
	Details               map[string]string `json:"details,omitempty"`
	Displaytext           string            `json:"displaytext,omitempty"`
	Domain                string            `json:"domain,omitempty"`
	Domainid              string            `json:"domainid,omitempty"`
	Format                string            `json:"format,omitempty"`
	Hostid                string            `json:"hostid,omitempty"`
	Hostname              string            `json:"hostname,omitempty"`
	Hypervisor            string            `json:"hypervisor,omitempty"`
	Id                    string            `json:"id,omitempty"`
	Isdynamicallyscalable bool              `json:"isdynamicallyscalable,omitempty"`
	Isextractable         bool              `json:"isextractable,omitempty"`
	Isfeatured            bool              `json:"isfeatured,omitempty"`
	Ispublic              bool              `json:"ispublic,omitempty"`
	Isready               bool              `json:"isready,omitempty"`
	Name                  string            `json:"name,omitempty"`
	Ostypeid              string            `json:"ostypeid,omitempty"`
	Ostypename            string            `json:"ostypename,omitempty"`
	Passwordenabled       bool              `json:"passwordenabled,omitempty"`
	Project               string            `json:"project,omitempty"`
	Projectid             string            `json:"projectid,omitempty"`
	Removed               string            `json:"removed,omitempty"`
	Size                  int64             `json:"size,omitempty"`
	Sourcetemplateid      string            `json:"sourcetemplateid,omitempty"`
	Sshkeyenabled         bool              `json:"sshkeyenabled,omitempty"`
	Status                string            `json:"status,omitempty"`
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
	Templatetag  string `json:"templatetag,omitempty"`
	Templatetype string `json:"templatetype,omitempty"`
	Zoneid       string `json:"zoneid,omitempty"`
	Zonename     string `json:"zonename,omitempty"`
}

type RegisterTemplateParams struct {
	p map[string]interface{}
}

func (p *RegisterTemplateParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["account"]; found {
		u.Set("account", v.(string))
	}
	if v, found := p.p["bits"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("bits", vv)
	}
	if v, found := p.p["checksum"]; found {
		u.Set("checksum", v.(string))
	}
	if v, found := p.p["details"]; found {
		i := 0
		for k, vv := range v.(map[string]string) {
			u.Set(fmt.Sprintf("details[%d].key", i), k)
			u.Set(fmt.Sprintf("details[%d].value", i), vv)
			i++
		}
	}
	if v, found := p.p["displaytext"]; found {
		u.Set("displaytext", v.(string))
	}
	if v, found := p.p["domainid"]; found {
		u.Set("domainid", v.(string))
	}
	if v, found := p.p["format"]; found {
		u.Set("format", v.(string))
	}
	if v, found := p.p["hypervisor"]; found {
		u.Set("hypervisor", v.(string))
	}
	if v, found := p.p["isdynamicallyscalable"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("isdynamicallyscalable", vv)
	}
	if v, found := p.p["isextractable"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("isextractable", vv)
	}
	if v, found := p.p["isfeatured"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("isfeatured", vv)
	}
	if v, found := p.p["ispublic"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("ispublic", vv)
	}
	if v, found := p.p["isrouting"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("isrouting", vv)
	}
	if v, found := p.p["name"]; found {
		u.Set("name", v.(string))
	}
	if v, found := p.p["ostypeid"]; found {
		u.Set("ostypeid", v.(string))
	}
	if v, found := p.p["passwordenabled"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("passwordenabled", vv)
	}
	if v, found := p.p["projectid"]; found {
		u.Set("projectid", v.(string))
	}
	if v, found := p.p["requireshvm"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("requireshvm", vv)
	}
	if v, found := p.p["sshkeyenabled"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("sshkeyenabled", vv)
	}
	if v, found := p.p["templatetag"]; found {
		u.Set("templatetag", v.(string))
	}
	if v, found := p.p["url"]; found {
		u.Set("url", v.(string))
	}
	if v, found := p.p["zoneid"]; found {
		u.Set("zoneid", v.(string))
	}
	return u
}

func (p *RegisterTemplateParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *RegisterTemplateParams) SetBits(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["bits"] = v
	return
}

func (p *RegisterTemplateParams) SetChecksum(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["checksum"] = v
	return
}

func (p *RegisterTemplateParams) SetDetails(v map[string]string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["details"] = v
	return
}

func (p *RegisterTemplateParams) SetDisplaytext(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["displaytext"] = v
	return
}

func (p *RegisterTemplateParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *RegisterTemplateParams) SetFormat(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["format"] = v
	return
}

func (p *RegisterTemplateParams) SetHypervisor(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["hypervisor"] = v
	return
}

func (p *RegisterTemplateParams) SetIsdynamicallyscalable(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isdynamicallyscalable"] = v
	return
}

func (p *RegisterTemplateParams) SetIsextractable(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isextractable"] = v
	return
}

func (p *RegisterTemplateParams) SetIsfeatured(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isfeatured"] = v
	return
}

func (p *RegisterTemplateParams) SetIspublic(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["ispublic"] = v
	return
}

func (p *RegisterTemplateParams) SetIsrouting(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isrouting"] = v
	return
}

func (p *RegisterTemplateParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *RegisterTemplateParams) SetOstypeid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["ostypeid"] = v
	return
}

func (p *RegisterTemplateParams) SetPasswordenabled(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["passwordenabled"] = v
	return
}

func (p *RegisterTemplateParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

func (p *RegisterTemplateParams) SetRequireshvm(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["requireshvm"] = v
	return
}

func (p *RegisterTemplateParams) SetSshkeyenabled(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["sshkeyenabled"] = v
	return
}

func (p *RegisterTemplateParams) SetTemplatetag(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["templatetag"] = v
	return
}

func (p *RegisterTemplateParams) SetUrl(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["url"] = v
	return
}

func (p *RegisterTemplateParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new RegisterTemplateParams instance,
// as then you are sure you have configured all required params
func (s *TemplateService) NewRegisterTemplateParams(displaytext string, format string, hypervisor string, name string, ostypeid string, url string, zoneid string) *RegisterTemplateParams {
	p := &RegisterTemplateParams{}
	p.p = make(map[string]interface{})
	p.p["displaytext"] = displaytext
	p.p["format"] = format
	p.p["hypervisor"] = hypervisor
	p.p["name"] = name
	p.p["ostypeid"] = ostypeid
	p.p["url"] = url
	p.p["zoneid"] = zoneid
	return p
}

// Registers an existing template into the CloudStack cloud.
func (s *TemplateService) RegisterTemplate(p *RegisterTemplateParams) (*RegisterTemplateResponse, error) {
	resp, err := s.cs.newRequest("registerTemplate", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r RegisterTemplateResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type RegisterTemplateResponse struct {
	Count            int                 `json:"count"`
	RegisterTemplate []*RegisterTemplate `json:"template"`
}

type RegisterTemplate struct {
	Account               string            `json:"account,omitempty"`
	Accountid             string            `json:"accountid,omitempty"`
	Bootable              bool              `json:"bootable,omitempty"`
	Checksum              string            `json:"checksum,omitempty"`
	Created               string            `json:"created,omitempty"`
	CrossZones            bool              `json:"crossZones,omitempty"`
	Details               map[string]string `json:"details,omitempty"`
	Displaytext           string            `json:"displaytext,omitempty"`
	Domain                string            `json:"domain,omitempty"`
	Domainid              string            `json:"domainid,omitempty"`
	Format                string            `json:"format,omitempty"`
	Hostid                string            `json:"hostid,omitempty"`
	Hostname              string            `json:"hostname,omitempty"`
	Hypervisor            string            `json:"hypervisor,omitempty"`
	Id                    string            `json:"id,omitempty"`
	Isdynamicallyscalable bool              `json:"isdynamicallyscalable,omitempty"`
	Isextractable         bool              `json:"isextractable,omitempty"`
	Isfeatured            bool              `json:"isfeatured,omitempty"`
	Ispublic              bool              `json:"ispublic,omitempty"`
	Isready               bool              `json:"isready,omitempty"`
	Name                  string            `json:"name,omitempty"`
	Ostypeid              string            `json:"ostypeid,omitempty"`
	Ostypename            string            `json:"ostypename,omitempty"`
	Passwordenabled       bool              `json:"passwordenabled,omitempty"`
	Project               string            `json:"project,omitempty"`
	Projectid             string            `json:"projectid,omitempty"`
	Removed               string            `json:"removed,omitempty"`
	Size                  int64             `json:"size,omitempty"`
	Sourcetemplateid      string            `json:"sourcetemplateid,omitempty"`
	Sshkeyenabled         bool              `json:"sshkeyenabled,omitempty"`
	Status                string            `json:"status,omitempty"`
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
	Templatetag  string `json:"templatetag,omitempty"`
	Templatetype string `json:"templatetype,omitempty"`
	Zoneid       string `json:"zoneid,omitempty"`
	Zonename     string `json:"zonename,omitempty"`
}

type UpdateTemplateParams struct {
	p map[string]interface{}
}

func (p *UpdateTemplateParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["bootable"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("bootable", vv)
	}
	if v, found := p.p["details"]; found {
		i := 0
		for k, vv := range v.(map[string]string) {
			u.Set(fmt.Sprintf("details[%d].key", i), k)
			u.Set(fmt.Sprintf("details[%d].value", i), vv)
			i++
		}
	}
	if v, found := p.p["displaytext"]; found {
		u.Set("displaytext", v.(string))
	}
	if v, found := p.p["format"]; found {
		u.Set("format", v.(string))
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	if v, found := p.p["isdynamicallyscalable"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("isdynamicallyscalable", vv)
	}
	if v, found := p.p["isrouting"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("isrouting", vv)
	}
	if v, found := p.p["name"]; found {
		u.Set("name", v.(string))
	}
	if v, found := p.p["ostypeid"]; found {
		u.Set("ostypeid", v.(string))
	}
	if v, found := p.p["passwordenabled"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("passwordenabled", vv)
	}
	if v, found := p.p["requireshvm"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("requireshvm", vv)
	}
	if v, found := p.p["sortkey"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("sortkey", vv)
	}
	return u
}

func (p *UpdateTemplateParams) SetBootable(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["bootable"] = v
	return
}

func (p *UpdateTemplateParams) SetDetails(v map[string]string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["details"] = v
	return
}

func (p *UpdateTemplateParams) SetDisplaytext(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["displaytext"] = v
	return
}

func (p *UpdateTemplateParams) SetFormat(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["format"] = v
	return
}

func (p *UpdateTemplateParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *UpdateTemplateParams) SetIsdynamicallyscalable(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isdynamicallyscalable"] = v
	return
}

func (p *UpdateTemplateParams) SetIsrouting(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isrouting"] = v
	return
}

func (p *UpdateTemplateParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *UpdateTemplateParams) SetOstypeid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["ostypeid"] = v
	return
}

func (p *UpdateTemplateParams) SetPasswordenabled(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["passwordenabled"] = v
	return
}

func (p *UpdateTemplateParams) SetRequireshvm(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["requireshvm"] = v
	return
}

func (p *UpdateTemplateParams) SetSortkey(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["sortkey"] = v
	return
}

// You should always use this function to get a new UpdateTemplateParams instance,
// as then you are sure you have configured all required params
func (s *TemplateService) NewUpdateTemplateParams(id string) *UpdateTemplateParams {
	p := &UpdateTemplateParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Updates attributes of a template.
func (s *TemplateService) UpdateTemplate(p *UpdateTemplateParams) (*UpdateTemplateResponse, error) {
	resp, err := s.cs.newRequest("updateTemplate", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r UpdateTemplateResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type UpdateTemplateResponse struct {
	Account               string            `json:"account,omitempty"`
	Accountid             string            `json:"accountid,omitempty"`
	Bootable              bool              `json:"bootable,omitempty"`
	Checksum              string            `json:"checksum,omitempty"`
	Created               string            `json:"created,omitempty"`
	CrossZones            bool              `json:"crossZones,omitempty"`
	Details               map[string]string `json:"details,omitempty"`
	Displaytext           string            `json:"displaytext,omitempty"`
	Domain                string            `json:"domain,omitempty"`
	Domainid              string            `json:"domainid,omitempty"`
	Format                string            `json:"format,omitempty"`
	Hostid                string            `json:"hostid,omitempty"`
	Hostname              string            `json:"hostname,omitempty"`
	Hypervisor            string            `json:"hypervisor,omitempty"`
	Id                    string            `json:"id,omitempty"`
	Isdynamicallyscalable bool              `json:"isdynamicallyscalable,omitempty"`
	Isextractable         bool              `json:"isextractable,omitempty"`
	Isfeatured            bool              `json:"isfeatured,omitempty"`
	Ispublic              bool              `json:"ispublic,omitempty"`
	Isready               bool              `json:"isready,omitempty"`
	Name                  string            `json:"name,omitempty"`
	Ostypeid              string            `json:"ostypeid,omitempty"`
	Ostypename            string            `json:"ostypename,omitempty"`
	Passwordenabled       bool              `json:"passwordenabled,omitempty"`
	Project               string            `json:"project,omitempty"`
	Projectid             string            `json:"projectid,omitempty"`
	Removed               string            `json:"removed,omitempty"`
	Size                  int64             `json:"size,omitempty"`
	Sourcetemplateid      string            `json:"sourcetemplateid,omitempty"`
	Sshkeyenabled         bool              `json:"sshkeyenabled,omitempty"`
	Status                string            `json:"status,omitempty"`
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
	Templatetag  string `json:"templatetag,omitempty"`
	Templatetype string `json:"templatetype,omitempty"`
	Zoneid       string `json:"zoneid,omitempty"`
	Zonename     string `json:"zonename,omitempty"`
}

type CopyTemplateParams struct {
	p map[string]interface{}
}

func (p *CopyTemplateParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["destzoneid"]; found {
		u.Set("destzoneid", v.(string))
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	if v, found := p.p["sourcezoneid"]; found {
		u.Set("sourcezoneid", v.(string))
	}
	return u
}

func (p *CopyTemplateParams) SetDestzoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["destzoneid"] = v
	return
}

func (p *CopyTemplateParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *CopyTemplateParams) SetSourcezoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["sourcezoneid"] = v
	return
}

// You should always use this function to get a new CopyTemplateParams instance,
// as then you are sure you have configured all required params
func (s *TemplateService) NewCopyTemplateParams(destzoneid string, id string) *CopyTemplateParams {
	p := &CopyTemplateParams{}
	p.p = make(map[string]interface{})
	p.p["destzoneid"] = destzoneid
	p.p["id"] = id
	return p
}

// Copies a template from one zone to another.
func (s *TemplateService) CopyTemplate(p *CopyTemplateParams) (*CopyTemplateResponse, error) {
	resp, err := s.cs.newRequest("copyTemplate", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r CopyTemplateResponse
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

type CopyTemplateResponse struct {
	JobID                 string            `json:"jobid,omitempty"`
	Account               string            `json:"account,omitempty"`
	Accountid             string            `json:"accountid,omitempty"`
	Bootable              bool              `json:"bootable,omitempty"`
	Checksum              string            `json:"checksum,omitempty"`
	Created               string            `json:"created,omitempty"`
	CrossZones            bool              `json:"crossZones,omitempty"`
	Details               map[string]string `json:"details,omitempty"`
	Displaytext           string            `json:"displaytext,omitempty"`
	Domain                string            `json:"domain,omitempty"`
	Domainid              string            `json:"domainid,omitempty"`
	Format                string            `json:"format,omitempty"`
	Hostid                string            `json:"hostid,omitempty"`
	Hostname              string            `json:"hostname,omitempty"`
	Hypervisor            string            `json:"hypervisor,omitempty"`
	Id                    string            `json:"id,omitempty"`
	Isdynamicallyscalable bool              `json:"isdynamicallyscalable,omitempty"`
	Isextractable         bool              `json:"isextractable,omitempty"`
	Isfeatured            bool              `json:"isfeatured,omitempty"`
	Ispublic              bool              `json:"ispublic,omitempty"`
	Isready               bool              `json:"isready,omitempty"`
	Name                  string            `json:"name,omitempty"`
	Ostypeid              string            `json:"ostypeid,omitempty"`
	Ostypename            string            `json:"ostypename,omitempty"`
	Passwordenabled       bool              `json:"passwordenabled,omitempty"`
	Project               string            `json:"project,omitempty"`
	Projectid             string            `json:"projectid,omitempty"`
	Removed               string            `json:"removed,omitempty"`
	Size                  int64             `json:"size,omitempty"`
	Sourcetemplateid      string            `json:"sourcetemplateid,omitempty"`
	Sshkeyenabled         bool              `json:"sshkeyenabled,omitempty"`
	Status                string            `json:"status,omitempty"`
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
	Templatetag  string `json:"templatetag,omitempty"`
	Templatetype string `json:"templatetype,omitempty"`
	Zoneid       string `json:"zoneid,omitempty"`
	Zonename     string `json:"zonename,omitempty"`
}

type DeleteTemplateParams struct {
	p map[string]interface{}
}

func (p *DeleteTemplateParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	if v, found := p.p["zoneid"]; found {
		u.Set("zoneid", v.(string))
	}
	return u
}

func (p *DeleteTemplateParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *DeleteTemplateParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new DeleteTemplateParams instance,
// as then you are sure you have configured all required params
func (s *TemplateService) NewDeleteTemplateParams(id string) *DeleteTemplateParams {
	p := &DeleteTemplateParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Deletes a template from the system. All virtual machines using the deleted template will not be affected.
func (s *TemplateService) DeleteTemplate(p *DeleteTemplateParams) (*DeleteTemplateResponse, error) {
	resp, err := s.cs.newRequest("deleteTemplate", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r DeleteTemplateResponse
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

type DeleteTemplateResponse struct {
	JobID       string `json:"jobid,omitempty"`
	Displaytext string `json:"displaytext,omitempty"`
	Success     bool   `json:"success,omitempty"`
}

type ListTemplatesParams struct {
	p map[string]interface{}
}

func (p *ListTemplatesParams) toURLValues() url.Values {
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
	if v, found := p.p["hypervisor"]; found {
		u.Set("hypervisor", v.(string))
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
	if v, found := p.p["showremoved"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("showremoved", vv)
	}
	if v, found := p.p["tags"]; found {
		i := 0
		for k, vv := range v.(map[string]string) {
			u.Set(fmt.Sprintf("tags[%d].key", i), k)
			u.Set(fmt.Sprintf("tags[%d].value", i), vv)
			i++
		}
	}
	if v, found := p.p["templatefilter"]; found {
		u.Set("templatefilter", v.(string))
	}
	if v, found := p.p["zoneid"]; found {
		u.Set("zoneid", v.(string))
	}
	return u
}

func (p *ListTemplatesParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *ListTemplatesParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *ListTemplatesParams) SetHypervisor(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["hypervisor"] = v
	return
}

func (p *ListTemplatesParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ListTemplatesParams) SetIsrecursive(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isrecursive"] = v
	return
}

func (p *ListTemplatesParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListTemplatesParams) SetListall(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["listall"] = v
	return
}

func (p *ListTemplatesParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *ListTemplatesParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListTemplatesParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

func (p *ListTemplatesParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

func (p *ListTemplatesParams) SetShowremoved(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["showremoved"] = v
	return
}

func (p *ListTemplatesParams) SetTags(v map[string]string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["tags"] = v
	return
}

func (p *ListTemplatesParams) SetTemplatefilter(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["templatefilter"] = v
	return
}

func (p *ListTemplatesParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new ListTemplatesParams instance,
// as then you are sure you have configured all required params
func (s *TemplateService) NewListTemplatesParams(templatefilter string) *ListTemplatesParams {
	p := &ListTemplatesParams{}
	p.p = make(map[string]interface{})
	p.p["templatefilter"] = templatefilter
	return p
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *TemplateService) GetTemplateID(name string, templatefilter string, zoneid string, opts ...OptionFunc) (string, int, error) {
	p := &ListTemplatesParams{}
	p.p = make(map[string]interface{})

	p.p["name"] = name
	p.p["templatefilter"] = templatefilter
	p.p["zoneid"] = zoneid

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return "", -1, err
		}
	}

	l, err := s.ListTemplates(p)
	if err != nil {
		return "", -1, err
	}

	if l.Count == 0 {
		return "", l.Count, fmt.Errorf("No match found for %s: %+v", name, l)
	}

	if l.Count == 1 {
		return l.Templates[0].Id, l.Count, nil
	}

	if l.Count > 1 {
		for _, v := range l.Templates {
			if v.Name == name {
				return v.Id, l.Count, nil
			}
		}
	}
	return "", l.Count, fmt.Errorf("Could not find an exact match for %s: %+v", name, l)
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *TemplateService) GetTemplateByName(name string, templatefilter string, zoneid string, opts ...OptionFunc) (*Template, int, error) {
	id, count, err := s.GetTemplateID(name, templatefilter, zoneid, opts...)
	if err != nil {
		return nil, count, err
	}

	r, count, err := s.GetTemplateByID(id, templatefilter, opts...)
	if err != nil {
		return nil, count, err
	}
	return r, count, nil
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *TemplateService) GetTemplateByID(id string, templatefilter string, opts ...OptionFunc) (*Template, int, error) {
	p := &ListTemplatesParams{}
	p.p = make(map[string]interface{})

	p.p["id"] = id
	p.p["templatefilter"] = templatefilter

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return nil, -1, err
		}
	}

	l, err := s.ListTemplates(p)
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
		return l.Templates[0], l.Count, nil
	}
	return nil, l.Count, fmt.Errorf("There is more then one result for Template UUID: %s!", id)
}

// List all public, private, and privileged templates.
func (s *TemplateService) ListTemplates(p *ListTemplatesParams) (*ListTemplatesResponse, error) {
	resp, err := s.cs.newRequest("listTemplates", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListTemplatesResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListTemplatesResponse struct {
	Count     int         `json:"count"`
	Templates []*Template `json:"template"`
}

type Template struct {
	Account               string            `json:"account,omitempty"`
	Accountid             string            `json:"accountid,omitempty"`
	Bootable              bool              `json:"bootable,omitempty"`
	Checksum              string            `json:"checksum,omitempty"`
	Created               string            `json:"created,omitempty"`
	CrossZones            bool              `json:"crossZones,omitempty"`
	Details               map[string]string `json:"details,omitempty"`
	Displaytext           string            `json:"displaytext,omitempty"`
	Domain                string            `json:"domain,omitempty"`
	Domainid              string            `json:"domainid,omitempty"`
	Format                string            `json:"format,omitempty"`
	Hostid                string            `json:"hostid,omitempty"`
	Hostname              string            `json:"hostname,omitempty"`
	Hypervisor            string            `json:"hypervisor,omitempty"`
	Id                    string            `json:"id,omitempty"`
	Isdynamicallyscalable bool              `json:"isdynamicallyscalable,omitempty"`
	Isextractable         bool              `json:"isextractable,omitempty"`
	Isfeatured            bool              `json:"isfeatured,omitempty"`
	Ispublic              bool              `json:"ispublic,omitempty"`
	Isready               bool              `json:"isready,omitempty"`
	Name                  string            `json:"name,omitempty"`
	Ostypeid              string            `json:"ostypeid,omitempty"`
	Ostypename            string            `json:"ostypename,omitempty"`
	Passwordenabled       bool              `json:"passwordenabled,omitempty"`
	Project               string            `json:"project,omitempty"`
	Projectid             string            `json:"projectid,omitempty"`
	Removed               string            `json:"removed,omitempty"`
	Size                  int64             `json:"size,omitempty"`
	Sourcetemplateid      string            `json:"sourcetemplateid,omitempty"`
	Sshkeyenabled         bool              `json:"sshkeyenabled,omitempty"`
	Status                string            `json:"status,omitempty"`
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
	Templatetag  string `json:"templatetag,omitempty"`
	Templatetype string `json:"templatetype,omitempty"`
	Zoneid       string `json:"zoneid,omitempty"`
	Zonename     string `json:"zonename,omitempty"`
}

type UpdateTemplatePermissionsParams struct {
	p map[string]interface{}
}

func (p *UpdateTemplatePermissionsParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["accounts"]; found {
		vv := strings.Join(v.([]string), ",")
		u.Set("accounts", vv)
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	if v, found := p.p["isextractable"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("isextractable", vv)
	}
	if v, found := p.p["isfeatured"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("isfeatured", vv)
	}
	if v, found := p.p["ispublic"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("ispublic", vv)
	}
	if v, found := p.p["op"]; found {
		u.Set("op", v.(string))
	}
	if v, found := p.p["projectids"]; found {
		vv := strings.Join(v.([]string), ",")
		u.Set("projectids", vv)
	}
	return u
}

func (p *UpdateTemplatePermissionsParams) SetAccounts(v []string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["accounts"] = v
	return
}

func (p *UpdateTemplatePermissionsParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *UpdateTemplatePermissionsParams) SetIsextractable(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isextractable"] = v
	return
}

func (p *UpdateTemplatePermissionsParams) SetIsfeatured(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isfeatured"] = v
	return
}

func (p *UpdateTemplatePermissionsParams) SetIspublic(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["ispublic"] = v
	return
}

func (p *UpdateTemplatePermissionsParams) SetOp(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["op"] = v
	return
}

func (p *UpdateTemplatePermissionsParams) SetProjectids(v []string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectids"] = v
	return
}

// You should always use this function to get a new UpdateTemplatePermissionsParams instance,
// as then you are sure you have configured all required params
func (s *TemplateService) NewUpdateTemplatePermissionsParams(id string) *UpdateTemplatePermissionsParams {
	p := &UpdateTemplatePermissionsParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Updates a template visibility permissions. A public template is visible to all accounts within the same domain. A private template is visible only to the owner of the template. A priviledged template is a private template with account permissions added. Only accounts specified under the template permissions are visible to them.
func (s *TemplateService) UpdateTemplatePermissions(p *UpdateTemplatePermissionsParams) (*UpdateTemplatePermissionsResponse, error) {
	resp, err := s.cs.newRequest("updateTemplatePermissions", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r UpdateTemplatePermissionsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type UpdateTemplatePermissionsResponse struct {
	Displaytext string `json:"displaytext,omitempty"`
	Success     string `json:"success,omitempty"`
}

type ListTemplatePermissionsParams struct {
	p map[string]interface{}
}

func (p *ListTemplatePermissionsParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	return u
}

func (p *ListTemplatePermissionsParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new ListTemplatePermissionsParams instance,
// as then you are sure you have configured all required params
func (s *TemplateService) NewListTemplatePermissionsParams(id string) *ListTemplatePermissionsParams {
	p := &ListTemplatePermissionsParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *TemplateService) GetTemplatePermissionByID(id string, opts ...OptionFunc) (*TemplatePermission, int, error) {
	p := &ListTemplatePermissionsParams{}
	p.p = make(map[string]interface{})

	p.p["id"] = id

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return nil, -1, err
		}
	}

	l, err := s.ListTemplatePermissions(p)
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
		return l.TemplatePermissions[0], l.Count, nil
	}
	return nil, l.Count, fmt.Errorf("There is more then one result for TemplatePermission UUID: %s!", id)
}

// List template visibility and all accounts that have permissions to view this template.
func (s *TemplateService) ListTemplatePermissions(p *ListTemplatePermissionsParams) (*ListTemplatePermissionsResponse, error) {
	resp, err := s.cs.newRequest("listTemplatePermissions", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListTemplatePermissionsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListTemplatePermissionsResponse struct {
	Count               int                   `json:"count"`
	TemplatePermissions []*TemplatePermission `json:"templatepermission"`
}

type TemplatePermission struct {
	Account    []string `json:"account,omitempty"`
	Domainid   string   `json:"domainid,omitempty"`
	Id         string   `json:"id,omitempty"`
	Ispublic   bool     `json:"ispublic,omitempty"`
	Projectids []string `json:"projectids,omitempty"`
}

type ExtractTemplateParams struct {
	p map[string]interface{}
}

func (p *ExtractTemplateParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	if v, found := p.p["mode"]; found {
		u.Set("mode", v.(string))
	}
	if v, found := p.p["url"]; found {
		u.Set("url", v.(string))
	}
	if v, found := p.p["zoneid"]; found {
		u.Set("zoneid", v.(string))
	}
	return u
}

func (p *ExtractTemplateParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ExtractTemplateParams) SetMode(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["mode"] = v
	return
}

func (p *ExtractTemplateParams) SetUrl(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["url"] = v
	return
}

func (p *ExtractTemplateParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new ExtractTemplateParams instance,
// as then you are sure you have configured all required params
func (s *TemplateService) NewExtractTemplateParams(id string, mode string) *ExtractTemplateParams {
	p := &ExtractTemplateParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	p.p["mode"] = mode
	return p
}

// Extracts a template
func (s *TemplateService) ExtractTemplate(p *ExtractTemplateParams) (*ExtractTemplateResponse, error) {
	resp, err := s.cs.newRequest("extractTemplate", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ExtractTemplateResponse
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

type ExtractTemplateResponse struct {
	JobID            string `json:"jobid,omitempty"`
	Accountid        string `json:"accountid,omitempty"`
	Created          string `json:"created,omitempty"`
	ExtractId        string `json:"extractId,omitempty"`
	ExtractMode      string `json:"extractMode,omitempty"`
	Id               string `json:"id,omitempty"`
	Name             string `json:"name,omitempty"`
	Resultstring     string `json:"resultstring,omitempty"`
	State            string `json:"state,omitempty"`
	Status           string `json:"status,omitempty"`
	Storagetype      string `json:"storagetype,omitempty"`
	Uploadpercentage int    `json:"uploadpercentage,omitempty"`
	Url              string `json:"url,omitempty"`
	Zoneid           string `json:"zoneid,omitempty"`
	Zonename         string `json:"zonename,omitempty"`
}

type PrepareTemplateParams struct {
	p map[string]interface{}
}

func (p *PrepareTemplateParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["storageid"]; found {
		u.Set("storageid", v.(string))
	}
	if v, found := p.p["templateid"]; found {
		u.Set("templateid", v.(string))
	}
	if v, found := p.p["zoneid"]; found {
		u.Set("zoneid", v.(string))
	}
	return u
}

func (p *PrepareTemplateParams) SetStorageid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["storageid"] = v
	return
}

func (p *PrepareTemplateParams) SetTemplateid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["templateid"] = v
	return
}

func (p *PrepareTemplateParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new PrepareTemplateParams instance,
// as then you are sure you have configured all required params
func (s *TemplateService) NewPrepareTemplateParams(templateid string, zoneid string) *PrepareTemplateParams {
	p := &PrepareTemplateParams{}
	p.p = make(map[string]interface{})
	p.p["templateid"] = templateid
	p.p["zoneid"] = zoneid
	return p
}

// load template into primary storage
func (s *TemplateService) PrepareTemplate(p *PrepareTemplateParams) (*PrepareTemplateResponse, error) {
	resp, err := s.cs.newRequest("prepareTemplate", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r PrepareTemplateResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type PrepareTemplateResponse struct {
	Account               string            `json:"account,omitempty"`
	Accountid             string            `json:"accountid,omitempty"`
	Bootable              bool              `json:"bootable,omitempty"`
	Checksum              string            `json:"checksum,omitempty"`
	Created               string            `json:"created,omitempty"`
	CrossZones            bool              `json:"crossZones,omitempty"`
	Details               map[string]string `json:"details,omitempty"`
	Displaytext           string            `json:"displaytext,omitempty"`
	Domain                string            `json:"domain,omitempty"`
	Domainid              string            `json:"domainid,omitempty"`
	Format                string            `json:"format,omitempty"`
	Hostid                string            `json:"hostid,omitempty"`
	Hostname              string            `json:"hostname,omitempty"`
	Hypervisor            string            `json:"hypervisor,omitempty"`
	Id                    string            `json:"id,omitempty"`
	Isdynamicallyscalable bool              `json:"isdynamicallyscalable,omitempty"`
	Isextractable         bool              `json:"isextractable,omitempty"`
	Isfeatured            bool              `json:"isfeatured,omitempty"`
	Ispublic              bool              `json:"ispublic,omitempty"`
	Isready               bool              `json:"isready,omitempty"`
	Name                  string            `json:"name,omitempty"`
	Ostypeid              string            `json:"ostypeid,omitempty"`
	Ostypename            string            `json:"ostypename,omitempty"`
	Passwordenabled       bool              `json:"passwordenabled,omitempty"`
	Project               string            `json:"project,omitempty"`
	Projectid             string            `json:"projectid,omitempty"`
	Removed               string            `json:"removed,omitempty"`
	Size                  int64             `json:"size,omitempty"`
	Sourcetemplateid      string            `json:"sourcetemplateid,omitempty"`
	Sshkeyenabled         bool              `json:"sshkeyenabled,omitempty"`
	Status                string            `json:"status,omitempty"`
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
	Templatetag  string `json:"templatetag,omitempty"`
	Templatetype string `json:"templatetype,omitempty"`
	Zoneid       string `json:"zoneid,omitempty"`
	Zonename     string `json:"zonename,omitempty"`
}

type UpgradeRouterTemplateParams struct {
	p map[string]interface{}
}

func (p *UpgradeRouterTemplateParams) toURLValues() url.Values {
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
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	if v, found := p.p["podid"]; found {
		u.Set("podid", v.(string))
	}
	if v, found := p.p["zoneid"]; found {
		u.Set("zoneid", v.(string))
	}
	return u
}

func (p *UpgradeRouterTemplateParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *UpgradeRouterTemplateParams) SetClusterid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["clusterid"] = v
	return
}

func (p *UpgradeRouterTemplateParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *UpgradeRouterTemplateParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *UpgradeRouterTemplateParams) SetPodid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["podid"] = v
	return
}

func (p *UpgradeRouterTemplateParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new UpgradeRouterTemplateParams instance,
// as then you are sure you have configured all required params
func (s *TemplateService) NewUpgradeRouterTemplateParams() *UpgradeRouterTemplateParams {
	p := &UpgradeRouterTemplateParams{}
	p.p = make(map[string]interface{})
	return p
}

// Upgrades router to use newer template
func (s *TemplateService) UpgradeRouterTemplate(p *UpgradeRouterTemplateParams) (*UpgradeRouterTemplateResponse, error) {
	resp, err := s.cs.newRequest("upgradeRouterTemplate", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r UpgradeRouterTemplateResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type UpgradeRouterTemplateResponse struct {
	Jobid     string `json:"jobid,omitempty"`
	Jobstatus int    `json:"jobstatus,omitempty"`
}

type GetUploadParamsForTemplateParams struct {
	p map[string]interface{}
}

func (p *GetUploadParamsForTemplateParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["account"]; found {
		u.Set("account", v.(string))
	}
	if v, found := p.p["bits"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("bits", vv)
	}
	if v, found := p.p["checksum"]; found {
		u.Set("checksum", v.(string))
	}
	if v, found := p.p["details"]; found {
		i := 0
		for k, vv := range v.(map[string]string) {
			u.Set(fmt.Sprintf("details[%d].key", i), k)
			u.Set(fmt.Sprintf("details[%d].value", i), vv)
			i++
		}
	}
	if v, found := p.p["displaytext"]; found {
		u.Set("displaytext", v.(string))
	}
	if v, found := p.p["domainid"]; found {
		u.Set("domainid", v.(string))
	}
	if v, found := p.p["format"]; found {
		u.Set("format", v.(string))
	}
	if v, found := p.p["hypervisor"]; found {
		u.Set("hypervisor", v.(string))
	}
	if v, found := p.p["isdynamicallyscalable"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("isdynamicallyscalable", vv)
	}
	if v, found := p.p["isextractable"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("isextractable", vv)
	}
	if v, found := p.p["isfeatured"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("isfeatured", vv)
	}
	if v, found := p.p["ispublic"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("ispublic", vv)
	}
	if v, found := p.p["isrouting"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("isrouting", vv)
	}
	if v, found := p.p["name"]; found {
		u.Set("name", v.(string))
	}
	if v, found := p.p["ostypeid"]; found {
		u.Set("ostypeid", v.(string))
	}
	if v, found := p.p["passwordenabled"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("passwordenabled", vv)
	}
	if v, found := p.p["projectid"]; found {
		u.Set("projectid", v.(string))
	}
	if v, found := p.p["requireshvm"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("requireshvm", vv)
	}
	if v, found := p.p["sshkeyenabled"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("sshkeyenabled", vv)
	}
	if v, found := p.p["templatetag"]; found {
		u.Set("templatetag", v.(string))
	}
	if v, found := p.p["zoneid"]; found {
		u.Set("zoneid", v.(string))
	}
	return u
}

func (p *GetUploadParamsForTemplateParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *GetUploadParamsForTemplateParams) SetBits(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["bits"] = v
	return
}

func (p *GetUploadParamsForTemplateParams) SetChecksum(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["checksum"] = v
	return
}

func (p *GetUploadParamsForTemplateParams) SetDetails(v map[string]string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["details"] = v
	return
}

func (p *GetUploadParamsForTemplateParams) SetDisplaytext(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["displaytext"] = v
	return
}

func (p *GetUploadParamsForTemplateParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *GetUploadParamsForTemplateParams) SetFormat(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["format"] = v
	return
}

func (p *GetUploadParamsForTemplateParams) SetHypervisor(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["hypervisor"] = v
	return
}

func (p *GetUploadParamsForTemplateParams) SetIsdynamicallyscalable(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isdynamicallyscalable"] = v
	return
}

func (p *GetUploadParamsForTemplateParams) SetIsextractable(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isextractable"] = v
	return
}

func (p *GetUploadParamsForTemplateParams) SetIsfeatured(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isfeatured"] = v
	return
}

func (p *GetUploadParamsForTemplateParams) SetIspublic(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["ispublic"] = v
	return
}

func (p *GetUploadParamsForTemplateParams) SetIsrouting(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isrouting"] = v
	return
}

func (p *GetUploadParamsForTemplateParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *GetUploadParamsForTemplateParams) SetOstypeid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["ostypeid"] = v
	return
}

func (p *GetUploadParamsForTemplateParams) SetPasswordenabled(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["passwordenabled"] = v
	return
}

func (p *GetUploadParamsForTemplateParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

func (p *GetUploadParamsForTemplateParams) SetRequireshvm(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["requireshvm"] = v
	return
}

func (p *GetUploadParamsForTemplateParams) SetSshkeyenabled(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["sshkeyenabled"] = v
	return
}

func (p *GetUploadParamsForTemplateParams) SetTemplatetag(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["templatetag"] = v
	return
}

func (p *GetUploadParamsForTemplateParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new GetUploadParamsForTemplateParams instance,
// as then you are sure you have configured all required params
func (s *TemplateService) NewGetUploadParamsForTemplateParams(displaytext string, format string, hypervisor string, name string, ostypeid string, zoneid string) *GetUploadParamsForTemplateParams {
	p := &GetUploadParamsForTemplateParams{}
	p.p = make(map[string]interface{})
	p.p["displaytext"] = displaytext
	p.p["format"] = format
	p.p["hypervisor"] = hypervisor
	p.p["name"] = name
	p.p["ostypeid"] = ostypeid
	p.p["zoneid"] = zoneid
	return p
}

// upload an existing template into the CloudStack cloud.
func (s *TemplateService) GetUploadParamsForTemplate(p *GetUploadParamsForTemplateParams) (*GetUploadParamsForTemplateResponse, error) {
	resp, err := s.cs.newRequest("getUploadParamsForTemplate", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r GetUploadParamsForTemplateResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type GetUploadParamsForTemplateResponse struct {
	Expires   string `json:"expires,omitempty"`
	Id        string `json:"id,omitempty"`
	Metadata  string `json:"metadata,omitempty"`
	PostURL   string `json:"postURL,omitempty"`
	Signature string `json:"signature,omitempty"`
}
