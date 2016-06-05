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

type AttachIsoParams struct {
	p map[string]interface{}
}

func (p *AttachIsoParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	if v, found := p.p["virtualmachineid"]; found {
		u.Set("virtualmachineid", v.(string))
	}
	return u
}

func (p *AttachIsoParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *AttachIsoParams) SetVirtualmachineid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["virtualmachineid"] = v
	return
}

// You should always use this function to get a new AttachIsoParams instance,
// as then you are sure you have configured all required params
func (s *ISOService) NewAttachIsoParams(id string, virtualmachineid string) *AttachIsoParams {
	p := &AttachIsoParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	p.p["virtualmachineid"] = virtualmachineid
	return p
}

// Attaches an ISO to a virtual machine.
func (s *ISOService) AttachIso(p *AttachIsoParams) (*AttachIsoResponse, error) {
	resp, err := s.cs.newRequest("attachIso", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r AttachIsoResponse
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

type AttachIsoResponse struct {
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

type DetachIsoParams struct {
	p map[string]interface{}
}

func (p *DetachIsoParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["virtualmachineid"]; found {
		u.Set("virtualmachineid", v.(string))
	}
	return u
}

func (p *DetachIsoParams) SetVirtualmachineid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["virtualmachineid"] = v
	return
}

// You should always use this function to get a new DetachIsoParams instance,
// as then you are sure you have configured all required params
func (s *ISOService) NewDetachIsoParams(virtualmachineid string) *DetachIsoParams {
	p := &DetachIsoParams{}
	p.p = make(map[string]interface{})
	p.p["virtualmachineid"] = virtualmachineid
	return p
}

// Detaches any ISO file (if any) currently attached to a virtual machine.
func (s *ISOService) DetachIso(p *DetachIsoParams) (*DetachIsoResponse, error) {
	resp, err := s.cs.newRequest("detachIso", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r DetachIsoResponse
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

type DetachIsoResponse struct {
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

type ListIsosParams struct {
	p map[string]interface{}
}

func (p *ListIsosParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["account"]; found {
		u.Set("account", v.(string))
	}
	if v, found := p.p["bootable"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("bootable", vv)
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
	if v, found := p.p["isofilter"]; found {
		u.Set("isofilter", v.(string))
	}
	if v, found := p.p["ispublic"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("ispublic", vv)
	}
	if v, found := p.p["isready"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("isready", vv)
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
	if v, found := p.p["zoneid"]; found {
		u.Set("zoneid", v.(string))
	}
	return u
}

func (p *ListIsosParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *ListIsosParams) SetBootable(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["bootable"] = v
	return
}

func (p *ListIsosParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *ListIsosParams) SetHypervisor(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["hypervisor"] = v
	return
}

func (p *ListIsosParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ListIsosParams) SetIsofilter(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isofilter"] = v
	return
}

func (p *ListIsosParams) SetIspublic(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["ispublic"] = v
	return
}

func (p *ListIsosParams) SetIsready(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isready"] = v
	return
}

func (p *ListIsosParams) SetIsrecursive(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isrecursive"] = v
	return
}

func (p *ListIsosParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListIsosParams) SetListall(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["listall"] = v
	return
}

func (p *ListIsosParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *ListIsosParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListIsosParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

func (p *ListIsosParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

func (p *ListIsosParams) SetShowremoved(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["showremoved"] = v
	return
}

func (p *ListIsosParams) SetTags(v map[string]string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["tags"] = v
	return
}

func (p *ListIsosParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new ListIsosParams instance,
// as then you are sure you have configured all required params
func (s *ISOService) NewListIsosParams() *ListIsosParams {
	p := &ListIsosParams{}
	p.p = make(map[string]interface{})
	return p
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *ISOService) GetIsoID(name string, isofilter string, zoneid string, opts ...OptionFunc) (string, error) {
	p := &ListIsosParams{}
	p.p = make(map[string]interface{})

	p.p["name"] = name
	p.p["isofilter"] = isofilter
	p.p["zoneid"] = zoneid

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return "", err
		}
	}

	l, err := s.ListIsos(p)
	if err != nil {
		return "", err
	}

	if l.Count == 0 {
		return "", fmt.Errorf("No match found for %s: %+v", name, l)
	}

	if l.Count == 1 {
		return l.Isos[0].Id, nil
	}

	if l.Count > 1 {
		for _, v := range l.Isos {
			if v.Name == name {
				return v.Id, nil
			}
		}
	}
	return "", fmt.Errorf("Could not find an exact match for %s: %+v", name, l)
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *ISOService) GetIsoByName(name string, isofilter string, zoneid string, opts ...OptionFunc) (*Iso, int, error) {
	id, err := s.GetIsoID(name, isofilter, zoneid, opts...)
	if err != nil {
		return nil, -1, err
	}

	r, count, err := s.GetIsoByID(id, opts...)
	if err != nil {
		return nil, count, err
	}
	return r, count, nil
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *ISOService) GetIsoByID(id string, opts ...OptionFunc) (*Iso, int, error) {
	p := &ListIsosParams{}
	p.p = make(map[string]interface{})

	p.p["id"] = id

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return nil, -1, err
		}
	}

	l, err := s.ListIsos(p)
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
		return l.Isos[0], l.Count, nil
	}
	return nil, l.Count, fmt.Errorf("There is more then one result for Iso UUID: %s!", id)
}

// Lists all available ISO files.
func (s *ISOService) ListIsos(p *ListIsosParams) (*ListIsosResponse, error) {
	resp, err := s.cs.newRequest("listIsos", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListIsosResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListIsosResponse struct {
	Count int    `json:"count"`
	Isos  []*Iso `json:"iso"`
}

type Iso struct {
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

type RegisterIsoParams struct {
	p map[string]interface{}
}

func (p *RegisterIsoParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["account"]; found {
		u.Set("account", v.(string))
	}
	if v, found := p.p["bootable"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("bootable", vv)
	}
	if v, found := p.p["checksum"]; found {
		u.Set("checksum", v.(string))
	}
	if v, found := p.p["displaytext"]; found {
		u.Set("displaytext", v.(string))
	}
	if v, found := p.p["domainid"]; found {
		u.Set("domainid", v.(string))
	}
	if v, found := p.p["imagestoreuuid"]; found {
		u.Set("imagestoreuuid", v.(string))
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
	if v, found := p.p["name"]; found {
		u.Set("name", v.(string))
	}
	if v, found := p.p["ostypeid"]; found {
		u.Set("ostypeid", v.(string))
	}
	if v, found := p.p["projectid"]; found {
		u.Set("projectid", v.(string))
	}
	if v, found := p.p["url"]; found {
		u.Set("url", v.(string))
	}
	if v, found := p.p["zoneid"]; found {
		u.Set("zoneid", v.(string))
	}
	return u
}

func (p *RegisterIsoParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *RegisterIsoParams) SetBootable(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["bootable"] = v
	return
}

func (p *RegisterIsoParams) SetChecksum(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["checksum"] = v
	return
}

func (p *RegisterIsoParams) SetDisplaytext(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["displaytext"] = v
	return
}

func (p *RegisterIsoParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *RegisterIsoParams) SetImagestoreuuid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["imagestoreuuid"] = v
	return
}

func (p *RegisterIsoParams) SetIsdynamicallyscalable(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isdynamicallyscalable"] = v
	return
}

func (p *RegisterIsoParams) SetIsextractable(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isextractable"] = v
	return
}

func (p *RegisterIsoParams) SetIsfeatured(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isfeatured"] = v
	return
}

func (p *RegisterIsoParams) SetIspublic(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["ispublic"] = v
	return
}

func (p *RegisterIsoParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *RegisterIsoParams) SetOstypeid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["ostypeid"] = v
	return
}

func (p *RegisterIsoParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

func (p *RegisterIsoParams) SetUrl(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["url"] = v
	return
}

func (p *RegisterIsoParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new RegisterIsoParams instance,
// as then you are sure you have configured all required params
func (s *ISOService) NewRegisterIsoParams(displaytext string, name string, url string, zoneid string) *RegisterIsoParams {
	p := &RegisterIsoParams{}
	p.p = make(map[string]interface{})
	p.p["displaytext"] = displaytext
	p.p["name"] = name
	p.p["url"] = url
	p.p["zoneid"] = zoneid
	return p
}

// Registers an existing ISO into the CloudStack Cloud.
func (s *ISOService) RegisterIso(p *RegisterIsoParams) (*RegisterIsoResponse, error) {
	resp, err := s.cs.newRequest("registerIso", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r RegisterIsoResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type RegisterIsoResponse struct {
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

type UpdateIsoParams struct {
	p map[string]interface{}
}

func (p *UpdateIsoParams) toURLValues() url.Values {
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

func (p *UpdateIsoParams) SetBootable(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["bootable"] = v
	return
}

func (p *UpdateIsoParams) SetDetails(v map[string]string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["details"] = v
	return
}

func (p *UpdateIsoParams) SetDisplaytext(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["displaytext"] = v
	return
}

func (p *UpdateIsoParams) SetFormat(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["format"] = v
	return
}

func (p *UpdateIsoParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *UpdateIsoParams) SetIsdynamicallyscalable(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isdynamicallyscalable"] = v
	return
}

func (p *UpdateIsoParams) SetIsrouting(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isrouting"] = v
	return
}

func (p *UpdateIsoParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *UpdateIsoParams) SetOstypeid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["ostypeid"] = v
	return
}

func (p *UpdateIsoParams) SetPasswordenabled(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["passwordenabled"] = v
	return
}

func (p *UpdateIsoParams) SetRequireshvm(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["requireshvm"] = v
	return
}

func (p *UpdateIsoParams) SetSortkey(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["sortkey"] = v
	return
}

// You should always use this function to get a new UpdateIsoParams instance,
// as then you are sure you have configured all required params
func (s *ISOService) NewUpdateIsoParams(id string) *UpdateIsoParams {
	p := &UpdateIsoParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Updates an ISO file.
func (s *ISOService) UpdateIso(p *UpdateIsoParams) (*UpdateIsoResponse, error) {
	resp, err := s.cs.newRequest("updateIso", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r UpdateIsoResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type UpdateIsoResponse struct {
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

type DeleteIsoParams struct {
	p map[string]interface{}
}

func (p *DeleteIsoParams) toURLValues() url.Values {
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

func (p *DeleteIsoParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *DeleteIsoParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new DeleteIsoParams instance,
// as then you are sure you have configured all required params
func (s *ISOService) NewDeleteIsoParams(id string) *DeleteIsoParams {
	p := &DeleteIsoParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Deletes an ISO file.
func (s *ISOService) DeleteIso(p *DeleteIsoParams) (*DeleteIsoResponse, error) {
	resp, err := s.cs.newRequest("deleteIso", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r DeleteIsoResponse
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

type DeleteIsoResponse struct {
	JobID       string `json:"jobid,omitempty"`
	Displaytext string `json:"displaytext,omitempty"`
	Success     bool   `json:"success,omitempty"`
}

type CopyIsoParams struct {
	p map[string]interface{}
}

func (p *CopyIsoParams) toURLValues() url.Values {
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

func (p *CopyIsoParams) SetDestzoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["destzoneid"] = v
	return
}

func (p *CopyIsoParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *CopyIsoParams) SetSourcezoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["sourcezoneid"] = v
	return
}

// You should always use this function to get a new CopyIsoParams instance,
// as then you are sure you have configured all required params
func (s *ISOService) NewCopyIsoParams(destzoneid string, id string) *CopyIsoParams {
	p := &CopyIsoParams{}
	p.p = make(map[string]interface{})
	p.p["destzoneid"] = destzoneid
	p.p["id"] = id
	return p
}

// Copies an ISO from one zone to another.
func (s *ISOService) CopyIso(p *CopyIsoParams) (*CopyIsoResponse, error) {
	resp, err := s.cs.newRequest("copyIso", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r CopyIsoResponse
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

type CopyIsoResponse struct {
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

type UpdateIsoPermissionsParams struct {
	p map[string]interface{}
}

func (p *UpdateIsoPermissionsParams) toURLValues() url.Values {
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

func (p *UpdateIsoPermissionsParams) SetAccounts(v []string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["accounts"] = v
	return
}

func (p *UpdateIsoPermissionsParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *UpdateIsoPermissionsParams) SetIsextractable(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isextractable"] = v
	return
}

func (p *UpdateIsoPermissionsParams) SetIsfeatured(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isfeatured"] = v
	return
}

func (p *UpdateIsoPermissionsParams) SetIspublic(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["ispublic"] = v
	return
}

func (p *UpdateIsoPermissionsParams) SetOp(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["op"] = v
	return
}

func (p *UpdateIsoPermissionsParams) SetProjectids(v []string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectids"] = v
	return
}

// You should always use this function to get a new UpdateIsoPermissionsParams instance,
// as then you are sure you have configured all required params
func (s *ISOService) NewUpdateIsoPermissionsParams(id string) *UpdateIsoPermissionsParams {
	p := &UpdateIsoPermissionsParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Updates ISO permissions
func (s *ISOService) UpdateIsoPermissions(p *UpdateIsoPermissionsParams) (*UpdateIsoPermissionsResponse, error) {
	resp, err := s.cs.newRequest("updateIsoPermissions", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r UpdateIsoPermissionsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type UpdateIsoPermissionsResponse struct {
	Displaytext string `json:"displaytext,omitempty"`
	Success     string `json:"success,omitempty"`
}

type ListIsoPermissionsParams struct {
	p map[string]interface{}
}

func (p *ListIsoPermissionsParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	return u
}

func (p *ListIsoPermissionsParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new ListIsoPermissionsParams instance,
// as then you are sure you have configured all required params
func (s *ISOService) NewListIsoPermissionsParams(id string) *ListIsoPermissionsParams {
	p := &ListIsoPermissionsParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *ISOService) GetIsoPermissionByID(id string, opts ...OptionFunc) (*IsoPermission, int, error) {
	p := &ListIsoPermissionsParams{}
	p.p = make(map[string]interface{})

	p.p["id"] = id
	p.p["id"] = id

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return nil, -1, err
		}
	}

	l, err := s.ListIsoPermissions(p)
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
		return l.IsoPermissions[0], l.Count, nil
	}
	return nil, l.Count, fmt.Errorf("There is more then one result for IsoPermission UUID: %s!", id)
}

// List iso visibility and all accounts that have permissions to view this iso.
func (s *ISOService) ListIsoPermissions(p *ListIsoPermissionsParams) (*ListIsoPermissionsResponse, error) {
	resp, err := s.cs.newRequest("listIsoPermissions", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListIsoPermissionsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListIsoPermissionsResponse struct {
	Count          int              `json:"count"`
	IsoPermissions []*IsoPermission `json:"isopermission"`
}

type IsoPermission struct {
	Account    []string `json:"account,omitempty"`
	Domainid   string   `json:"domainid,omitempty"`
	Id         string   `json:"id,omitempty"`
	Ispublic   bool     `json:"ispublic,omitempty"`
	Projectids []string `json:"projectids,omitempty"`
}

type ExtractIsoParams struct {
	p map[string]interface{}
}

func (p *ExtractIsoParams) toURLValues() url.Values {
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

func (p *ExtractIsoParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ExtractIsoParams) SetMode(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["mode"] = v
	return
}

func (p *ExtractIsoParams) SetUrl(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["url"] = v
	return
}

func (p *ExtractIsoParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new ExtractIsoParams instance,
// as then you are sure you have configured all required params
func (s *ISOService) NewExtractIsoParams(id string, mode string) *ExtractIsoParams {
	p := &ExtractIsoParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	p.p["mode"] = mode
	return p
}

// Extracts an ISO
func (s *ISOService) ExtractIso(p *ExtractIsoParams) (*ExtractIsoResponse, error) {
	resp, err := s.cs.newRequest("extractIso", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ExtractIsoResponse
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

type ExtractIsoResponse struct {
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
