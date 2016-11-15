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
)

type LdapCreateAccountParams struct {
	p map[string]interface{}
}

func (p *LdapCreateAccountParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["account"]; found {
		u.Set("account", v.(string))
	}
	if v, found := p.p["accountdetails"]; found {
		i := 0
		for k, vv := range v.(map[string]string) {
			u.Set(fmt.Sprintf("accountdetails[%d].key", i), k)
			u.Set(fmt.Sprintf("accountdetails[%d].value", i), vv)
			i++
		}
	}
	if v, found := p.p["accountid"]; found {
		u.Set("accountid", v.(string))
	}
	if v, found := p.p["accounttype"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("accounttype", vv)
	}
	if v, found := p.p["domainid"]; found {
		u.Set("domainid", v.(string))
	}
	if v, found := p.p["networkdomain"]; found {
		u.Set("networkdomain", v.(string))
	}
	if v, found := p.p["timezone"]; found {
		u.Set("timezone", v.(string))
	}
	if v, found := p.p["userid"]; found {
		u.Set("userid", v.(string))
	}
	if v, found := p.p["username"]; found {
		u.Set("username", v.(string))
	}
	return u
}

func (p *LdapCreateAccountParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *LdapCreateAccountParams) SetAccountdetails(v map[string]string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["accountdetails"] = v
	return
}

func (p *LdapCreateAccountParams) SetAccountid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["accountid"] = v
	return
}

func (p *LdapCreateAccountParams) SetAccounttype(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["accounttype"] = v
	return
}

func (p *LdapCreateAccountParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *LdapCreateAccountParams) SetNetworkdomain(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["networkdomain"] = v
	return
}

func (p *LdapCreateAccountParams) SetTimezone(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["timezone"] = v
	return
}

func (p *LdapCreateAccountParams) SetUserid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["userid"] = v
	return
}

func (p *LdapCreateAccountParams) SetUsername(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["username"] = v
	return
}

// You should always use this function to get a new LdapCreateAccountParams instance,
// as then you are sure you have configured all required params
func (s *LDAPService) NewLdapCreateAccountParams(accounttype int, username string) *LdapCreateAccountParams {
	p := &LdapCreateAccountParams{}
	p.p = make(map[string]interface{})
	p.p["accounttype"] = accounttype
	p.p["username"] = username
	return p
}

// Creates an account from an LDAP user
func (s *LDAPService) LdapCreateAccount(p *LdapCreateAccountParams) (*LdapCreateAccountResponse, error) {
	resp, err := s.cs.newRequest("ldapCreateAccount", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r LdapCreateAccountResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type LdapCreateAccountResponse struct {
	Accountdetails            map[string]string `json:"accountdetails,omitempty"`
	Accounttype               int               `json:"accounttype,omitempty"`
	Cpuavailable              string            `json:"cpuavailable,omitempty"`
	Cpulimit                  string            `json:"cpulimit,omitempty"`
	Cputotal                  int64             `json:"cputotal,omitempty"`
	Defaultzoneid             string            `json:"defaultzoneid,omitempty"`
	Domain                    string            `json:"domain,omitempty"`
	Domainid                  string            `json:"domainid,omitempty"`
	Groups                    []string          `json:"groups,omitempty"`
	Id                        string            `json:"id,omitempty"`
	Ipavailable               string            `json:"ipavailable,omitempty"`
	Iplimit                   string            `json:"iplimit,omitempty"`
	Iptotal                   int64             `json:"iptotal,omitempty"`
	Iscleanuprequired         bool              `json:"iscleanuprequired,omitempty"`
	Isdefault                 bool              `json:"isdefault,omitempty"`
	Memoryavailable           string            `json:"memoryavailable,omitempty"`
	Memorylimit               string            `json:"memorylimit,omitempty"`
	Memorytotal               int64             `json:"memorytotal,omitempty"`
	Name                      string            `json:"name,omitempty"`
	Networkavailable          string            `json:"networkavailable,omitempty"`
	Networkdomain             string            `json:"networkdomain,omitempty"`
	Networklimit              string            `json:"networklimit,omitempty"`
	Networktotal              int64             `json:"networktotal,omitempty"`
	Primarystorageavailable   string            `json:"primarystorageavailable,omitempty"`
	Primarystoragelimit       string            `json:"primarystoragelimit,omitempty"`
	Primarystoragetotal       int64             `json:"primarystoragetotal,omitempty"`
	Projectavailable          string            `json:"projectavailable,omitempty"`
	Projectlimit              string            `json:"projectlimit,omitempty"`
	Projecttotal              int64             `json:"projecttotal,omitempty"`
	Receivedbytes             int64             `json:"receivedbytes,omitempty"`
	Secondarystorageavailable string            `json:"secondarystorageavailable,omitempty"`
	Secondarystoragelimit     string            `json:"secondarystoragelimit,omitempty"`
	Secondarystoragetotal     int64             `json:"secondarystoragetotal,omitempty"`
	Sentbytes                 int64             `json:"sentbytes,omitempty"`
	Snapshotavailable         string            `json:"snapshotavailable,omitempty"`
	Snapshotlimit             string            `json:"snapshotlimit,omitempty"`
	Snapshottotal             int64             `json:"snapshottotal,omitempty"`
	State                     string            `json:"state,omitempty"`
	Templateavailable         string            `json:"templateavailable,omitempty"`
	Templatelimit             string            `json:"templatelimit,omitempty"`
	Templatetotal             int64             `json:"templatetotal,omitempty"`
	User                      []struct {
		Account             string `json:"account,omitempty"`
		Accountid           string `json:"accountid,omitempty"`
		Accounttype         int    `json:"accounttype,omitempty"`
		Apikey              string `json:"apikey,omitempty"`
		Created             string `json:"created,omitempty"`
		Domain              string `json:"domain,omitempty"`
		Domainid            string `json:"domainid,omitempty"`
		Email               string `json:"email,omitempty"`
		Firstname           string `json:"firstname,omitempty"`
		Id                  string `json:"id,omitempty"`
		Iscallerchilddomain bool   `json:"iscallerchilddomain,omitempty"`
		Isdefault           bool   `json:"isdefault,omitempty"`
		Lastname            string `json:"lastname,omitempty"`
		Secretkey           string `json:"secretkey,omitempty"`
		State               string `json:"state,omitempty"`
		Timezone            string `json:"timezone,omitempty"`
		Username            string `json:"username,omitempty"`
	} `json:"user,omitempty"`
	Vmavailable     string `json:"vmavailable,omitempty"`
	Vmlimit         string `json:"vmlimit,omitempty"`
	Vmrunning       int    `json:"vmrunning,omitempty"`
	Vmstopped       int    `json:"vmstopped,omitempty"`
	Vmtotal         int64  `json:"vmtotal,omitempty"`
	Volumeavailable string `json:"volumeavailable,omitempty"`
	Volumelimit     string `json:"volumelimit,omitempty"`
	Volumetotal     int64  `json:"volumetotal,omitempty"`
	Vpcavailable    string `json:"vpcavailable,omitempty"`
	Vpclimit        string `json:"vpclimit,omitempty"`
	Vpctotal        int64  `json:"vpctotal,omitempty"`
}
