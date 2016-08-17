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

type CreateAccountParams struct {
	p map[string]interface{}
}

func (p *CreateAccountParams) toURLValues() url.Values {
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
	if v, found := p.p["email"]; found {
		u.Set("email", v.(string))
	}
	if v, found := p.p["firstname"]; found {
		u.Set("firstname", v.(string))
	}
	if v, found := p.p["lastname"]; found {
		u.Set("lastname", v.(string))
	}
	if v, found := p.p["networkdomain"]; found {
		u.Set("networkdomain", v.(string))
	}
	if v, found := p.p["password"]; found {
		u.Set("password", v.(string))
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

func (p *CreateAccountParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *CreateAccountParams) SetAccountdetails(v map[string]string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["accountdetails"] = v
	return
}

func (p *CreateAccountParams) SetAccountid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["accountid"] = v
	return
}

func (p *CreateAccountParams) SetAccounttype(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["accounttype"] = v
	return
}

func (p *CreateAccountParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *CreateAccountParams) SetEmail(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["email"] = v
	return
}

func (p *CreateAccountParams) SetFirstname(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["firstname"] = v
	return
}

func (p *CreateAccountParams) SetLastname(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["lastname"] = v
	return
}

func (p *CreateAccountParams) SetNetworkdomain(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["networkdomain"] = v
	return
}

func (p *CreateAccountParams) SetPassword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["password"] = v
	return
}

func (p *CreateAccountParams) SetTimezone(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["timezone"] = v
	return
}

func (p *CreateAccountParams) SetUserid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["userid"] = v
	return
}

func (p *CreateAccountParams) SetUsername(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["username"] = v
	return
}

// You should always use this function to get a new CreateAccountParams instance,
// as then you are sure you have configured all required params
func (s *AccountService) NewCreateAccountParams(accounttype int, email string, firstname string, lastname string, password string, username string) *CreateAccountParams {
	p := &CreateAccountParams{}
	p.p = make(map[string]interface{})
	p.p["accounttype"] = accounttype
	p.p["email"] = email
	p.p["firstname"] = firstname
	p.p["lastname"] = lastname
	p.p["password"] = password
	p.p["username"] = username
	return p
}

// Creates an account
func (s *AccountService) CreateAccount(p *CreateAccountParams) (*CreateAccountResponse, error) {
	resp, err := s.cs.newRequest("createAccount", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r CreateAccountResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type CreateAccountResponse struct {
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

type DeleteAccountParams struct {
	p map[string]interface{}
}

func (p *DeleteAccountParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	return u
}

func (p *DeleteAccountParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new DeleteAccountParams instance,
// as then you are sure you have configured all required params
func (s *AccountService) NewDeleteAccountParams(id string) *DeleteAccountParams {
	p := &DeleteAccountParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Deletes a account, and all users associated with this account
func (s *AccountService) DeleteAccount(p *DeleteAccountParams) (*DeleteAccountResponse, error) {
	resp, err := s.cs.newRequest("deleteAccount", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r DeleteAccountResponse
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

type DeleteAccountResponse struct {
	JobID       string `json:"jobid,omitempty"`
	Displaytext string `json:"displaytext,omitempty"`
	Success     bool   `json:"success,omitempty"`
}

type UpdateAccountParams struct {
	p map[string]interface{}
}

func (p *UpdateAccountParams) toURLValues() url.Values {
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
	if v, found := p.p["domainid"]; found {
		u.Set("domainid", v.(string))
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	if v, found := p.p["networkdomain"]; found {
		u.Set("networkdomain", v.(string))
	}
	if v, found := p.p["newname"]; found {
		u.Set("newname", v.(string))
	}
	return u
}

func (p *UpdateAccountParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *UpdateAccountParams) SetAccountdetails(v map[string]string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["accountdetails"] = v
	return
}

func (p *UpdateAccountParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *UpdateAccountParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *UpdateAccountParams) SetNetworkdomain(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["networkdomain"] = v
	return
}

func (p *UpdateAccountParams) SetNewname(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["newname"] = v
	return
}

// You should always use this function to get a new UpdateAccountParams instance,
// as then you are sure you have configured all required params
func (s *AccountService) NewUpdateAccountParams(newname string) *UpdateAccountParams {
	p := &UpdateAccountParams{}
	p.p = make(map[string]interface{})
	p.p["newname"] = newname
	return p
}

// Updates account information for the authenticated user
func (s *AccountService) UpdateAccount(p *UpdateAccountParams) (*UpdateAccountResponse, error) {
	resp, err := s.cs.newRequest("updateAccount", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r UpdateAccountResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type UpdateAccountResponse struct {
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

type DisableAccountParams struct {
	p map[string]interface{}
}

func (p *DisableAccountParams) toURLValues() url.Values {
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
	if v, found := p.p["lock"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("lock", vv)
	}
	return u
}

func (p *DisableAccountParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *DisableAccountParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *DisableAccountParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *DisableAccountParams) SetLock(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["lock"] = v
	return
}

// You should always use this function to get a new DisableAccountParams instance,
// as then you are sure you have configured all required params
func (s *AccountService) NewDisableAccountParams(lock bool) *DisableAccountParams {
	p := &DisableAccountParams{}
	p.p = make(map[string]interface{})
	p.p["lock"] = lock
	return p
}

// Disables an account
func (s *AccountService) DisableAccount(p *DisableAccountParams) (*DisableAccountResponse, error) {
	resp, err := s.cs.newRequest("disableAccount", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r DisableAccountResponse
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

type DisableAccountResponse struct {
	JobID                     string            `json:"jobid,omitempty"`
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

type EnableAccountParams struct {
	p map[string]interface{}
}

func (p *EnableAccountParams) toURLValues() url.Values {
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
	return u
}

func (p *EnableAccountParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *EnableAccountParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *EnableAccountParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new EnableAccountParams instance,
// as then you are sure you have configured all required params
func (s *AccountService) NewEnableAccountParams() *EnableAccountParams {
	p := &EnableAccountParams{}
	p.p = make(map[string]interface{})
	return p
}

// Enables an account
func (s *AccountService) EnableAccount(p *EnableAccountParams) (*EnableAccountResponse, error) {
	resp, err := s.cs.newRequest("enableAccount", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r EnableAccountResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type EnableAccountResponse struct {
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

type LockAccountParams struct {
	p map[string]interface{}
}

func (p *LockAccountParams) toURLValues() url.Values {
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
	return u
}

func (p *LockAccountParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *LockAccountParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

// You should always use this function to get a new LockAccountParams instance,
// as then you are sure you have configured all required params
func (s *AccountService) NewLockAccountParams(account string, domainid string) *LockAccountParams {
	p := &LockAccountParams{}
	p.p = make(map[string]interface{})
	p.p["account"] = account
	p.p["domainid"] = domainid
	return p
}

// This deprecated function used to locks an account. Look for the API DisableAccount instead
func (s *AccountService) LockAccount(p *LockAccountParams) (*LockAccountResponse, error) {
	resp, err := s.cs.newRequest("lockAccount", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r LockAccountResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type LockAccountResponse struct {
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

type ListAccountsParams struct {
	p map[string]interface{}
}

func (p *ListAccountsParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["accounttype"]; found {
		vv := strconv.FormatInt(v.(int64), 10)
		u.Set("accounttype", vv)
	}
	if v, found := p.p["domainid"]; found {
		u.Set("domainid", v.(string))
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	if v, found := p.p["iscleanuprequired"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("iscleanuprequired", vv)
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
	if v, found := p.p["state"]; found {
		u.Set("state", v.(string))
	}
	return u
}

func (p *ListAccountsParams) SetAccounttype(v int64) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["accounttype"] = v
	return
}

func (p *ListAccountsParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *ListAccountsParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ListAccountsParams) SetIscleanuprequired(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["iscleanuprequired"] = v
	return
}

func (p *ListAccountsParams) SetIsrecursive(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isrecursive"] = v
	return
}

func (p *ListAccountsParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListAccountsParams) SetListall(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["listall"] = v
	return
}

func (p *ListAccountsParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *ListAccountsParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListAccountsParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

func (p *ListAccountsParams) SetState(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["state"] = v
	return
}

// You should always use this function to get a new ListAccountsParams instance,
// as then you are sure you have configured all required params
func (s *AccountService) NewListAccountsParams() *ListAccountsParams {
	p := &ListAccountsParams{}
	p.p = make(map[string]interface{})
	return p
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *AccountService) GetAccountID(name string, opts ...OptionFunc) (string, int, error) {
	p := &ListAccountsParams{}
	p.p = make(map[string]interface{})

	p.p["name"] = name

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return "", -1, err
		}
	}

	l, err := s.ListAccounts(p)
	if err != nil {
		return "", -1, err
	}

	if l.Count == 0 {
		return "", l.Count, fmt.Errorf("No match found for %s: %+v", name, l)
	}

	if l.Count == 1 {
		return l.Accounts[0].Id, l.Count, nil
	}

	if l.Count > 1 {
		for _, v := range l.Accounts {
			if v.Name == name {
				return v.Id, l.Count, nil
			}
		}
	}
	return "", l.Count, fmt.Errorf("Could not find an exact match for %s: %+v", name, l)
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *AccountService) GetAccountByName(name string, opts ...OptionFunc) (*Account, int, error) {
	id, count, err := s.GetAccountID(name, opts...)
	if err != nil {
		return nil, count, err
	}

	r, count, err := s.GetAccountByID(id, opts...)
	if err != nil {
		return nil, count, err
	}
	return r, count, nil
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *AccountService) GetAccountByID(id string, opts ...OptionFunc) (*Account, int, error) {
	p := &ListAccountsParams{}
	p.p = make(map[string]interface{})

	p.p["id"] = id

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return nil, -1, err
		}
	}

	l, err := s.ListAccounts(p)
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
		return l.Accounts[0], l.Count, nil
	}
	return nil, l.Count, fmt.Errorf("There is more then one result for Account UUID: %s!", id)
}

// Lists accounts and provides detailed account information for listed accounts
func (s *AccountService) ListAccounts(p *ListAccountsParams) (*ListAccountsResponse, error) {
	resp, err := s.cs.newRequest("listAccounts", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListAccountsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListAccountsResponse struct {
	Count    int        `json:"count"`
	Accounts []*Account `json:"account"`
}

type Account struct {
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

type MarkDefaultZoneForAccountParams struct {
	p map[string]interface{}
}

func (p *MarkDefaultZoneForAccountParams) toURLValues() url.Values {
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
	if v, found := p.p["zoneid"]; found {
		u.Set("zoneid", v.(string))
	}
	return u
}

func (p *MarkDefaultZoneForAccountParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *MarkDefaultZoneForAccountParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *MarkDefaultZoneForAccountParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new MarkDefaultZoneForAccountParams instance,
// as then you are sure you have configured all required params
func (s *AccountService) NewMarkDefaultZoneForAccountParams(account string, domainid string, zoneid string) *MarkDefaultZoneForAccountParams {
	p := &MarkDefaultZoneForAccountParams{}
	p.p = make(map[string]interface{})
	p.p["account"] = account
	p.p["domainid"] = domainid
	p.p["zoneid"] = zoneid
	return p
}

// Marks a default zone for this account
func (s *AccountService) MarkDefaultZoneForAccount(p *MarkDefaultZoneForAccountParams) (*MarkDefaultZoneForAccountResponse, error) {
	resp, err := s.cs.newRequest("markDefaultZoneForAccount", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r MarkDefaultZoneForAccountResponse
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

type MarkDefaultZoneForAccountResponse struct {
	JobID                     string            `json:"jobid,omitempty"`
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

type AddAccountToProjectParams struct {
	p map[string]interface{}
}

func (p *AddAccountToProjectParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["account"]; found {
		u.Set("account", v.(string))
	}
	if v, found := p.p["email"]; found {
		u.Set("email", v.(string))
	}
	if v, found := p.p["projectid"]; found {
		u.Set("projectid", v.(string))
	}
	return u
}

func (p *AddAccountToProjectParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *AddAccountToProjectParams) SetEmail(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["email"] = v
	return
}

func (p *AddAccountToProjectParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

// You should always use this function to get a new AddAccountToProjectParams instance,
// as then you are sure you have configured all required params
func (s *AccountService) NewAddAccountToProjectParams(projectid string) *AddAccountToProjectParams {
	p := &AddAccountToProjectParams{}
	p.p = make(map[string]interface{})
	p.p["projectid"] = projectid
	return p
}

// Adds account to a project
func (s *AccountService) AddAccountToProject(p *AddAccountToProjectParams) (*AddAccountToProjectResponse, error) {
	resp, err := s.cs.newRequest("addAccountToProject", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r AddAccountToProjectResponse
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

type AddAccountToProjectResponse struct {
	JobID       string `json:"jobid,omitempty"`
	Displaytext string `json:"displaytext,omitempty"`
	Success     bool   `json:"success,omitempty"`
}

type DeleteAccountFromProjectParams struct {
	p map[string]interface{}
}

func (p *DeleteAccountFromProjectParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["account"]; found {
		u.Set("account", v.(string))
	}
	if v, found := p.p["projectid"]; found {
		u.Set("projectid", v.(string))
	}
	return u
}

func (p *DeleteAccountFromProjectParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *DeleteAccountFromProjectParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

// You should always use this function to get a new DeleteAccountFromProjectParams instance,
// as then you are sure you have configured all required params
func (s *AccountService) NewDeleteAccountFromProjectParams(account string, projectid string) *DeleteAccountFromProjectParams {
	p := &DeleteAccountFromProjectParams{}
	p.p = make(map[string]interface{})
	p.p["account"] = account
	p.p["projectid"] = projectid
	return p
}

// Deletes account from the project
func (s *AccountService) DeleteAccountFromProject(p *DeleteAccountFromProjectParams) (*DeleteAccountFromProjectResponse, error) {
	resp, err := s.cs.newRequest("deleteAccountFromProject", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r DeleteAccountFromProjectResponse
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

type DeleteAccountFromProjectResponse struct {
	JobID       string `json:"jobid,omitempty"`
	Displaytext string `json:"displaytext,omitempty"`
	Success     bool   `json:"success,omitempty"`
}

type ListProjectAccountsParams struct {
	p map[string]interface{}
}

func (p *ListProjectAccountsParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["account"]; found {
		u.Set("account", v.(string))
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
	if v, found := p.p["projectid"]; found {
		u.Set("projectid", v.(string))
	}
	if v, found := p.p["role"]; found {
		u.Set("role", v.(string))
	}
	return u
}

func (p *ListProjectAccountsParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *ListProjectAccountsParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListProjectAccountsParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListProjectAccountsParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

func (p *ListProjectAccountsParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

func (p *ListProjectAccountsParams) SetRole(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["role"] = v
	return
}

// You should always use this function to get a new ListProjectAccountsParams instance,
// as then you are sure you have configured all required params
func (s *AccountService) NewListProjectAccountsParams(projectid string) *ListProjectAccountsParams {
	p := &ListProjectAccountsParams{}
	p.p = make(map[string]interface{})
	p.p["projectid"] = projectid
	return p
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *AccountService) GetProjectAccountID(keyword string, projectid string, opts ...OptionFunc) (string, int, error) {
	p := &ListProjectAccountsParams{}
	p.p = make(map[string]interface{})

	p.p["keyword"] = keyword
	p.p["projectid"] = projectid

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return "", -1, err
		}
	}

	l, err := s.ListProjectAccounts(p)
	if err != nil {
		return "", -1, err
	}

	if l.Count == 0 {
		return "", l.Count, fmt.Errorf("No match found for %s: %+v", keyword, l)
	}

	if l.Count == 1 {
		return l.ProjectAccounts[0].Id, l.Count, nil
	}

	if l.Count > 1 {
		for _, v := range l.ProjectAccounts {
			if v.Name == keyword {
				return v.Id, l.Count, nil
			}
		}
	}
	return "", l.Count, fmt.Errorf("Could not find an exact match for %s: %+v", keyword, l)
}

// Lists project's accounts
func (s *AccountService) ListProjectAccounts(p *ListProjectAccountsParams) (*ListProjectAccountsResponse, error) {
	resp, err := s.cs.newRequest("listProjectAccounts", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListProjectAccountsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListProjectAccountsResponse struct {
	Count           int               `json:"count"`
	ProjectAccounts []*ProjectAccount `json:"projectaccount"`
}

type ProjectAccount struct {
	Account                   string `json:"account,omitempty"`
	Cpuavailable              string `json:"cpuavailable,omitempty"`
	Cpulimit                  string `json:"cpulimit,omitempty"`
	Cputotal                  int64  `json:"cputotal,omitempty"`
	Displaytext               string `json:"displaytext,omitempty"`
	Domain                    string `json:"domain,omitempty"`
	Domainid                  string `json:"domainid,omitempty"`
	Id                        string `json:"id,omitempty"`
	Ipavailable               string `json:"ipavailable,omitempty"`
	Iplimit                   string `json:"iplimit,omitempty"`
	Iptotal                   int64  `json:"iptotal,omitempty"`
	Memoryavailable           string `json:"memoryavailable,omitempty"`
	Memorylimit               string `json:"memorylimit,omitempty"`
	Memorytotal               int64  `json:"memorytotal,omitempty"`
	Name                      string `json:"name,omitempty"`
	Networkavailable          string `json:"networkavailable,omitempty"`
	Networklimit              string `json:"networklimit,omitempty"`
	Networktotal              int64  `json:"networktotal,omitempty"`
	Primarystorageavailable   string `json:"primarystorageavailable,omitempty"`
	Primarystoragelimit       string `json:"primarystoragelimit,omitempty"`
	Primarystoragetotal       int64  `json:"primarystoragetotal,omitempty"`
	Secondarystorageavailable string `json:"secondarystorageavailable,omitempty"`
	Secondarystoragelimit     string `json:"secondarystoragelimit,omitempty"`
	Secondarystoragetotal     int64  `json:"secondarystoragetotal,omitempty"`
	Snapshotavailable         string `json:"snapshotavailable,omitempty"`
	Snapshotlimit             string `json:"snapshotlimit,omitempty"`
	Snapshottotal             int64  `json:"snapshottotal,omitempty"`
	State                     string `json:"state,omitempty"`
	Tags                      []struct {
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
	Templateavailable string `json:"templateavailable,omitempty"`
	Templatelimit     string `json:"templatelimit,omitempty"`
	Templatetotal     int64  `json:"templatetotal,omitempty"`
	Vmavailable       string `json:"vmavailable,omitempty"`
	Vmlimit           string `json:"vmlimit,omitempty"`
	Vmrunning         int    `json:"vmrunning,omitempty"`
	Vmstopped         int    `json:"vmstopped,omitempty"`
	Vmtotal           int64  `json:"vmtotal,omitempty"`
	Volumeavailable   string `json:"volumeavailable,omitempty"`
	Volumelimit       string `json:"volumelimit,omitempty"`
	Volumetotal       int64  `json:"volumetotal,omitempty"`
	Vpcavailable      string `json:"vpcavailable,omitempty"`
	Vpclimit          string `json:"vpclimit,omitempty"`
	Vpctotal          int64  `json:"vpctotal,omitempty"`
}

type GetSolidFireAccountIdParams struct {
	p map[string]interface{}
}

func (p *GetSolidFireAccountIdParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["accountid"]; found {
		u.Set("accountid", v.(string))
	}
	if v, found := p.p["storageid"]; found {
		u.Set("storageid", v.(string))
	}
	return u
}

func (p *GetSolidFireAccountIdParams) SetAccountid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["accountid"] = v
	return
}

func (p *GetSolidFireAccountIdParams) SetStorageid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["storageid"] = v
	return
}

// You should always use this function to get a new GetSolidFireAccountIdParams instance,
// as then you are sure you have configured all required params
func (s *AccountService) NewGetSolidFireAccountIdParams(accountid string, storageid string) *GetSolidFireAccountIdParams {
	p := &GetSolidFireAccountIdParams{}
	p.p = make(map[string]interface{})
	p.p["accountid"] = accountid
	p.p["storageid"] = storageid
	return p
}

// Get SolidFire Account ID
func (s *AccountService) GetSolidFireAccountId(p *GetSolidFireAccountIdParams) (*GetSolidFireAccountIdResponse, error) {
	resp, err := s.cs.newRequest("getSolidFireAccountId", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r GetSolidFireAccountIdResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type GetSolidFireAccountIdResponse struct {
	SolidFireAccountId int64 `json:"solidFireAccountId,omitempty"`
}
