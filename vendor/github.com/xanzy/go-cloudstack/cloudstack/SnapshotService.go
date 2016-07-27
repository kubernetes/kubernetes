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

type CreateSnapshotParams struct {
	p map[string]interface{}
}

func (p *CreateSnapshotParams) toURLValues() url.Values {
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
	if v, found := p.p["name"]; found {
		u.Set("name", v.(string))
	}
	if v, found := p.p["policyid"]; found {
		u.Set("policyid", v.(string))
	}
	if v, found := p.p["quiescevm"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("quiescevm", vv)
	}
	if v, found := p.p["volumeid"]; found {
		u.Set("volumeid", v.(string))
	}
	return u
}

func (p *CreateSnapshotParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *CreateSnapshotParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *CreateSnapshotParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *CreateSnapshotParams) SetPolicyid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["policyid"] = v
	return
}

func (p *CreateSnapshotParams) SetQuiescevm(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["quiescevm"] = v
	return
}

func (p *CreateSnapshotParams) SetVolumeid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["volumeid"] = v
	return
}

// You should always use this function to get a new CreateSnapshotParams instance,
// as then you are sure you have configured all required params
func (s *SnapshotService) NewCreateSnapshotParams(volumeid string) *CreateSnapshotParams {
	p := &CreateSnapshotParams{}
	p.p = make(map[string]interface{})
	p.p["volumeid"] = volumeid
	return p
}

// Creates an instant snapshot of a volume.
func (s *SnapshotService) CreateSnapshot(p *CreateSnapshotParams) (*CreateSnapshotResponse, error) {
	resp, err := s.cs.newRequest("createSnapshot", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r CreateSnapshotResponse
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

type CreateSnapshotResponse struct {
	JobID        string `json:"jobid,omitempty"`
	Account      string `json:"account,omitempty"`
	Created      string `json:"created,omitempty"`
	Domain       string `json:"domain,omitempty"`
	Domainid     string `json:"domainid,omitempty"`
	Id           string `json:"id,omitempty"`
	Intervaltype string `json:"intervaltype,omitempty"`
	Name         string `json:"name,omitempty"`
	Physicalsize int64  `json:"physicalsize,omitempty"`
	Project      string `json:"project,omitempty"`
	Projectid    string `json:"projectid,omitempty"`
	Revertable   bool   `json:"revertable,omitempty"`
	Snapshottype string `json:"snapshottype,omitempty"`
	State        string `json:"state,omitempty"`
	Tags         []struct {
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
	Volumeid   string `json:"volumeid,omitempty"`
	Volumename string `json:"volumename,omitempty"`
	Volumetype string `json:"volumetype,omitempty"`
	Zoneid     string `json:"zoneid,omitempty"`
}

type ListSnapshotsParams struct {
	p map[string]interface{}
}

func (p *ListSnapshotsParams) toURLValues() url.Values {
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
	if v, found := p.p["intervaltype"]; found {
		u.Set("intervaltype", v.(string))
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
	if v, found := p.p["snapshottype"]; found {
		u.Set("snapshottype", v.(string))
	}
	if v, found := p.p["tags"]; found {
		i := 0
		for k, vv := range v.(map[string]string) {
			u.Set(fmt.Sprintf("tags[%d].key", i), k)
			u.Set(fmt.Sprintf("tags[%d].value", i), vv)
			i++
		}
	}
	if v, found := p.p["volumeid"]; found {
		u.Set("volumeid", v.(string))
	}
	if v, found := p.p["zoneid"]; found {
		u.Set("zoneid", v.(string))
	}
	return u
}

func (p *ListSnapshotsParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *ListSnapshotsParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *ListSnapshotsParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ListSnapshotsParams) SetIntervaltype(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["intervaltype"] = v
	return
}

func (p *ListSnapshotsParams) SetIsrecursive(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isrecursive"] = v
	return
}

func (p *ListSnapshotsParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListSnapshotsParams) SetListall(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["listall"] = v
	return
}

func (p *ListSnapshotsParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *ListSnapshotsParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListSnapshotsParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

func (p *ListSnapshotsParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

func (p *ListSnapshotsParams) SetSnapshottype(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["snapshottype"] = v
	return
}

func (p *ListSnapshotsParams) SetTags(v map[string]string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["tags"] = v
	return
}

func (p *ListSnapshotsParams) SetVolumeid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["volumeid"] = v
	return
}

func (p *ListSnapshotsParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new ListSnapshotsParams instance,
// as then you are sure you have configured all required params
func (s *SnapshotService) NewListSnapshotsParams() *ListSnapshotsParams {
	p := &ListSnapshotsParams{}
	p.p = make(map[string]interface{})
	return p
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *SnapshotService) GetSnapshotID(name string, opts ...OptionFunc) (string, error) {
	p := &ListSnapshotsParams{}
	p.p = make(map[string]interface{})

	p.p["name"] = name

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return "", err
		}
	}

	l, err := s.ListSnapshots(p)
	if err != nil {
		return "", err
	}

	if l.Count == 0 {
		return "", fmt.Errorf("No match found for %s: %+v", name, l)
	}

	if l.Count == 1 {
		return l.Snapshots[0].Id, nil
	}

	if l.Count > 1 {
		for _, v := range l.Snapshots {
			if v.Name == name {
				return v.Id, nil
			}
		}
	}
	return "", fmt.Errorf("Could not find an exact match for %s: %+v", name, l)
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *SnapshotService) GetSnapshotByName(name string, opts ...OptionFunc) (*Snapshot, int, error) {
	id, err := s.GetSnapshotID(name, opts...)
	if err != nil {
		return nil, -1, err
	}

	r, count, err := s.GetSnapshotByID(id, opts...)
	if err != nil {
		return nil, count, err
	}
	return r, count, nil
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *SnapshotService) GetSnapshotByID(id string, opts ...OptionFunc) (*Snapshot, int, error) {
	p := &ListSnapshotsParams{}
	p.p = make(map[string]interface{})

	p.p["id"] = id

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return nil, -1, err
		}
	}

	l, err := s.ListSnapshots(p)
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
		return l.Snapshots[0], l.Count, nil
	}
	return nil, l.Count, fmt.Errorf("There is more then one result for Snapshot UUID: %s!", id)
}

// Lists all available snapshots for the account.
func (s *SnapshotService) ListSnapshots(p *ListSnapshotsParams) (*ListSnapshotsResponse, error) {
	resp, err := s.cs.newRequest("listSnapshots", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListSnapshotsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListSnapshotsResponse struct {
	Count     int         `json:"count"`
	Snapshots []*Snapshot `json:"snapshot"`
}

type Snapshot struct {
	Account      string `json:"account,omitempty"`
	Created      string `json:"created,omitempty"`
	Domain       string `json:"domain,omitempty"`
	Domainid     string `json:"domainid,omitempty"`
	Id           string `json:"id,omitempty"`
	Intervaltype string `json:"intervaltype,omitempty"`
	Name         string `json:"name,omitempty"`
	Physicalsize int64  `json:"physicalsize,omitempty"`
	Project      string `json:"project,omitempty"`
	Projectid    string `json:"projectid,omitempty"`
	Revertable   bool   `json:"revertable,omitempty"`
	Snapshottype string `json:"snapshottype,omitempty"`
	State        string `json:"state,omitempty"`
	Tags         []struct {
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
	Volumeid   string `json:"volumeid,omitempty"`
	Volumename string `json:"volumename,omitempty"`
	Volumetype string `json:"volumetype,omitempty"`
	Zoneid     string `json:"zoneid,omitempty"`
}

type DeleteSnapshotParams struct {
	p map[string]interface{}
}

func (p *DeleteSnapshotParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	return u
}

func (p *DeleteSnapshotParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new DeleteSnapshotParams instance,
// as then you are sure you have configured all required params
func (s *SnapshotService) NewDeleteSnapshotParams(id string) *DeleteSnapshotParams {
	p := &DeleteSnapshotParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Deletes a snapshot of a disk volume.
func (s *SnapshotService) DeleteSnapshot(p *DeleteSnapshotParams) (*DeleteSnapshotResponse, error) {
	resp, err := s.cs.newRequest("deleteSnapshot", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r DeleteSnapshotResponse
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

type DeleteSnapshotResponse struct {
	JobID       string `json:"jobid,omitempty"`
	Displaytext string `json:"displaytext,omitempty"`
	Success     bool   `json:"success,omitempty"`
}

type CreateSnapshotPolicyParams struct {
	p map[string]interface{}
}

func (p *CreateSnapshotPolicyParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["fordisplay"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("fordisplay", vv)
	}
	if v, found := p.p["intervaltype"]; found {
		u.Set("intervaltype", v.(string))
	}
	if v, found := p.p["maxsnaps"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("maxsnaps", vv)
	}
	if v, found := p.p["schedule"]; found {
		u.Set("schedule", v.(string))
	}
	if v, found := p.p["timezone"]; found {
		u.Set("timezone", v.(string))
	}
	if v, found := p.p["volumeid"]; found {
		u.Set("volumeid", v.(string))
	}
	return u
}

func (p *CreateSnapshotPolicyParams) SetFordisplay(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["fordisplay"] = v
	return
}

func (p *CreateSnapshotPolicyParams) SetIntervaltype(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["intervaltype"] = v
	return
}

func (p *CreateSnapshotPolicyParams) SetMaxsnaps(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["maxsnaps"] = v
	return
}

func (p *CreateSnapshotPolicyParams) SetSchedule(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["schedule"] = v
	return
}

func (p *CreateSnapshotPolicyParams) SetTimezone(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["timezone"] = v
	return
}

func (p *CreateSnapshotPolicyParams) SetVolumeid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["volumeid"] = v
	return
}

// You should always use this function to get a new CreateSnapshotPolicyParams instance,
// as then you are sure you have configured all required params
func (s *SnapshotService) NewCreateSnapshotPolicyParams(intervaltype string, maxsnaps int, schedule string, timezone string, volumeid string) *CreateSnapshotPolicyParams {
	p := &CreateSnapshotPolicyParams{}
	p.p = make(map[string]interface{})
	p.p["intervaltype"] = intervaltype
	p.p["maxsnaps"] = maxsnaps
	p.p["schedule"] = schedule
	p.p["timezone"] = timezone
	p.p["volumeid"] = volumeid
	return p
}

// Creates a snapshot policy for the account.
func (s *SnapshotService) CreateSnapshotPolicy(p *CreateSnapshotPolicyParams) (*CreateSnapshotPolicyResponse, error) {
	resp, err := s.cs.newRequest("createSnapshotPolicy", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r CreateSnapshotPolicyResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type CreateSnapshotPolicyResponse struct {
	Fordisplay   bool   `json:"fordisplay,omitempty"`
	Id           string `json:"id,omitempty"`
	Intervaltype int    `json:"intervaltype,omitempty"`
	Maxsnaps     int    `json:"maxsnaps,omitempty"`
	Schedule     string `json:"schedule,omitempty"`
	Timezone     string `json:"timezone,omitempty"`
	Volumeid     string `json:"volumeid,omitempty"`
}

type UpdateSnapshotPolicyParams struct {
	p map[string]interface{}
}

func (p *UpdateSnapshotPolicyParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["customid"]; found {
		u.Set("customid", v.(string))
	}
	if v, found := p.p["fordisplay"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("fordisplay", vv)
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	return u
}

func (p *UpdateSnapshotPolicyParams) SetCustomid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["customid"] = v
	return
}

func (p *UpdateSnapshotPolicyParams) SetFordisplay(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["fordisplay"] = v
	return
}

func (p *UpdateSnapshotPolicyParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new UpdateSnapshotPolicyParams instance,
// as then you are sure you have configured all required params
func (s *SnapshotService) NewUpdateSnapshotPolicyParams() *UpdateSnapshotPolicyParams {
	p := &UpdateSnapshotPolicyParams{}
	p.p = make(map[string]interface{})
	return p
}

// Updates the snapshot policy.
func (s *SnapshotService) UpdateSnapshotPolicy(p *UpdateSnapshotPolicyParams) (*UpdateSnapshotPolicyResponse, error) {
	resp, err := s.cs.newRequest("updateSnapshotPolicy", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r UpdateSnapshotPolicyResponse
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

type UpdateSnapshotPolicyResponse struct {
	JobID        string `json:"jobid,omitempty"`
	Fordisplay   bool   `json:"fordisplay,omitempty"`
	Id           string `json:"id,omitempty"`
	Intervaltype int    `json:"intervaltype,omitempty"`
	Maxsnaps     int    `json:"maxsnaps,omitempty"`
	Schedule     string `json:"schedule,omitempty"`
	Timezone     string `json:"timezone,omitempty"`
	Volumeid     string `json:"volumeid,omitempty"`
}

type DeleteSnapshotPoliciesParams struct {
	p map[string]interface{}
}

func (p *DeleteSnapshotPoliciesParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	if v, found := p.p["ids"]; found {
		vv := strings.Join(v.([]string), ",")
		u.Set("ids", vv)
	}
	return u
}

func (p *DeleteSnapshotPoliciesParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *DeleteSnapshotPoliciesParams) SetIds(v []string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["ids"] = v
	return
}

// You should always use this function to get a new DeleteSnapshotPoliciesParams instance,
// as then you are sure you have configured all required params
func (s *SnapshotService) NewDeleteSnapshotPoliciesParams() *DeleteSnapshotPoliciesParams {
	p := &DeleteSnapshotPoliciesParams{}
	p.p = make(map[string]interface{})
	return p
}

// Deletes snapshot policies for the account.
func (s *SnapshotService) DeleteSnapshotPolicies(p *DeleteSnapshotPoliciesParams) (*DeleteSnapshotPoliciesResponse, error) {
	resp, err := s.cs.newRequest("deleteSnapshotPolicies", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r DeleteSnapshotPoliciesResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type DeleteSnapshotPoliciesResponse struct {
	Displaytext string `json:"displaytext,omitempty"`
	Success     string `json:"success,omitempty"`
}

type ListSnapshotPoliciesParams struct {
	p map[string]interface{}
}

func (p *ListSnapshotPoliciesParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["fordisplay"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("fordisplay", vv)
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
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
	if v, found := p.p["volumeid"]; found {
		u.Set("volumeid", v.(string))
	}
	return u
}

func (p *ListSnapshotPoliciesParams) SetFordisplay(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["fordisplay"] = v
	return
}

func (p *ListSnapshotPoliciesParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ListSnapshotPoliciesParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListSnapshotPoliciesParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListSnapshotPoliciesParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

func (p *ListSnapshotPoliciesParams) SetVolumeid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["volumeid"] = v
	return
}

// You should always use this function to get a new ListSnapshotPoliciesParams instance,
// as then you are sure you have configured all required params
func (s *SnapshotService) NewListSnapshotPoliciesParams() *ListSnapshotPoliciesParams {
	p := &ListSnapshotPoliciesParams{}
	p.p = make(map[string]interface{})
	return p
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *SnapshotService) GetSnapshotPolicyByID(id string, opts ...OptionFunc) (*SnapshotPolicy, int, error) {
	p := &ListSnapshotPoliciesParams{}
	p.p = make(map[string]interface{})

	p.p["id"] = id

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return nil, -1, err
		}
	}

	l, err := s.ListSnapshotPolicies(p)
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
		return l.SnapshotPolicies[0], l.Count, nil
	}
	return nil, l.Count, fmt.Errorf("There is more then one result for SnapshotPolicy UUID: %s!", id)
}

// Lists snapshot policies.
func (s *SnapshotService) ListSnapshotPolicies(p *ListSnapshotPoliciesParams) (*ListSnapshotPoliciesResponse, error) {
	resp, err := s.cs.newRequest("listSnapshotPolicies", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListSnapshotPoliciesResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListSnapshotPoliciesResponse struct {
	Count            int               `json:"count"`
	SnapshotPolicies []*SnapshotPolicy `json:"snapshotpolicy"`
}

type SnapshotPolicy struct {
	Fordisplay   bool   `json:"fordisplay,omitempty"`
	Id           string `json:"id,omitempty"`
	Intervaltype int    `json:"intervaltype,omitempty"`
	Maxsnaps     int    `json:"maxsnaps,omitempty"`
	Schedule     string `json:"schedule,omitempty"`
	Timezone     string `json:"timezone,omitempty"`
	Volumeid     string `json:"volumeid,omitempty"`
}

type RevertSnapshotParams struct {
	p map[string]interface{}
}

func (p *RevertSnapshotParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	return u
}

func (p *RevertSnapshotParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new RevertSnapshotParams instance,
// as then you are sure you have configured all required params
func (s *SnapshotService) NewRevertSnapshotParams(id string) *RevertSnapshotParams {
	p := &RevertSnapshotParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// revert a volume snapshot.
func (s *SnapshotService) RevertSnapshot(p *RevertSnapshotParams) (*RevertSnapshotResponse, error) {
	resp, err := s.cs.newRequest("revertSnapshot", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r RevertSnapshotResponse
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

type RevertSnapshotResponse struct {
	JobID        string `json:"jobid,omitempty"`
	Account      string `json:"account,omitempty"`
	Created      string `json:"created,omitempty"`
	Domain       string `json:"domain,omitempty"`
	Domainid     string `json:"domainid,omitempty"`
	Id           string `json:"id,omitempty"`
	Intervaltype string `json:"intervaltype,omitempty"`
	Name         string `json:"name,omitempty"`
	Physicalsize int64  `json:"physicalsize,omitempty"`
	Project      string `json:"project,omitempty"`
	Projectid    string `json:"projectid,omitempty"`
	Revertable   bool   `json:"revertable,omitempty"`
	Snapshottype string `json:"snapshottype,omitempty"`
	State        string `json:"state,omitempty"`
	Tags         []struct {
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
	Volumeid   string `json:"volumeid,omitempty"`
	Volumename string `json:"volumename,omitempty"`
	Volumetype string `json:"volumetype,omitempty"`
	Zoneid     string `json:"zoneid,omitempty"`
}

type ListVMSnapshotParams struct {
	p map[string]interface{}
}

func (p *ListVMSnapshotParams) toURLValues() url.Values {
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
	if v, found := p.p["state"]; found {
		u.Set("state", v.(string))
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
	if v, found := p.p["vmsnapshotid"]; found {
		u.Set("vmsnapshotid", v.(string))
	}
	return u
}

func (p *ListVMSnapshotParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *ListVMSnapshotParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *ListVMSnapshotParams) SetIsrecursive(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isrecursive"] = v
	return
}

func (p *ListVMSnapshotParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListVMSnapshotParams) SetListall(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["listall"] = v
	return
}

func (p *ListVMSnapshotParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *ListVMSnapshotParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListVMSnapshotParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

func (p *ListVMSnapshotParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

func (p *ListVMSnapshotParams) SetState(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["state"] = v
	return
}

func (p *ListVMSnapshotParams) SetTags(v map[string]string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["tags"] = v
	return
}

func (p *ListVMSnapshotParams) SetVirtualmachineid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["virtualmachineid"] = v
	return
}

func (p *ListVMSnapshotParams) SetVmsnapshotid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["vmsnapshotid"] = v
	return
}

// You should always use this function to get a new ListVMSnapshotParams instance,
// as then you are sure you have configured all required params
func (s *SnapshotService) NewListVMSnapshotParams() *ListVMSnapshotParams {
	p := &ListVMSnapshotParams{}
	p.p = make(map[string]interface{})
	return p
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *SnapshotService) GetVMSnapshotID(name string, opts ...OptionFunc) (string, error) {
	p := &ListVMSnapshotParams{}
	p.p = make(map[string]interface{})

	p.p["name"] = name

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return "", err
		}
	}

	l, err := s.ListVMSnapshot(p)
	if err != nil {
		return "", err
	}

	if l.Count == 0 {
		return "", fmt.Errorf("No match found for %s: %+v", name, l)
	}

	if l.Count == 1 {
		return l.VMSnapshot[0].Id, nil
	}

	if l.Count > 1 {
		for _, v := range l.VMSnapshot {
			if v.Name == name {
				return v.Id, nil
			}
		}
	}
	return "", fmt.Errorf("Could not find an exact match for %s: %+v", name, l)
}

// List virtual machine snapshot by conditions
func (s *SnapshotService) ListVMSnapshot(p *ListVMSnapshotParams) (*ListVMSnapshotResponse, error) {
	resp, err := s.cs.newRequest("listVMSnapshot", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListVMSnapshotResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListVMSnapshotResponse struct {
	Count      int           `json:"count"`
	VMSnapshot []*VMSnapshot `json:"vmsnapshot"`
}

type VMSnapshot struct {
	Account          string `json:"account,omitempty"`
	Created          string `json:"created,omitempty"`
	Current          bool   `json:"current,omitempty"`
	Description      string `json:"description,omitempty"`
	Displayname      string `json:"displayname,omitempty"`
	Domain           string `json:"domain,omitempty"`
	Domainid         string `json:"domainid,omitempty"`
	Id               string `json:"id,omitempty"`
	Name             string `json:"name,omitempty"`
	Parent           string `json:"parent,omitempty"`
	ParentName       string `json:"parentName,omitempty"`
	Project          string `json:"project,omitempty"`
	Projectid        string `json:"projectid,omitempty"`
	State            string `json:"state,omitempty"`
	Type             string `json:"type,omitempty"`
	Virtualmachineid string `json:"virtualmachineid,omitempty"`
	Zoneid           string `json:"zoneid,omitempty"`
}

type CreateVMSnapshotParams struct {
	p map[string]interface{}
}

func (p *CreateVMSnapshotParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["description"]; found {
		u.Set("description", v.(string))
	}
	if v, found := p.p["name"]; found {
		u.Set("name", v.(string))
	}
	if v, found := p.p["quiescevm"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("quiescevm", vv)
	}
	if v, found := p.p["snapshotmemory"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("snapshotmemory", vv)
	}
	if v, found := p.p["virtualmachineid"]; found {
		u.Set("virtualmachineid", v.(string))
	}
	return u
}

func (p *CreateVMSnapshotParams) SetDescription(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["description"] = v
	return
}

func (p *CreateVMSnapshotParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *CreateVMSnapshotParams) SetQuiescevm(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["quiescevm"] = v
	return
}

func (p *CreateVMSnapshotParams) SetSnapshotmemory(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["snapshotmemory"] = v
	return
}

func (p *CreateVMSnapshotParams) SetVirtualmachineid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["virtualmachineid"] = v
	return
}

// You should always use this function to get a new CreateVMSnapshotParams instance,
// as then you are sure you have configured all required params
func (s *SnapshotService) NewCreateVMSnapshotParams(virtualmachineid string) *CreateVMSnapshotParams {
	p := &CreateVMSnapshotParams{}
	p.p = make(map[string]interface{})
	p.p["virtualmachineid"] = virtualmachineid
	return p
}

// Creates snapshot for a vm.
func (s *SnapshotService) CreateVMSnapshot(p *CreateVMSnapshotParams) (*CreateVMSnapshotResponse, error) {
	resp, err := s.cs.newRequest("createVMSnapshot", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r CreateVMSnapshotResponse
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

type CreateVMSnapshotResponse struct {
	JobID            string `json:"jobid,omitempty"`
	Account          string `json:"account,omitempty"`
	Created          string `json:"created,omitempty"`
	Current          bool   `json:"current,omitempty"`
	Description      string `json:"description,omitempty"`
	Displayname      string `json:"displayname,omitempty"`
	Domain           string `json:"domain,omitempty"`
	Domainid         string `json:"domainid,omitempty"`
	Id               string `json:"id,omitempty"`
	Name             string `json:"name,omitempty"`
	Parent           string `json:"parent,omitempty"`
	ParentName       string `json:"parentName,omitempty"`
	Project          string `json:"project,omitempty"`
	Projectid        string `json:"projectid,omitempty"`
	State            string `json:"state,omitempty"`
	Type             string `json:"type,omitempty"`
	Virtualmachineid string `json:"virtualmachineid,omitempty"`
	Zoneid           string `json:"zoneid,omitempty"`
}

type DeleteVMSnapshotParams struct {
	p map[string]interface{}
}

func (p *DeleteVMSnapshotParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["vmsnapshotid"]; found {
		u.Set("vmsnapshotid", v.(string))
	}
	return u
}

func (p *DeleteVMSnapshotParams) SetVmsnapshotid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["vmsnapshotid"] = v
	return
}

// You should always use this function to get a new DeleteVMSnapshotParams instance,
// as then you are sure you have configured all required params
func (s *SnapshotService) NewDeleteVMSnapshotParams(vmsnapshotid string) *DeleteVMSnapshotParams {
	p := &DeleteVMSnapshotParams{}
	p.p = make(map[string]interface{})
	p.p["vmsnapshotid"] = vmsnapshotid
	return p
}

// Deletes a vmsnapshot.
func (s *SnapshotService) DeleteVMSnapshot(p *DeleteVMSnapshotParams) (*DeleteVMSnapshotResponse, error) {
	resp, err := s.cs.newRequest("deleteVMSnapshot", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r DeleteVMSnapshotResponse
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

type DeleteVMSnapshotResponse struct {
	JobID       string `json:"jobid,omitempty"`
	Displaytext string `json:"displaytext,omitempty"`
	Success     bool   `json:"success,omitempty"`
}

type RevertToVMSnapshotParams struct {
	p map[string]interface{}
}

func (p *RevertToVMSnapshotParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["vmsnapshotid"]; found {
		u.Set("vmsnapshotid", v.(string))
	}
	return u
}

func (p *RevertToVMSnapshotParams) SetVmsnapshotid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["vmsnapshotid"] = v
	return
}

// You should always use this function to get a new RevertToVMSnapshotParams instance,
// as then you are sure you have configured all required params
func (s *SnapshotService) NewRevertToVMSnapshotParams(vmsnapshotid string) *RevertToVMSnapshotParams {
	p := &RevertToVMSnapshotParams{}
	p.p = make(map[string]interface{})
	p.p["vmsnapshotid"] = vmsnapshotid
	return p
}

// Revert VM from a vmsnapshot.
func (s *SnapshotService) RevertToVMSnapshot(p *RevertToVMSnapshotParams) (*RevertToVMSnapshotResponse, error) {
	resp, err := s.cs.newRequest("revertToVMSnapshot", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r RevertToVMSnapshotResponse
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

type RevertToVMSnapshotResponse struct {
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
