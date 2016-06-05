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

type ListAlertsParams struct {
	p map[string]interface{}
}

func (p *ListAlertsParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
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
	if v, found := p.p["type"]; found {
		u.Set("type", v.(string))
	}
	return u
}

func (p *ListAlertsParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ListAlertsParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListAlertsParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *ListAlertsParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListAlertsParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

func (p *ListAlertsParams) SetType(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["alertType"] = v
	return
}

// You should always use this function to get a new ListAlertsParams instance,
// as then you are sure you have configured all required params
func (s *AlertService) NewListAlertsParams() *ListAlertsParams {
	p := &ListAlertsParams{}
	p.p = make(map[string]interface{})
	return p
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *AlertService) GetAlertID(name string, opts ...OptionFunc) (string, error) {
	p := &ListAlertsParams{}
	p.p = make(map[string]interface{})

	p.p["name"] = name

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return "", err
		}
	}

	l, err := s.ListAlerts(p)
	if err != nil {
		return "", err
	}

	if l.Count == 0 {
		return "", fmt.Errorf("No match found for %s: %+v", name, l)
	}

	if l.Count == 1 {
		return l.Alerts[0].Id, nil
	}

	if l.Count > 1 {
		for _, v := range l.Alerts {
			if v.Name == name {
				return v.Id, nil
			}
		}
	}
	return "", fmt.Errorf("Could not find an exact match for %s: %+v", name, l)
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *AlertService) GetAlertByName(name string, opts ...OptionFunc) (*Alert, int, error) {
	id, err := s.GetAlertID(name, opts...)
	if err != nil {
		return nil, -1, err
	}

	r, count, err := s.GetAlertByID(id, opts...)
	if err != nil {
		return nil, count, err
	}
	return r, count, nil
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *AlertService) GetAlertByID(id string, opts ...OptionFunc) (*Alert, int, error) {
	p := &ListAlertsParams{}
	p.p = make(map[string]interface{})

	p.p["id"] = id

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return nil, -1, err
		}
	}

	l, err := s.ListAlerts(p)
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
		return l.Alerts[0], l.Count, nil
	}
	return nil, l.Count, fmt.Errorf("There is more then one result for Alert UUID: %s!", id)
}

// Lists all alerts.
func (s *AlertService) ListAlerts(p *ListAlertsParams) (*ListAlertsResponse, error) {
	resp, err := s.cs.newRequest("listAlerts", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListAlertsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListAlertsResponse struct {
	Count  int      `json:"count"`
	Alerts []*Alert `json:"alert"`
}

type Alert struct {
	Description string `json:"description,omitempty"`
	Id          string `json:"id,omitempty"`
	Name        string `json:"name,omitempty"`
	Sent        string `json:"sent,omitempty"`
	Type        int    `json:"type,omitempty"`
}

type ArchiveAlertsParams struct {
	p map[string]interface{}
}

func (p *ArchiveAlertsParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["enddate"]; found {
		u.Set("enddate", v.(string))
	}
	if v, found := p.p["ids"]; found {
		vv := strings.Join(v.([]string), ",")
		u.Set("ids", vv)
	}
	if v, found := p.p["startdate"]; found {
		u.Set("startdate", v.(string))
	}
	if v, found := p.p["type"]; found {
		u.Set("type", v.(string))
	}
	return u
}

func (p *ArchiveAlertsParams) SetEnddate(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["enddate"] = v
	return
}

func (p *ArchiveAlertsParams) SetIds(v []string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["ids"] = v
	return
}

func (p *ArchiveAlertsParams) SetStartdate(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["startdate"] = v
	return
}

func (p *ArchiveAlertsParams) SetType(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["alertType"] = v
	return
}

// You should always use this function to get a new ArchiveAlertsParams instance,
// as then you are sure you have configured all required params
func (s *AlertService) NewArchiveAlertsParams() *ArchiveAlertsParams {
	p := &ArchiveAlertsParams{}
	p.p = make(map[string]interface{})
	return p
}

// Archive one or more alerts.
func (s *AlertService) ArchiveAlerts(p *ArchiveAlertsParams) (*ArchiveAlertsResponse, error) {
	resp, err := s.cs.newRequest("archiveAlerts", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ArchiveAlertsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ArchiveAlertsResponse struct {
	Displaytext string `json:"displaytext,omitempty"`
	Success     string `json:"success,omitempty"`
}

type DeleteAlertsParams struct {
	p map[string]interface{}
}

func (p *DeleteAlertsParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["enddate"]; found {
		u.Set("enddate", v.(string))
	}
	if v, found := p.p["ids"]; found {
		vv := strings.Join(v.([]string), ",")
		u.Set("ids", vv)
	}
	if v, found := p.p["startdate"]; found {
		u.Set("startdate", v.(string))
	}
	if v, found := p.p["type"]; found {
		u.Set("type", v.(string))
	}
	return u
}

func (p *DeleteAlertsParams) SetEnddate(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["enddate"] = v
	return
}

func (p *DeleteAlertsParams) SetIds(v []string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["ids"] = v
	return
}

func (p *DeleteAlertsParams) SetStartdate(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["startdate"] = v
	return
}

func (p *DeleteAlertsParams) SetType(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["alertType"] = v
	return
}

// You should always use this function to get a new DeleteAlertsParams instance,
// as then you are sure you have configured all required params
func (s *AlertService) NewDeleteAlertsParams() *DeleteAlertsParams {
	p := &DeleteAlertsParams{}
	p.p = make(map[string]interface{})
	return p
}

// Delete one or more alerts.
func (s *AlertService) DeleteAlerts(p *DeleteAlertsParams) (*DeleteAlertsResponse, error) {
	resp, err := s.cs.newRequest("deleteAlerts", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r DeleteAlertsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type DeleteAlertsResponse struct {
	Displaytext string `json:"displaytext,omitempty"`
	Success     string `json:"success,omitempty"`
}

type GenerateAlertParams struct {
	p map[string]interface{}
}

func (p *GenerateAlertParams) toURLValues() url.Values {
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
	if v, found := p.p["podid"]; found {
		u.Set("podid", v.(string))
	}
	if v, found := p.p["type"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("type", vv)
	}
	if v, found := p.p["zoneid"]; found {
		u.Set("zoneid", v.(string))
	}
	return u
}

func (p *GenerateAlertParams) SetDescription(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["description"] = v
	return
}

func (p *GenerateAlertParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *GenerateAlertParams) SetPodid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["podid"] = v
	return
}

func (p *GenerateAlertParams) SetType(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["alertType"] = v
	return
}

func (p *GenerateAlertParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new GenerateAlertParams instance,
// as then you are sure you have configured all required params
func (s *AlertService) NewGenerateAlertParams(description string, name string, alertType int) *GenerateAlertParams {
	p := &GenerateAlertParams{}
	p.p = make(map[string]interface{})
	p.p["description"] = description
	p.p["name"] = name
	p.p["alertType"] = alertType
	return p
}

// Generates an alert
func (s *AlertService) GenerateAlert(p *GenerateAlertParams) (*GenerateAlertResponse, error) {
	resp, err := s.cs.newRequest("generateAlert", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r GenerateAlertResponse
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

type GenerateAlertResponse struct {
	JobID       string `json:"jobid,omitempty"`
	Displaytext string `json:"displaytext,omitempty"`
	Success     bool   `json:"success,omitempty"`
}
