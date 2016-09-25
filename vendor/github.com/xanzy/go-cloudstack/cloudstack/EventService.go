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

type ListEventsParams struct {
	p map[string]interface{}
}

func (p *ListEventsParams) toURLValues() url.Values {
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
	if v, found := p.p["duration"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("duration", vv)
	}
	if v, found := p.p["enddate"]; found {
		u.Set("enddate", v.(string))
	}
	if v, found := p.p["entrytime"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("entrytime", vv)
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
	if v, found := p.p["level"]; found {
		u.Set("level", v.(string))
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
	if v, found := p.p["startdate"]; found {
		u.Set("startdate", v.(string))
	}
	if v, found := p.p["type"]; found {
		u.Set("type", v.(string))
	}
	return u
}

func (p *ListEventsParams) SetAccount(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["account"] = v
	return
}

func (p *ListEventsParams) SetDomainid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainid"] = v
	return
}

func (p *ListEventsParams) SetDuration(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["duration"] = v
	return
}

func (p *ListEventsParams) SetEnddate(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["enddate"] = v
	return
}

func (p *ListEventsParams) SetEntrytime(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["entrytime"] = v
	return
}

func (p *ListEventsParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ListEventsParams) SetIsrecursive(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["isrecursive"] = v
	return
}

func (p *ListEventsParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListEventsParams) SetLevel(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["level"] = v
	return
}

func (p *ListEventsParams) SetListall(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["listall"] = v
	return
}

func (p *ListEventsParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListEventsParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

func (p *ListEventsParams) SetProjectid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["projectid"] = v
	return
}

func (p *ListEventsParams) SetStartdate(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["startdate"] = v
	return
}

func (p *ListEventsParams) SetType(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["eventType"] = v
	return
}

// You should always use this function to get a new ListEventsParams instance,
// as then you are sure you have configured all required params
func (s *EventService) NewListEventsParams() *ListEventsParams {
	p := &ListEventsParams{}
	p.p = make(map[string]interface{})
	return p
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *EventService) GetEventByID(id string, opts ...OptionFunc) (*Event, int, error) {
	p := &ListEventsParams{}
	p.p = make(map[string]interface{})

	p.p["id"] = id

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return nil, -1, err
		}
	}

	l, err := s.ListEvents(p)
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
		return l.Events[0], l.Count, nil
	}
	return nil, l.Count, fmt.Errorf("There is more then one result for Event UUID: %s!", id)
}

// A command to list events.
func (s *EventService) ListEvents(p *ListEventsParams) (*ListEventsResponse, error) {
	resp, err := s.cs.newRequest("listEvents", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListEventsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListEventsResponse struct {
	Count  int      `json:"count"`
	Events []*Event `json:"event"`
}

type Event struct {
	Account     string `json:"account,omitempty"`
	Created     string `json:"created,omitempty"`
	Description string `json:"description,omitempty"`
	Domain      string `json:"domain,omitempty"`
	Domainid    string `json:"domainid,omitempty"`
	Id          string `json:"id,omitempty"`
	Level       string `json:"level,omitempty"`
	Parentid    string `json:"parentid,omitempty"`
	Project     string `json:"project,omitempty"`
	Projectid   string `json:"projectid,omitempty"`
	State       string `json:"state,omitempty"`
	Type        string `json:"type,omitempty"`
	Username    string `json:"username,omitempty"`
}

type ListEventTypesParams struct {
	p map[string]interface{}
}

func (p *ListEventTypesParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	return u
}

// You should always use this function to get a new ListEventTypesParams instance,
// as then you are sure you have configured all required params
func (s *EventService) NewListEventTypesParams() *ListEventTypesParams {
	p := &ListEventTypesParams{}
	p.p = make(map[string]interface{})
	return p
}

// List Event Types
func (s *EventService) ListEventTypes(p *ListEventTypesParams) (*ListEventTypesResponse, error) {
	resp, err := s.cs.newRequest("listEventTypes", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListEventTypesResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListEventTypesResponse struct {
	Count      int          `json:"count"`
	EventTypes []*EventType `json:"eventtype"`
}

type EventType struct {
	Name string `json:"name,omitempty"`
}

type ArchiveEventsParams struct {
	p map[string]interface{}
}

func (p *ArchiveEventsParams) toURLValues() url.Values {
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

func (p *ArchiveEventsParams) SetEnddate(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["enddate"] = v
	return
}

func (p *ArchiveEventsParams) SetIds(v []string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["ids"] = v
	return
}

func (p *ArchiveEventsParams) SetStartdate(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["startdate"] = v
	return
}

func (p *ArchiveEventsParams) SetType(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["eventType"] = v
	return
}

// You should always use this function to get a new ArchiveEventsParams instance,
// as then you are sure you have configured all required params
func (s *EventService) NewArchiveEventsParams() *ArchiveEventsParams {
	p := &ArchiveEventsParams{}
	p.p = make(map[string]interface{})
	return p
}

// Archive one or more events.
func (s *EventService) ArchiveEvents(p *ArchiveEventsParams) (*ArchiveEventsResponse, error) {
	resp, err := s.cs.newRequest("archiveEvents", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ArchiveEventsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ArchiveEventsResponse struct {
	Displaytext string `json:"displaytext,omitempty"`
	Success     string `json:"success,omitempty"`
}

type DeleteEventsParams struct {
	p map[string]interface{}
}

func (p *DeleteEventsParams) toURLValues() url.Values {
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

func (p *DeleteEventsParams) SetEnddate(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["enddate"] = v
	return
}

func (p *DeleteEventsParams) SetIds(v []string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["ids"] = v
	return
}

func (p *DeleteEventsParams) SetStartdate(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["startdate"] = v
	return
}

func (p *DeleteEventsParams) SetType(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["eventType"] = v
	return
}

// You should always use this function to get a new DeleteEventsParams instance,
// as then you are sure you have configured all required params
func (s *EventService) NewDeleteEventsParams() *DeleteEventsParams {
	p := &DeleteEventsParams{}
	p.p = make(map[string]interface{})
	return p
}

// Delete one or more events.
func (s *EventService) DeleteEvents(p *DeleteEventsParams) (*DeleteEventsResponse, error) {
	resp, err := s.cs.newRequest("deleteEvents", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r DeleteEventsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type DeleteEventsResponse struct {
	Displaytext string `json:"displaytext,omitempty"`
	Success     string `json:"success,omitempty"`
}
