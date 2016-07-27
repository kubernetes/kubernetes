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

type ListStoragePoolsParams struct {
	p map[string]interface{}
}

func (p *ListStoragePoolsParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["clusterid"]; found {
		u.Set("clusterid", v.(string))
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	if v, found := p.p["ipaddress"]; found {
		u.Set("ipaddress", v.(string))
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
	if v, found := p.p["path"]; found {
		u.Set("path", v.(string))
	}
	if v, found := p.p["podid"]; found {
		u.Set("podid", v.(string))
	}
	if v, found := p.p["scope"]; found {
		u.Set("scope", v.(string))
	}
	if v, found := p.p["zoneid"]; found {
		u.Set("zoneid", v.(string))
	}
	return u
}

func (p *ListStoragePoolsParams) SetClusterid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["clusterid"] = v
	return
}

func (p *ListStoragePoolsParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *ListStoragePoolsParams) SetIpaddress(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["ipaddress"] = v
	return
}

func (p *ListStoragePoolsParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *ListStoragePoolsParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *ListStoragePoolsParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *ListStoragePoolsParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

func (p *ListStoragePoolsParams) SetPath(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["path"] = v
	return
}

func (p *ListStoragePoolsParams) SetPodid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["podid"] = v
	return
}

func (p *ListStoragePoolsParams) SetScope(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["scope"] = v
	return
}

func (p *ListStoragePoolsParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new ListStoragePoolsParams instance,
// as then you are sure you have configured all required params
func (s *PoolService) NewListStoragePoolsParams() *ListStoragePoolsParams {
	p := &ListStoragePoolsParams{}
	p.p = make(map[string]interface{})
	return p
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *PoolService) GetStoragePoolID(name string, opts ...OptionFunc) (string, error) {
	p := &ListStoragePoolsParams{}
	p.p = make(map[string]interface{})

	p.p["name"] = name

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return "", err
		}
	}

	l, err := s.ListStoragePools(p)
	if err != nil {
		return "", err
	}

	if l.Count == 0 {
		return "", fmt.Errorf("No match found for %s: %+v", name, l)
	}

	if l.Count == 1 {
		return l.StoragePools[0].Id, nil
	}

	if l.Count > 1 {
		for _, v := range l.StoragePools {
			if v.Name == name {
				return v.Id, nil
			}
		}
	}
	return "", fmt.Errorf("Could not find an exact match for %s: %+v", name, l)
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *PoolService) GetStoragePoolByName(name string, opts ...OptionFunc) (*StoragePool, int, error) {
	id, err := s.GetStoragePoolID(name, opts...)
	if err != nil {
		return nil, -1, err
	}

	r, count, err := s.GetStoragePoolByID(id, opts...)
	if err != nil {
		return nil, count, err
	}
	return r, count, nil
}

// This is a courtesy helper function, which in some cases may not work as expected!
func (s *PoolService) GetStoragePoolByID(id string, opts ...OptionFunc) (*StoragePool, int, error) {
	p := &ListStoragePoolsParams{}
	p.p = make(map[string]interface{})

	p.p["id"] = id

	for _, fn := range opts {
		if err := fn(s.cs, p); err != nil {
			return nil, -1, err
		}
	}

	l, err := s.ListStoragePools(p)
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
		return l.StoragePools[0], l.Count, nil
	}
	return nil, l.Count, fmt.Errorf("There is more then one result for StoragePool UUID: %s!", id)
}

// Lists storage pools.
func (s *PoolService) ListStoragePools(p *ListStoragePoolsParams) (*ListStoragePoolsResponse, error) {
	resp, err := s.cs.newRequest("listStoragePools", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r ListStoragePoolsResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type ListStoragePoolsResponse struct {
	Count        int            `json:"count"`
	StoragePools []*StoragePool `json:"storagepool"`
}

type StoragePool struct {
	Capacityiops         int64             `json:"capacityiops,omitempty"`
	Clusterid            string            `json:"clusterid,omitempty"`
	Clustername          string            `json:"clustername,omitempty"`
	Created              string            `json:"created,omitempty"`
	Disksizeallocated    int64             `json:"disksizeallocated,omitempty"`
	Disksizetotal        int64             `json:"disksizetotal,omitempty"`
	Disksizeused         int64             `json:"disksizeused,omitempty"`
	Hypervisor           string            `json:"hypervisor,omitempty"`
	Id                   string            `json:"id,omitempty"`
	Ipaddress            string            `json:"ipaddress,omitempty"`
	Name                 string            `json:"name,omitempty"`
	Overprovisionfactor  string            `json:"overprovisionfactor,omitempty"`
	Path                 string            `json:"path,omitempty"`
	Podid                string            `json:"podid,omitempty"`
	Podname              string            `json:"podname,omitempty"`
	Scope                string            `json:"scope,omitempty"`
	State                string            `json:"state,omitempty"`
	Storagecapabilities  map[string]string `json:"storagecapabilities,omitempty"`
	Suitableformigration bool              `json:"suitableformigration,omitempty"`
	Tags                 string            `json:"tags,omitempty"`
	Type                 string            `json:"type,omitempty"`
	Zoneid               string            `json:"zoneid,omitempty"`
	Zonename             string            `json:"zonename,omitempty"`
}

type CreateStoragePoolParams struct {
	p map[string]interface{}
}

func (p *CreateStoragePoolParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["capacitybytes"]; found {
		vv := strconv.FormatInt(v.(int64), 10)
		u.Set("capacitybytes", vv)
	}
	if v, found := p.p["capacityiops"]; found {
		vv := strconv.FormatInt(v.(int64), 10)
		u.Set("capacityiops", vv)
	}
	if v, found := p.p["clusterid"]; found {
		u.Set("clusterid", v.(string))
	}
	if v, found := p.p["details"]; found {
		i := 0
		for k, vv := range v.(map[string]string) {
			u.Set(fmt.Sprintf("details[%d].key", i), k)
			u.Set(fmt.Sprintf("details[%d].value", i), vv)
			i++
		}
	}
	if v, found := p.p["hypervisor"]; found {
		u.Set("hypervisor", v.(string))
	}
	if v, found := p.p["managed"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("managed", vv)
	}
	if v, found := p.p["name"]; found {
		u.Set("name", v.(string))
	}
	if v, found := p.p["podid"]; found {
		u.Set("podid", v.(string))
	}
	if v, found := p.p["provider"]; found {
		u.Set("provider", v.(string))
	}
	if v, found := p.p["scope"]; found {
		u.Set("scope", v.(string))
	}
	if v, found := p.p["tags"]; found {
		u.Set("tags", v.(string))
	}
	if v, found := p.p["url"]; found {
		u.Set("url", v.(string))
	}
	if v, found := p.p["zoneid"]; found {
		u.Set("zoneid", v.(string))
	}
	return u
}

func (p *CreateStoragePoolParams) SetCapacitybytes(v int64) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["capacitybytes"] = v
	return
}

func (p *CreateStoragePoolParams) SetCapacityiops(v int64) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["capacityiops"] = v
	return
}

func (p *CreateStoragePoolParams) SetClusterid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["clusterid"] = v
	return
}

func (p *CreateStoragePoolParams) SetDetails(v map[string]string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["details"] = v
	return
}

func (p *CreateStoragePoolParams) SetHypervisor(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["hypervisor"] = v
	return
}

func (p *CreateStoragePoolParams) SetManaged(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["managed"] = v
	return
}

func (p *CreateStoragePoolParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *CreateStoragePoolParams) SetPodid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["podid"] = v
	return
}

func (p *CreateStoragePoolParams) SetProvider(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["provider"] = v
	return
}

func (p *CreateStoragePoolParams) SetScope(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["scope"] = v
	return
}

func (p *CreateStoragePoolParams) SetTags(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["tags"] = v
	return
}

func (p *CreateStoragePoolParams) SetUrl(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["url"] = v
	return
}

func (p *CreateStoragePoolParams) SetZoneid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["zoneid"] = v
	return
}

// You should always use this function to get a new CreateStoragePoolParams instance,
// as then you are sure you have configured all required params
func (s *PoolService) NewCreateStoragePoolParams(name string, url string, zoneid string) *CreateStoragePoolParams {
	p := &CreateStoragePoolParams{}
	p.p = make(map[string]interface{})
	p.p["name"] = name
	p.p["url"] = url
	p.p["zoneid"] = zoneid
	return p
}

// Creates a storage pool.
func (s *PoolService) CreateStoragePool(p *CreateStoragePoolParams) (*CreateStoragePoolResponse, error) {
	resp, err := s.cs.newRequest("createStoragePool", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r CreateStoragePoolResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type CreateStoragePoolResponse struct {
	Capacityiops         int64             `json:"capacityiops,omitempty"`
	Clusterid            string            `json:"clusterid,omitempty"`
	Clustername          string            `json:"clustername,omitempty"`
	Created              string            `json:"created,omitempty"`
	Disksizeallocated    int64             `json:"disksizeallocated,omitempty"`
	Disksizetotal        int64             `json:"disksizetotal,omitempty"`
	Disksizeused         int64             `json:"disksizeused,omitempty"`
	Hypervisor           string            `json:"hypervisor,omitempty"`
	Id                   string            `json:"id,omitempty"`
	Ipaddress            string            `json:"ipaddress,omitempty"`
	Name                 string            `json:"name,omitempty"`
	Overprovisionfactor  string            `json:"overprovisionfactor,omitempty"`
	Path                 string            `json:"path,omitempty"`
	Podid                string            `json:"podid,omitempty"`
	Podname              string            `json:"podname,omitempty"`
	Scope                string            `json:"scope,omitempty"`
	State                string            `json:"state,omitempty"`
	Storagecapabilities  map[string]string `json:"storagecapabilities,omitempty"`
	Suitableformigration bool              `json:"suitableformigration,omitempty"`
	Tags                 string            `json:"tags,omitempty"`
	Type                 string            `json:"type,omitempty"`
	Zoneid               string            `json:"zoneid,omitempty"`
	Zonename             string            `json:"zonename,omitempty"`
}

type UpdateStoragePoolParams struct {
	p map[string]interface{}
}

func (p *UpdateStoragePoolParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["capacitybytes"]; found {
		vv := strconv.FormatInt(v.(int64), 10)
		u.Set("capacitybytes", vv)
	}
	if v, found := p.p["capacityiops"]; found {
		vv := strconv.FormatInt(v.(int64), 10)
		u.Set("capacityiops", vv)
	}
	if v, found := p.p["enabled"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("enabled", vv)
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	if v, found := p.p["tags"]; found {
		vv := strings.Join(v.([]string), ",")
		u.Set("tags", vv)
	}
	return u
}

func (p *UpdateStoragePoolParams) SetCapacitybytes(v int64) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["capacitybytes"] = v
	return
}

func (p *UpdateStoragePoolParams) SetCapacityiops(v int64) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["capacityiops"] = v
	return
}

func (p *UpdateStoragePoolParams) SetEnabled(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["enabled"] = v
	return
}

func (p *UpdateStoragePoolParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *UpdateStoragePoolParams) SetTags(v []string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["tags"] = v
	return
}

// You should always use this function to get a new UpdateStoragePoolParams instance,
// as then you are sure you have configured all required params
func (s *PoolService) NewUpdateStoragePoolParams(id string) *UpdateStoragePoolParams {
	p := &UpdateStoragePoolParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Updates a storage pool.
func (s *PoolService) UpdateStoragePool(p *UpdateStoragePoolParams) (*UpdateStoragePoolResponse, error) {
	resp, err := s.cs.newRequest("updateStoragePool", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r UpdateStoragePoolResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type UpdateStoragePoolResponse struct {
	Capacityiops         int64             `json:"capacityiops,omitempty"`
	Clusterid            string            `json:"clusterid,omitempty"`
	Clustername          string            `json:"clustername,omitempty"`
	Created              string            `json:"created,omitempty"`
	Disksizeallocated    int64             `json:"disksizeallocated,omitempty"`
	Disksizetotal        int64             `json:"disksizetotal,omitempty"`
	Disksizeused         int64             `json:"disksizeused,omitempty"`
	Hypervisor           string            `json:"hypervisor,omitempty"`
	Id                   string            `json:"id,omitempty"`
	Ipaddress            string            `json:"ipaddress,omitempty"`
	Name                 string            `json:"name,omitempty"`
	Overprovisionfactor  string            `json:"overprovisionfactor,omitempty"`
	Path                 string            `json:"path,omitempty"`
	Podid                string            `json:"podid,omitempty"`
	Podname              string            `json:"podname,omitempty"`
	Scope                string            `json:"scope,omitempty"`
	State                string            `json:"state,omitempty"`
	Storagecapabilities  map[string]string `json:"storagecapabilities,omitempty"`
	Suitableformigration bool              `json:"suitableformigration,omitempty"`
	Tags                 string            `json:"tags,omitempty"`
	Type                 string            `json:"type,omitempty"`
	Zoneid               string            `json:"zoneid,omitempty"`
	Zonename             string            `json:"zonename,omitempty"`
}

type DeleteStoragePoolParams struct {
	p map[string]interface{}
}

func (p *DeleteStoragePoolParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["forced"]; found {
		vv := strconv.FormatBool(v.(bool))
		u.Set("forced", vv)
	}
	if v, found := p.p["id"]; found {
		u.Set("id", v.(string))
	}
	return u
}

func (p *DeleteStoragePoolParams) SetForced(v bool) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["forced"] = v
	return
}

func (p *DeleteStoragePoolParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

// You should always use this function to get a new DeleteStoragePoolParams instance,
// as then you are sure you have configured all required params
func (s *PoolService) NewDeleteStoragePoolParams(id string) *DeleteStoragePoolParams {
	p := &DeleteStoragePoolParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Deletes a storage pool.
func (s *PoolService) DeleteStoragePool(p *DeleteStoragePoolParams) (*DeleteStoragePoolResponse, error) {
	resp, err := s.cs.newRequest("deleteStoragePool", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r DeleteStoragePoolResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type DeleteStoragePoolResponse struct {
	Displaytext string `json:"displaytext,omitempty"`
	Success     string `json:"success,omitempty"`
}

type FindStoragePoolsForMigrationParams struct {
	p map[string]interface{}
}

func (p *FindStoragePoolsForMigrationParams) toURLValues() url.Values {
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

func (p *FindStoragePoolsForMigrationParams) SetId(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *FindStoragePoolsForMigrationParams) SetKeyword(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["keyword"] = v
	return
}

func (p *FindStoragePoolsForMigrationParams) SetPage(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["page"] = v
	return
}

func (p *FindStoragePoolsForMigrationParams) SetPagesize(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["pagesize"] = v
	return
}

// You should always use this function to get a new FindStoragePoolsForMigrationParams instance,
// as then you are sure you have configured all required params
func (s *PoolService) NewFindStoragePoolsForMigrationParams(id string) *FindStoragePoolsForMigrationParams {
	p := &FindStoragePoolsForMigrationParams{}
	p.p = make(map[string]interface{})
	p.p["id"] = id
	return p
}

// Lists storage pools available for migration of a volume.
func (s *PoolService) FindStoragePoolsForMigration(p *FindStoragePoolsForMigrationParams) (*FindStoragePoolsForMigrationResponse, error) {
	resp, err := s.cs.newRequest("findStoragePoolsForMigration", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r FindStoragePoolsForMigrationResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type FindStoragePoolsForMigrationResponse struct {
	Capacityiops         int64             `json:"capacityiops,omitempty"`
	Clusterid            string            `json:"clusterid,omitempty"`
	Clustername          string            `json:"clustername,omitempty"`
	Created              string            `json:"created,omitempty"`
	Disksizeallocated    int64             `json:"disksizeallocated,omitempty"`
	Disksizetotal        int64             `json:"disksizetotal,omitempty"`
	Disksizeused         int64             `json:"disksizeused,omitempty"`
	Hypervisor           string            `json:"hypervisor,omitempty"`
	Id                   string            `json:"id,omitempty"`
	Ipaddress            string            `json:"ipaddress,omitempty"`
	Name                 string            `json:"name,omitempty"`
	Overprovisionfactor  string            `json:"overprovisionfactor,omitempty"`
	Path                 string            `json:"path,omitempty"`
	Podid                string            `json:"podid,omitempty"`
	Podname              string            `json:"podname,omitempty"`
	Scope                string            `json:"scope,omitempty"`
	State                string            `json:"state,omitempty"`
	Storagecapabilities  map[string]string `json:"storagecapabilities,omitempty"`
	Suitableformigration bool              `json:"suitableformigration,omitempty"`
	Tags                 string            `json:"tags,omitempty"`
	Type                 string            `json:"type,omitempty"`
	Zoneid               string            `json:"zoneid,omitempty"`
	Zonename             string            `json:"zonename,omitempty"`
}
