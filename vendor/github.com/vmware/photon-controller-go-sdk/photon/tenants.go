// Copyright (c) 2016 VMware, Inc. All Rights Reserved.
//
// This product is licensed to you under the Apache License, Version 2.0 (the "License").
// You may not use this product except in compliance with the License.
//
// This product may include a number of subcomponents with separate copyright notices and
// license terms. Your use of these subcomponents is subject to the terms and conditions
// of the subcomponent's license, as noted in the LICENSE file.

package photon

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// Contains functionality for tenants API.
type TenantsAPI struct {
	client *Client
}

// Options for GetResourceTickets API.
type ResourceTicketGetOptions struct {
	Name string `urlParam:"name"`
}

// Options for GetProjects API.
type ProjectGetOptions struct {
	Name string `urlParam:"name"`
}

var tenantUrl string = rootUrl + "/tenants"

// Returns all tenants on an photon instance.
func (api *TenantsAPI) GetAll() (result *Tenants, err error) {
	uri := api.client.Endpoint + tenantUrl
	res, err := api.client.restClient.GetList(api.client.Endpoint, uri, api.client.options.TokenOptions)
	if err != nil {
		return
	}

	result = &Tenants{}
	err = json.Unmarshal(res, result)
	return
}

// Creates a tenant.
func (api *TenantsAPI) Create(tenantSpec *TenantCreateSpec) (task *Task, err error) {
	body, err := json.Marshal(tenantSpec)
	if err != nil {
		return
	}
	res, err := api.client.restClient.Post(
		api.client.Endpoint+tenantUrl,
		"application/json",
		bytes.NewReader(body),
		api.client.options.TokenOptions)
	if err != nil {
		return
	}
	defer res.Body.Close()
	task, err = getTask(getError(res))
	return
}

// Deletes the tenant with specified ID. Any projects, VMs, disks, etc., owned by the tenant must be deleted first.
func (api *TenantsAPI) Delete(id string) (task *Task, err error) {
	res, err := api.client.restClient.Delete(api.client.Endpoint+tenantUrl+"/"+id, api.client.options.TokenOptions)
	if err != nil {
		return
	}
	defer res.Body.Close()
	task, err = getTask(getError(res))
	return
}

// Creates a resource ticket on the specified tenant.
func (api *TenantsAPI) CreateResourceTicket(tenantId string, spec *ResourceTicketCreateSpec) (task *Task, err error) {
	body, err := json.Marshal(spec)
	if err != nil {
		return
	}
	res, err := api.client.restClient.Post(
		api.client.Endpoint+tenantUrl+"/"+tenantId+"/resource-tickets",
		"application/json",
		bytes.NewReader(body),
		api.client.options.TokenOptions)
	if err != nil {
		return
	}
	defer res.Body.Close()
	task, err = getTask(getError(res))
	return
}

// Gets resource tickets for tenant with the specified ID, using options to filter the results.
// If options is nil, no filtering will occur.
func (api *TenantsAPI) GetResourceTickets(tenantId string, options *ResourceTicketGetOptions) (tickets *ResourceList, err error) {
	uri := api.client.Endpoint + tenantUrl + "/" + tenantId + "/resource-tickets"
	if options != nil {
		uri += getQueryString(options)
	}
	res, err := api.client.restClient.GetList(api.client.Endpoint, uri, api.client.options.TokenOptions)
	if err != nil {
		return
	}

	tickets = &ResourceList{}
	err = json.Unmarshal(res, tickets)
	return
}

// Creates a project on the specified tenant.
func (api *TenantsAPI) CreateProject(tenantId string, spec *ProjectCreateSpec) (task *Task, err error) {
	body, err := json.Marshal(spec)
	if err != nil {
		return
	}
	res, err := api.client.restClient.Post(
		api.client.Endpoint+tenantUrl+"/"+tenantId+"/projects",
		"application/json",
		bytes.NewReader(body),
		api.client.options.TokenOptions)
	if err != nil {
		return
	}
	defer res.Body.Close()
	task, err = getTask(getError(res))
	return
}

// Gets the projects for tenant with the specified ID, using options to filter the results.
// If options is nil, no filtering will occur.
func (api *TenantsAPI) GetProjects(tenantId string, options *ProjectGetOptions) (result *ProjectList, err error) {
	uri := api.client.Endpoint + tenantUrl + "/" + tenantId + "/projects"
	if options != nil {
		uri += getQueryString(options)
	}
	res, err := api.client.restClient.GetList(api.client.Endpoint, uri, api.client.options.TokenOptions)
	if err != nil {
		return
	}

	result = &(ProjectList{})
	err = json.Unmarshal(res, result)
	return
}

// Gets all tasks with the specified tenant ID, using options to filter the results.
// If options is nil, no filtering will occur.
func (api *TenantsAPI) GetTasks(id string, options *TaskGetOptions) (result *TaskList, err error) {
	uri := api.client.Endpoint + tenantUrl + "/" + id + "/tasks"
	if options != nil {
		uri += getQueryString(options)
	}
	res, err := api.client.restClient.GetList(api.client.Endpoint, uri, api.client.options.TokenOptions)
	if err != nil {
		return
	}

	result = &TaskList{}
	err = json.Unmarshal(res, result)
	return
}

// Gets a tenant with the specified ID or name
func (api *TenantsAPI) Get(identity string) (tenant *Tenant, err error) {
	res, err := api.client.restClient.Get(api.getEntityUrl(identity), api.client.options.TokenOptions)
	if err != nil {
		return
	}
	defer res.Body.Close()
	res, err = getError(res)
	tenant = &Tenant{}
	if res != nil {
		err = json.NewDecoder(res.Body).Decode(tenant)
		// ID corresponds to the tenant ID found, return tenant
		if err == nil {
			return
		}
	}
	// Find by Name
	uri := api.client.Endpoint + tenantUrl + "?name=" + identity
	res2, err := api.client.restClient.GetList(api.client.Endpoint, uri, api.client.options.TokenOptions)

	if err != nil {
		return
	}

	tenants := &Tenants{}
	err = json.Unmarshal(res2, tenants)
	if err != nil {
		return
	}

	if len(tenants.Items) < 1 {
		err = fmt.Errorf("Cannot find a tenant with id or name match %s", identity)
		return
	}

	tenant = &(tenants.Items[0])
	return
}

// Set security groups for this tenant, overwriting any existing ones.
func (api *TenantsAPI) SetSecurityGroups(id string, securityGroups *SecurityGroupsSpec) (*Task, error) {
	return setSecurityGroups(api.client, api.getEntityUrl(id), securityGroups)
}

func (api *TenantsAPI) getEntityUrl(id string) (url string) {
	return api.client.Endpoint + tenantUrl + "/" + id
}
