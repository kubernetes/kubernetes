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
)

// Contains functionality for projects API.
type ProjectsAPI struct {
	client *Client
}

// Options for GetDisks API.
type DiskGetOptions struct {
	Name string `urlParam:"name"`
}

// Options for GetDisks API.
type VmGetOptions struct {
	Name string `urlParam:"name"`
}

var projectUrl string = rootUrl + "/projects/"

// Deletes the project with specified ID. Any VMs, disks, etc., owned by the project must be deleted first.
func (api *ProjectsAPI) Delete(projectID string) (task *Task, err error) {
	res, err := api.client.restClient.Delete(api.client.Endpoint+projectUrl+projectID, api.client.options.TokenOptions)
	if err != nil {
		return
	}
	defer res.Body.Close()
	task, err = getTask(getError(res))
	return
}

// Creates a disk on the specified project.
func (api *ProjectsAPI) CreateDisk(projectID string, spec *DiskCreateSpec) (task *Task, err error) {
	body, err := json.Marshal(spec)
	if err != nil {
		return
	}
	res, err := api.client.restClient.Post(
		api.client.Endpoint+projectUrl+projectID+"/disks",
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

// Gets disks for project with the specified ID, using options to filter the results.
// If options is nil, no filtering will occur.
func (api *ProjectsAPI) GetDisks(projectID string, options *DiskGetOptions) (result *DiskList, err error) {
	uri := api.client.Endpoint + projectUrl + projectID + "/disks"
	if options != nil {
		uri += getQueryString(options)
	}
	res, err := api.client.restClient.GetList(api.client.Endpoint, uri, api.client.options.TokenOptions)
	if err != nil {
		return
	}

	result = &DiskList{}
	err = json.Unmarshal(res, result)
	return
}

// Creates a VM on the specified project.
func (api *ProjectsAPI) CreateVM(projectID string, spec *VmCreateSpec) (task *Task, err error) {
	body, err := json.Marshal(spec)
	if err != nil {
		return
	}
	res, err := api.client.restClient.Post(
		api.client.Endpoint+projectUrl+projectID+"/vms",
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

// Gets all tasks with the specified project ID, using options to filter the results.
// If options is nil, no filtering will occur.
func (api *ProjectsAPI) GetTasks(id string, options *TaskGetOptions) (result *TaskList, err error) {
	uri := api.client.Endpoint + projectUrl + id + "/tasks"
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

// Gets vms for project with the specified ID, using options to filter the results.
// If options is nil, no filtering will occur.
func (api *ProjectsAPI) GetVMs(projectID string, options *VmGetOptions) (result *VMs, err error) {
	uri := api.client.Endpoint + projectUrl + projectID + "/vms"
	if options != nil {
		uri += getQueryString(options)
	}
	res, err := api.client.restClient.GetList(api.client.Endpoint, uri, api.client.options.TokenOptions)
	if err != nil {
		return
	}

	result = &VMs{}
	err = json.Unmarshal(res, result)
	return
}

// Creates a service on the specified project.
func (api *ProjectsAPI) CreateService(projectID string, spec *ServiceCreateSpec) (task *Task, err error) {
	body, err := json.Marshal(spec)
	if err != nil {
		return
	}
	res, err := api.client.restClient.Post(
		api.client.Endpoint+projectUrl+projectID+"/services",
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

// Gets services for project with the specified ID
func (api *ProjectsAPI) GetServices(projectID string) (result *Services, err error) {
	uri := api.client.Endpoint + projectUrl + projectID + "/services"
	res, err := api.client.restClient.GetList(api.client.Endpoint, uri, api.client.options.TokenOptions)
	if err != nil {
		return
	}

	result = &Services{}
	err = json.Unmarshal(res, result)
	return
}

// Gets the project with a specified ID.
func (api *ProjectsAPI) Get(id string) (project *ProjectCompact, err error) {
	res, err := api.client.restClient.Get(api.getEntityUrl(id), api.client.options.TokenOptions)
	if err != nil {
		return
	}
	defer res.Body.Close()
	res, err = getError(res)
	if err != nil {
		return
	}
	project = &ProjectCompact{}
	err = json.NewDecoder(res.Body).Decode(project)
	return
}

// Set security groups for this project, overwriting any existing ones.
func (api *ProjectsAPI) SetSecurityGroups(projectID string, securityGroups *SecurityGroupsSpec) (*Task, error) {
	return setSecurityGroups(api.client, api.getEntityUrl(projectID), securityGroups)
}

func (api *ProjectsAPI) getEntityUrl(id string) string {
	return api.client.Endpoint + projectUrl + id
}

// Creates a router on the specified project.
func (api *ProjectsAPI) CreateRouter(projectID string, spec *RouterCreateSpec) (task *Task, err error) {
	body, err := json.Marshal(spec)
	if err != nil {
		return
	}
	res, err := api.client.restClient.Post(
		api.client.Endpoint+projectUrl+projectID+"/routers",
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
