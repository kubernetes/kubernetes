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
	"io"
)

// Contains functionality for VMs API.
type VmAPI struct {
	client *Client
}

var vmUrl string = "/vms/"

func (api *VmAPI) Get(id string) (vm *VM, err error) {
	res, err := api.client.restClient.Get(api.client.Endpoint+vmUrl+id, api.client.options.TokenOptions.AccessToken)
	if err != nil {
		return
	}
	defer res.Body.Close()
	res, err = getError(res)
	if err != nil {
		return
	}
	vm = &VM{}
	err = json.NewDecoder(res.Body).Decode(vm)
	return
}

func (api *VmAPI) Delete(id string) (task *Task, err error) {
	res, err := api.client.restClient.Delete(api.client.Endpoint+vmUrl+id, api.client.options.TokenOptions.AccessToken)

	if err != nil {
		return
	}
	defer res.Body.Close()
	task, err = getTask(getError(res))
	return
}

func (api *VmAPI) AttachDisk(id string, op *VmDiskOperation) (task *Task, err error) {
	body, err := json.Marshal(op)
	if err != nil {
		return
	}
	res, err := api.client.restClient.Post(
		api.client.Endpoint+vmUrl+id+"/attach_disk",
		"application/json",
		bytes.NewReader(body),
		api.client.options.TokenOptions.AccessToken)
	if err != nil {
		return
	}
	defer res.Body.Close()
	task, err = getTask(getError(res))
	return
}

func (api *VmAPI) DetachDisk(id string, op *VmDiskOperation) (task *Task, err error) {
	body, err := json.Marshal(op)
	if err != nil {
		return
	}
	res, err := api.client.restClient.Post(
		api.client.Endpoint+vmUrl+id+"/detach_disk",
		"application/json",
		bytes.NewReader(body),
		api.client.options.TokenOptions.AccessToken)
	if err != nil {
		return
	}
	defer res.Body.Close()
	task, err = getTask(getError(res))
	return
}

func (api *VmAPI) AttachISO(id string, reader io.Reader, name string) (task *Task, err error) {
	res, err := api.client.restClient.MultipartUpload(
		api.client.Endpoint+vmUrl+id+"/attach_iso", reader, name, nil, api.client.options.TokenOptions.AccessToken)
	if err != nil {
		return
	}
	defer res.Body.Close()
	result, err := getTask(getError(res))
	return result, err
}

func (api *VmAPI) DetachISO(id string) (task *Task, err error) {
	body := []byte{}
	if err != nil {
		return
	}
	res, err := api.client.restClient.Post(
		api.client.Endpoint+vmUrl+id+"/detach_iso",
		"application/json",
		bytes.NewReader(body),
		api.client.options.TokenOptions.AccessToken)
	if err != nil {
		return
	}
	defer res.Body.Close()
	task, err = getTask(getError(res))
	return
}

func (api *VmAPI) Start(id string) (task *Task, err error) {
	body := []byte{}
	if err != nil {
		return
	}
	res, err := api.client.restClient.Post(
		api.client.Endpoint+vmUrl+id+"/start",
		"application/json",
		bytes.NewReader(body),
		api.client.options.TokenOptions.AccessToken)
	if err != nil {
		return
	}
	defer res.Body.Close()
	task, err = getTask(getError(res))
	return
}

func (api *VmAPI) Stop(id string) (task *Task, err error) {
	body := []byte{}
	if err != nil {
		return
	}
	res, err := api.client.restClient.Post(
		api.client.Endpoint+vmUrl+id+"/stop",
		"application/json",
		bytes.NewReader(body),
		api.client.options.TokenOptions.AccessToken)
	if err != nil {
		return
	}
	defer res.Body.Close()
	task, err = getTask(getError(res))
	return
}

func (api *VmAPI) Restart(id string) (task *Task, err error) {
	body := []byte{}
	if err != nil {
		return
	}
	res, err := api.client.restClient.Post(
		api.client.Endpoint+vmUrl+id+"/restart",
		"application/json",
		bytes.NewReader(body),
		api.client.options.TokenOptions.AccessToken)
	if err != nil {
		return
	}
	defer res.Body.Close()
	task, err = getTask(getError(res))
	return
}

func (api *VmAPI) Resume(id string) (task *Task, err error) {
	body := []byte{}
	if err != nil {
		return
	}
	res, err := api.client.restClient.Post(
		api.client.Endpoint+vmUrl+id+"/resume",
		"application/json",
		bytes.NewReader(body),
		api.client.options.TokenOptions.AccessToken)
	if err != nil {
		return
	}
	defer res.Body.Close()
	task, err = getTask(getError(res))
	return
}

func (api *VmAPI) Suspend(id string) (task *Task, err error) {
	body := []byte{}
	if err != nil {
		return
	}
	res, err := api.client.restClient.Post(
		api.client.Endpoint+vmUrl+id+"/suspend",
		"application/json",
		bytes.NewReader(body),
		api.client.options.TokenOptions.AccessToken)
	if err != nil {
		return
	}
	defer res.Body.Close()
	task, err = getTask(getError(res))
	return
}

func (api *VmAPI) SetMetadata(id string, metadata *VmMetadata) (task *Task, err error) {
	body, err := json.Marshal(metadata)
	if err != nil {
		return
	}
	res, err := api.client.restClient.Post(
		api.client.Endpoint+vmUrl+id+"/set_metadata",
		"application/json",
		bytes.NewReader(body),
		api.client.options.TokenOptions.AccessToken)
	if err != nil {
		return
	}
	defer res.Body.Close()
	task, err = getTask(getError(res))
	return
}

// Gets all tasks with the specified vm ID, using options to filter the results.
// If options is nil, no filtering will occur.
func (api *VmAPI) GetTasks(id string, options *TaskGetOptions) (result *TaskList, err error) {
	uri := api.client.Endpoint + vmUrl + id + "/tasks"
	if options != nil {
		uri += getQueryString(options)
	}
	res, err := api.client.restClient.GetList(api.client.Endpoint, uri, api.client.options.TokenOptions.AccessToken)
	if err != nil {
		return
	}

	result = &TaskList{}
	err = json.Unmarshal(res, result)
	return
}

func (api *VmAPI) GetNetworks(id string) (task *Task, err error) {
	res, err := api.client.restClient.Get(api.client.Endpoint+vmUrl+id+"/subnets", api.client.options.TokenOptions.AccessToken)
	if err != nil {
		return
	}
	defer res.Body.Close()
	task, err = getTask(getError(res))
	return
}

func (api *VmAPI) AcquireFloatingIp(id string, spec *VmFloatingIpSpec) (task *Task, err error) {
	body, err := json.Marshal(spec)
	if err != nil {
		return
	}

	res, err := api.client.restClient.Post(
		api.client.Endpoint+vmUrl+id+"/acquire_floating_ip",
		"application/json",
		bytes.NewReader(body),
		api.client.options.TokenOptions.AccessToken)
	if err != nil {
		return
	}
	defer res.Body.Close()
	task, err = getTask(getError(res))
	return
}

func (api *VmAPI) ReleaseFloatingIp(id string) (task *Task, err error) {

	res, err := api.client.restClient.Delete(
		api.client.Endpoint+vmUrl+id+"/release_floating_ip",
		api.client.options.TokenOptions.AccessToken)
	if err != nil {
		return
	}
	defer res.Body.Close()
	task, err = getTask(getError(res))
	return
}

func (api *VmAPI) GetMKSTicket(id string) (task *Task, err error) {
	res, err := api.client.restClient.Get(api.client.Endpoint+vmUrl+id+"/mks_ticket", api.client.options.TokenOptions.AccessToken)
	if err != nil {
		return
	}
	defer res.Body.Close()
	task, err = getTask(getError(res))
	return
}

func (api *VmAPI) SetTag(id string, tag *VmTag) (task *Task, err error) {
	body, err := json.Marshal(tag)
	if err != nil {
		return
	}
	res, err := api.client.restClient.Post(
		api.client.Endpoint+vmUrl+id+"/tags",
		"application/json",
		bytes.NewReader(body),
		api.client.options.TokenOptions.AccessToken)
	if err != nil {
		return
	}
	defer res.Body.Close()
	task, err = getTask(getError(res))
	return
}

func (api *VmAPI) CreateImage(id string, options *ImageCreateSpec) (task *Task, err error) {
	body, err := json.Marshal(options)
	if err != nil {
		return
	}
	res, err := api.client.restClient.Post(
		api.client.Endpoint+vmUrl+id+"/create_image",
		"application/json",
		bytes.NewReader(body),
		api.client.options.TokenOptions.AccessToken)
	if err != nil {
		return
	}
	defer res.Body.Close()
	task, err = getTask(getError(res))
	return
}
