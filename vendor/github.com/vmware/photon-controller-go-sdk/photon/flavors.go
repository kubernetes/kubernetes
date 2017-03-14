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

// Contains functionality for flavors API.
type FlavorsAPI struct {
	client *Client
}

// Options used for find/get APIs
type FlavorGetOptions struct {
	Name string `urlParam:"name"`
	Kind string `urlParam:"kind"`
}

var flavorUrl string = rootUrl + "/flavors"

// Creates a flavor.
func (api *FlavorsAPI) Create(spec *FlavorCreateSpec) (task *Task, err error) {
	body, err := json.Marshal(spec)
	if err != nil {
		return
	}
	res, err := api.client.restClient.Post(
		api.client.Endpoint+flavorUrl,
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

// Gets details of flavor with specified ID.
func (api *FlavorsAPI) Get(flavorID string) (flavor *Flavor, err error) {
	res, err := api.client.restClient.Get(api.client.Endpoint+flavorUrl+"/"+flavorID, api.client.options.TokenOptions)
	if err != nil {
		return
	}
	defer res.Body.Close()
	res, err = getError(res)
	if err != nil {
		return
	}
	flavor = &Flavor{}
	err = json.NewDecoder(res.Body).Decode(flavor)
	return
}

// Gets flavors using options to filter results. Returns all flavors if options is nil.
func (api *FlavorsAPI) GetAll(options *FlavorGetOptions) (flavors *FlavorList, err error) {
	uri := api.client.Endpoint + flavorUrl
	if options != nil {
		uri += getQueryString(options)
	}
	res, err := api.client.restClient.GetList(api.client.Endpoint, uri, api.client.options.TokenOptions)
	if err != nil {
		return
	}

	flavors = &FlavorList{}
	err = json.Unmarshal(res, flavors)
	return
}

// Deletes flavor with specified ID.
func (api *FlavorsAPI) Delete(flavorID string) (task *Task, err error) {
	res, err := api.client.restClient.Delete(api.client.Endpoint+flavorUrl+"/"+flavorID, api.client.options.TokenOptions)
	if err != nil {
		return
	}
	defer res.Body.Close()
	task, err = getTask(getError(res))
	return
}

// Gets all tasks with the specified flavor ID, using options to filter the results.
// If options is nil, no filtering will occur.
func (api *FlavorsAPI) GetTasks(id string, options *TaskGetOptions) (result *TaskList, err error) {
	uri := api.client.Endpoint + flavorUrl + "/" + id + "/tasks"
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
