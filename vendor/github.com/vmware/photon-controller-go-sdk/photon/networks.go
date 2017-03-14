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

// Contains functionality for networks API.
type NetworksAPI struct {
	client *Client
}

// Options used for GetAll API
type NetworkGetOptions struct {
	Name string `urlParam:"name"`
}

var networkUrl string = rootUrl + "/subnets"

// Creates a network.
func (api *NetworksAPI) Create(networkSpec *NetworkCreateSpec) (task *Task, err error) {
	body, err := json.Marshal(networkSpec)
	if err != nil {
		return
	}
	res, err := api.client.restClient.Post(
		api.client.Endpoint+networkUrl,
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

// Deletes a network with specified ID.
func (api *NetworksAPI) Delete(id string) (task *Task, err error) {
	res, err := api.client.restClient.Delete(api.client.Endpoint+networkUrl+"/"+id, api.client.options.TokenOptions)
	if err != nil {
		return
	}
	defer res.Body.Close()
	task, err = getTask(getError(res))
	return
}

// Gets a network with the specified ID.
func (api *NetworksAPI) Get(id string) (network *Network, err error) {
	res, err := api.client.restClient.Get(api.client.Endpoint+networkUrl+"/"+id, api.client.options.TokenOptions)
	if err != nil {
		return
	}
	defer res.Body.Close()
	res, err = getError(res)
	if err != nil {
		return
	}
	var result Network
	err = json.NewDecoder(res.Body).Decode(&result)
	return &result, nil
}

// Returns all networks
func (api *NetworksAPI) GetAll(options *NetworkGetOptions) (result *Networks, err error) {
	uri := api.client.Endpoint + networkUrl
	if options != nil {
		uri += getQueryString(options)
	}
	res, err := api.client.restClient.GetList(api.client.Endpoint, uri, api.client.options.TokenOptions)

	result = &Networks{}
	err = json.Unmarshal(res, result)
	return
}

// Sets default network.
func (api *NetworksAPI) SetDefault(id string) (task *Task, err error) {
	res, err := api.client.restClient.Post(
		api.client.Endpoint+networkUrl+"/"+id+"/set_default",
		"application/json",
		bytes.NewReader([]byte("")),
		api.client.options.TokenOptions)
	if err != nil {
		return
	}
	defer res.Body.Close()
	task, err = getTask(getError(res))
	return
}
