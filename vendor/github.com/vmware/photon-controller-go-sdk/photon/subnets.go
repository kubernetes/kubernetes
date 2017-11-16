// Copyright (c) 2017 VMware, Inc. All Rights Reserved.
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

// Contains functionality for subnets API.
type SubnetsAPI struct {
	client *Client
}

var subnetUrl string = "/temp-subnets/"

// Deletes a subnet with the specified ID.
func (api *SubnetsAPI) Delete(id string) (task *Task, err error) {
	res, err := api.client.restClient.Delete(api.client.Endpoint+subnetUrl+id, api.client.options.TokenOptions)
	if err != nil {
		return
	}

	defer res.Body.Close()

	task, err = getTask(getError(res))
	return
}

// Gets a subnet with the specified ID.
func (api *SubnetsAPI) Get(id string) (subnet *Subnet, err error) {
	res, err := api.client.restClient.Get(api.client.Endpoint+subnetUrl+id, api.client.options.TokenOptions)
	if err != nil {
		return
	}
	defer res.Body.Close()
	res, err = getError(res)
	if err != nil {
		return
	}
	var result Subnet
	err = json.NewDecoder(res.Body).Decode(&result)
	return &result, nil
}

// Updates subnet's attributes.
func (api *SubnetsAPI) Update(id string, subnetSpec *SubnetUpdateSpec) (task *Task, err error) {
	body, err := json.Marshal(subnetSpec)
	if err != nil {
		return
	}

	res, err := api.client.restClient.Put(
		api.client.Endpoint+subnetUrl+id,
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
