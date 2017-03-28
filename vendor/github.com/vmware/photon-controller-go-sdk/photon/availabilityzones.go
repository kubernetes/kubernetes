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

// Contains functionality for availability zones API.
type AvailabilityZonesAPI struct {
	client *Client
}

var availabilityzoneUrl string = rootUrl + "/availabilityzones"

// Creates availability zone.
func (api *AvailabilityZonesAPI) Create(availabilityzoneSpec *AvailabilityZoneCreateSpec) (task *Task, err error) {
	body, err := json.Marshal(availabilityzoneSpec)
	if err != nil {
		return
	}
	res, err := api.client.restClient.Post(
		api.client.Endpoint+availabilityzoneUrl,
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

// Gets availability zone with the specified ID.
func (api *AvailabilityZonesAPI) Get(id string) (availabilityzone *AvailabilityZone, err error) {
	res, err := api.client.restClient.Get(api.getEntityUrl(id), api.client.options.TokenOptions)
	if err != nil {
		return
	}
	defer res.Body.Close()
	res, err = getError(res)
	if err != nil {
		return
	}
	availabilityzone = &AvailabilityZone{}
	err = json.NewDecoder(res.Body).Decode(availabilityzone)
	return
}

// Returns all availability zones on an photon instance.
func (api *AvailabilityZonesAPI) GetAll() (result *AvailabilityZones, err error) {
	uri := api.client.Endpoint + availabilityzoneUrl
	res, err := api.client.restClient.GetList(api.client.Endpoint, uri, api.client.options.TokenOptions)
	if err != nil {
		return
	}

	result = &AvailabilityZones{}
	err = json.Unmarshal(res, result)
	return
}

// Deletes the availability zone with specified ID.
func (api *AvailabilityZonesAPI) Delete(id string) (task *Task, err error) {
	res, err := api.client.restClient.Delete(api.client.Endpoint+availabilityzoneUrl+"/"+id, api.client.options.TokenOptions)
	if err != nil {
		return
	}
	defer res.Body.Close()
	task, err = getTask(getError(res))
	return
}

// Gets all tasks with the specified availability zone ID, using options to filter the results.
// If options is nil, no filtering will occur.
func (api *AvailabilityZonesAPI) GetTasks(id string, options *TaskGetOptions) (result *TaskList, err error) {
	uri := api.client.Endpoint + availabilityzoneUrl + "/" + id + "/tasks"
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

func (api *AvailabilityZonesAPI) getEntityUrl(id string) (url string) {
	return api.client.Endpoint + availabilityzoneUrl + "/" + id
}
