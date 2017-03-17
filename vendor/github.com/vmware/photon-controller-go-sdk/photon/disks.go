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
	"encoding/json"
)

// Contains functionality for disks API.
type DisksAPI struct {
	client *Client
}

var diskUrl string = rootUrl + "/disks/"

// Gets a PersistentDisk for the disk with specified ID.
func (api *DisksAPI) Get(diskID string) (disk *PersistentDisk, err error) {
	res, err := api.client.restClient.Get(api.client.Endpoint+diskUrl+diskID, api.client.options.TokenOptions)
	if err != nil {
		return
	}
	defer res.Body.Close()
	res, err = getError(res)
	if err != nil {
		return
	}
	disk = &PersistentDisk{}
	err = json.NewDecoder(res.Body).Decode(disk)
	return
}

// Deletes a disk with the specified ID.
func (api *DisksAPI) Delete(diskID string) (task *Task, err error) {
	res, err := api.client.restClient.Delete(api.client.Endpoint+diskUrl+diskID, api.client.options.TokenOptions)
	if err != nil {
		return
	}
	defer res.Body.Close()
	task, err = getTask(getError(res))
	return
}

// Gets all tasks with the specified disk ID, using options to filter the results.
// If options is nil, no filtering will occur.
func (api *DisksAPI) GetTasks(id string, options *TaskGetOptions) (result *TaskList, err error) {
	uri := api.client.Endpoint + diskUrl + id + "/tasks"
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
