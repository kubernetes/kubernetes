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

// Contains functionality for hosts API.
type ResourceTicketsAPI struct {
	client *Client
}

var resourceTicketUrl string = "/resource-tickets/"

// Gets all tasks with the specified resource ticket ID, using options to filter the results.
// If options is nil, no filtering will occur.
func (api *ResourceTicketsAPI) GetTasks(id string, options *TaskGetOptions) (result *TaskList, err error) {
	uri := api.client.Endpoint + resourceTicketUrl + id + "/tasks"
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

// Gets the resource ticket with a specified ID.
func (api *ResourceTicketsAPI) Get(id string) (resourceTicket *ResourceTicket, err error) {
	res, err := api.client.restClient.Get(api.client.Endpoint+resourceTicketUrl+"/"+id,
		api.client.options.TokenOptions.AccessToken)
	if err != nil {
		return
	}
	defer res.Body.Close()
	res, err = getError(res)
	if err != nil {
		return
	}
	resourceTicket = &ResourceTicket{}
	err = json.NewDecoder(res.Body).Decode(resourceTicket)
	return
}
