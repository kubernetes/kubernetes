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

// Contains functionality for status API.
type StatusAPI struct {
	client *Client
}

var statusUrl string = rootUrl + "/status"

// Returns the status of an photon endpoint.
func (api *StatusAPI) Get() (status *Status, err error) {
	res, err := api.client.restClient.Get(api.client.Endpoint+statusUrl, api.client.options.TokenOptions)
	if err != nil {
		return
	}
	defer res.Body.Close()
	res, err = getError(res)
	if err != nil {
		return
	}
	status = &Status{}
	err = json.NewDecoder(res.Body).Decode(status)
	return
}
