// +build go1.7

// Package location provides a client for Locations.
package location

// Copyright 2017 Microsoft Corporation
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

import (
	"encoding/xml"

	"github.com/Azure/azure-sdk-for-go/services/classic/management"
)

const (
	azureLocationListURL = "locations"
	errParamNotSpecified = "Parameter %s is not specified."
)

//NewClient is used to instantiate a new LocationClient from an Azure client
func NewClient(client management.Client) LocationClient {
	return LocationClient{client: client}
}

func (c LocationClient) ListLocations() (ListLocationsResponse, error) {
	var l ListLocationsResponse

	response, err := c.client.SendAzureGetRequest(azureLocationListURL)
	if err != nil {
		return l, err
	}

	err = xml.Unmarshal(response, &l)
	return l, err
}
