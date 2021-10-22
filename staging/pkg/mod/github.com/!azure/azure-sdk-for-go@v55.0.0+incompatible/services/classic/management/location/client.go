// +build go1.7

// Package location provides a client for Locations.
package location

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

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
