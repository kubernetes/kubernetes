/*
Copyright 2021 The Kubernetes Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package zoneclient

import (
	"bytes"
	"context"
	"io/ioutil"
	"net/http"
	"testing"

	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"

	azclients "k8s.io/legacy-cloud-providers/azure/clients"
	"k8s.io/legacy-cloud-providers/azure/clients/armclient"
	"k8s.io/legacy-cloud-providers/azure/clients/armclient/mockarmclient"
)

const (
	responseString = `
{
	"value": [
		{
			"id": "/subscriptions/subscriptionID/providers/Microsoft.Network",
			"resourceTypes": [
				{
					"resourceType": "foo",
					"zoneMappings": [
						{
							"location": "East US 2",
							"zones": [
								"1",
								"2",
								"3"
							]
						}
					]
				},
				{
					"resourceType": "virtualMachines",
					"zoneMappings": [
						{
							"location": "East US 2",
							"zones": [
								"1",
								"2",
								"3"
							]
						}
					]
				}
			]
		},
		{
			"id": "/subscriptions/subscriptionID/providers/Microsoft.Compute",
			"resourceTypes": [
				{
					"resourceType": "foo",
					"zoneMappings": [
						{
							"location": "East US 2",
							"zones": [
								"1",
								"2",
								"3"
							]
						}
					]
				},
				{
					"resourceType": "virtualMachines",
					"zoneMappings": [
						{
							"location": "East US 2",
							"zones": [
								"1",
								"2",
								"3"
							]
						}
					]
				}
			]
		}
	]
}
`
	testProviderResourcesListID = "/subscriptions/subscriptionID/providers"
)

func TestNew(t *testing.T) {
	config := &azclients.ClientConfig{
		SubscriptionID:          "sub",
		ResourceManagerEndpoint: "endpoint",
		Location:                "eastus",
	}

	zoneClient := New(config)
	assert.Equal(t, "sub", zoneClient.subscriptionID)
}

func TestGetZones(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	response := &http.Response{
		StatusCode: http.StatusOK,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte(responseString))),
	}
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().GetResource(gomock.Any(), testProviderResourcesListID, "").Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	zoneClient := getTestZoneClient(armClient)
	expected := map[string][]string{"eastus2": {"1", "2", "3"}}
	zones, rerr := zoneClient.GetZones(context.TODO(), zoneClient.subscriptionID)
	assert.Nil(t, rerr)
	assert.Equal(t, expected, zones)
}

func getTestZoneClient(armClient armclient.Interface) *Client {
	return &Client{
		armClient:      armClient,
		subscriptionID: "subscriptionID",
	}
}

func TestGetNotFound(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	response := &http.Response{
		StatusCode: http.StatusNotFound,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
	}
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().GetResource(gomock.Any(), testProviderResourcesListID, "").Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	zoneClient := getTestZoneClient(armClient)
	result, rerr := zoneClient.GetZones(context.TODO(), zoneClient.subscriptionID)
	assert.Nil(t, result)
	assert.NotNil(t, rerr)
	assert.Equal(t, http.StatusNotFound, rerr.HTTPStatusCode)
}

func TestGetInternalError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	response := &http.Response{
		StatusCode: http.StatusInternalServerError,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
	}
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().GetResource(gomock.Any(), testProviderResourcesListID, "").Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	zoneClient := getTestZoneClient(armClient)
	result, rerr := zoneClient.GetZones(context.TODO(), zoneClient.subscriptionID)
	assert.Nil(t, result)
	assert.NotNil(t, rerr)
	assert.Equal(t, http.StatusInternalServerError, rerr.HTTPStatusCode)
}
