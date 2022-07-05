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
	"context"
	"fmt"
	"net/http"
	"strings"

	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/azure"

	"k8s.io/klog/v2"

	azclients "k8s.io/legacy-cloud-providers/azure/clients"
	"k8s.io/legacy-cloud-providers/azure/clients/armclient"
	"k8s.io/legacy-cloud-providers/azure/retry"
)

var _ Interface = &Client{}

type resourceTypeMetadata struct {
	ResourceType string         `json:"resourceType"`
	ZoneMappings []zoneMappings `json:"zoneMappings"`
}

type zoneMappings struct {
	Location string   `json:"location"`
	Zones    []string `json:"zones"`
}

type providerListDataProperty struct {
	ID            string                 `json:"id"`
	ResourceTypes []resourceTypeMetadata `json:"resourceTypes"`
}

type providerListData struct {
	ProviderListDataProperties []providerListDataProperty `json:"value"`
}

// Client implements zone client Interface.
type Client struct {
	armClient      armclient.Interface
	subscriptionID string
	cloudName      string
}

// New creates a new zone client with ratelimiting.
func New(config *azclients.ClientConfig) *Client {
	baseURI := config.ResourceManagerEndpoint
	authorizer := config.Authorizer
	apiVersion := APIVersion

	armClient := armclient.New(authorizer, baseURI, config.UserAgent, apiVersion, config.Location, config.Backoff)
	client := &Client{
		armClient:      armClient,
		subscriptionID: config.SubscriptionID,
		cloudName:      config.CloudName,
	}

	return client
}

// GetZones gets the region-zone map for the subscription specified
func (c *Client) GetZones(ctx context.Context, subscriptionID string) (map[string][]string, *retry.Error) {
	result, rerr := c.getZones(ctx, subscriptionID)
	if rerr != nil {

		return result, rerr
	}

	return result, nil
}

// getZones gets the region-zone map for the subscription specified
func (c *Client) getZones(ctx context.Context, subscriptionID string) (map[string][]string, *retry.Error) {
	resourceID := armclient.GetProviderResourcesListID(subscriptionID)

	response, rerr := c.armClient.GetResource(ctx, resourceID, "")
	defer c.armClient.CloseResponse(ctx, response)
	if rerr != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "zone.get.request", resourceID, rerr.Error())
		return nil, rerr
	}

	result := providerListData{}
	err := autorest.Respond(
		response,
		azure.WithErrorUnlessStatusCode(http.StatusOK),
		autorest.ByUnmarshallingJSON(&result))
	if err != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "zone.get.respond", resourceID, err)
		return nil, retry.GetError(response, err)
	}

	regionZoneMap := make(map[string][]string)
	expectedID := fmt.Sprintf("/subscriptions/%s/providers/Microsoft.Compute", subscriptionID)
	if len(result.ProviderListDataProperties) != 0 {
		for _, property := range result.ProviderListDataProperties {
			if strings.EqualFold(property.ID, expectedID) {
				for _, resourceType := range property.ResourceTypes {
					if strings.EqualFold(resourceType.ResourceType, "virtualMachines") {
						if len(resourceType.ZoneMappings) != 0 {
							for _, zoneMapping := range resourceType.ZoneMappings {
								location := strings.ToLower(strings.ReplaceAll(zoneMapping.Location, " ", ""))
								regionZoneMap[location] = zoneMapping.Zones
							}
						}
					}
				}
			}
		}
	}

	return regionZoneMap, nil
}
