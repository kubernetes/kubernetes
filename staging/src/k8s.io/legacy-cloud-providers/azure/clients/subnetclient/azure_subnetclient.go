// +build !providerless

/*
Copyright 2020 The Kubernetes Authors.

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

package subnetclient

import (
	"context"
	"net/http"
	"time"

	"github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-06-01/network"
	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/azure"
	"github.com/Azure/go-autorest/autorest/to"

	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/klog/v2"
	azclients "k8s.io/legacy-cloud-providers/azure/clients"
	"k8s.io/legacy-cloud-providers/azure/clients/armclient"
	"k8s.io/legacy-cloud-providers/azure/metrics"
	"k8s.io/legacy-cloud-providers/azure/retry"
)

var _ Interface = &Client{}

// Client implements Subnet client Interface.
type Client struct {
	armClient      armclient.Interface
	subscriptionID string

	// Rate limiting configures.
	rateLimiterReader flowcontrol.RateLimiter
	rateLimiterWriter flowcontrol.RateLimiter

	// ARM throttling configures.
	RetryAfterReader time.Time
	RetryAfterWriter time.Time
}

// New creates a new Subnet client with ratelimiting.
func New(config *azclients.ClientConfig) *Client {
	baseURI := config.ResourceManagerEndpoint
	authorizer := config.Authorizer
	armClient := armclient.New(authorizer, baseURI, config.UserAgent, APIVersion, config.Location, config.Backoff)
	rateLimiterReader, rateLimiterWriter := azclients.NewRateLimiter(config.RateLimitConfig)

	klog.V(2).Infof("Azure SubnetsClient (read ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPS,
		config.RateLimitConfig.CloudProviderRateLimitBucket)
	klog.V(2).Infof("Azure SubnetsClient (write ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPSWrite,
		config.RateLimitConfig.CloudProviderRateLimitBucketWrite)

	client := &Client{
		armClient:         armClient,
		rateLimiterReader: rateLimiterReader,
		rateLimiterWriter: rateLimiterWriter,
		subscriptionID:    config.SubscriptionID,
	}

	return client
}

// Get gets a Subnet.
func (c *Client) Get(ctx context.Context, resourceGroupName string, virtualNetworkName string, subnetName string, expand string) (network.Subnet, *retry.Error) {
	mc := metrics.NewMetricContext("subnets", "get", resourceGroupName, c.subscriptionID, "")

	// Report errors if the client is rate limited.
	if !c.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		return network.Subnet{}, retry.GetRateLimitError(false, "SubnetGet")
	}

	// Report errors if the client is throttled.
	if c.RetryAfterReader.After(time.Now()) {
		mc.ThrottledCount()
		rerr := retry.GetThrottlingError("SubnetGet", "client throttled", c.RetryAfterReader)
		return network.Subnet{}, rerr
	}

	result, rerr := c.getSubnet(ctx, resourceGroupName, virtualNetworkName, subnetName, expand)
	mc.Observe(rerr.Error())
	if rerr != nil {
		if rerr.IsThrottled() {
			// Update RetryAfterReader so that no more requests would be sent until RetryAfter expires.
			c.RetryAfterReader = rerr.RetryAfter
		}

		return result, rerr
	}

	return result, nil
}

// getSubnet gets a Subnet.
func (c *Client) getSubnet(ctx context.Context, resourceGroupName string, virtualNetworkName string, subnetName string, expand string) (network.Subnet, *retry.Error) {
	resourceID := armclient.GetChildResourceID(
		c.subscriptionID,
		resourceGroupName,
		"Microsoft.Network/virtualNetworks",
		virtualNetworkName,
		"subnets",
		subnetName,
	)
	result := network.Subnet{}

	response, rerr := c.armClient.GetResource(ctx, resourceID, expand)
	defer c.armClient.CloseResponse(ctx, response)
	if rerr != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "subnet.get.request", resourceID, rerr.Error())
		return result, rerr
	}

	err := autorest.Respond(
		response,
		azure.WithErrorUnlessStatusCode(http.StatusOK),
		autorest.ByUnmarshallingJSON(&result))
	if err != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "subnet.get.respond", resourceID, err)
		return result, retry.GetError(response, err)
	}

	result.Response = autorest.Response{Response: response}
	return result, nil
}

// List gets a list of Subnets in the VNet.
func (c *Client) List(ctx context.Context, resourceGroupName string, virtualNetworkName string) ([]network.Subnet, *retry.Error) {
	mc := metrics.NewMetricContext("subnets", "list", resourceGroupName, c.subscriptionID, "")

	// Report errors if the client is rate limited.
	if !c.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		return nil, retry.GetRateLimitError(false, "SubnetList")
	}

	// Report errors if the client is throttled.
	if c.RetryAfterReader.After(time.Now()) {
		mc.ThrottledCount()
		rerr := retry.GetThrottlingError("SubnetList", "client throttled", c.RetryAfterReader)
		return nil, rerr
	}

	result, rerr := c.listSubnet(ctx, resourceGroupName, virtualNetworkName)
	mc.Observe(rerr.Error())
	if rerr != nil {
		if rerr.IsThrottled() {
			// Update RetryAfterReader so that no more requests would be sent until RetryAfter expires.
			c.RetryAfterReader = rerr.RetryAfter
		}

		return result, rerr
	}

	return result, nil
}

// listSubnet gets a list of Subnets in the VNet.
func (c *Client) listSubnet(ctx context.Context, resourceGroupName string, virtualNetworkName string) ([]network.Subnet, *retry.Error) {
	resourceID := armclient.GetChildResourcesListID(
		c.subscriptionID,
		resourceGroupName,
		"Microsoft.Network/virtualNetworks",
		virtualNetworkName,
		"subnets")

	result := make([]network.Subnet, 0)
	page := &SubnetListResultPage{}
	page.fn = c.listNextResults

	resp, rerr := c.armClient.GetResource(ctx, resourceID, "")
	defer c.armClient.CloseResponse(ctx, resp)
	if rerr != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "subnet.list.request", resourceID, rerr.Error())
		return result, rerr
	}

	var err error
	page.slr, err = c.listResponder(resp)
	if err != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "subnet.list.respond", resourceID, err)
		return result, retry.GetError(resp, err)
	}

	for page.NotDone() {
		result = append(result, *page.Response().Value...)
		if err = page.NextWithContext(ctx); err != nil {
			klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "subnet.list.next", resourceID, err)
			return result, retry.GetError(page.Response().Response.Response, err)
		}
	}

	return result, nil
}

// CreateOrUpdate creates or updates a Subnet.
func (c *Client) CreateOrUpdate(ctx context.Context, resourceGroupName string, virtualNetworkName string, subnetName string, subnetParameters network.Subnet) *retry.Error {
	mc := metrics.NewMetricContext("subnets", "create_or_update", resourceGroupName, c.subscriptionID, "")

	// Report errors if the client is rate limited.
	if !c.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return retry.GetRateLimitError(true, "SubnetCreateOrUpdate")
	}

	// Report errors if the client is throttled.
	if c.RetryAfterWriter.After(time.Now()) {
		mc.ThrottledCount()
		rerr := retry.GetThrottlingError("SubnetCreateOrUpdate", "client throttled", c.RetryAfterWriter)
		return rerr
	}

	rerr := c.createOrUpdateSubnet(ctx, resourceGroupName, virtualNetworkName, subnetName, subnetParameters)
	mc.Observe(rerr.Error())
	if rerr != nil {
		if rerr.IsThrottled() {
			// Update RetryAfterReader so that no more requests would be sent until RetryAfter expires.
			c.RetryAfterWriter = rerr.RetryAfter
		}

		return rerr
	}

	return nil
}

// createOrUpdateSubnet creates or updates a Subnet.
func (c *Client) createOrUpdateSubnet(ctx context.Context, resourceGroupName string, virtualNetworkName string, subnetName string, subnetParameters network.Subnet) *retry.Error {
	resourceID := armclient.GetChildResourceID(
		c.subscriptionID,
		resourceGroupName,
		"Microsoft.Network/virtualNetworks",
		virtualNetworkName,
		"subnets",
		subnetName)

	response, rerr := c.armClient.PutResource(ctx, resourceID, subnetParameters)
	defer c.armClient.CloseResponse(ctx, response)
	if rerr != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "subnet.put.request", resourceID, rerr.Error())
		return rerr
	}

	if response != nil && response.StatusCode != http.StatusNoContent {
		_, rerr = c.createOrUpdateResponder(response)
		if rerr != nil {
			klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "subnet.put.respond", resourceID, rerr.Error())
			return rerr
		}
	}

	return nil
}

func (c *Client) createOrUpdateResponder(resp *http.Response) (*network.Subnet, *retry.Error) {
	result := &network.Subnet{}
	err := autorest.Respond(
		resp,
		azure.WithErrorUnlessStatusCode(http.StatusOK, http.StatusCreated),
		autorest.ByUnmarshallingJSON(&result))
	result.Response = autorest.Response{Response: resp}
	return result, retry.GetError(resp, err)
}

// Delete deletes a Subnet by name.
func (c *Client) Delete(ctx context.Context, resourceGroupName string, virtualNetworkName string, subnetName string) *retry.Error {
	mc := metrics.NewMetricContext("subnets", "delete", resourceGroupName, c.subscriptionID, "")

	// Report errors if the client is rate limited.
	if !c.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return retry.GetRateLimitError(true, "SubnetDelete")
	}

	// Report errors if the client is throttled.
	if c.RetryAfterWriter.After(time.Now()) {
		mc.ThrottledCount()
		rerr := retry.GetThrottlingError("SubnetDelete", "client throttled", c.RetryAfterWriter)
		return rerr
	}

	rerr := c.deleteSubnet(ctx, resourceGroupName, virtualNetworkName, subnetName)
	mc.Observe(rerr.Error())
	if rerr != nil {
		if rerr.IsThrottled() {
			// Update RetryAfterReader so that no more requests would be sent until RetryAfter expires.
			c.RetryAfterWriter = rerr.RetryAfter
		}

		return rerr
	}

	return nil
}

// deleteSubnet deletes a PublicIPAddress by name.
func (c *Client) deleteSubnet(ctx context.Context, resourceGroupName string, virtualNetworkName string, subnetName string) *retry.Error {
	resourceID := armclient.GetChildResourceID(
		c.subscriptionID,
		resourceGroupName,
		"Microsoft.Network/virtualNetworks",
		virtualNetworkName,
		"subnets",
		subnetName)

	return c.armClient.DeleteResource(ctx, resourceID, "")
}

func (c *Client) listResponder(resp *http.Response) (result network.SubnetListResult, err error) {
	err = autorest.Respond(
		resp,
		autorest.ByIgnoring(),
		azure.WithErrorUnlessStatusCode(http.StatusOK),
		autorest.ByUnmarshallingJSON(&result))
	result.Response = autorest.Response{Response: resp}
	return
}

// subnetListResultPreparer prepares a request to retrieve the next set of results.
// It returns nil if no more results exist.
func (c *Client) subnetListResultPreparer(ctx context.Context, lblr network.SubnetListResult) (*http.Request, error) {
	if lblr.NextLink == nil || len(to.String(lblr.NextLink)) < 1 {
		return nil, nil
	}

	decorators := []autorest.PrepareDecorator{
		autorest.WithBaseURL(to.String(lblr.NextLink)),
	}
	return c.armClient.PrepareGetRequest(ctx, decorators...)
}

// listNextResults retrieves the next set of results, if any.
func (c *Client) listNextResults(ctx context.Context, lastResults network.SubnetListResult) (result network.SubnetListResult, err error) {
	req, err := c.subnetListResultPreparer(ctx, lastResults)
	if err != nil {
		return result, autorest.NewErrorWithError(err, "subnetclient", "listNextResults", nil, "Failure preparing next results request")
	}
	if req == nil {
		return
	}

	resp, rerr := c.armClient.Send(ctx, req)
	defer c.armClient.CloseResponse(ctx, resp)
	if rerr != nil {
		result.Response = autorest.Response{Response: resp}
		return result, autorest.NewErrorWithError(rerr.Error(), "subnetclient", "listNextResults", resp, "Failure sending next results request")
	}

	result, err = c.listResponder(resp)
	if err != nil {
		err = autorest.NewErrorWithError(err, "subnetclient", "listNextResults", resp, "Failure responding to next results request")
	}

	return
}

// SubnetListResultPage contains a page of Subnet values.
type SubnetListResultPage struct {
	fn  func(context.Context, network.SubnetListResult) (network.SubnetListResult, error)
	slr network.SubnetListResult
}

// NextWithContext advances to the next page of values.  If there was an error making
// the request the page does not advance and the error is returned.
func (page *SubnetListResultPage) NextWithContext(ctx context.Context) (err error) {
	next, err := page.fn(ctx, page.slr)
	if err != nil {
		return err
	}
	page.slr = next
	return nil
}

// Next advances to the next page of values.  If there was an error making
// the request the page does not advance and the error is returned.
// Deprecated: Use NextWithContext() instead.
func (page *SubnetListResultPage) Next() error {
	return page.NextWithContext(context.Background())
}

// NotDone returns true if the page enumeration should be started or is not yet complete.
func (page SubnetListResultPage) NotDone() bool {
	return !page.slr.IsEmpty()
}

// Response returns the raw server response from the last page request.
func (page SubnetListResultPage) Response() network.SubnetListResult {
	return page.slr
}

// Values returns the slice of values for the current page or nil if there are no values.
func (page SubnetListResultPage) Values() []network.Subnet {
	if page.slr.IsEmpty() {
		return nil
	}
	return *page.slr.Value
}
