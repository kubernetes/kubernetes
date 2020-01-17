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

package loadbalancerclient

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-06-01/network"
	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/azure"
	"github.com/Azure/go-autorest/autorest/to"

	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/klog"
	azclients "k8s.io/legacy-cloud-providers/azure/clients"
	"k8s.io/legacy-cloud-providers/azure/clients/armclient"
	"k8s.io/legacy-cloud-providers/azure/metrics"
	"k8s.io/legacy-cloud-providers/azure/retry"
)

var _ Interface = &Client{}

// Client implements LoadBalancer client Interface.
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

// New creates a new LoadBalancer client with ratelimiting.
func New(config *azclients.ClientConfig) *Client {
	baseURI := config.ResourceManagerEndpoint
	authorizer := autorest.NewBearerAuthorizer(config.ServicePrincipalToken)
	armClient := armclient.New(authorizer, baseURI, "", APIVersion, config.Location, config.Backoff)
	rateLimiterReader, rateLimiterWriter := azclients.NewRateLimiter(config.RateLimitConfig)

	klog.V(2).Infof("Azure LoadBalancersClient (read ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPS,
		config.RateLimitConfig.CloudProviderRateLimitBucket)
	klog.V(2).Infof("Azure LoadBalancersClient (write ops) using rate limit config: QPS=%g, bucket=%d",
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

// Get gets a LoadBalancer.
func (c *Client) Get(ctx context.Context, resourceGroupName string, loadBalancerName string, expand string) (network.LoadBalancer, *retry.Error) {
	mc := metrics.NewMetricContext("load_balancers", "get", resourceGroupName, c.subscriptionID, "")

	// Report errors if the client is rate limited.
	if !c.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		return network.LoadBalancer{}, retry.GetRateLimitError(false, "LBGet")
	}

	// Report errors if the client is throttled.
	if c.RetryAfterReader.After(time.Now()) {
		mc.ThrottledCount()
		rerr := retry.GetThrottlingError("LBGet", "client throttled", c.RetryAfterReader)
		return network.LoadBalancer{}, rerr
	}

	result, rerr := c.getLB(ctx, resourceGroupName, loadBalancerName, expand)
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

// getLB gets a LoadBalancer.
func (c *Client) getLB(ctx context.Context, resourceGroupName string, loadBalancerName string, expand string) (network.LoadBalancer, *retry.Error) {
	resourceID := armclient.GetResourceID(
		c.subscriptionID,
		resourceGroupName,
		"Microsoft.Network/loadBalancers",
		loadBalancerName,
	)
	result := network.LoadBalancer{}

	response, rerr := c.armClient.GetResource(ctx, resourceID, expand)
	defer c.armClient.CloseResponse(ctx, response)
	if rerr != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "loadbalancer.get.request", resourceID, rerr.Error())
		return result, rerr
	}

	err := autorest.Respond(
		response,
		azure.WithErrorUnlessStatusCode(http.StatusOK),
		autorest.ByUnmarshallingJSON(&result))
	if err != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "loadbalancer.get.respond", resourceID, err)
		return result, retry.GetError(response, err)
	}

	result.Response = autorest.Response{Response: response}
	return result, nil
}

// List gets a list of LoadBalancer in the resource group.
func (c *Client) List(ctx context.Context, resourceGroupName string) ([]network.LoadBalancer, *retry.Error) {
	mc := metrics.NewMetricContext("load_balancers", "list", resourceGroupName, c.subscriptionID, "")

	// Report errors if the client is rate limited.
	if !c.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		return nil, retry.GetRateLimitError(false, "LBList")
	}

	// Report errors if the client is throttled.
	if c.RetryAfterReader.After(time.Now()) {
		mc.ThrottledCount()
		rerr := retry.GetThrottlingError("LBList", "client throttled", c.RetryAfterReader)
		return nil, rerr
	}

	result, rerr := c.listLB(ctx, resourceGroupName)
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

// listLB gets a list of LoadBalancers in the resource group.
func (c *Client) listLB(ctx context.Context, resourceGroupName string) ([]network.LoadBalancer, *retry.Error) {
	resourceID := fmt.Sprintf("/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Network/loadBalancers",
		autorest.Encode("path", c.subscriptionID),
		autorest.Encode("path", resourceGroupName))
	result := make([]network.LoadBalancer, 0)
	page := &LoadBalancerListResultPage{}
	page.fn = c.listNextResults

	resp, rerr := c.armClient.GetResource(ctx, resourceID, "")
	defer c.armClient.CloseResponse(ctx, resp)
	if rerr != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "loadbalancer.list.request", resourceID, rerr.Error())
		return result, rerr
	}

	var err error
	page.lblr, err = c.listResponder(resp)
	if err != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "loadbalancer.list.respond", resourceID, err)
		return result, retry.GetError(resp, err)
	}

	for page.NotDone() {
		result = append(result, *page.Response().Value...)
		if err = page.NextWithContext(ctx); err != nil {
			klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "loadbalancer.list.next", resourceID, err)
			return result, retry.GetError(page.Response().Response.Response, err)
		}
	}

	return result, nil
}

// CreateOrUpdate creates or updates a LoadBalancer.
func (c *Client) CreateOrUpdate(ctx context.Context, resourceGroupName string, loadBalancerName string, parameters network.LoadBalancer, etag string) *retry.Error {
	mc := metrics.NewMetricContext("load_balancers", "create_or_update", resourceGroupName, c.subscriptionID, "")

	// Report errors if the client is rate limited.
	if !c.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return retry.GetRateLimitError(true, "LBCreateOrUpdate")
	}

	// Report errors if the client is throttled.
	if c.RetryAfterWriter.After(time.Now()) {
		mc.ThrottledCount()
		rerr := retry.GetThrottlingError("LBCreateOrUpdate", "client throttled", c.RetryAfterWriter)
		return rerr
	}

	rerr := c.createOrUpdateLB(ctx, resourceGroupName, loadBalancerName, parameters, etag)
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

// createOrUpdateLB creates or updates a LoadBalancer.
func (c *Client) createOrUpdateLB(ctx context.Context, resourceGroupName string, loadBalancerName string, parameters network.LoadBalancer, etag string) *retry.Error {
	resourceID := armclient.GetResourceID(
		c.subscriptionID,
		resourceGroupName,
		"Microsoft.Network/loadBalancers",
		loadBalancerName,
	)
	decorators := []autorest.PrepareDecorator{
		autorest.WithPathParameters("{resourceID}", map[string]interface{}{"resourceID": resourceID}),
		autorest.WithJSON(parameters),
	}
	if etag != "" {
		decorators = append(decorators, autorest.WithHeader("If-Match", autorest.String(etag)))
	}

	response, rerr := c.armClient.PutResourceWithDecorators(ctx, resourceID, parameters, decorators)
	defer c.armClient.CloseResponse(ctx, response)
	if rerr != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "loadbalancer.put.request", resourceID, rerr.Error())
		return rerr
	}

	if response != nil && response.StatusCode != http.StatusNoContent {
		_, rerr = c.createOrUpdateResponder(response)
		if rerr != nil {
			klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "loadbalancer.put.respond", resourceID, rerr.Error())
			return rerr
		}
	}

	return nil
}

func (c *Client) createOrUpdateResponder(resp *http.Response) (*network.LoadBalancer, *retry.Error) {
	result := &network.LoadBalancer{}
	err := autorest.Respond(
		resp,
		azure.WithErrorUnlessStatusCode(http.StatusOK, http.StatusCreated),
		autorest.ByUnmarshallingJSON(&result))
	result.Response = autorest.Response{Response: resp}
	return result, retry.GetError(resp, err)
}

// Delete deletes a LoadBalancer by name.
func (c *Client) Delete(ctx context.Context, resourceGroupName string, loadBalancerName string) *retry.Error {
	mc := metrics.NewMetricContext("load_balancers", "delete", resourceGroupName, c.subscriptionID, "")

	// Report errors if the client is rate limited.
	if !c.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return retry.GetRateLimitError(true, "LBDelete")
	}

	// Report errors if the client is throttled.
	if c.RetryAfterWriter.After(time.Now()) {
		mc.ThrottledCount()
		rerr := retry.GetThrottlingError("LBDelete", "client throttled", c.RetryAfterWriter)
		return rerr
	}

	rerr := c.deleteLB(ctx, resourceGroupName, loadBalancerName)
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

// deleteLB deletes a LoadBalancer by name.
func (c *Client) deleteLB(ctx context.Context, resourceGroupName string, loadBalancerName string) *retry.Error {
	resourceID := armclient.GetResourceID(
		c.subscriptionID,
		resourceGroupName,
		"Microsoft.Network/loadBalancers",
		loadBalancerName,
	)

	return c.armClient.DeleteResource(ctx, resourceID, "")
}

func (c *Client) listResponder(resp *http.Response) (result network.LoadBalancerListResult, err error) {
	err = autorest.Respond(
		resp,
		autorest.ByIgnoring(),
		azure.WithErrorUnlessStatusCode(http.StatusOK),
		autorest.ByUnmarshallingJSON(&result))
	result.Response = autorest.Response{Response: resp}
	return
}

// loadBalancerListResultPreparer prepares a request to retrieve the next set of results.
// It returns nil if no more results exist.
func (c *Client) loadBalancerListResultPreparer(ctx context.Context, lblr network.LoadBalancerListResult) (*http.Request, error) {
	if lblr.NextLink == nil || len(to.String(lblr.NextLink)) < 1 {
		return nil, nil
	}

	decorators := []autorest.PrepareDecorator{
		autorest.WithBaseURL(to.String(lblr.NextLink)),
	}
	return c.armClient.PrepareGetRequest(ctx, decorators...)
}

// listNextResults retrieves the next set of results, if any.
func (c *Client) listNextResults(ctx context.Context, lastResults network.LoadBalancerListResult) (result network.LoadBalancerListResult, err error) {
	req, err := c.loadBalancerListResultPreparer(ctx, lastResults)
	if err != nil {
		return result, autorest.NewErrorWithError(err, "loadbalancerclient", "listNextResults", nil, "Failure preparing next results request")
	}
	if req == nil {
		return
	}

	resp, rerr := c.armClient.Send(ctx, req)
	defer c.armClient.CloseResponse(ctx, resp)
	if rerr != nil {
		result.Response = autorest.Response{Response: resp}
		return result, autorest.NewErrorWithError(rerr.Error(), "loadbalancerclient", "listNextResults", resp, "Failure sending next results request")
	}

	result, err = c.listResponder(resp)
	if err != nil {
		err = autorest.NewErrorWithError(err, "loadbalancerclient", "listNextResults", resp, "Failure responding to next results request")
	}

	return
}

// LoadBalancerListResultPage contains a page of LoadBalancer values.
type LoadBalancerListResultPage struct {
	fn   func(context.Context, network.LoadBalancerListResult) (network.LoadBalancerListResult, error)
	lblr network.LoadBalancerListResult
}

// NextWithContext advances to the next page of values.  If there was an error making
// the request the page does not advance and the error is returned.
func (page *LoadBalancerListResultPage) NextWithContext(ctx context.Context) (err error) {
	next, err := page.fn(ctx, page.lblr)
	if err != nil {
		return err
	}
	page.lblr = next
	return nil
}

// Next advances to the next page of values.  If there was an error making
// the request the page does not advance and the error is returned.
// Deprecated: Use NextWithContext() instead.
func (page *LoadBalancerListResultPage) Next() error {
	return page.NextWithContext(context.Background())
}

// NotDone returns true if the page enumeration should be started or is not yet complete.
func (page LoadBalancerListResultPage) NotDone() bool {
	return !page.lblr.IsEmpty()
}

// Response returns the raw server response from the last page request.
func (page LoadBalancerListResultPage) Response() network.LoadBalancerListResult {
	return page.lblr
}

// Values returns the slice of values for the current page or nil if there are no values.
func (page LoadBalancerListResultPage) Values() []network.LoadBalancer {
	if page.lblr.IsEmpty() {
		return nil
	}
	return *page.lblr.Value
}
