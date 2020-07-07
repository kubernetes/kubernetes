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

package containerserviceclient

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-04-01/containerservice"
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

// Client implements ContainerService client Interface.
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

// New creates a new ContainerServiceClient client with ratelimiting.
func New(config *azclients.ClientConfig) *Client {
	baseURI := config.ResourceManagerEndpoint
	authorizer := config.Authorizer
	armClient := armclient.New(authorizer, baseURI, config.UserAgent, APIVersion, config.Location, config.Backoff)
	rateLimiterReader, rateLimiterWriter := azclients.NewRateLimiter(config.RateLimitConfig)

	klog.V(2).Infof("Azure ContainerServiceClient (read ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPS,
		config.RateLimitConfig.CloudProviderRateLimitBucket)
	klog.V(2).Infof("Azure ContainerServiceClient (write ops) using rate limit config: QPS=%g, bucket=%d",
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

// Get gets a ManagedCluster.
func (c *Client) Get(ctx context.Context, resourceGroupName string, managedClusterName string) (containerservice.ManagedCluster, *retry.Error) {
	mc := metrics.NewMetricContext("managed_clusters", "get", resourceGroupName, c.subscriptionID, "")

	// Report errors if the client is rate limited.
	if !c.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		return containerservice.ManagedCluster{}, retry.GetRateLimitError(false, "GetManagedCluster")
	}

	// Report errors if the client is throttled.
	if c.RetryAfterReader.After(time.Now()) {
		mc.ThrottledCount()
		rerr := retry.GetThrottlingError("GetManagedCluster", "client throttled", c.RetryAfterReader)
		return containerservice.ManagedCluster{}, rerr
	}

	result, rerr := c.getManagedCluster(ctx, resourceGroupName, managedClusterName)
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

// getManagedCluster gets a ManagedCluster.
func (c *Client) getManagedCluster(ctx context.Context, resourceGroupName string, managedClusterName string) (containerservice.ManagedCluster, *retry.Error) {
	resourceID := armclient.GetResourceID(
		c.subscriptionID,
		resourceGroupName,
		"Microsoft.ContainerService/managedClusters",
		managedClusterName,
	)
	result := containerservice.ManagedCluster{}

	response, rerr := c.armClient.GetResource(ctx, resourceID, "")
	defer c.armClient.CloseResponse(ctx, response)
	if rerr != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "managedcluster.get.request", resourceID, rerr.Error())
		return result, rerr
	}

	err := autorest.Respond(
		response,
		azure.WithErrorUnlessStatusCode(http.StatusOK),
		autorest.ByUnmarshallingJSON(&result))
	if err != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "managedcluster.get.respond", resourceID, err)
		return result, retry.GetError(response, err)
	}

	result.Response = autorest.Response{Response: response}
	return result, nil
}

// List gets a list of ManagedClusters in the resource group.
func (c *Client) List(ctx context.Context, resourceGroupName string) ([]containerservice.ManagedCluster, *retry.Error) {
	mc := metrics.NewMetricContext("managed_clusters", "list", resourceGroupName, c.subscriptionID, "")

	// Report errors if the client is rate limited.
	if !c.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		return nil, retry.GetRateLimitError(false, "ListManagedCluster")
	}

	// Report errors if the client is throttled.
	if c.RetryAfterReader.After(time.Now()) {
		mc.ThrottledCount()
		rerr := retry.GetThrottlingError("ListManagedCluster", "client throttled", c.RetryAfterReader)
		return nil, rerr
	}

	result, rerr := c.listManagedCluster(ctx, resourceGroupName)
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

// listManagedCluster gets a list of ManagedClusters in the resource group.
func (c *Client) listManagedCluster(ctx context.Context, resourceGroupName string) ([]containerservice.ManagedCluster, *retry.Error) {
	resourceID := fmt.Sprintf("/subscriptions/%s/resourceGroups/%s/providers/Microsoft.ContainerService/managedClusters",
		autorest.Encode("path", c.subscriptionID),
		autorest.Encode("path", resourceGroupName))
	result := make([]containerservice.ManagedCluster, 0)
	page := &ManagedClusterResultPage{}
	page.fn = c.listNextResults

	resp, rerr := c.armClient.GetResource(ctx, resourceID, "")
	defer c.armClient.CloseResponse(ctx, resp)
	if rerr != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "managedcluster.list.request", resourceID, rerr.Error())
		return result, rerr
	}

	var err error
	page.mclr, err = c.listResponder(resp)
	if err != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "managedcluster.list.respond", resourceID, err)
		return result, retry.GetError(resp, err)
	}

	for page.NotDone() {
		result = append(result, *page.Response().Value...)
		if err = page.NextWithContext(ctx); err != nil {
			klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "managedcluster.list.next", resourceID, err)
			return result, retry.GetError(page.Response().Response.Response, err)
		}
	}

	return result, nil
}

func (c *Client) listResponder(resp *http.Response) (result containerservice.ManagedClusterListResult, err error) {
	err = autorest.Respond(
		resp,
		autorest.ByIgnoring(),
		azure.WithErrorUnlessStatusCode(http.StatusOK),
		autorest.ByUnmarshallingJSON(&result))
	result.Response = autorest.Response{Response: resp}
	return
}

// managedClusterListResultPreparer prepares a request to retrieve the next set of results.
// It returns nil if no more results exist.
func (c *Client) managedClusterListResultPreparer(ctx context.Context, mclr containerservice.ManagedClusterListResult) (*http.Request, error) {
	if mclr.NextLink == nil || len(to.String(mclr.NextLink)) < 1 {
		return nil, nil
	}

	decorators := []autorest.PrepareDecorator{
		autorest.WithBaseURL(to.String(mclr.NextLink)),
	}
	return c.armClient.PrepareGetRequest(ctx, decorators...)
}

// listNextResults retrieves the next set of results, if any.
func (c *Client) listNextResults(ctx context.Context, lastResults containerservice.ManagedClusterListResult) (result containerservice.ManagedClusterListResult, err error) {
	req, err := c.managedClusterListResultPreparer(ctx, lastResults)
	if err != nil {
		return result, autorest.NewErrorWithError(err, "managedclusterclient", "listNextResults", nil, "Failure preparing next results request")
	}
	if req == nil {
		return
	}

	resp, rerr := c.armClient.Send(ctx, req)
	defer c.armClient.CloseResponse(ctx, resp)
	if rerr != nil {
		result.Response = autorest.Response{Response: resp}
		return result, autorest.NewErrorWithError(rerr.Error(), "managedclusterclient", "listNextResults", resp, "Failure sending next results request")
	}

	result, err = c.listResponder(resp)
	if err != nil {
		err = autorest.NewErrorWithError(err, "managedclusterclient", "listNextResults", resp, "Failure responding to next results request")
	}

	return
}

// ManagedClusterResultPage contains a page of ManagedCluster values.
type ManagedClusterResultPage struct {
	fn   func(context.Context, containerservice.ManagedClusterListResult) (containerservice.ManagedClusterListResult, error)
	mclr containerservice.ManagedClusterListResult
}

// NextWithContext advances to the next page of values.  If there was an error making
// the request the page does not advance and the error is returned.
func (page *ManagedClusterResultPage) NextWithContext(ctx context.Context) (err error) {
	next, err := page.fn(ctx, page.mclr)
	if err != nil {
		return err
	}
	page.mclr = next
	return nil
}

// Next advances to the next page of values.  If there was an error making
// the request the page does not advance and the error is returned.
// Deprecated: Use NextWithContext() instead.
func (page *ManagedClusterResultPage) Next() error {
	return page.NextWithContext(context.Background())
}

// NotDone returns true if the page enumeration should be started or is not yet complete.
func (page ManagedClusterResultPage) NotDone() bool {
	return !page.mclr.IsEmpty()
}

// Response returns the raw server response from the last page request.
func (page ManagedClusterResultPage) Response() containerservice.ManagedClusterListResult {
	return page.mclr
}

// Values returns the slice of values for the current page or nil if there are no values.
func (page ManagedClusterResultPage) Values() []containerservice.ManagedCluster {
	if page.mclr.IsEmpty() {
		return nil
	}
	return *page.mclr.Value
}

// CreateOrUpdate creates or updates a ManagedCluster.
func (c *Client) CreateOrUpdate(ctx context.Context, resourceGroupName string, managedClusterName string, parameters containerservice.ManagedCluster, etag string) *retry.Error {
	mc := metrics.NewMetricContext("managed_clusters", "create_or_update", resourceGroupName, c.subscriptionID, "")

	// Report errors if the client is rate limited.
	if !c.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return retry.GetRateLimitError(true, "CreateOrUpdateManagedCluster")
	}

	// Report errors if the client is throttled.
	if c.RetryAfterWriter.After(time.Now()) {
		mc.ThrottledCount()
		rerr := retry.GetThrottlingError("CreateOrUpdateManagedCluster", "client throttled", c.RetryAfterWriter)
		return rerr
	}

	rerr := c.createOrUpdateManagedCluster(ctx, resourceGroupName, managedClusterName, parameters, etag)
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

// createOrUpdateManagedCluster creates or updates a ManagedCluster.
func (c *Client) createOrUpdateManagedCluster(ctx context.Context, resourceGroupName string, managedClusterName string, parameters containerservice.ManagedCluster, etag string) *retry.Error {
	resourceID := armclient.GetResourceID(
		c.subscriptionID,
		resourceGroupName,
		"Microsoft.ContainerService/managedClusters",
		managedClusterName,
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
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "managedCluster.put.request", resourceID, rerr.Error())
		return rerr
	}

	if response != nil && response.StatusCode != http.StatusNoContent {
		_, rerr = c.createOrUpdateResponder(response)
		if rerr != nil {
			klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "managedCluster.put.respond", resourceID, rerr.Error())
			return rerr
		}
	}

	return nil
}

func (c *Client) createOrUpdateResponder(resp *http.Response) (*containerservice.ManagedCluster, *retry.Error) {
	result := &containerservice.ManagedCluster{}
	err := autorest.Respond(
		resp,
		azure.WithErrorUnlessStatusCode(http.StatusOK, http.StatusCreated),
		autorest.ByUnmarshallingJSON(&result))
	result.Response = autorest.Response{Response: resp}
	return result, retry.GetError(resp, err)
}

// Delete deletes a ManagedCluster by name.
func (c *Client) Delete(ctx context.Context, resourceGroupName string, managedClusterName string) *retry.Error {
	mc := metrics.NewMetricContext("managed_clusters", "delete", resourceGroupName, c.subscriptionID, "")

	// Report errors if the client is rate limited.
	if !c.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return retry.GetRateLimitError(true, "DeleteManagedCluster")
	}

	// Report errors if the client is throttled.
	if c.RetryAfterWriter.After(time.Now()) {
		mc.ThrottledCount()
		rerr := retry.GetThrottlingError("DeleteManagedCluster", "client throttled", c.RetryAfterWriter)
		return rerr
	}

	rerr := c.deleteManagedCluster(ctx, resourceGroupName, managedClusterName)
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

// deleteManagedCluster deletes a ManagedCluster by name.
func (c *Client) deleteManagedCluster(ctx context.Context, resourceGroupName string, managedClusterName string) *retry.Error {
	resourceID := armclient.GetResourceID(
		c.subscriptionID,
		resourceGroupName,
		"Microsoft.ContainerService/managedClusters",
		managedClusterName,
	)

	return c.armClient.DeleteResource(ctx, resourceID, "")
}
