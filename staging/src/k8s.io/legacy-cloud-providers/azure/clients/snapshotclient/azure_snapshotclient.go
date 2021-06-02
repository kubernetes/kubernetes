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

package snapshotclient

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-12-01/compute"
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

// Client implements Snapshot client Interface.
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

// New creates a new Snapshot client with ratelimiting.
func New(config *azclients.ClientConfig) *Client {
	baseURI := config.ResourceManagerEndpoint
	authorizer := config.Authorizer
	armClient := armclient.New(authorizer, baseURI, config.UserAgent, APIVersion, config.Location, config.Backoff)
	rateLimiterReader, rateLimiterWriter := azclients.NewRateLimiter(config.RateLimitConfig)

	klog.V(2).Infof("Azure SnapshotClient (read ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPS,
		config.RateLimitConfig.CloudProviderRateLimitBucket)
	klog.V(2).Infof("Azure SnapshotClient (write ops) using rate limit config: QPS=%g, bucket=%d",
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

// Get gets a Snapshot.
func (c *Client) Get(ctx context.Context, resourceGroupName string, snapshotName string) (compute.Snapshot, *retry.Error) {
	mc := metrics.NewMetricContext("snapshot", "get", resourceGroupName, c.subscriptionID, "")

	// Report errors if the client is rate limited.
	if !c.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		return compute.Snapshot{}, retry.GetRateLimitError(false, "SnapshotGet")
	}

	// Report errors if the client is throttled.
	if c.RetryAfterReader.After(time.Now()) {
		mc.ThrottledCount()
		rerr := retry.GetThrottlingError("SnapshotGet", "client throttled", c.RetryAfterReader)
		return compute.Snapshot{}, rerr
	}

	result, rerr := c.getSnapshot(ctx, resourceGroupName, snapshotName)
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

// getSnapshot gets a Snapshot.
func (c *Client) getSnapshot(ctx context.Context, resourceGroupName string, snapshotName string) (compute.Snapshot, *retry.Error) {
	resourceID := armclient.GetResourceID(
		c.subscriptionID,
		resourceGroupName,
		"Microsoft.Compute/snapshots",
		snapshotName,
	)
	result := compute.Snapshot{}

	response, rerr := c.armClient.GetResource(ctx, resourceID, "")
	defer c.armClient.CloseResponse(ctx, response)
	if rerr != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "snapshot.get.request", resourceID, rerr.Error())
		return result, rerr
	}

	err := autorest.Respond(
		response,
		azure.WithErrorUnlessStatusCode(http.StatusOK),
		autorest.ByUnmarshallingJSON(&result))
	if err != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "snapshot.get.respond", resourceID, err)
		return result, retry.GetError(response, err)
	}

	result.Response = autorest.Response{Response: response}
	return result, nil
}

// Delete deletes a Snapshot by name.
func (c *Client) Delete(ctx context.Context, resourceGroupName string, snapshotName string) *retry.Error {
	mc := metrics.NewMetricContext("snapshot", "delete", resourceGroupName, c.subscriptionID, "")

	// Report errors if the client is rate limited.
	if !c.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return retry.GetRateLimitError(true, "SnapshotDelete")
	}

	// Report errors if the client is throttled.
	if c.RetryAfterWriter.After(time.Now()) {
		mc.ThrottledCount()
		rerr := retry.GetThrottlingError("SnapshotDelete", "client throttled", c.RetryAfterWriter)
		return rerr
	}

	rerr := c.deleteSnapshot(ctx, resourceGroupName, snapshotName)
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

// deleteSnapshot deletes a PublicIPAddress by name.
func (c *Client) deleteSnapshot(ctx context.Context, resourceGroupName string, snapshotName string) *retry.Error {
	resourceID := armclient.GetResourceID(
		c.subscriptionID,
		resourceGroupName,
		"Microsoft.Compute/snapshots",
		snapshotName,
	)

	return c.armClient.DeleteResource(ctx, resourceID, "")
}

// CreateOrUpdate creates or updates a Snapshot.
func (c *Client) CreateOrUpdate(ctx context.Context, resourceGroupName string, snapshotName string, snapshot compute.Snapshot) *retry.Error {
	mc := metrics.NewMetricContext("snapshot", "create_or_update", resourceGroupName, c.subscriptionID, "")

	// Report errors if the client is rate limited.
	if !c.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return retry.GetRateLimitError(true, "SnapshotCreateOrUpdate")
	}

	// Report errors if the client is throttled.
	if c.RetryAfterWriter.After(time.Now()) {
		mc.ThrottledCount()
		rerr := retry.GetThrottlingError("SnapshotCreateOrUpdate", "client throttled", c.RetryAfterWriter)
		return rerr
	}

	rerr := c.createOrUpdateSnapshot(ctx, resourceGroupName, snapshotName, snapshot)
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

// createOrUpdateSnapshot creates or updates a Snapshot.
func (c *Client) createOrUpdateSnapshot(ctx context.Context, resourceGroupName string, snapshotName string, snapshot compute.Snapshot) *retry.Error {
	resourceID := armclient.GetResourceID(
		c.subscriptionID,
		resourceGroupName,
		"Microsoft.Compute/snapshots",
		snapshotName,
	)

	response, rerr := c.armClient.PutResource(ctx, resourceID, snapshot)
	defer c.armClient.CloseResponse(ctx, response)
	if rerr != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "snapshot.put.request", resourceID, rerr.Error())
		return rerr
	}

	if response != nil && response.StatusCode != http.StatusNoContent {
		_, rerr = c.createOrUpdateResponder(response)
		if rerr != nil {
			klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "snapshot.put.respond", resourceID, rerr.Error())
			return rerr
		}
	}

	return nil
}

func (c *Client) createOrUpdateResponder(resp *http.Response) (*compute.Snapshot, *retry.Error) {
	result := &compute.Snapshot{}
	err := autorest.Respond(
		resp,
		azure.WithErrorUnlessStatusCode(http.StatusOK, http.StatusCreated),
		autorest.ByUnmarshallingJSON(&result))
	result.Response = autorest.Response{Response: resp}
	return result, retry.GetError(resp, err)
}

// ListByResourceGroup get a list snapshots by resourceGroup.
func (c *Client) ListByResourceGroup(ctx context.Context, resourceGroupName string) ([]compute.Snapshot, *retry.Error) {
	mc := metrics.NewMetricContext("snapshot", "list_by_resource_group", resourceGroupName, c.subscriptionID, "")

	// Report errors if the client is rate limited.
	if !c.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		return nil, retry.GetRateLimitError(false, "SnapshotListByResourceGroup")
	}

	// Report errors if the client is throttled.
	if c.RetryAfterReader.After(time.Now()) {
		mc.ThrottledCount()
		rerr := retry.GetThrottlingError("SnapshotListByResourceGroup", "client throttled", c.RetryAfterReader)
		return nil, rerr
	}

	result, rerr := c.listSnapshotsByResourceGroup(ctx, resourceGroupName)
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

// listSnapshotsByResourceGroup gets a list of snapshots in the resource group.
func (c *Client) listSnapshotsByResourceGroup(ctx context.Context, resourceGroupName string) ([]compute.Snapshot, *retry.Error) {
	resourceID := fmt.Sprintf("/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Compute/snapshots",
		autorest.Encode("path", c.subscriptionID),
		autorest.Encode("path", resourceGroupName))
	result := make([]compute.Snapshot, 0)
	page := &SnapshotListPage{}
	page.fn = c.listNextResults

	resp, rerr := c.armClient.GetResource(ctx, resourceID, "")
	defer c.armClient.CloseResponse(ctx, resp)
	if rerr != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "snapshot.list.request", resourceID, rerr.Error())
		return result, rerr
	}

	var err error
	page.sl, err = c.listResponder(resp)
	if err != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "snapshot.list.respond", resourceID, err)
		return result, retry.GetError(resp, err)
	}

	for {
		result = append(result, page.Values()...)

		// Abort the loop when there's no nextLink in the response.
		if to.String(page.Response().NextLink) == "" {
			break
		}

		if err = page.NextWithContext(ctx); err != nil {
			klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "snapshot.list.next", resourceID, err)
			return result, retry.GetError(page.Response().Response.Response, err)
		}
	}

	return result, nil
}

func (c *Client) listResponder(resp *http.Response) (result compute.SnapshotList, err error) {
	err = autorest.Respond(
		resp,
		autorest.ByIgnoring(),
		azure.WithErrorUnlessStatusCode(http.StatusOK),
		autorest.ByUnmarshallingJSON(&result))
	result.Response = autorest.Response{Response: resp}
	return
}

// SnapshotListResultPreparer prepares a request to retrieve the next set of results.
// It returns nil if no more results exist.
func (c *Client) SnapshotListResultPreparer(ctx context.Context, lr compute.SnapshotList) (*http.Request, error) {
	if lr.NextLink == nil || len(to.String(lr.NextLink)) < 1 {
		return nil, nil
	}

	decorators := []autorest.PrepareDecorator{
		autorest.WithBaseURL(to.String(lr.NextLink)),
	}
	return c.armClient.PrepareGetRequest(ctx, decorators...)
}

// listNextResults retrieves the next set of results, if any.
func (c *Client) listNextResults(ctx context.Context, lastResults compute.SnapshotList) (result compute.SnapshotList, err error) {
	req, err := c.SnapshotListResultPreparer(ctx, lastResults)
	if err != nil {
		return result, autorest.NewErrorWithError(err, "snapshotclient", "listNextResults", nil, "Failure preparing next results request")
	}
	if req == nil {
		return
	}

	resp, rerr := c.armClient.Send(ctx, req)
	defer c.armClient.CloseResponse(ctx, resp)
	if rerr != nil {
		result.Response = autorest.Response{Response: resp}
		return result, autorest.NewErrorWithError(rerr.Error(), "snapshotclient", "listNextResults", resp, "Failure sending next results request")
	}

	result, err = c.listResponder(resp)
	if err != nil {
		err = autorest.NewErrorWithError(err, "snapshotclient", "listNextResults", resp, "Failure responding to next results request")
	}

	return
}

// SnapshotListPage contains a page of Snapshot values.
type SnapshotListPage struct {
	fn func(context.Context, compute.SnapshotList) (compute.SnapshotList, error)
	sl compute.SnapshotList
}

// NextWithContext advances to the next page of values.  If there was an error making
// the request the page does not advance and the error is returned.
func (page *SnapshotListPage) NextWithContext(ctx context.Context) (err error) {
	next, err := page.fn(ctx, page.sl)
	if err != nil {
		return err
	}
	page.sl = next
	return nil
}

// Next advances to the next page of values.  If there was an error making
// the request the page does not advance and the error is returned.
// Deprecated: Use NextWithContext() instead.
func (page *SnapshotListPage) Next() error {
	return page.NextWithContext(context.Background())
}

// NotDone returns true if the page enumeration should be started or is not yet complete.
func (page SnapshotListPage) NotDone() bool {
	return !page.sl.IsEmpty()
}

// Response returns the raw server response from the last page request.
func (page SnapshotListPage) Response() compute.SnapshotList {
	return page.sl
}

// Values returns the slice of values for the current page or nil if there are no values.
func (page SnapshotListPage) Values() []compute.Snapshot {
	if page.sl.IsEmpty() {
		return nil
	}
	return *page.sl.Value
}
