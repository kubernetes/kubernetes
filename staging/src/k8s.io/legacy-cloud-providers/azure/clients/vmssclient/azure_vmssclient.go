// +build !providerless

/*
Copyright 2019 The Kubernetes Authors.

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

package vmssclient

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

// Client implements VMSS client Interface.
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

// New creates a new VMSS client with ratelimiting.
func New(config *azclients.ClientConfig) *Client {
	baseURI := config.ResourceManagerEndpoint
	authorizer := config.Authorizer
	armClient := armclient.New(authorizer, baseURI, "", APIVersion, config.Location, config.Backoff)
	rateLimiterReader, rateLimiterWriter := azclients.NewRateLimiter(config.RateLimitConfig)

	klog.V(2).Infof("Azure VirtualMachineScaleSetClient (read ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPS,
		config.RateLimitConfig.CloudProviderRateLimitBucket)
	klog.V(2).Infof("Azure VirtualMachineScaleSetClient (write ops) using rate limit config: QPS=%g, bucket=%d",
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

// Get gets a VirtualMachineScaleSet.
func (c *Client) Get(ctx context.Context, resourceGroupName string, VMScaleSetName string) (compute.VirtualMachineScaleSet, *retry.Error) {
	mc := metrics.NewMetricContext("vmss", "get", resourceGroupName, c.subscriptionID, "")

	// Report errors if the client is rate limited.
	if !c.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		return compute.VirtualMachineScaleSet{}, retry.GetRateLimitError(false, "VMSSGet")
	}

	// Report errors if the client is throttled.
	if c.RetryAfterReader.After(time.Now()) {
		mc.ThrottledCount()
		rerr := retry.GetThrottlingError("VMSSGet", "client throttled", c.RetryAfterReader)
		return compute.VirtualMachineScaleSet{}, rerr
	}

	result, rerr := c.getVMSS(ctx, resourceGroupName, VMScaleSetName)
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

// getVMSS gets a VirtualMachineScaleSet.
func (c *Client) getVMSS(ctx context.Context, resourceGroupName string, VMScaleSetName string) (compute.VirtualMachineScaleSet, *retry.Error) {
	resourceID := armclient.GetResourceID(
		c.subscriptionID,
		resourceGroupName,
		"Microsoft.Compute/virtualMachineScaleSets",
		VMScaleSetName,
	)
	result := compute.VirtualMachineScaleSet{}

	response, rerr := c.armClient.GetResource(ctx, resourceID, "")
	defer c.armClient.CloseResponse(ctx, response)
	if rerr != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "vmss.get.request", resourceID, rerr.Error())
		return result, rerr
	}

	err := autorest.Respond(
		response,
		azure.WithErrorUnlessStatusCode(http.StatusOK),
		autorest.ByUnmarshallingJSON(&result))
	if err != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "vmss.get.respond", resourceID, err)
		return result, retry.GetError(response, err)
	}

	result.Response = autorest.Response{Response: response}
	return result, nil
}

// List gets a list of VirtualMachineScaleSets in the resource group.
func (c *Client) List(ctx context.Context, resourceGroupName string) ([]compute.VirtualMachineScaleSet, *retry.Error) {
	mc := metrics.NewMetricContext("vmss", "list", resourceGroupName, c.subscriptionID, "")

	// Report errors if the client is rate limited.
	if !c.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		return nil, retry.GetRateLimitError(false, "VMSSList")
	}

	// Report errors if the client is throttled.
	if c.RetryAfterReader.After(time.Now()) {
		mc.ThrottledCount()
		rerr := retry.GetThrottlingError("VMSSList", "client throttled", c.RetryAfterReader)
		return nil, rerr
	}

	result, rerr := c.listVMSS(ctx, resourceGroupName)
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

// listVMSS gets a list of VirtualMachineScaleSets in the resource group.
func (c *Client) listVMSS(ctx context.Context, resourceGroupName string) ([]compute.VirtualMachineScaleSet, *retry.Error) {
	resourceID := fmt.Sprintf("/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Compute/virtualMachineScaleSets",
		autorest.Encode("path", c.subscriptionID),
		autorest.Encode("path", resourceGroupName))
	result := make([]compute.VirtualMachineScaleSet, 0)
	page := &VirtualMachineScaleSetListResultPage{}
	page.fn = c.listNextResults

	resp, rerr := c.armClient.GetResource(ctx, resourceID, "")
	defer c.armClient.CloseResponse(ctx, resp)
	if rerr != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "vmss.list.request", resourceID, rerr.Error())
		return result, rerr
	}

	var err error
	page.vmsslr, err = c.listResponder(resp)
	if err != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "vmss.list.respond", resourceID, err)
		return result, retry.GetError(resp, err)
	}

	for page.NotDone() {
		result = append(result, *page.Response().Value...)
		if err = page.NextWithContext(ctx); err != nil {
			klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "vmss.list.next", resourceID, err)
			return result, retry.GetError(page.Response().Response.Response, err)
		}
	}

	return result, nil
}

// CreateOrUpdate creates or updates a VirtualMachineScaleSet.
func (c *Client) CreateOrUpdate(ctx context.Context, resourceGroupName string, VMScaleSetName string, parameters compute.VirtualMachineScaleSet) *retry.Error {
	mc := metrics.NewMetricContext("vmss", "create_or_update", resourceGroupName, c.subscriptionID, "")

	// Report errors if the client is rate limited.
	if !c.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return retry.GetRateLimitError(true, "VMSSCreateOrUpdate")
	}

	// Report errors if the client is throttled.
	if c.RetryAfterWriter.After(time.Now()) {
		mc.ThrottledCount()
		rerr := retry.GetThrottlingError("VMSSCreateOrUpdate", "client throttled", c.RetryAfterWriter)
		return rerr
	}

	rerr := c.createOrUpdateVMSS(ctx, resourceGroupName, VMScaleSetName, parameters)
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

// CreateOrUpdateAsync sends the request to arm client and DO NOT wait for the response
func (c *Client) CreateOrUpdateAsync(ctx context.Context, resourceGroupName string, VMScaleSetName string, parameters compute.VirtualMachineScaleSet) (*azure.Future, *retry.Error) {
	mc := metrics.NewMetricContext("vmss", "create_or_update_async", resourceGroupName, c.subscriptionID, "")

	// Report errors if the client is rate limited.
	if !c.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return nil, retry.GetRateLimitError(true, "VMSSCreateOrUpdateAsync")
	}

	// Report errors if the client is throttled.
	if c.RetryAfterWriter.After(time.Now()) {
		mc.ThrottledCount()
		rerr := retry.GetThrottlingError("VMSSCreateOrUpdateAsync", "client throttled", c.RetryAfterWriter)
		return nil, rerr
	}

	resourceID := armclient.GetResourceID(
		c.subscriptionID,
		resourceGroupName,
		"Microsoft.Compute/virtualMachineScaleSets",
		VMScaleSetName,
	)

	future, rerr := c.armClient.PutResourceAsync(ctx, resourceID, parameters)
	mc.Observe(rerr.Error())
	if rerr != nil {
		if rerr.IsThrottled() {
			// Update RetryAfterReader so that no more requests would be sent until RetryAfter expires.
			c.RetryAfterWriter = rerr.RetryAfter
		}

		return nil, rerr
	}

	return future, nil
}

// WaitForAsyncOperationResult waits for the response of the request
func (c *Client) WaitForAsyncOperationResult(ctx context.Context, future *azure.Future) (*http.Response, error) {
	return c.armClient.WaitForAsyncOperationResult(ctx, future, "VMSSWaitForAsyncOperationResult")
}

// createOrUpdateVMSS creates or updates a VirtualMachineScaleSet.
func (c *Client) createOrUpdateVMSS(ctx context.Context, resourceGroupName string, VMScaleSetName string, parameters compute.VirtualMachineScaleSet) *retry.Error {
	resourceID := armclient.GetResourceID(
		c.subscriptionID,
		resourceGroupName,
		"Microsoft.Compute/virtualMachineScaleSets",
		VMScaleSetName,
	)
	response, rerr := c.armClient.PutResource(ctx, resourceID, parameters)
	defer c.armClient.CloseResponse(ctx, response)
	if rerr != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "vmss.put.request", resourceID, rerr.Error())
		return rerr
	}

	if response != nil && response.StatusCode != http.StatusNoContent {
		_, rerr = c.createOrUpdateResponder(response)
		if rerr != nil {
			klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "vmss.put.respond", resourceID, rerr.Error())
			return rerr
		}
	}

	return nil
}

func (c *Client) createOrUpdateResponder(resp *http.Response) (*compute.VirtualMachineScaleSet, *retry.Error) {
	result := &compute.VirtualMachineScaleSet{}
	err := autorest.Respond(
		resp,
		azure.WithErrorUnlessStatusCode(http.StatusOK, http.StatusCreated),
		autorest.ByUnmarshallingJSON(&result))
	result.Response = autorest.Response{Response: resp}
	return result, retry.GetError(resp, err)
}

func (c *Client) listResponder(resp *http.Response) (result compute.VirtualMachineScaleSetListResult, err error) {
	err = autorest.Respond(
		resp,
		autorest.ByIgnoring(),
		azure.WithErrorUnlessStatusCode(http.StatusOK),
		autorest.ByUnmarshallingJSON(&result))
	result.Response = autorest.Response{Response: resp}
	return
}

// virtualMachineScaleSetListResultPreparer prepares a request to retrieve the next set of results.
// It returns nil if no more results exist.
func (c *Client) virtualMachineScaleSetListResultPreparer(ctx context.Context, vmsslr compute.VirtualMachineScaleSetListResult) (*http.Request, error) {
	if vmsslr.NextLink == nil || len(to.String(vmsslr.NextLink)) < 1 {
		return nil, nil
	}

	decorators := []autorest.PrepareDecorator{
		autorest.WithBaseURL(to.String(vmsslr.NextLink)),
	}
	return c.armClient.PrepareGetRequest(ctx, decorators...)
}

// listNextResults retrieves the next set of results, if any.
func (c *Client) listNextResults(ctx context.Context, lastResults compute.VirtualMachineScaleSetListResult) (result compute.VirtualMachineScaleSetListResult, err error) {
	req, err := c.virtualMachineScaleSetListResultPreparer(ctx, lastResults)
	if err != nil {
		return result, autorest.NewErrorWithError(err, "vmssclient", "listNextResults", nil, "Failure preparing next results request")
	}
	if req == nil {
		return
	}

	resp, rerr := c.armClient.Send(ctx, req)
	defer c.armClient.CloseResponse(ctx, resp)
	if rerr != nil {
		result.Response = autorest.Response{Response: resp}
		return result, autorest.NewErrorWithError(rerr.Error(), "vmssclient", "listNextResults", resp, "Failure sending next results request")
	}

	result, err = c.listResponder(resp)
	if err != nil {
		err = autorest.NewErrorWithError(err, "vmssclient", "listNextResults", resp, "Failure responding to next results request")
	}

	return
}

// VirtualMachineScaleSetListResultPage contains a page of VirtualMachineScaleSet values.
type VirtualMachineScaleSetListResultPage struct {
	fn     func(context.Context, compute.VirtualMachineScaleSetListResult) (compute.VirtualMachineScaleSetListResult, error)
	vmsslr compute.VirtualMachineScaleSetListResult
}

// NextWithContext advances to the next page of values.  If there was an error making
// the request the page does not advance and the error is returned.
func (page *VirtualMachineScaleSetListResultPage) NextWithContext(ctx context.Context) (err error) {
	next, err := page.fn(ctx, page.vmsslr)
	if err != nil {
		return err
	}
	page.vmsslr = next
	return nil
}

// Next advances to the next page of values.  If there was an error making
// the request the page does not advance and the error is returned.
// Deprecated: Use NextWithContext() instead.
func (page *VirtualMachineScaleSetListResultPage) Next() error {
	return page.NextWithContext(context.Background())
}

// NotDone returns true if the page enumeration should be started or is not yet complete.
func (page VirtualMachineScaleSetListResultPage) NotDone() bool {
	return !page.vmsslr.IsEmpty()
}

// Response returns the raw server response from the last page request.
func (page VirtualMachineScaleSetListResultPage) Response() compute.VirtualMachineScaleSetListResult {
	return page.vmsslr
}

// Values returns the slice of values for the current page or nil if there are no values.
func (page VirtualMachineScaleSetListResultPage) Values() []compute.VirtualMachineScaleSet {
	if page.vmsslr.IsEmpty() {
		return nil
	}
	return *page.vmsslr.Value
}

// DeleteInstances deletes the instances for a VirtualMachineScaleSet.
func (c *Client) DeleteInstances(ctx context.Context, resourceGroupName string, vmScaleSetName string, vmInstanceIDs compute.VirtualMachineScaleSetVMInstanceRequiredIDs) *retry.Error {
	mc := metrics.NewMetricContext("vmss", "delete_instances", resourceGroupName, c.subscriptionID, "")

	// Report errors if the client is rate limited.
	if !c.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return retry.GetRateLimitError(true, "VMSSDeleteInstances")
	}

	// Report errors if the client is throttled.
	if c.RetryAfterWriter.After(time.Now()) {
		mc.ThrottledCount()
		rerr := retry.GetThrottlingError("VMSSDeleteInstances", "client throttled", c.RetryAfterWriter)
		return rerr
	}

	rerr := c.deleteVMSSInstances(ctx, resourceGroupName, vmScaleSetName, vmInstanceIDs)
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

// DeleteInstancesAsync sends the delete request to ARM client and DOEST NOT wait on the future
func (c *Client) DeleteInstancesAsync(ctx context.Context, resourceGroupName string, vmScaleSetName string, vmInstanceIDs compute.VirtualMachineScaleSetVMInstanceRequiredIDs) (*azure.Future, *retry.Error) {
	mc := metrics.NewMetricContext("vmss", "delete_instances_async", resourceGroupName, c.subscriptionID, "")

	// Report errors if the client is rate limited.
	if !c.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return nil, retry.GetRateLimitError(true, "VMSSDeleteInstancesAsync")
	}

	// Report errors if the client is throttled.
	if c.RetryAfterWriter.After(time.Now()) {
		mc.ThrottledCount()
		rerr := retry.GetThrottlingError("VMSSDeleteInstancesAsync", "client throttled", c.RetryAfterWriter)
		return nil, rerr
	}

	resourceID := armclient.GetResourceID(
		c.subscriptionID,
		resourceGroupName,
		"Microsoft.Compute/virtualMachineScaleSets",
		vmScaleSetName,
	)

	response, rerr := c.armClient.PostResource(ctx, resourceID, "delete", vmInstanceIDs)
	defer c.armClient.CloseResponse(ctx, response)

	if rerr != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "vmss.deletevms.request", resourceID, rerr.Error())
		return nil, rerr
	}

	err := autorest.Respond(response, azure.WithErrorUnlessStatusCode(http.StatusOK, http.StatusAccepted))
	if err != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "vmss.deletevms.respond", resourceID, rerr.Error())
		return nil, retry.GetError(response, err)
	}

	future, err := azure.NewFutureFromResponse(response)
	if err != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "vmss.deletevms.future", resourceID, rerr.Error())
		return nil, retry.NewError(false, err)
	}

	return &future, nil
}

// deleteVMSSInstances deletes the instances for a VirtualMachineScaleSet.
func (c *Client) deleteVMSSInstances(ctx context.Context, resourceGroupName string, vmScaleSetName string, vmInstanceIDs compute.VirtualMachineScaleSetVMInstanceRequiredIDs) *retry.Error {
	resourceID := armclient.GetResourceID(
		c.subscriptionID,
		resourceGroupName,
		"Microsoft.Compute/virtualMachineScaleSets",
		vmScaleSetName,
	)
	response, rerr := c.armClient.PostResource(ctx, resourceID, "delete", vmInstanceIDs)
	defer c.armClient.CloseResponse(ctx, response)
	if rerr != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "vmss.deletevms.request", resourceID, rerr.Error())
		return rerr
	}

	err := autorest.Respond(response, azure.WithErrorUnlessStatusCode(http.StatusOK, http.StatusAccepted))
	if err != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "vmss.deletevms.respond", resourceID, rerr.Error())
		return retry.GetError(response, err)
	}

	future, err := azure.NewFutureFromResponse(response)
	if err != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "vmss.deletevms.future", resourceID, rerr.Error())
		return retry.NewError(false, err)
	}

	if err := c.armClient.WaitForAsyncOperationCompletion(ctx, &future, "vmssclient.DeleteInstances"); err != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "vmss.deletevms.wait", resourceID, rerr.Error())
		return retry.NewError(false, err)
	}

	return nil
}
