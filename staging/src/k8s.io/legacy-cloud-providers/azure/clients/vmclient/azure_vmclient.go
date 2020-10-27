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

package vmclient

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

// Client implements VirtualMachine client Interface.
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

// New creates a new VirtualMachine client with ratelimiting.
func New(config *azclients.ClientConfig) *Client {
	baseURI := config.ResourceManagerEndpoint
	authorizer := config.Authorizer
	armClient := armclient.New(authorizer, baseURI, config.UserAgent, APIVersion, config.Location, config.Backoff)
	rateLimiterReader, rateLimiterWriter := azclients.NewRateLimiter(config.RateLimitConfig)

	klog.V(2).Infof("Azure VirtualMachine client (read ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPS,
		config.RateLimitConfig.CloudProviderRateLimitBucket)
	klog.V(2).Infof("Azure VirtualMachine client (write ops) using rate limit config: QPS=%g, bucket=%d",
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

// Get gets a VirtualMachine.
func (c *Client) Get(ctx context.Context, resourceGroupName string, VMName string, expand compute.InstanceViewTypes) (compute.VirtualMachine, *retry.Error) {
	mc := metrics.NewMetricContext("vm", "get", resourceGroupName, c.subscriptionID, "")

	// Report errors if the client is rate limited.
	if !c.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		return compute.VirtualMachine{}, retry.GetRateLimitError(false, "VMGet")
	}

	// Report errors if the client is throttled.
	if c.RetryAfterReader.After(time.Now()) {
		mc.ThrottledCount()
		rerr := retry.GetThrottlingError("VMGet", "client throttled", c.RetryAfterReader)
		return compute.VirtualMachine{}, rerr
	}

	result, rerr := c.getVM(ctx, resourceGroupName, VMName, expand)
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

// getVM gets a VirtualMachine.
func (c *Client) getVM(ctx context.Context, resourceGroupName string, VMName string, expand compute.InstanceViewTypes) (compute.VirtualMachine, *retry.Error) {
	resourceID := armclient.GetResourceID(
		c.subscriptionID,
		resourceGroupName,
		"Microsoft.Compute/virtualMachines",
		VMName,
	)
	result := compute.VirtualMachine{}

	response, rerr := c.armClient.GetResource(ctx, resourceID, string(expand))
	defer c.armClient.CloseResponse(ctx, response)
	if rerr != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "vm.get.request", resourceID, rerr.Error())
		return result, rerr
	}

	err := autorest.Respond(
		response,
		azure.WithErrorUnlessStatusCode(http.StatusOK),
		autorest.ByUnmarshallingJSON(&result))
	if err != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "vm.get.respond", resourceID, err)
		return result, retry.GetError(response, err)
	}

	result.Response = autorest.Response{Response: response}
	return result, nil
}

// List gets a list of VirtualMachine in the resourceGroupName.
func (c *Client) List(ctx context.Context, resourceGroupName string) ([]compute.VirtualMachine, *retry.Error) {
	mc := metrics.NewMetricContext("vm", "list", resourceGroupName, c.subscriptionID, "")

	// Report errors if the client is rate limited.
	if !c.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		return nil, retry.GetRateLimitError(false, "VMList")
	}

	// Report errors if the client is throttled.
	if c.RetryAfterReader.After(time.Now()) {
		mc.ThrottledCount()
		rerr := retry.GetThrottlingError("VMList", "client throttled", c.RetryAfterReader)
		return nil, rerr
	}

	result, rerr := c.listVM(ctx, resourceGroupName)
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

// listVM gets a list of VirtualMachines in the resourceGroupName.
func (c *Client) listVM(ctx context.Context, resourceGroupName string) ([]compute.VirtualMachine, *retry.Error) {
	resourceID := fmt.Sprintf("/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Compute/virtualMachines",
		autorest.Encode("path", c.subscriptionID),
		autorest.Encode("path", resourceGroupName),
	)

	result := make([]compute.VirtualMachine, 0)
	page := &VirtualMachineListResultPage{}
	page.fn = c.listNextResults

	resp, rerr := c.armClient.GetResource(ctx, resourceID, "")
	defer c.armClient.CloseResponse(ctx, resp)
	if rerr != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "vm.list.request", resourceID, rerr.Error())
		return result, rerr
	}

	var err error
	page.vmlr, err = c.listResponder(resp)
	if err != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "vm.list.respond", resourceID, err)
		return result, retry.GetError(resp, err)
	}

	for page.NotDone() {
		result = append(result, *page.Response().Value...)
		if err = page.NextWithContext(ctx); err != nil {
			klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "vm.list.next", resourceID, err)
			return result, retry.GetError(page.Response().Response.Response, err)
		}
	}

	return result, nil
}

// Update updates a VirtualMachine.
func (c *Client) Update(ctx context.Context, resourceGroupName string, VMName string, parameters compute.VirtualMachineUpdate, source string) *retry.Error {
	mc := metrics.NewMetricContext("vm", "update", resourceGroupName, c.subscriptionID, source)

	// Report errors if the client is rate limited.
	if !c.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return retry.GetRateLimitError(true, "VMUpdate")
	}

	// Report errors if the client is throttled.
	if c.RetryAfterWriter.After(time.Now()) {
		mc.ThrottledCount()
		rerr := retry.GetThrottlingError("VMUpdate", "client throttled", c.RetryAfterWriter)
		return rerr
	}

	rerr := c.updateVM(ctx, resourceGroupName, VMName, parameters, source)
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

// updateVM updates a VirtualMachine.
func (c *Client) updateVM(ctx context.Context, resourceGroupName string, VMName string, parameters compute.VirtualMachineUpdate, source string) *retry.Error {
	resourceID := armclient.GetResourceID(
		c.subscriptionID,
		resourceGroupName,
		"Microsoft.Compute/virtualMachines",
		VMName,
	)

	response, rerr := c.armClient.PatchResource(ctx, resourceID, parameters)
	defer c.armClient.CloseResponse(ctx, response)
	if rerr != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "vm.put.request", resourceID, rerr.Error())
		return rerr
	}

	if response != nil && response.StatusCode != http.StatusNoContent {
		_, rerr = c.updateResponder(response)
		if rerr != nil {
			klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "vm.put.respond", resourceID, rerr.Error())
			return rerr
		}
	}

	return nil
}

func (c *Client) updateResponder(resp *http.Response) (*compute.VirtualMachine, *retry.Error) {
	result := &compute.VirtualMachine{}
	err := autorest.Respond(
		resp,
		azure.WithErrorUnlessStatusCode(http.StatusOK, http.StatusCreated),
		autorest.ByUnmarshallingJSON(&result),
		autorest.ByClosing())
	result.Response = autorest.Response{Response: resp}
	return result, retry.GetError(resp, err)
}

func (c *Client) listResponder(resp *http.Response) (result compute.VirtualMachineListResult, err error) {
	err = autorest.Respond(
		resp,
		autorest.ByIgnoring(),
		azure.WithErrorUnlessStatusCode(http.StatusOK),
		autorest.ByUnmarshallingJSON(&result),
		autorest.ByClosing(),
	)
	result.Response = autorest.Response{Response: resp}
	return
}

// vmListResultPreparer prepares a request to retrieve the next set of results.
// It returns nil if no more results exist.
func (c *Client) vmListResultPreparer(ctx context.Context, vmlr compute.VirtualMachineListResult) (*http.Request, error) {
	if vmlr.NextLink == nil || len(to.String(vmlr.NextLink)) < 1 {
		return nil, nil
	}

	decorators := []autorest.PrepareDecorator{
		autorest.WithBaseURL(to.String(vmlr.NextLink)),
	}
	return c.armClient.PrepareGetRequest(ctx, decorators...)
}

// listNextResults retrieves the next set of results, if any.
func (c *Client) listNextResults(ctx context.Context, lastResults compute.VirtualMachineListResult) (result compute.VirtualMachineListResult, err error) {
	req, err := c.vmListResultPreparer(ctx, lastResults)
	if err != nil {
		return result, autorest.NewErrorWithError(err, "vmclient", "listNextResults", nil, "Failure preparing next results request")
	}
	if req == nil {
		return
	}

	resp, rerr := c.armClient.Send(ctx, req)
	defer c.armClient.CloseResponse(ctx, resp)
	if rerr != nil {
		result.Response = autorest.Response{Response: resp}
		return result, autorest.NewErrorWithError(rerr.Error(), "vmclient", "listNextResults", resp, "Failure sending next results request")
	}

	result, err = c.listResponder(resp)
	if err != nil {
		err = autorest.NewErrorWithError(err, "vmclient", "listNextResults", resp, "Failure responding to next results request")
	}

	return
}

// VirtualMachineListResultPage contains a page of VirtualMachine values.
type VirtualMachineListResultPage struct {
	fn   func(context.Context, compute.VirtualMachineListResult) (compute.VirtualMachineListResult, error)
	vmlr compute.VirtualMachineListResult
}

// NextWithContext advances to the next page of values.  If there was an error making
// the request the page does not advance and the error is returned.
func (page *VirtualMachineListResultPage) NextWithContext(ctx context.Context) (err error) {
	next, err := page.fn(ctx, page.vmlr)
	if err != nil {
		return err
	}
	page.vmlr = next
	return nil
}

// Next advances to the next page of values.  If there was an error making
// the request the page does not advance and the error is returned.
// Deprecated: Use NextWithContext() instead.
func (page *VirtualMachineListResultPage) Next() error {
	return page.NextWithContext(context.Background())
}

// NotDone returns true if the page enumeration should be started or is not yet complete.
func (page VirtualMachineListResultPage) NotDone() bool {
	return !page.vmlr.IsEmpty()
}

// Response returns the raw server response from the last page request.
func (page VirtualMachineListResultPage) Response() compute.VirtualMachineListResult {
	return page.vmlr
}

// Values returns the slice of values for the current page or nil if there are no values.
func (page VirtualMachineListResultPage) Values() []compute.VirtualMachine {
	if page.vmlr.IsEmpty() {
		return nil
	}
	return *page.vmlr.Value
}

// CreateOrUpdate creates or updates a VirtualMachine.
func (c *Client) CreateOrUpdate(ctx context.Context, resourceGroupName string, VMName string, parameters compute.VirtualMachine, source string) *retry.Error {
	mc := metrics.NewMetricContext("vm", "create_or_update", resourceGroupName, c.subscriptionID, source)

	// Report errors if the client is rate limited.
	if !c.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return retry.GetRateLimitError(true, "VMCreateOrUpdate")
	}

	// Report errors if the client is throttled.
	if c.RetryAfterWriter.After(time.Now()) {
		mc.ThrottledCount()
		rerr := retry.GetThrottlingError("VMCreateOrUpdate", "client throttled", c.RetryAfterWriter)
		return rerr
	}

	rerr := c.createOrUpdateVM(ctx, resourceGroupName, VMName, parameters, source)
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

// createOrUpdateVM creates or updates a VirtualMachine.
func (c *Client) createOrUpdateVM(ctx context.Context, resourceGroupName string, VMName string, parameters compute.VirtualMachine, source string) *retry.Error {
	resourceID := armclient.GetResourceID(
		c.subscriptionID,
		resourceGroupName,
		"Microsoft.Compute/virtualMachines",
		VMName,
	)

	response, rerr := c.armClient.PutResource(ctx, resourceID, parameters)
	defer c.armClient.CloseResponse(ctx, response)
	if rerr != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "vm.put.request", resourceID, rerr.Error())
		return rerr
	}

	if response != nil && response.StatusCode != http.StatusNoContent {
		_, rerr = c.createOrUpdateResponder(response)
		if rerr != nil {
			klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "vm.put.respond", resourceID, rerr.Error())
			return rerr
		}
	}

	return nil
}

func (c *Client) createOrUpdateResponder(resp *http.Response) (*compute.VirtualMachine, *retry.Error) {
	result := &compute.VirtualMachine{}
	err := autorest.Respond(
		resp,
		azure.WithErrorUnlessStatusCode(http.StatusOK, http.StatusCreated),
		autorest.ByUnmarshallingJSON(&result),
		autorest.ByClosing())
	result.Response = autorest.Response{Response: resp}
	return result, retry.GetError(resp, err)
}

// Delete deletes a VirtualMachine.
func (c *Client) Delete(ctx context.Context, resourceGroupName string, VMName string) *retry.Error {
	mc := metrics.NewMetricContext("vm", "delete", resourceGroupName, c.subscriptionID, "")

	// Report errors if the client is rate limited.
	if !c.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return retry.GetRateLimitError(true, "VMDelete")
	}

	// Report errors if the client is throttled.
	if c.RetryAfterWriter.After(time.Now()) {
		mc.ThrottledCount()
		rerr := retry.GetThrottlingError("VMDelete", "client throttled", c.RetryAfterWriter)
		return rerr
	}

	rerr := c.deleteVM(ctx, resourceGroupName, VMName)
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

// deleteVM deletes a VirtualMachine.
func (c *Client) deleteVM(ctx context.Context, resourceGroupName string, VMName string) *retry.Error {
	resourceID := armclient.GetResourceID(
		c.subscriptionID,
		resourceGroupName,
		"Microsoft.Compute/virtualMachines",
		VMName,
	)

	return c.armClient.DeleteResource(ctx, resourceID, "")
}
