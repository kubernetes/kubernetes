//go:build !providerless
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

package vmssvmclient

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-12-01/compute"
	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/azure"
	"github.com/Azure/go-autorest/autorest/to"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
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

// New creates a new vmssVM client with ratelimiting.
func New(config *azclients.ClientConfig) *Client {
	baseURI := config.ResourceManagerEndpoint
	authorizer := config.Authorizer
	armClient := armclient.New(authorizer, baseURI, config.UserAgent, APIVersion, config.Location, config.Backoff)
	rateLimiterReader, rateLimiterWriter := azclients.NewRateLimiter(config.RateLimitConfig)

	klog.V(2).Infof("Azure vmssVM client (read ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPS,
		config.RateLimitConfig.CloudProviderRateLimitBucket)
	klog.V(2).Infof("Azure vmssVM client (write ops) using rate limit config: QPS=%g, bucket=%d",
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

// Get gets a VirtualMachineScaleSetVM.
func (c *Client) Get(ctx context.Context, resourceGroupName string, VMScaleSetName string, instanceID string, expand compute.InstanceViewTypes) (compute.VirtualMachineScaleSetVM, *retry.Error) {
	mc := metrics.NewMetricContext("vmssvm", "get", resourceGroupName, c.subscriptionID, "")

	// Report errors if the client is rate limited.
	if !c.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		return compute.VirtualMachineScaleSetVM{}, retry.GetRateLimitError(false, "VMSSVMGet")
	}

	// Report errors if the client is throttled.
	if c.RetryAfterReader.After(time.Now()) {
		mc.ThrottledCount()
		rerr := retry.GetThrottlingError("VMSSVMGet", "client throttled", c.RetryAfterReader)
		return compute.VirtualMachineScaleSetVM{}, rerr
	}

	result, rerr := c.getVMSSVM(ctx, resourceGroupName, VMScaleSetName, instanceID, expand)
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

// getVMSSVM gets a VirtualMachineScaleSetVM.
func (c *Client) getVMSSVM(ctx context.Context, resourceGroupName string, VMScaleSetName string, instanceID string, expand compute.InstanceViewTypes) (compute.VirtualMachineScaleSetVM, *retry.Error) {
	resourceID := armclient.GetChildResourceID(
		c.subscriptionID,
		resourceGroupName,
		"Microsoft.Compute/virtualMachineScaleSets",
		VMScaleSetName,
		"virtualMachines",
		instanceID,
	)
	result := compute.VirtualMachineScaleSetVM{}

	response, rerr := c.armClient.GetResource(ctx, resourceID, string(expand))
	defer c.armClient.CloseResponse(ctx, response)
	if rerr != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "vmssvm.get.request", resourceID, rerr.Error())
		return result, rerr
	}

	err := autorest.Respond(
		response,
		azure.WithErrorUnlessStatusCode(http.StatusOK),
		autorest.ByUnmarshallingJSON(&result))
	if err != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "vmssvm.get.respond", resourceID, err)
		return result, retry.GetError(response, err)
	}

	result.Response = autorest.Response{Response: response}
	return result, nil
}

// List gets a list of VirtualMachineScaleSetVMs in the virtualMachineScaleSet.
func (c *Client) List(ctx context.Context, resourceGroupName string, virtualMachineScaleSetName string, expand string) ([]compute.VirtualMachineScaleSetVM, *retry.Error) {
	mc := metrics.NewMetricContext("vmssvm", "list", resourceGroupName, c.subscriptionID, "")

	// Report errors if the client is rate limited.
	if !c.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		return nil, retry.GetRateLimitError(false, "VMSSVMList")
	}

	// Report errors if the client is throttled.
	if c.RetryAfterReader.After(time.Now()) {
		mc.ThrottledCount()
		rerr := retry.GetThrottlingError("VMSSVMList", "client throttled", c.RetryAfterReader)
		return nil, rerr
	}

	result, rerr := c.listVMSSVM(ctx, resourceGroupName, virtualMachineScaleSetName, expand)
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

// listVMSSVM gets a list of VirtualMachineScaleSetVMs in the virtualMachineScaleSet.
func (c *Client) listVMSSVM(ctx context.Context, resourceGroupName string, virtualMachineScaleSetName string, expand string) ([]compute.VirtualMachineScaleSetVM, *retry.Error) {
	resourceID := fmt.Sprintf("/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Compute/virtualMachineScaleSets/%s/virtualMachines",
		autorest.Encode("path", c.subscriptionID),
		autorest.Encode("path", resourceGroupName),
		autorest.Encode("path", virtualMachineScaleSetName),
	)

	result := make([]compute.VirtualMachineScaleSetVM, 0)
	page := &VirtualMachineScaleSetVMListResultPage{}
	page.fn = c.listNextResults

	resp, rerr := c.armClient.GetResource(ctx, resourceID, expand)
	defer c.armClient.CloseResponse(ctx, resp)
	if rerr != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "vmssvm.list.request", resourceID, rerr.Error())
		return result, rerr
	}

	var err error
	page.vmssvlr, err = c.listResponder(resp)
	if err != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "vmssvm.list.respond", resourceID, err)
		return result, retry.GetError(resp, err)
	}

	for {
		result = append(result, page.Values()...)

		// Abort the loop when there's no nextLink in the response.
		if to.String(page.Response().NextLink) == "" {
			break
		}

		if err = page.NextWithContext(ctx); err != nil {
			klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "vmssvm.list.next", resourceID, err)
			return result, retry.GetError(page.Response().Response.Response, err)
		}
	}

	return result, nil
}

// Update updates a VirtualMachineScaleSetVM.
func (c *Client) Update(ctx context.Context, resourceGroupName string, VMScaleSetName string, instanceID string, parameters compute.VirtualMachineScaleSetVM, source string) *retry.Error {
	mc := metrics.NewMetricContext("vmssvm", "update", resourceGroupName, c.subscriptionID, source)

	// Report errors if the client is rate limited.
	if !c.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return retry.GetRateLimitError(true, "VMSSVMUpdate")
	}

	// Report errors if the client is throttled.
	if c.RetryAfterWriter.After(time.Now()) {
		mc.ThrottledCount()
		rerr := retry.GetThrottlingError("VMSSVMUpdate", "client throttled", c.RetryAfterWriter)
		return rerr
	}

	rerr := c.updateVMSSVM(ctx, resourceGroupName, VMScaleSetName, instanceID, parameters)
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

// updateVMSSVM updates a VirtualMachineScaleSetVM.
func (c *Client) updateVMSSVM(ctx context.Context, resourceGroupName string, VMScaleSetName string, instanceID string, parameters compute.VirtualMachineScaleSetVM) *retry.Error {
	resourceID := armclient.GetChildResourceID(
		c.subscriptionID,
		resourceGroupName,
		"Microsoft.Compute/virtualMachineScaleSets",
		VMScaleSetName,
		"virtualMachines",
		instanceID,
	)

	response, rerr := c.armClient.PutResource(ctx, resourceID, parameters)
	defer c.armClient.CloseResponse(ctx, response)
	if rerr != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "vmssvm.put.request", resourceID, rerr.Error())
		return rerr
	}

	if response != nil && response.StatusCode != http.StatusNoContent {
		_, rerr = c.updateResponder(response)
		if rerr != nil {
			klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "vmssvm.put.respond", resourceID, rerr.Error())
			return rerr
		}
	}

	return nil
}

func (c *Client) updateResponder(resp *http.Response) (*compute.VirtualMachineScaleSetVM, *retry.Error) {
	result := &compute.VirtualMachineScaleSetVM{}
	err := autorest.Respond(
		resp,
		azure.WithErrorUnlessStatusCode(http.StatusOK, http.StatusCreated),
		autorest.ByUnmarshallingJSON(&result))
	result.Response = autorest.Response{Response: resp}
	return result, retry.GetError(resp, err)
}

func (c *Client) listResponder(resp *http.Response) (result compute.VirtualMachineScaleSetVMListResult, err error) {
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
func (c *Client) virtualMachineScaleSetVMListResultPreparer(ctx context.Context, vmssvmlr compute.VirtualMachineScaleSetVMListResult) (*http.Request, error) {
	if vmssvmlr.NextLink == nil || len(to.String(vmssvmlr.NextLink)) < 1 {
		return nil, nil
	}

	decorators := []autorest.PrepareDecorator{
		autorest.WithBaseURL(to.String(vmssvmlr.NextLink)),
	}
	return c.armClient.PrepareGetRequest(ctx, decorators...)
}

// listNextResults retrieves the next set of results, if any.
func (c *Client) listNextResults(ctx context.Context, lastResults compute.VirtualMachineScaleSetVMListResult) (result compute.VirtualMachineScaleSetVMListResult, err error) {
	req, err := c.virtualMachineScaleSetVMListResultPreparer(ctx, lastResults)
	if err != nil {
		return result, autorest.NewErrorWithError(err, "vmssvmclient", "listNextResults", nil, "Failure preparing next results request")
	}
	if req == nil {
		return
	}

	resp, rerr := c.armClient.Send(ctx, req)
	defer c.armClient.CloseResponse(ctx, resp)
	if rerr != nil {
		result.Response = autorest.Response{Response: resp}
		return result, autorest.NewErrorWithError(rerr.Error(), "vmssvmclient", "listNextResults", resp, "Failure sending next results request")
	}

	result, err = c.listResponder(resp)
	if err != nil {
		err = autorest.NewErrorWithError(err, "vmssvmclient", "listNextResults", resp, "Failure responding to next results request")
	}

	return
}

// VirtualMachineScaleSetVMListResultPage contains a page of VirtualMachineScaleSetVM values.
type VirtualMachineScaleSetVMListResultPage struct {
	fn      func(context.Context, compute.VirtualMachineScaleSetVMListResult) (compute.VirtualMachineScaleSetVMListResult, error)
	vmssvlr compute.VirtualMachineScaleSetVMListResult
}

// NextWithContext advances to the next page of values.  If there was an error making
// the request the page does not advance and the error is returned.
func (page *VirtualMachineScaleSetVMListResultPage) NextWithContext(ctx context.Context) (err error) {
	next, err := page.fn(ctx, page.vmssvlr)
	if err != nil {
		return err
	}
	page.vmssvlr = next
	return nil
}

// Next advances to the next page of values.  If there was an error making
// the request the page does not advance and the error is returned.
// Deprecated: Use NextWithContext() instead.
func (page *VirtualMachineScaleSetVMListResultPage) Next() error {
	return page.NextWithContext(context.Background())
}

// NotDone returns true if the page enumeration should be started or is not yet complete.
func (page VirtualMachineScaleSetVMListResultPage) NotDone() bool {
	return !page.vmssvlr.IsEmpty()
}

// Response returns the raw server response from the last page request.
func (page VirtualMachineScaleSetVMListResultPage) Response() compute.VirtualMachineScaleSetVMListResult {
	return page.vmssvlr
}

// Values returns the slice of values for the current page or nil if there are no values.
func (page VirtualMachineScaleSetVMListResultPage) Values() []compute.VirtualMachineScaleSetVM {
	if page.vmssvlr.IsEmpty() {
		return nil
	}
	return *page.vmssvlr.Value
}

// UpdateVMs updates a list of VirtualMachineScaleSetVM from map[instanceID]compute.VirtualMachineScaleSetVM.
func (c *Client) UpdateVMs(ctx context.Context, resourceGroupName string, VMScaleSetName string, instances map[string]compute.VirtualMachineScaleSetVM, source string) *retry.Error {
	mc := metrics.NewMetricContext("vmssvm", "update_vms", resourceGroupName, c.subscriptionID, source)

	// Report errors if the client is rate limited.
	if !c.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return retry.GetRateLimitError(true, "VMSSVMUpdateVMs")
	}

	// Report errors if the client is throttled.
	if c.RetryAfterWriter.After(time.Now()) {
		mc.ThrottledCount()
		rerr := retry.GetThrottlingError("VMSSVMUpdateVMs", "client throttled", c.RetryAfterWriter)
		return rerr
	}

	rerr := c.updateVMSSVMs(ctx, resourceGroupName, VMScaleSetName, instances)
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

// updateVMSSVMs updates a list of VirtualMachineScaleSetVM from map[instanceID]compute.VirtualMachineScaleSetVM.
func (c *Client) updateVMSSVMs(ctx context.Context, resourceGroupName string, VMScaleSetName string, instances map[string]compute.VirtualMachineScaleSetVM) *retry.Error {
	resources := make(map[string]interface{})
	for instanceID, parameter := range instances {
		resourceID := armclient.GetChildResourceID(
			c.subscriptionID,
			resourceGroupName,
			"Microsoft.Compute/virtualMachineScaleSets",
			VMScaleSetName,
			"virtualMachines",
			instanceID,
		)
		resources[resourceID] = parameter
	}

	responses := c.armClient.PutResources(ctx, resources)
	errors := make([]*retry.Error, 0)
	for resourceID, resp := range responses {
		if resp == nil {
			continue
		}

		defer c.armClient.CloseResponse(ctx, resp.Response)
		if resp.Error != nil {
			klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "vmssvm.put.request", resourceID, resp.Error.Error())
			errors = append(errors, resp.Error)
			continue
		}

		if resp.Response != nil && resp.Response.StatusCode != http.StatusNoContent {
			_, rerr := c.updateResponder(resp.Response)
			if rerr != nil {
				klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "vmssvm.put.respond", resourceID, rerr.Error())
				errors = append(errors, rerr)
			}
		}
	}

	// Aggregate errors.
	if len(errors) > 0 {
		rerr := &retry.Error{}
		errs := make([]error, 0)
		for _, err := range errors {
			if err.IsThrottled() && err.RetryAfter.After(err.RetryAfter) {
				rerr.RetryAfter = err.RetryAfter
			}
			errs = append(errs, err.Error())
		}
		rerr.RawError = utilerrors.Flatten(utilerrors.NewAggregate(errs))
		return rerr
	}

	return nil
}
