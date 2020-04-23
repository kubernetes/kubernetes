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

package armclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/azure"

	"k8s.io/client-go/pkg/version"
	"k8s.io/klog"
	"k8s.io/legacy-cloud-providers/azure/retry"
)

var _ Interface = &Client{}

// Client implements ARM client Interface.
type Client struct {
	client  autorest.Client
	backoff *retry.Backoff

	baseURI      string
	apiVersion   string
	clientRegion string
}

// New creates a ARM client
func New(authorizer autorest.Authorizer, baseURI, userAgent, apiVersion, clientRegion string, clientBackoff *retry.Backoff) *Client {
	restClient := autorest.NewClientWithUserAgent(userAgent)
	restClient.PollingDelay = 5 * time.Second
	restClient.RetryAttempts = 3
	restClient.RetryDuration = time.Second * 1
	restClient.Authorizer = authorizer

	if userAgent == "" {
		restClient.UserAgent = GetUserAgent(restClient)
	}

	backoff := clientBackoff
	if backoff == nil {
		// 1 steps means no retry.
		backoff = &retry.Backoff{
			Steps: 1,
		}
	}

	return &Client{
		client:       restClient,
		baseURI:      baseURI,
		backoff:      backoff,
		apiVersion:   apiVersion,
		clientRegion: NormalizeAzureRegion(clientRegion),
	}
}

// GetUserAgent gets the autorest client with a user agent that
// includes "kubernetes" and the full kubernetes git version string
// example:
// Azure-SDK-for-Go/7.0.1 arm-network/2016-09-01; kubernetes-cloudprovider/v1.17.0;
func GetUserAgent(client autorest.Client) string {
	k8sVersion := version.Get().GitVersion
	return fmt.Sprintf("%s; kubernetes-cloudprovider/%s", client.UserAgent, k8sVersion)
}

// NormalizeAzureRegion returns a normalized Azure region with white spaces removed and converted to lower case
func NormalizeAzureRegion(name string) string {
	region := ""
	for _, runeValue := range name {
		if !unicode.IsSpace(runeValue) {
			region += string(runeValue)
		}
	}
	return strings.ToLower(region)
}

// sendRequest sends a http request to ARM service.
// Although Azure SDK supports retries per https://github.com/azure/azure-sdk-for-go#request-retry-policy, we
// disable it since we want to fully control the retry policies.
func (c *Client) sendRequest(ctx context.Context, request *http.Request) (*http.Response, *retry.Error) {
	sendBackoff := *c.backoff
	response, err := autorest.SendWithSender(
		c.client,
		request,
		retry.DoExponentialBackoffRetry(&sendBackoff),
	)
	return response, retry.GetError(response, err)
}

// Send sends a http request to ARM service with possible retry to regional ARM endpoint.
func (c *Client) Send(ctx context.Context, request *http.Request) (*http.Response, *retry.Error) {
	response, rerr := c.sendRequest(ctx, request)
	if rerr != nil {
		return response, rerr
	}

	if response.StatusCode != http.StatusNotFound || c.clientRegion == "" {
		return response, rerr
	}

	bodyBytes, _ := ioutil.ReadAll(response.Body)
	defer func() {
		response.Body = ioutil.NopCloser(bytes.NewBuffer(bodyBytes))
	}()

	bodyString := string(bodyBytes)
	klog.V(5).Infof("Send.sendRequest original error message: %s", bodyString)

	// Hack: retry the regional ARM endpoint in case of ARM traffic split and arm resource group replication is too slow
	var body map[string]interface{}
	if e := json.Unmarshal(bodyBytes, &body); e != nil {
		klog.V(5).Infof("Send.sendRequest: error in parsing response body string: %s, Skip retrying regional host", e)
		return response, rerr
	}

	if err, ok := body["error"].(map[string]interface{}); !ok ||
		err["code"] == nil ||
		!strings.EqualFold(err["code"].(string), "ResourceGroupNotFound") {
		klog.V(5).Infof("Send.sendRequest: response body does not contain ResourceGroupNotFound error code. Skip retrying regional host")
		return response, rerr
	}

	currentHost := request.URL.Host
	if request.Host != "" {
		currentHost = request.Host
	}

	if strings.HasPrefix(strings.ToLower(currentHost), c.clientRegion) {
		klog.V(5).Infof("Send.sendRequest: current host %s is regional host. Skip retrying regional host.", currentHost)
		return response, rerr
	}

	request.Host = fmt.Sprintf("%s.%s", c.clientRegion, strings.ToLower(currentHost))
	klog.V(5).Infof("Send.sendRegionalRequest on ResourceGroupNotFound error. Retrying regional host: %s", request.Host)
	regionalResponse, regionalError := c.sendRequest(ctx, request)

	// only use the result if the regional request actually goes through and returns 2xx status code, for two reasons:
	// 1. the retry on regional ARM host approach is a hack.
	// 2. the concatted regional uri could be wrong as the rule is not officially declared by ARM.
	if regionalResponse == nil || regionalResponse.StatusCode > 299 {
		regionalErrStr := ""
		if regionalError != nil {
			regionalErrStr = regionalError.Error().Error()
		}

		klog.V(5).Infof("Send.sendRegionalRequest failed to get response from regional host, error: '%s'. Ignoring the result.", regionalErrStr)
		return response, rerr
	}

	return regionalResponse, regionalError
}

// PreparePutRequest prepares put request
func (c *Client) PreparePutRequest(ctx context.Context, decorators ...autorest.PrepareDecorator) (*http.Request, error) {
	decorators = append(
		[]autorest.PrepareDecorator{
			autorest.AsContentType("application/json; charset=utf-8"),
			autorest.AsPut(),
			autorest.WithBaseURL(c.baseURI)},
		decorators...)
	return c.prepareRequest(ctx, decorators...)
}

// PreparePatchRequest prepares patch request
func (c *Client) PreparePatchRequest(ctx context.Context, decorators ...autorest.PrepareDecorator) (*http.Request, error) {
	decorators = append(
		[]autorest.PrepareDecorator{
			autorest.AsContentType("application/json; charset=utf-8"),
			autorest.AsPatch(),
			autorest.WithBaseURL(c.baseURI)},
		decorators...)
	return c.prepareRequest(ctx, decorators...)
}

// PreparePostRequest prepares post request
func (c *Client) PreparePostRequest(ctx context.Context, decorators ...autorest.PrepareDecorator) (*http.Request, error) {
	decorators = append(
		[]autorest.PrepareDecorator{
			autorest.AsContentType("application/json; charset=utf-8"),
			autorest.AsPost(),
			autorest.WithBaseURL(c.baseURI)},
		decorators...)
	return c.prepareRequest(ctx, decorators...)
}

// PrepareGetRequest prepares get request
func (c *Client) PrepareGetRequest(ctx context.Context, decorators ...autorest.PrepareDecorator) (*http.Request, error) {
	decorators = append(
		[]autorest.PrepareDecorator{
			autorest.AsGet(),
			autorest.WithBaseURL(c.baseURI)},
		decorators...)
	return c.prepareRequest(ctx, decorators...)
}

// PrepareDeleteRequest preparse delete request
func (c *Client) PrepareDeleteRequest(ctx context.Context, decorators ...autorest.PrepareDecorator) (*http.Request, error) {
	decorators = append(
		[]autorest.PrepareDecorator{
			autorest.AsDelete(),
			autorest.WithBaseURL(c.baseURI)},
		decorators...)
	return c.prepareRequest(ctx, decorators...)
}

// PrepareHeadRequest prepares head request
func (c *Client) PrepareHeadRequest(ctx context.Context, decorators ...autorest.PrepareDecorator) (*http.Request, error) {
	decorators = append(
		[]autorest.PrepareDecorator{
			autorest.AsHead(),
			autorest.WithBaseURL(c.baseURI)},
		decorators...)
	return c.prepareRequest(ctx, decorators...)
}

// WaitForAsyncOperationCompletion waits for an operation completion
func (c *Client) WaitForAsyncOperationCompletion(ctx context.Context, future *azure.Future, asyncOperationName string) error {
	err := future.WaitForCompletionRef(ctx, c.client)
	if err != nil {
		klog.V(5).Infof("Received error in WaitForCompletionRef: '%v'", err)
		return err
	}

	var done bool
	done, err = future.DoneWithContext(ctx, c.client)
	if err != nil {
		klog.V(5).Infof("Received error in DoneWithContext: '%v'", err)
		return autorest.NewErrorWithError(err, asyncOperationName, "Result", future.Response(), "Polling failure")
	}
	if !done {
		return azure.NewAsyncOpIncompleteError(asyncOperationName)
	}

	return nil
}

// WaitForAsyncOperationResult waits for an operation result.
func (c *Client) WaitForAsyncOperationResult(ctx context.Context, future *azure.Future, asyncOperationName string) (*http.Response, error) {
	err := c.WaitForAsyncOperationCompletion(ctx, future, asyncOperationName)
	if err != nil {
		klog.V(5).Infof("Received error in WaitForAsyncOperationCompletion: '%v'", err)
		return nil, err
	}

	sendBackoff := *c.backoff
	sender := autorest.DecorateSender(
		c.client,
		retry.DoExponentialBackoffRetry(&sendBackoff),
	)
	return future.GetResult(sender)
}

// SendAsync send a request and return a future object representing the async result as well as the origin http response
func (c *Client) SendAsync(ctx context.Context, request *http.Request) (*azure.Future, *http.Response, *retry.Error) {
	asyncResponse, rerr := c.Send(ctx, request)
	if rerr != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "sendAsync.send", request.URL.String(), rerr.Error())
		return nil, nil, rerr
	}

	future, err := azure.NewFutureFromResponse(asyncResponse)
	if err != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "sendAsync.respond", request.URL.String(), err)
		return nil, asyncResponse, retry.GetError(asyncResponse, err)
	}

	return &future, asyncResponse, nil
}

// GetResource get a resource by resource ID
func (c *Client) GetResource(ctx context.Context, resourceID, expand string) (*http.Response, *retry.Error) {
	decorators := []autorest.PrepareDecorator{
		autorest.WithPathParameters("{resourceID}", map[string]interface{}{"resourceID": resourceID}),
	}
	if expand != "" {
		queryParameters := map[string]interface{}{
			"$expand": autorest.Encode("query", expand),
		}
		decorators = append(decorators, autorest.WithQueryParameters(queryParameters))
	}
	request, err := c.PrepareGetRequest(ctx, decorators...)
	if err != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "get.prepare", resourceID, err)
		return nil, retry.NewError(false, err)
	}

	return c.Send(ctx, request)
}

// GetResourceWithDecorators get a resource with decorators by resource ID
func (c *Client) GetResourceWithDecorators(ctx context.Context, resourceID string, decorators []autorest.PrepareDecorator) (*http.Response, *retry.Error) {
	getDecorators := []autorest.PrepareDecorator{
		autorest.WithPathParameters("{resourceID}", map[string]interface{}{"resourceID": resourceID}),
	}
	getDecorators = append(getDecorators, decorators...)
	request, err := c.PrepareGetRequest(ctx, getDecorators...)
	if err != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "get.prepare", resourceID, err)
		return nil, retry.NewError(false, err)
	}

	return c.Send(ctx, request)
}

// PutResource puts a resource by resource ID
func (c *Client) PutResource(ctx context.Context, resourceID string, parameters interface{}) (*http.Response, *retry.Error) {
	putDecorators := []autorest.PrepareDecorator{
		autorest.WithPathParameters("{resourceID}", map[string]interface{}{"resourceID": resourceID}),
		autorest.WithJSON(parameters),
	}
	return c.PutResourceWithDecorators(ctx, resourceID, parameters, putDecorators)
}

// PutResources puts a list of resources from resources map[resourceID]parameters.
// Those resources sync requests are sequential while async requests are concurrent. It's especially
// useful when the ARM API doesn't support concurrent requests.
func (c *Client) PutResources(ctx context.Context, resources map[string]interface{}) map[string]*PutResourcesResponse {
	if len(resources) == 0 {
		return nil
	}

	// Sequential sync requests.
	futures := make(map[string]*azure.Future)
	responses := make(map[string]*PutResourcesResponse)
	for resourceID, parameters := range resources {
		decorators := []autorest.PrepareDecorator{
			autorest.WithPathParameters("{resourceID}", map[string]interface{}{"resourceID": resourceID}),
			autorest.WithJSON(parameters),
		}
		request, err := c.PreparePutRequest(ctx, decorators...)
		if err != nil {
			klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "put.prepare", resourceID, err)
			responses[resourceID] = &PutResourcesResponse{
				Error: retry.NewError(false, err),
			}
			continue
		}

		future, resp, clientErr := c.SendAsync(ctx, request)
		defer c.CloseResponse(ctx, resp)
		if clientErr != nil {
			klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "put.send", resourceID, clientErr.Error())
			responses[resourceID] = &PutResourcesResponse{
				Error: clientErr,
			}
			continue
		}

		futures[resourceID] = future
	}

	// Concurrent async requests.
	wg := sync.WaitGroup{}
	var responseLock sync.Mutex
	for resourceID, future := range futures {
		wg.Add(1)
		go func(resourceID string, future *azure.Future) {
			defer wg.Done()
			response, err := c.WaitForAsyncOperationResult(ctx, future, "armclient.PutResource")
			if err != nil {
				if response != nil {
					klog.V(5).Infof("Received error in WaitForAsyncOperationResult: '%s', response code %d", err.Error(), response.StatusCode)
				} else {
					klog.V(5).Infof("Received error in WaitForAsyncOperationResult: '%s', no response", err.Error())
				}

				retriableErr := retry.GetError(response, err)
				if !retriableErr.Retriable &&
					strings.Contains(strings.ToUpper(err.Error()), strings.ToUpper("InternalServerError")) {
					klog.V(5).Infof("Received InternalServerError in WaitForAsyncOperationResult: '%s', setting error retriable", err.Error())
					retriableErr.Retriable = true
				}

				responseLock.Lock()
				responses[resourceID] = &PutResourcesResponse{
					Error: retriableErr,
				}
				responseLock.Unlock()
				return
			}

			responseLock.Lock()
			responses[resourceID] = &PutResourcesResponse{
				Response: response,
			}
			responseLock.Unlock()
		}(resourceID, future)
	}

	wg.Wait()
	return responses
}

// PutResourceWithDecorators puts a resource by resource ID
func (c *Client) PutResourceWithDecorators(ctx context.Context, resourceID string, parameters interface{}, decorators []autorest.PrepareDecorator) (*http.Response, *retry.Error) {
	request, err := c.PreparePutRequest(ctx, decorators...)
	if err != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "put.prepare", resourceID, err)
		return nil, retry.NewError(false, err)
	}

	future, resp, clientErr := c.SendAsync(ctx, request)
	defer c.CloseResponse(ctx, resp)
	if clientErr != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "put.send", resourceID, clientErr.Error())
		return nil, clientErr
	}

	response, err := c.WaitForAsyncOperationResult(ctx, future, "armclient.PutResource")
	if err != nil {
		if response != nil {
			klog.V(5).Infof("Received error in WaitForAsyncOperationResult: '%s', response code %d", err.Error(), response.StatusCode)
		} else {
			klog.V(5).Infof("Received error in WaitForAsyncOperationResult: '%s', no response", err.Error())
		}

		retriableErr := retry.GetError(response, err)
		if !retriableErr.Retriable &&
			strings.Contains(strings.ToUpper(err.Error()), strings.ToUpper("InternalServerError")) {
			klog.V(5).Infof("Received InternalServerError in WaitForAsyncOperationResult: '%s', setting error retriable", err.Error())
			retriableErr.Retriable = true
		}
		return nil, retriableErr
	}

	return response, nil
}

// PatchResource patches a resource by resource ID
func (c *Client) PatchResource(ctx context.Context, resourceID string, parameters interface{}) (*http.Response, *retry.Error) {
	decorators := []autorest.PrepareDecorator{
		autorest.WithPathParameters("{resourceID}", map[string]interface{}{"resourceID": resourceID}),
		autorest.WithJSON(parameters),
	}

	request, err := c.PreparePatchRequest(ctx, decorators...)
	if err != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "patch.prepare", resourceID, err)
		return nil, retry.NewError(false, err)
	}

	future, resp, clientErr := c.SendAsync(ctx, request)
	defer c.CloseResponse(ctx, resp)
	if clientErr != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "patch.send", resourceID, clientErr.Error())
		return nil, clientErr
	}

	response, err := c.WaitForAsyncOperationResult(ctx, future, "armclient.PatchResource")
	if err != nil {
		if response != nil {
			klog.V(5).Infof("Received error in WaitForAsyncOperationResult: '%s', response code %d", err.Error(), response.StatusCode)
		} else {
			klog.V(5).Infof("Received error in WaitForAsyncOperationResult: '%s', no response", err.Error())
		}

		retriableErr := retry.GetError(response, err)
		if !retriableErr.Retriable &&
			strings.Contains(strings.ToUpper(err.Error()), strings.ToUpper("InternalServerError")) {
			klog.V(5).Infof("Received InternalServerError in WaitForAsyncOperationResult: '%s', setting error retriable", err.Error())
			retriableErr.Retriable = true
		}
		return nil, retriableErr
	}

	return response, nil
}

// PutResourceAsync puts a resource by resource ID in async mode
func (c *Client) PutResourceAsync(ctx context.Context, resourceID string, parameters interface{}) (*azure.Future, *retry.Error) {
	decorators := []autorest.PrepareDecorator{
		autorest.WithPathParameters("{resourceID}", map[string]interface{}{"resourceID": resourceID}),
		autorest.WithJSON(parameters),
	}

	request, err := c.PreparePutRequest(ctx, decorators...)
	if err != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "put.prepare", resourceID, err)
		return nil, retry.NewError(false, err)
	}

	future, resp, rErr := c.SendAsync(ctx, request)
	defer c.CloseResponse(ctx, resp)
	if rErr != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "put.send", resourceID, err)
		return nil, rErr
	}

	return future, nil
}

// PostResource posts a resource by resource ID
func (c *Client) PostResource(ctx context.Context, resourceID, action string, parameters interface{}) (*http.Response, *retry.Error) {
	pathParameters := map[string]interface{}{
		"resourceID": resourceID,
		"action":     action,
	}

	decorators := []autorest.PrepareDecorator{
		autorest.WithPathParameters("{resourceID}/{action}", pathParameters),
		autorest.WithJSON(parameters),
	}
	request, err := c.PreparePostRequest(ctx, decorators...)
	if err != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "post.prepare", resourceID, err)
		return nil, retry.NewError(false, err)
	}

	return c.sendRequest(ctx, request)
}

// DeleteResource deletes a resource by resource ID
func (c *Client) DeleteResource(ctx context.Context, resourceID, ifMatch string) *retry.Error {
	future, clientErr := c.DeleteResourceAsync(ctx, resourceID, ifMatch)
	if clientErr != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "delete.request", resourceID, clientErr.Error())
		return clientErr
	}

	if future == nil {
		return nil
	}

	if err := c.WaitForAsyncOperationCompletion(ctx, future, "armclient.DeleteResource"); err != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "delete.wait", resourceID, clientErr.Error())
		return retry.NewError(true, err)
	}

	return nil
}

// HeadResource heads a resource by resource ID
func (c *Client) HeadResource(ctx context.Context, resourceID string) (*http.Response, *retry.Error) {
	decorators := []autorest.PrepareDecorator{
		autorest.WithPathParameters("{resourceID}", map[string]interface{}{"resourceID": resourceID}),
	}
	request, err := c.PrepareHeadRequest(ctx, decorators...)
	if err != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "head.prepare", resourceID, err)
		return nil, retry.NewError(false, err)
	}

	return c.sendRequest(ctx, request)
}

// DeleteResourceAsync delete a resource by resource ID and returns a future representing the async result
func (c *Client) DeleteResourceAsync(ctx context.Context, resourceID, ifMatch string) (*azure.Future, *retry.Error) {
	decorators := []autorest.PrepareDecorator{
		autorest.WithPathParameters("{resourceID}", map[string]interface{}{"resourceID": resourceID}),
	}
	if len(ifMatch) > 0 {
		decorators = append(decorators, autorest.WithHeader("If-Match", autorest.String(ifMatch)))
	}

	deleteRequest, err := c.PrepareDeleteRequest(ctx, decorators...)
	if err != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "deleteAsync.prepare", resourceID, err)
		return nil, retry.NewError(false, err)
	}

	resp, rerr := c.sendRequest(ctx, deleteRequest)
	defer c.CloseResponse(ctx, resp)
	if rerr != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "deleteAsync.send", resourceID, rerr.Error())
		return nil, rerr
	}

	err = autorest.Respond(
		resp,
		azure.WithErrorUnlessStatusCode(http.StatusOK, http.StatusAccepted, http.StatusNoContent, http.StatusNotFound))
	if err != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "deleteAsync.respond", resourceID, err)
		return nil, retry.GetError(resp, err)
	}

	if resp.StatusCode == http.StatusNotFound {
		return nil, nil
	}

	future, err := azure.NewFutureFromResponse(resp)
	if err != nil {
		klog.V(5).Infof("Received error in %s: resourceID: %s, error: %s", "deleteAsync.future", resourceID, err)
		return nil, retry.GetError(resp, err)
	}

	return &future, nil
}

// CloseResponse closes a response
func (c *Client) CloseResponse(ctx context.Context, response *http.Response) {
	if response != nil && response.Body != nil {
		if err := response.Body.Close(); err != nil {
			klog.Errorf("Error closing the response body: %v", err)
		}
	}
}

func (c *Client) prepareRequest(ctx context.Context, decorators ...autorest.PrepareDecorator) (*http.Request, error) {
	decorators = append(
		decorators,
		withAPIVersion(c.apiVersion))
	preparer := autorest.CreatePreparer(decorators...)
	return preparer.Prepare((&http.Request{}).WithContext(ctx))
}

func withAPIVersion(apiVersion string) autorest.PrepareDecorator {
	const apiVersionKey = "api-version"
	return func(p autorest.Preparer) autorest.Preparer {
		return autorest.PreparerFunc(func(r *http.Request) (*http.Request, error) {
			r, err := p.Prepare(r)
			if err == nil {
				if r.URL == nil {
					return r, fmt.Errorf("Error in withAPIVersion: Invoked with a nil URL")
				}

				v := r.URL.Query()
				if len(v.Get(apiVersionKey)) > 0 {
					return r, nil
				}

				v.Add(apiVersionKey, apiVersion)
				r.URL.RawQuery = v.Encode()
			}
			return r, err
		})
	}
}

// GetResourceID gets Azure resource ID
func GetResourceID(subscriptionID, resourceGroupName, resourceType, resourceName string) string {
	return fmt.Sprintf("/subscriptions/%s/resourceGroups/%s/providers/%s/%s",
		autorest.Encode("path", subscriptionID),
		autorest.Encode("path", resourceGroupName),
		resourceType,
		autorest.Encode("path", resourceName))
}

// GetChildResourceID gets Azure child resource ID
func GetChildResourceID(subscriptionID, resourceGroupName, resourceType, resourceName, childResourceType, childResourceName string) string {
	return fmt.Sprintf("/subscriptions/%s/resourceGroups/%s/providers/%s/%s/%s/%s",
		autorest.Encode("path", subscriptionID),
		autorest.Encode("path", resourceGroupName),
		resourceType,
		autorest.Encode("path", resourceName),
		childResourceType,
		autorest.Encode("path", childResourceName))
}

// GetChildResourcesListID gets Azure child resources list ID
func GetChildResourcesListID(subscriptionID, resourceGroupName, resourceType, resourceName, childResourceType string) string {
	return fmt.Sprintf("/subscriptions/%s/resourceGroups/%s/providers/%s/%s/%s",
		autorest.Encode("path", subscriptionID),
		autorest.Encode("path", resourceGroupName),
		resourceType,
		autorest.Encode("path", resourceName),
		childResourceType)
}

// GetProviderResourceID gets Azure RP resource ID
func GetProviderResourceID(subscriptionID, providerNamespace string) string {
	return fmt.Sprintf("/subscriptions/%s/providers/%s",
		autorest.Encode("path", subscriptionID),
		providerNamespace)
}

// GetProviderResourcesListID gets Azure RP resources list ID
func GetProviderResourcesListID(subscriptionID string) string {
	return fmt.Sprintf("/subscriptions/%s/providers", autorest.Encode("path", subscriptionID))
}
