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
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Azure/go-autorest/autorest"
	"github.com/stretchr/testify/assert"
	"k8s.io/legacy-cloud-providers/azure/retry"
)

func TestSend(t *testing.T) {
	count := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if count <= 1 {
			http.Error(w, "failed", http.StatusInternalServerError)
			count++
		}
	}))

	backoff := &retry.Backoff{Steps: 3}
	armClient := New(nil, server.URL, "test", "2019-01-01", "eastus", backoff)
	pathParameters := map[string]interface{}{
		"resourceGroupName": autorest.Encode("path", "testgroup"),
		"subscriptionId":    autorest.Encode("path", "testid"),
		"resourceName":      autorest.Encode("path", "testname"),
	}

	decorators := []autorest.PrepareDecorator{
		autorest.WithPathParameters(
			"/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Network/vNets/{resourceName}", pathParameters),
	}

	ctx := context.Background()
	request, err := armClient.PrepareGetRequest(ctx, decorators...)
	assert.NoError(t, err)

	response, rerr := armClient.Send(ctx, request)
	assert.Nil(t, rerr)
	assert.Equal(t, 2, count)
	assert.Equal(t, http.StatusOK, response.StatusCode)
}

func TestSendFailure(t *testing.T) {
	count := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "failed", http.StatusInternalServerError)
		count++
	}))

	backoff := &retry.Backoff{Steps: 3}
	armClient := New(nil, server.URL, "test", "2019-01-01", "eastus", backoff)
	pathParameters := map[string]interface{}{
		"resourceGroupName": autorest.Encode("path", "testgroup"),
		"subscriptionId":    autorest.Encode("path", "testid"),
		"resourceName":      autorest.Encode("path", "testname"),
	}

	decorators := []autorest.PrepareDecorator{
		autorest.WithPathParameters(
			"/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Network/vNets/{resourceName}", pathParameters),
	}

	ctx := context.Background()
	request, err := armClient.PrepareGetRequest(ctx, decorators...)
	assert.NoError(t, err)

	response, rerr := armClient.Send(ctx, request)
	assert.NotNil(t, rerr)
	assert.Equal(t, 3, count)
	assert.Equal(t, http.StatusInternalServerError, response.StatusCode)
}

func TestSendThrottled(t *testing.T) {
	count := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set(retry.RetryAfterHeaderKey, "30")
		http.Error(w, "failed", http.StatusTooManyRequests)
		count++
	}))

	backoff := &retry.Backoff{Steps: 3}
	armClient := New(nil, server.URL, "test", "2019-01-01", "eastus", backoff)
	pathParameters := map[string]interface{}{
		"resourceGroupName": autorest.Encode("path", "testgroup"),
		"subscriptionId":    autorest.Encode("path", "testid"),
		"resourceName":      autorest.Encode("path", "testname"),
	}
	decorators := []autorest.PrepareDecorator{
		autorest.WithPathParameters(
			"/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Network/vNets/{resourceName}", pathParameters),
	}

	ctx := context.Background()
	request, err := armClient.PrepareGetRequest(ctx, decorators...)
	assert.NoError(t, err)

	response, rerr := armClient.Send(ctx, request)
	assert.NotNil(t, rerr)
	assert.Equal(t, 1, count)
	assert.Equal(t, http.StatusTooManyRequests, response.StatusCode)
}

func TestSendAsync(t *testing.T) {
	count := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count++
		http.Error(w, "failed", http.StatusForbidden)

	}))

	backoff := &retry.Backoff{Steps: 1}
	armClient := New(nil, server.URL, "test", "2019-01-01", "eastus", backoff)
	armClient.client.RetryDuration = time.Millisecond * 1

	pathParameters := map[string]interface{}{
		"resourceGroupName": autorest.Encode("path", "testgroup"),
		"subscriptionId":    autorest.Encode("path", "testid"),
		"resourceName":      autorest.Encode("path", "testname"),
	}
	decorators := []autorest.PrepareDecorator{
		autorest.WithPathParameters(
			"/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Network/vNets/{resourceName}", pathParameters),
	}

	ctx := context.Background()
	request, err := armClient.PreparePutRequest(ctx, decorators...)
	assert.NoError(t, err)

	future, response, rerr := armClient.SendAsync(ctx, request)
	assert.Nil(t, future)
	assert.Nil(t, response)
	assert.Equal(t, 1, count)
	assert.NotNil(t, rerr)
	assert.Equal(t, true, rerr.Retriable)
}

func TestNormalizeAzureRegion(t *testing.T) {
	tests := []struct {
		region   string
		expected string
	}{
		{
			region:   "eastus",
			expected: "eastus",
		},
		{
			region:   " eastus ",
			expected: "eastus",
		},
		{
			region:   " eastus\t",
			expected: "eastus",
		},
		{
			region:   " eastus\v",
			expected: "eastus",
		},
		{
			region:   " eastus\v\r\f\n",
			expected: "eastus",
		},
	}

	for i, test := range tests {
		real := NormalizeAzureRegion(test.region)
		assert.Equal(t, test.expected, real, "test[%d]: NormalizeAzureRegion(%q) != %q", i, test.region, test.expected)
	}
}

func TestPutResource(t *testing.T) {
	expectedURI := "/subscriptions/subscription/resourceGroups/rg/providers/Microsoft.Network/publicIPAddresses/testPIP?api-version=2019-01-01"
	operationURI := "/subscriptions/subscription/providers/Microsoft.Network/locations/eastus/operations/op?api-version=2019-01-01"
	handlers := []func(http.ResponseWriter, *http.Request){
		func(rw http.ResponseWriter, req *http.Request) {
			assert.Equal(t, "PUT", req.Method)
			assert.Equal(t, expectedURI, req.URL.String())
			rw.Header().Set(http.CanonicalHeaderKey("Azure-AsyncOperation"),
				fmt.Sprintf("http://%s%s", req.Host, operationURI))
			rw.WriteHeader(http.StatusCreated)
		},

		func(rw http.ResponseWriter, req *http.Request) {
			assert.Equal(t, "GET", req.Method)
			assert.Equal(t, operationURI, req.URL.String())

			rw.WriteHeader(http.StatusOK)
			rw.Write([]byte(`{"error":{"code":"InternalServerError"},"status":"Failed"}`))
		},
	}

	count := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		handlers[count](w, r)
		count++
		if count > 1 {
			count = 1
		}
	}))

	backoff := &retry.Backoff{Steps: 1}
	armClient := New(nil, server.URL, "test", "2019-01-01", "eastus", backoff)
	armClient.client.RetryDuration = time.Millisecond * 1

	ctx := context.Background()
	response, rerr := armClient.PutResource(ctx, "/subscriptions/subscription/resourceGroups/rg/providers/Microsoft.Network/publicIPAddresses/testPIP", nil)
	assert.Equal(t, 1, count)
	assert.Nil(t, response)
	assert.NotNil(t, rerr)
	assert.Equal(t, true, rerr.Retriable)
}

func TestDeleteResourceAsync(t *testing.T) {
	count := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count++
		http.Error(w, "failed", http.StatusForbidden)
	}))

	backoff := &retry.Backoff{Steps: 3}
	armClient := New(nil, server.URL, "test", "2019-01-01", "eastus", backoff)
	armClient.client.RetryDuration = time.Millisecond * 1

	ctx := context.Background()
	resourceID := "/subscriptions/subscription/resourceGroups/rg/providers/Microsoft.Network/publicIPAddresses/testPIP"
	future, rerr := armClient.DeleteResourceAsync(ctx, resourceID, "")
	assert.Equal(t, 3, count)
	assert.Nil(t, future)
	assert.NotNil(t, rerr)
	assert.Equal(t, true, rerr.Retriable)
}
