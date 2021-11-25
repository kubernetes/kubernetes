//go:build !providerless
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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"testing"
	"time"

	"github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-06-01/network"
	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/to"
	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"

	azclients "k8s.io/legacy-cloud-providers/azure/clients"
	"k8s.io/legacy-cloud-providers/azure/clients/armclient"
	"k8s.io/legacy-cloud-providers/azure/clients/armclient/mockarmclient"
	"k8s.io/legacy-cloud-providers/azure/retry"
)

func TestNew(t *testing.T) {
	config := &azclients.ClientConfig{
		SubscriptionID:          "sub",
		ResourceManagerEndpoint: "endpoint",
		Location:                "eastus",
		RateLimitConfig: &azclients.RateLimitConfig{
			CloudProviderRateLimit:            true,
			CloudProviderRateLimitQPS:         0.5,
			CloudProviderRateLimitBucket:      1,
			CloudProviderRateLimitQPSWrite:    0.5,
			CloudProviderRateLimitBucketWrite: 1,
		},
		Backoff: &retry.Backoff{Steps: 1},
	}

	lbClient := New(config)
	assert.Equal(t, "sub", lbClient.subscriptionID)
	assert.NotEmpty(t, lbClient.rateLimiterReader)
	assert.NotEmpty(t, lbClient.rateLimiterWriter)
}

func TestGetNotFound(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Network/loadBalancers/lb1"
	response := &http.Response{
		StatusCode: http.StatusNotFound,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
	}
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	lbClient := getTestLoadBalancerClient(armClient)
	expected := network.LoadBalancer{Response: autorest.Response{}}
	result, rerr := lbClient.Get(context.TODO(), "rg", "lb1", "")
	assert.Equal(t, expected, result)
	assert.NotNil(t, rerr)
	assert.Equal(t, http.StatusNotFound, rerr.HTTPStatusCode)
}

func TestGetInternalError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Network/loadBalancers/lb1"
	response := &http.Response{
		StatusCode: http.StatusInternalServerError,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
	}
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	lbClient := getTestLoadBalancerClient(armClient)
	expected := network.LoadBalancer{Response: autorest.Response{}}
	result, rerr := lbClient.Get(context.TODO(), "rg", "lb1", "")
	assert.Equal(t, expected, result)
	assert.NotNil(t, rerr)
	assert.Equal(t, http.StatusInternalServerError, rerr.HTTPStatusCode)
}

func TestGetThrottle(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Network/loadBalancers/lb1"
	response := &http.Response{
		StatusCode: http.StatusTooManyRequests,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
	}
	throttleErr := &retry.Error{
		HTTPStatusCode: http.StatusTooManyRequests,
		RawError:       fmt.Errorf("error"),
		Retriable:      true,
		RetryAfter:     time.Unix(100, 0),
	}
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(response, throttleErr).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	lbClient := getTestLoadBalancerClient(armClient)
	result, rerr := lbClient.Get(context.TODO(), "rg", "lb1", "")
	assert.Empty(t, result)
	assert.Equal(t, throttleErr, rerr)
}

func TestList(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Network/loadBalancers"
	armClient := mockarmclient.NewMockInterface(ctrl)
	lbList := []network.LoadBalancer{getTestLoadBalancer("lb1"), getTestLoadBalancer("lb2"), getTestLoadBalancer("lb3")}
	responseBody, err := json.Marshal(network.LoadBalancerListResult{Value: &lbList})
	assert.NoError(t, err)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(
		&http.Response{
			StatusCode: http.StatusOK,
			Body:       ioutil.NopCloser(bytes.NewReader(responseBody)),
		}, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	lbClient := getTestLoadBalancerClient(armClient)
	result, rerr := lbClient.List(context.TODO(), "rg")
	assert.Nil(t, rerr)
	assert.Equal(t, 3, len(result))

	response := &http.Response{
		StatusCode: http.StatusTooManyRequests,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
	}
	throttleErr := &retry.Error{
		HTTPStatusCode: http.StatusTooManyRequests,
		RawError:       fmt.Errorf("error"),
		Retriable:      true,
		RetryAfter:     time.Unix(100, 0),
	}
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(response, throttleErr).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)
	_, rerr = lbClient.List(context.TODO(), "rg")
	assert.Equal(t, throttleErr, rerr)
}

func TestListWithNextPage(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Network/loadBalancers"
	armClient := mockarmclient.NewMockInterface(ctrl)
	lbList := []network.LoadBalancer{getTestLoadBalancer("lb1"), getTestLoadBalancer("lb2"), getTestLoadBalancer("lb3")}
	// LoadBalancerListResult.MarshalJson() doesn't include "nextLink" in its result, hence partialResponse is composed manually below.
	partialResponse, err := json.Marshal(map[string]interface{}{"value": lbList, "nextLink": "nextLink"})
	assert.NoError(t, err)
	pagedResponse, err := json.Marshal(network.LoadBalancerListResult{Value: &lbList})
	assert.NoError(t, err)
	armClient.EXPECT().PrepareGetRequest(gomock.Any(), gomock.Any()).Return(&http.Request{}, nil)
	armClient.EXPECT().Send(gomock.Any(), gomock.Any()).Return(
		&http.Response{
			StatusCode: http.StatusOK,
			Body:       ioutil.NopCloser(bytes.NewReader(pagedResponse)),
		}, nil)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(
		&http.Response{
			StatusCode: http.StatusOK,
			Body:       ioutil.NopCloser(bytes.NewReader(partialResponse)),
		}, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(2)

	lbClient := getTestLoadBalancerClient(armClient)
	result, rerr := lbClient.List(context.TODO(), "rg")
	assert.Nil(t, rerr)
	assert.Equal(t, 6, len(result))
}

func TestListNextResultsMultiPages(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	tests := []struct {
		prepareErr error
		sendErr    *retry.Error
	}{
		{
			prepareErr: nil,
			sendErr:    nil,
		},
		{
			prepareErr: fmt.Errorf("error"),
		},
		{
			sendErr: &retry.Error{RawError: fmt.Errorf("error")},
		},
	}

	lastResult := network.LoadBalancerListResult{
		NextLink: to.StringPtr("next"),
	}

	for _, test := range tests {
		armClient := mockarmclient.NewMockInterface(ctrl)
		req := &http.Request{
			Method: "GET",
		}
		armClient.EXPECT().PrepareGetRequest(gomock.Any(), gomock.Any()).Return(req, test.prepareErr)
		if test.prepareErr == nil {
			armClient.EXPECT().Send(gomock.Any(), req).Return(&http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewReader([]byte(`{"foo":"bar"}`))),
			}, test.sendErr)
			armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any())
		}

		lbClient := getTestLoadBalancerClient(armClient)
		result, err := lbClient.listNextResults(context.TODO(), lastResult)
		if test.prepareErr != nil || test.sendErr != nil {
			assert.Error(t, err)
		} else {
			assert.NoError(t, err)
		}
		if test.prepareErr != nil {
			assert.Empty(t, result)
		} else {
			assert.NotEmpty(t, result)
		}
	}
}

func TestCreateOrUpdate(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	lb := getTestLoadBalancer("lb1")
	armClient := mockarmclient.NewMockInterface(ctrl)
	response := &http.Response{
		StatusCode: http.StatusOK,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte(""))),
	}
	armClient.EXPECT().PutResourceWithDecorators(gomock.Any(), to.String(lb.ID), lb, gomock.Any()).Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	lbClient := getTestLoadBalancerClient(armClient)
	rerr := lbClient.CreateOrUpdate(context.TODO(), "rg", "lb1", lb, "")
	assert.Nil(t, rerr)
}

func TestDelete(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	tests := []struct {
		description  string
		armClientErr *retry.Error
		expectedErr  *retry.Error
	}{
		{
			description:  "Delete should report the throttling error",
			armClientErr: &retry.Error{HTTPStatusCode: http.StatusTooManyRequests},
			expectedErr:  &retry.Error{HTTPStatusCode: http.StatusTooManyRequests},
		},
		{
			description: "Delete should not report any error if there's no error from arm client",
		},
	}

	lb := getTestLoadBalancer("lb1")

	for _, test := range tests {
		armClient := mockarmclient.NewMockInterface(ctrl)
		armClient.EXPECT().DeleteResource(gomock.Any(), to.String(lb.ID), "").Return(test.armClientErr)

		lbClient := getTestLoadBalancerClient(armClient)
		rerr := lbClient.Delete(context.TODO(), "rg", "lb1")
		assert.Equal(t, test.expectedErr, rerr)
	}
}

func getTestLoadBalancer(name string) network.LoadBalancer {
	return network.LoadBalancer{
		ID:       to.StringPtr(fmt.Sprintf("/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Network/loadBalancers/%s", name)),
		Name:     to.StringPtr(name),
		Location: to.StringPtr("eastus"),
	}
}

func getTestLoadBalancerClient(armClient armclient.Interface) *Client {
	rateLimiterReader, rateLimiterWriter := azclients.NewRateLimiter(&azclients.RateLimitConfig{})
	return &Client{
		armClient:         armClient,
		subscriptionID:    "subscriptionID",
		rateLimiterReader: rateLimiterReader,
		rateLimiterWriter: rateLimiterWriter,
	}
}
