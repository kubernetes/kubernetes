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

package routeclient

import (
	"bytes"
	"context"
	"fmt"
	"io/ioutil"
	"net/http"
	"testing"
	"time"

	"github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-06-01/network"
	"github.com/Azure/go-autorest/autorest/to"
	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"

	"k8s.io/client-go/util/flowcontrol"
	azclients "k8s.io/legacy-cloud-providers/azure/clients"
	"k8s.io/legacy-cloud-providers/azure/clients/armclient"
	"k8s.io/legacy-cloud-providers/azure/clients/armclient/mockarmclient"
	"k8s.io/legacy-cloud-providers/azure/retry"
)

// 2065-01-24 05:20:00 +0000 UTC
func getFutureTime() time.Time {
	return time.Unix(3000000000, 0)
}

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

	routeClient := New(config)
	assert.Equal(t, "sub", routeClient.subscriptionID)
	assert.NotEmpty(t, routeClient.rateLimiterReader)
	assert.NotEmpty(t, routeClient.rateLimiterWriter)
}

func TestCreateOrUpdate(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	r := getTestRoute("r1")
	armClient := mockarmclient.NewMockInterface(ctrl)
	response := &http.Response{
		StatusCode: http.StatusOK,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte(""))),
	}
	armClient.EXPECT().PutResourceWithDecorators(gomock.Any(), to.String(r.ID), r, gomock.Any()).Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	routeClient := getTestRouteClient(armClient)
	rerr := routeClient.CreateOrUpdate(context.TODO(), "rg", "rt", "r1", r, "*")
	assert.Nil(t, rerr)
}

func TestCreateOrUpdateWithNeverRateLimiter(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	rcCreateOrUpdateErr := &retry.Error{
		RawError:  fmt.Errorf("azure cloud provider rate limited(%s) for operation %q", "write", "RouteCreateOrUpdate"),
		Retriable: true,
	}

	r := getTestRoute("r1")
	armClient := mockarmclient.NewMockInterface(ctrl)

	routeClient := getTestRouteClientWithNeverRateLimiter(armClient)
	rerr := routeClient.CreateOrUpdate(context.TODO(), "rg", "rt", "r1", r, "")
	assert.Equal(t, rcCreateOrUpdateErr, rerr)
}

func TestCreateOrUpdateRetryAfterReader(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	rcCreateOrUpdateErr := &retry.Error{
		RawError:   fmt.Errorf("azure cloud provider throttled for operation %s with reason %q", "RouteCreateOrUpdate", "client throttled"),
		Retriable:  true,
		RetryAfter: getFutureTime(),
	}

	r := getTestRoute("r1")
	armClient := mockarmclient.NewMockInterface(ctrl)

	routeClient := getTestRouteClientWithRetryAfterReader(armClient)
	rerr := routeClient.CreateOrUpdate(context.TODO(), "rg", "rt", "r1", r, "")
	assert.NotNil(t, rerr)
	assert.Equal(t, rcCreateOrUpdateErr, rerr)
}

func TestCreateOrUpdateThrottle(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

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

	r := getTestRoute("r1")
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().PutResourceWithDecorators(gomock.Any(), to.String(r.ID), r, gomock.Any()).Return(response, throttleErr).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	routeClient := getTestRouteClient(armClient)
	rerr := routeClient.CreateOrUpdate(context.TODO(), "rg", "rt", "r1", r, "")
	assert.Equal(t, throttleErr, rerr)
}

func TestCreateOrUpdateWithCreateOrUpdateResponderError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	r := getTestRoute("r1")
	armClient := mockarmclient.NewMockInterface(ctrl)
	response := &http.Response{
		StatusCode: http.StatusNotFound,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte(""))),
	}

	armClient.EXPECT().PutResourceWithDecorators(gomock.Any(), to.String(r.ID), r, gomock.Any()).Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	routeClient := getTestRouteClient(armClient)
	rerr := routeClient.CreateOrUpdate(context.TODO(), "rg", "rt", "r1", r, "")
	assert.NotNil(t, rerr)
}

func TestDelete(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	r := getTestRoute("r1")
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().DeleteResource(gomock.Any(), to.String(r.ID), "").Return(nil).Times(1)

	routeClient := getTestRouteClient(armClient)
	rerr := routeClient.Delete(context.TODO(), "rg", "rt", "r1")
	assert.Nil(t, rerr)
}

func TestDeleteWithNeverRateLimiter(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	rcDeleteErr := &retry.Error{
		RawError:  fmt.Errorf("azure cloud provider rate limited(%s) for operation %q", "write", "RouteDelete"),
		Retriable: true,
	}

	armClient := mockarmclient.NewMockInterface(ctrl)

	routeClient := getTestRouteClientWithNeverRateLimiter(armClient)
	rerr := routeClient.Delete(context.TODO(), "rg", "rt", "r1")
	assert.Equal(t, rcDeleteErr, rerr)
}

func TestDeleteRetryAfterReader(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	rcDeleteErr := &retry.Error{
		RawError:   fmt.Errorf("azure cloud provider throttled for operation %s with reason %q", "RouteDelete", "client throttled"),
		Retriable:  true,
		RetryAfter: getFutureTime(),
	}

	armClient := mockarmclient.NewMockInterface(ctrl)

	routeClient := getTestRouteClientWithRetryAfterReader(armClient)
	rerr := routeClient.Delete(context.TODO(), "rg", "rt", "r1")
	assert.Equal(t, rcDeleteErr, rerr)
}

func TestDeleteThrottle(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	throttleErr := &retry.Error{
		HTTPStatusCode: http.StatusTooManyRequests,
		RawError:       fmt.Errorf("error"),
		Retriable:      true,
		RetryAfter:     time.Unix(100, 0),
	}

	r := getTestRoute("r1")
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().DeleteResource(gomock.Any(), to.String(r.ID), "").Return(throttleErr).Times(1)

	routeClient := getTestRouteClient(armClient)
	rerr := routeClient.Delete(context.TODO(), "rg", "rt", "r1")
	assert.Equal(t, throttleErr, rerr)
}

func getTestRoute(name string) network.Route {
	return network.Route{
		ID:   to.StringPtr(fmt.Sprintf("/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Network/routeTables/rt/routes/%s", name)),
		Name: to.StringPtr(name),
	}
}

func getTestRouteClient(armClient armclient.Interface) *Client {
	rateLimiterReader, rateLimiterWriter := azclients.NewRateLimiter(&azclients.RateLimitConfig{})
	return &Client{
		armClient:         armClient,
		subscriptionID:    "subscriptionID",
		rateLimiterReader: rateLimiterReader,
		rateLimiterWriter: rateLimiterWriter,
	}
}

func getTestRouteClientWithNeverRateLimiter(armClient armclient.Interface) *Client {
	rateLimiterReader := flowcontrol.NewFakeNeverRateLimiter()
	rateLimiterWriter := flowcontrol.NewFakeNeverRateLimiter()
	return &Client{
		armClient:         armClient,
		subscriptionID:    "subscriptionID",
		rateLimiterReader: rateLimiterReader,
		rateLimiterWriter: rateLimiterWriter,
	}
}

func getTestRouteClientWithRetryAfterReader(armClient armclient.Interface) *Client {
	rateLimiterReader := flowcontrol.NewFakeAlwaysRateLimiter()
	rateLimiterWriter := flowcontrol.NewFakeAlwaysRateLimiter()
	return &Client{
		armClient:         armClient,
		subscriptionID:    "subscriptionID",
		rateLimiterReader: rateLimiterReader,
		rateLimiterWriter: rateLimiterWriter,
		RetryAfterReader:  getFutureTime(),
		RetryAfterWriter:  getFutureTime(),
	}
}
