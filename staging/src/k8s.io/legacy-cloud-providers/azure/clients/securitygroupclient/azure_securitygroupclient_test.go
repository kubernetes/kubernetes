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

package securitygroupclient

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

	nsgClient := New(config)
	assert.Equal(t, "sub", nsgClient.subscriptionID)
	assert.NotEmpty(t, nsgClient.rateLimiterReader)
	assert.NotEmpty(t, nsgClient.rateLimiterWriter)
}

func TestGet(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Network/networkSecurityGroups/nsg1"
	response := &http.Response{
		StatusCode: http.StatusOK,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
	}

	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	expected := network.SecurityGroup{}
	expected.Response = autorest.Response{Response: response}
	nsgClient := getTestSecurityGroupClient(armClient)
	result, rerr := nsgClient.Get(context.TODO(), "rg", "nsg1", "")
	assert.Equal(t, expected, result)
	assert.Nil(t, rerr)
}

func TestGetNeverRateLimiter(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	nsgGetErr := &retry.Error{
		RawError:  fmt.Errorf("azure cloud provider rate limited(%s) for operation %q", "read", "NSGGet"),
		Retriable: true,
	}

	armClient := mockarmclient.NewMockInterface(ctrl)

	nsgClient := getTestSecurityGroupClientWithNeverRateLimiter(armClient)
	expected := network.SecurityGroup{}
	result, rerr := nsgClient.Get(context.TODO(), "rg", "nsg1", "")
	assert.Equal(t, expected, result)
	assert.Equal(t, nsgGetErr, rerr)
}

func TestGetRetryAfterReader(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	nsgGetErr := &retry.Error{
		RawError:   fmt.Errorf("azure cloud provider throttled for operation %s with reason %q", "NSGGet", "client throttled"),
		Retriable:  true,
		RetryAfter: getFutureTime(),
	}

	armClient := mockarmclient.NewMockInterface(ctrl)

	nsgClient := getTestSecurityGroupClientWithRetryAfterReader(armClient)
	expected := network.SecurityGroup{}
	result, rerr := nsgClient.Get(context.TODO(), "rg", "nsg1", "")
	assert.Equal(t, expected, result)
	assert.Equal(t, nsgGetErr, rerr)
}

func TestGetThrottle(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Network/networkSecurityGroups/nsg1"
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

	nsgClient := getTestSecurityGroupClient(armClient)
	result, rerr := nsgClient.Get(context.TODO(), "rg", "nsg1", "")
	assert.Empty(t, result)
	assert.Equal(t, throttleErr, rerr)
}

func TestGetNotFound(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Network/networkSecurityGroups/nsg1"
	response := &http.Response{
		StatusCode: http.StatusNotFound,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
	}
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	nsgClient := getTestSecurityGroupClient(armClient)
	expected := network.SecurityGroup{Response: autorest.Response{}}
	result, rerr := nsgClient.Get(context.TODO(), "rg", "nsg1", "")
	assert.Equal(t, expected, result)
	assert.NotNil(t, rerr)
	assert.Equal(t, http.StatusNotFound, rerr.HTTPStatusCode)
}

func TestGetInternalError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Network/networkSecurityGroups/nsg1"
	response := &http.Response{
		StatusCode: http.StatusInternalServerError,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
	}
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	nsgClient := getTestSecurityGroupClient(armClient)
	expected := network.SecurityGroup{Response: autorest.Response{}}
	result, rerr := nsgClient.Get(context.TODO(), "rg", "nsg1", "")
	assert.Equal(t, expected, result)
	assert.NotNil(t, rerr)
	assert.Equal(t, http.StatusInternalServerError, rerr.HTTPStatusCode)
}

func TestList(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Network/networkSecurityGroups"
	armClient := mockarmclient.NewMockInterface(ctrl)
	nsgList := []network.SecurityGroup{getTestSecurityGroup("nsg1"), getTestSecurityGroup("nsg2"), getTestSecurityGroup("nsg3")}
	responseBody, err := json.Marshal(network.SecurityGroupListResult{Value: &nsgList})
	assert.NoError(t, err)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(
		&http.Response{
			StatusCode: http.StatusOK,
			Body:       ioutil.NopCloser(bytes.NewReader(responseBody)),
		}, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	nsgClient := getTestSecurityGroupClient(armClient)
	result, rerr := nsgClient.List(context.TODO(), "rg")
	assert.Nil(t, rerr)
	assert.Equal(t, 3, len(result))
}

func TestListNextResultsMultiPages(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	tests := []struct {
		prepareErr error
		sendErr    *retry.Error
		statusCode int
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

	lastResult := network.SecurityGroupListResult{
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

		sgClient := getTestSecurityGroupClient(armClient)
		result, err := sgClient.listNextResults(context.TODO(), lastResult)
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

func TestListNextResultsMultiPagesWithListResponderError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	test := struct {
		prepareErr error
		sendErr    *retry.Error
	}{
		prepareErr: nil,
		sendErr:    nil,
	}

	lastResult := network.SecurityGroupListResult{
		NextLink: to.StringPtr("next"),
	}

	armClient := mockarmclient.NewMockInterface(ctrl)
	req := &http.Request{
		Method: "GET",
	}
	armClient.EXPECT().PrepareGetRequest(gomock.Any(), gomock.Any()).Return(req, test.prepareErr)
	if test.prepareErr == nil {
		armClient.EXPECT().Send(gomock.Any(), req).Return(&http.Response{
			StatusCode: http.StatusNotFound,
			Body:       ioutil.NopCloser(bytes.NewReader([]byte(`{"foo":"bar"}`))),
		}, test.sendErr)
		armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any())
	}

	response := &http.Response{
		StatusCode: http.StatusNotFound,
		Body:       ioutil.NopCloser(bytes.NewBuffer([]byte(`{"foo":"bar"}`))),
	}
	expected := network.SecurityGroupListResult{}
	expected.Response = autorest.Response{Response: response}
	sgClient := getTestSecurityGroupClient(armClient)
	result, err := sgClient.listNextResults(context.TODO(), lastResult)
	assert.Error(t, err)
	assert.Equal(t, expected, result)
}

func TestListWithListResponderError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Network/networkSecurityGroups"
	armClient := mockarmclient.NewMockInterface(ctrl)
	nsgList := []network.SecurityGroup{getTestSecurityGroup("nsg1"), getTestSecurityGroup("nsg2"), getTestSecurityGroup("nsg3")}
	responseBody, err := json.Marshal(network.SecurityGroupListResult{Value: &nsgList})
	assert.NoError(t, err)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(
		&http.Response{
			StatusCode: http.StatusNotFound,
			Body:       ioutil.NopCloser(bytes.NewReader(responseBody)),
		}, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)
	nsgClient := getTestSecurityGroupClient(armClient)
	result, rerr := nsgClient.List(context.TODO(), "rg")
	assert.NotNil(t, rerr)
	assert.Equal(t, 0, len(result))
}

func TestListWithNextPage(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Network/networkSecurityGroups"
	armClient := mockarmclient.NewMockInterface(ctrl)
	nsgList := []network.SecurityGroup{getTestSecurityGroup("nsg1"), getTestSecurityGroup("nsg2"), getTestSecurityGroup("nsg3")}
	partialResponse, err := json.Marshal(network.SecurityGroupListResult{Value: &nsgList, NextLink: to.StringPtr("nextLink")})
	assert.NoError(t, err)
	pagedResponse, err := json.Marshal(network.SecurityGroupListResult{Value: &nsgList})
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
	nsgClient := getTestSecurityGroupClient(armClient)
	result, rerr := nsgClient.List(context.TODO(), "rg")
	assert.Nil(t, rerr)
	assert.Equal(t, 6, len(result))
}

func TestListNeverRateLimiter(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	nsgListErr := &retry.Error{
		RawError:  fmt.Errorf("azure cloud provider rate limited(%s) for operation %q", "read", "NSGList"),
		Retriable: true,
	}

	armClient := mockarmclient.NewMockInterface(ctrl)
	nsgClient := getTestSecurityGroupClientWithNeverRateLimiter(armClient)
	result, rerr := nsgClient.List(context.TODO(), "rg")
	assert.Equal(t, 0, len(result))
	assert.NotNil(t, rerr)
	assert.Equal(t, nsgListErr, rerr)
}

func TestListRetryAfterReader(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	nsgListErr := &retry.Error{
		RawError:   fmt.Errorf("azure cloud provider throttled for operation %s with reason %q", "NSGList", "client throttled"),
		Retriable:  true,
		RetryAfter: getFutureTime(),
	}

	armClient := mockarmclient.NewMockInterface(ctrl)
	nsgClient := getTestSecurityGroupClientWithRetryAfterReader(armClient)
	result, rerr := nsgClient.List(context.TODO(), "rg")
	assert.Equal(t, 0, len(result))
	assert.NotNil(t, rerr)
	assert.Equal(t, nsgListErr, rerr)
}

func TestListThrottle(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Network/networkSecurityGroups"
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

	nsgClient := getTestSecurityGroupClient(armClient)
	result, rerr := nsgClient.List(context.TODO(), "rg")
	assert.Empty(t, result)
	assert.NotNil(t, rerr)
	assert.Equal(t, throttleErr, rerr)
}

func TestCreateOrUpdate(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	nsg := getTestSecurityGroup("nsg1")
	armClient := mockarmclient.NewMockInterface(ctrl)
	response := &http.Response{
		StatusCode: http.StatusOK,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte(""))),
	}
	armClient.EXPECT().PutResourceWithDecorators(gomock.Any(), to.String(nsg.ID), nsg, gomock.Any()).Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	nsgClient := getTestSecurityGroupClient(armClient)
	rerr := nsgClient.CreateOrUpdate(context.TODO(), "rg", "nsg1", nsg, "*")
	assert.Nil(t, rerr)
}

func TestCreateOrUpdateWithCreateOrUpdateResponderError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	nsg := getTestSecurityGroup("nsg1")
	armClient := mockarmclient.NewMockInterface(ctrl)
	response := &http.Response{
		StatusCode: http.StatusNotFound,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte(""))),
	}
	armClient.EXPECT().PutResourceWithDecorators(gomock.Any(), to.String(nsg.ID), nsg, gomock.Any()).Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	nsgClient := getTestSecurityGroupClient(armClient)
	rerr := nsgClient.CreateOrUpdate(context.TODO(), "rg", "nsg1", nsg, "")
	assert.NotNil(t, rerr)
}

func TestCreateOrUpdateNeverRateLimiter(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	nsgCreateOrUpdateErr := &retry.Error{
		RawError:  fmt.Errorf("azure cloud provider rate limited(%s) for operation %q", "write", "NSGCreateOrUpdate"),
		Retriable: true,
	}

	armClient := mockarmclient.NewMockInterface(ctrl)

	nsgClient := getTestSecurityGroupClientWithNeverRateLimiter(armClient)
	nsg := getTestSecurityGroup("nsg1")
	rerr := nsgClient.CreateOrUpdate(context.TODO(), "rg", "nsg1", nsg, "")
	assert.NotNil(t, rerr)
	assert.Equal(t, nsgCreateOrUpdateErr, rerr)
}

func TestCreateOrUpdateRetryAfterReader(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	nsgCreateOrUpdateErr := &retry.Error{
		RawError:   fmt.Errorf("azure cloud provider throttled for operation %s with reason %q", "NSGCreateOrUpdate", "client throttled"),
		Retriable:  true,
		RetryAfter: getFutureTime(),
	}

	nsg := getTestSecurityGroup("nsg1")
	armClient := mockarmclient.NewMockInterface(ctrl)

	nsgClient := getTestSecurityGroupClientWithRetryAfterReader(armClient)
	rerr := nsgClient.CreateOrUpdate(context.TODO(), "rg", "nsg1", nsg, "")
	assert.NotNil(t, rerr)
	assert.Equal(t, nsgCreateOrUpdateErr, rerr)
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

	nsg := getTestSecurityGroup("nsg1")
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().PutResourceWithDecorators(gomock.Any(), to.String(nsg.ID), nsg, gomock.Any()).Return(response, throttleErr).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	nsgClient := getTestSecurityGroupClient(armClient)
	rerr := nsgClient.CreateOrUpdate(context.TODO(), "rg", "nsg1", nsg, "")
	assert.NotNil(t, rerr)
	assert.Equal(t, throttleErr, rerr)
}

func TestDelete(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	r := getTestSecurityGroup("nsg1")
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().DeleteResource(gomock.Any(), to.String(r.ID), "").Return(nil).Times(1)

	rtClient := getTestSecurityGroupClient(armClient)
	rerr := rtClient.Delete(context.TODO(), "rg", "nsg1")
	assert.Nil(t, rerr)
}

func TestDeleteNeverRateLimiter(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	nsgDeleteErr := &retry.Error{
		RawError:  fmt.Errorf("azure cloud provider rate limited(%s) for operation %q", "write", "NSGDelete"),
		Retriable: true,
	}

	armClient := mockarmclient.NewMockInterface(ctrl)

	nsgClient := getTestSecurityGroupClientWithNeverRateLimiter(armClient)
	rerr := nsgClient.Delete(context.TODO(), "rg", "nsg1")
	assert.NotNil(t, rerr)
	assert.Equal(t, nsgDeleteErr, rerr)
}

func TestDeleteRetryAfterReader(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	nsgDeleteErr := &retry.Error{
		RawError:   fmt.Errorf("azure cloud provider throttled for operation %s with reason %q", "NSGDelete", "client throttled"),
		Retriable:  true,
		RetryAfter: getFutureTime(),
	}

	armClient := mockarmclient.NewMockInterface(ctrl)

	nsgClient := getTestSecurityGroupClientWithRetryAfterReader(armClient)
	rerr := nsgClient.Delete(context.TODO(), "rg", "nsg1")
	assert.NotNil(t, rerr)
	assert.Equal(t, nsgDeleteErr, rerr)
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

	nsg := getTestSecurityGroup("nsg1")
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().DeleteResource(gomock.Any(), to.String(nsg.ID), "").Return(throttleErr).Times(1)

	nsgClient := getTestSecurityGroupClient(armClient)
	rerr := nsgClient.Delete(context.TODO(), "rg", "nsg1")
	assert.NotNil(t, rerr)
	assert.Equal(t, throttleErr, rerr)
}

func getTestSecurityGroup(name string) network.SecurityGroup {
	return network.SecurityGroup{
		ID:       to.StringPtr(fmt.Sprintf("/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Network/networkSecurityGroups/%s", name)),
		Name:     to.StringPtr(name),
		Location: to.StringPtr("eastus"),
	}
}

func getTestSecurityGroupClient(armClient armclient.Interface) *Client {
	rateLimiterReader, rateLimiterWriter := azclients.NewRateLimiter(&azclients.RateLimitConfig{})
	return &Client{
		armClient:         armClient,
		subscriptionID:    "subscriptionID",
		rateLimiterReader: rateLimiterReader,
		rateLimiterWriter: rateLimiterWriter,
	}
}

func getTestSecurityGroupClientWithNeverRateLimiter(armClient armclient.Interface) *Client {
	rateLimiterReader := flowcontrol.NewFakeNeverRateLimiter()
	rateLimiterWriter := flowcontrol.NewFakeNeverRateLimiter()
	return &Client{
		armClient:         armClient,
		subscriptionID:    "subscriptionID",
		rateLimiterReader: rateLimiterReader,
		rateLimiterWriter: rateLimiterWriter,
	}
}

func getTestSecurityGroupClientWithRetryAfterReader(armClient armclient.Interface) *Client {
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
