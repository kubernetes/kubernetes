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

package subnetclient

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

	subnetClient := New(config)
	assert.Equal(t, "sub", subnetClient.subscriptionID)
	assert.NotEmpty(t, subnetClient.rateLimiterReader)
	assert.NotEmpty(t, subnetClient.rateLimiterWriter)
}

func TestGet(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Network/virtualNetworks/vnet/subnets/subnet1"
	testSubnet := network.Subnet{
		Name: to.StringPtr("subnet1"),
	}
	subnet, err := testSubnet.MarshalJSON()
	assert.Nil(t, err)
	response := &http.Response{
		StatusCode: http.StatusOK,
		Body:       ioutil.NopCloser(bytes.NewReader(subnet)),
	}

	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	expected := network.Subnet{
		Response: autorest.Response{Response: response},
		Name:     to.StringPtr("subnet1"),
	}
	subnetClient := getTestSubnetClient(armClient)
	result, rerr := subnetClient.Get(context.TODO(), "rg", "vnet", "subnet1", "")
	assert.Equal(t, expected, result)
	assert.Nil(t, rerr)
}

func TestGetNotFound(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Network/virtualNetworks/vnet/subnets/subnet1"
	response := &http.Response{
		StatusCode: http.StatusNotFound,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
	}
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	subnetClient := getTestSubnetClient(armClient)
	expected := network.Subnet{Response: autorest.Response{}}
	result, rerr := subnetClient.Get(context.TODO(), "rg", "vnet", "subnet1", "")
	assert.Equal(t, expected, result)
	assert.NotNil(t, rerr)
	assert.Equal(t, http.StatusNotFound, rerr.HTTPStatusCode)
}

func TestGetInternalError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Network/virtualNetworks/vnet/subnets/subnet1"
	response := &http.Response{
		StatusCode: http.StatusInternalServerError,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
	}
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	subnetClient := getTestSubnetClient(armClient)
	expected := network.Subnet{Response: autorest.Response{}}
	result, rerr := subnetClient.Get(context.TODO(), "rg", "vnet", "subnet1", "")
	assert.Equal(t, expected, result)
	assert.NotNil(t, rerr)
	assert.Equal(t, http.StatusInternalServerError, rerr.HTTPStatusCode)
}

func TestGetNeverRateLimiter(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	subnetGetErr := &retry.Error{
		RawError:  fmt.Errorf("azure cloud provider rate limited(%s) for operation %q", "read", "SubnetGet"),
		Retriable: true,
	}

	armClient := mockarmclient.NewMockInterface(ctrl)
	subnetClient := getTestSubnetClientWithNeverRateLimiter(armClient)
	expected := network.Subnet{}
	result, rerr := subnetClient.Get(context.TODO(), "rg", "vnet", "subnet1", "")
	assert.Equal(t, expected, result)
	assert.Equal(t, subnetGetErr, rerr)
}

func TestGetRetryAfterReader(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	subnetGetErr := &retry.Error{
		RawError:   fmt.Errorf("azure cloud provider throttled for operation %s with reason %q", "SubnetGet", "client throttled"),
		Retriable:  true,
		RetryAfter: getFutureTime(),
	}

	armClient := mockarmclient.NewMockInterface(ctrl)
	subnetClient := getTestSubnetClientWithRetryAfterReader(armClient)
	expected := network.Subnet{}
	result, rerr := subnetClient.Get(context.TODO(), "rg", "vnet", "subnet1", "")
	assert.Equal(t, expected, result)
	assert.Equal(t, subnetGetErr, rerr)
}

func TestGetThrottle(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Network/virtualNetworks/vnet/subnets/subnet1"
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

	subnetClient := getTestSubnetClient(armClient)
	result, rerr := subnetClient.Get(context.TODO(), "rg", "vnet", "subnet1", "")
	assert.Empty(t, result)
	assert.Equal(t, throttleErr, rerr)
}

func TestList(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Network/virtualNetworks/vnet/subnets"
	armClient := mockarmclient.NewMockInterface(ctrl)
	subnetList := []network.Subnet{getTestSubnet("subnet1"), getTestSubnet("subnet2"), getTestSubnet("subnet3")}
	responseBody, err := json.Marshal(network.SubnetListResult{Value: &subnetList})
	assert.Nil(t, err)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(
		&http.Response{
			StatusCode: http.StatusOK,
			Body:       ioutil.NopCloser(bytes.NewReader(responseBody)),
		}, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	subnetClient := getTestSubnetClient(armClient)
	result, rerr := subnetClient.List(context.TODO(), "rg", "vnet")
	assert.Nil(t, rerr)
	assert.Equal(t, 3, len(result))
}

func TestListNotFound(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Network/virtualNetworks/vnet/subnets"
	response := &http.Response{
		StatusCode: http.StatusNotFound,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
	}
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	subnetClient := getTestSubnetClient(armClient)
	expected := []network.Subnet{}
	result, rerr := subnetClient.List(context.TODO(), "rg", "vnet")
	assert.Equal(t, expected, result)
	assert.NotNil(t, rerr)
	assert.Equal(t, http.StatusNotFound, rerr.HTTPStatusCode)
}

func TestListInternalError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Network/virtualNetworks/vnet/subnets"
	response := &http.Response{
		StatusCode: http.StatusInternalServerError,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
	}
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	subnetClient := getTestSubnetClient(armClient)
	expected := []network.Subnet{}
	result, rerr := subnetClient.List(context.TODO(), "rg", "vnet")
	assert.Equal(t, expected, result)
	assert.NotNil(t, rerr)
	assert.Equal(t, http.StatusInternalServerError, rerr.HTTPStatusCode)
}

func TestListThrottle(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Network/virtualNetworks/vnet/subnets"
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

	subnetClient := getTestSubnetClient(armClient)
	result, rerr := subnetClient.List(context.TODO(), "rg", "vnet")
	assert.Empty(t, result)
	assert.NotNil(t, rerr)
	assert.Equal(t, throttleErr, rerr)
}

func TestListWithListResponderError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Network/virtualNetworks/vnet/subnets"
	armClient := mockarmclient.NewMockInterface(ctrl)
	subnetList := []network.Subnet{getTestSubnet("subnet1"), getTestSubnet("subnet2"), getTestSubnet("subnet3")}
	responseBody, err := json.Marshal(network.SubnetListResult{Value: &subnetList})
	assert.Nil(t, err)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(
		&http.Response{
			StatusCode: http.StatusNotFound,
			Body:       ioutil.NopCloser(bytes.NewReader(responseBody)),
		}, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)
	subnetClient := getTestSubnetClient(armClient)
	result, rerr := subnetClient.List(context.TODO(), "rg", "vnet")
	assert.NotNil(t, rerr)
	assert.Equal(t, 0, len(result))
}

func TestListNeverRateLimiter(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	subnetListErr := &retry.Error{
		RawError:  fmt.Errorf("azure cloud provider rate limited(%s) for operation %q", "read", "SubnetList"),
		Retriable: true,
	}

	armClient := mockarmclient.NewMockInterface(ctrl)
	subnetClient := getTestSubnetClientWithNeverRateLimiter(armClient)
	result, rerr := subnetClient.List(context.TODO(), "rg", "vnet")
	assert.Equal(t, 0, len(result))
	assert.NotNil(t, rerr)
	assert.Equal(t, subnetListErr, rerr)
}

func TestListRetryAfterReader(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	subnetListErr := &retry.Error{
		RawError:   fmt.Errorf("azure cloud provider throttled for operation %s with reason %q", "SubnetList", "client throttled"),
		Retriable:  true,
		RetryAfter: getFutureTime(),
	}

	armClient := mockarmclient.NewMockInterface(ctrl)
	subnetClient := getTestSubnetClientWithRetryAfterReader(armClient)
	result, rerr := subnetClient.List(context.TODO(), "rg", "vnet")
	assert.Equal(t, 0, len(result))
	assert.NotNil(t, rerr)
	assert.Equal(t, subnetListErr, rerr)
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

	lastResult := network.SubnetListResult{
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

		subnetClient := getTestSubnetClient(armClient)
		result, err := subnetClient.listNextResults(context.TODO(), lastResult)
		if test.prepareErr != nil || test.sendErr != nil {
			assert.NotNil(t, err)
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

	lastResult := network.SubnetListResult{
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
	expected := network.SubnetListResult{}
	expected.Response = autorest.Response{Response: response}
	subnetClient := getTestSubnetClient(armClient)
	result, err := subnetClient.listNextResults(context.TODO(), lastResult)
	assert.NotNil(t, err)
	assert.Equal(t, expected, result)
}

func TestCreateOrUpdate(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	subnet := getTestSubnet("subnet1")
	armClient := mockarmclient.NewMockInterface(ctrl)
	response := &http.Response{
		StatusCode: http.StatusOK,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte(""))),
	}
	armClient.EXPECT().PutResource(gomock.Any(), to.String(subnet.ID), subnet).Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	subnetClient := getTestSubnetClient(armClient)
	rerr := subnetClient.CreateOrUpdate(context.TODO(), "rg", "vnet", "subnet1", subnet)
	assert.Nil(t, rerr)
}

func TestCreateOrUpdateWithCreateOrUpdateResponderError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	subnet := getTestSubnet("subnet1")
	armClient := mockarmclient.NewMockInterface(ctrl)
	response := &http.Response{
		StatusCode: http.StatusNotFound,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte(""))),
	}
	armClient.EXPECT().PutResource(gomock.Any(), to.String(subnet.ID), subnet).Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	subnetClient := getTestSubnetClient(armClient)
	rerr := subnetClient.CreateOrUpdate(context.TODO(), "rg", "vnet", "subnet1", subnet)
	assert.NotNil(t, rerr)
}

func TestCreateOrUpdateNeverRateLimiter(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	subnetCreateOrUpdateErr := &retry.Error{
		RawError:  fmt.Errorf("azure cloud provider rate limited(%s) for operation %q", "write", "SubnetCreateOrUpdate"),
		Retriable: true,
	}

	armClient := mockarmclient.NewMockInterface(ctrl)
	subnetClient := getTestSubnetClientWithNeverRateLimiter(armClient)
	subnet := getTestSubnet("subnet1")
	rerr := subnetClient.CreateOrUpdate(context.TODO(), "rg", "vnet", "subnet1", subnet)
	assert.NotNil(t, rerr)
	assert.Equal(t, subnetCreateOrUpdateErr, rerr)
}

func TestCreateOrUpdateRetryAfterReader(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	subnetCreateOrUpdateErr := &retry.Error{
		RawError:   fmt.Errorf("azure cloud provider throttled for operation %s with reason %q", "SubnetCreateOrUpdate", "client throttled"),
		Retriable:  true,
		RetryAfter: getFutureTime(),
	}

	subnet := getTestSubnet("subnet1")
	armClient := mockarmclient.NewMockInterface(ctrl)
	subnetClient := getTestSubnetClientWithRetryAfterReader(armClient)
	rerr := subnetClient.CreateOrUpdate(context.TODO(), "rg", "vnet", "subnet1", subnet)
	assert.NotNil(t, rerr)
	assert.Equal(t, subnetCreateOrUpdateErr, rerr)
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

	subnet := getTestSubnet("subnet1")
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().PutResource(gomock.Any(), to.String(subnet.ID), subnet).Return(response, throttleErr).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	subnetClient := getTestSubnetClient(armClient)
	rerr := subnetClient.CreateOrUpdate(context.TODO(), "rg", "vnet", "subnet1", subnet)
	assert.NotNil(t, rerr)
	assert.Equal(t, throttleErr, rerr)
}

func TestDelete(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	r := getTestSubnet("subnet1")
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().DeleteResource(gomock.Any(), to.String(r.ID), "").Return(nil).Times(1)

	subnetClient := getTestSubnetClient(armClient)
	rerr := subnetClient.Delete(context.TODO(), "rg", "vnet", "subnet1")
	assert.Nil(t, rerr)
}

func TestDeleteNeverRateLimiter(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	subnetDeleteErr := &retry.Error{
		RawError:  fmt.Errorf("azure cloud provider rate limited(%s) for operation %q", "write", "SubnetDelete"),
		Retriable: true,
	}

	armClient := mockarmclient.NewMockInterface(ctrl)
	subnetClient := getTestSubnetClientWithNeverRateLimiter(armClient)
	rerr := subnetClient.Delete(context.TODO(), "rg", "vnet", "subnet1")
	assert.NotNil(t, rerr)
	assert.Equal(t, subnetDeleteErr, rerr)
}

func TestDeleteRetryAfterReader(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	subnetDeleteErr := &retry.Error{
		RawError:   fmt.Errorf("azure cloud provider throttled for operation %s with reason %q", "SubnetDelete", "client throttled"),
		Retriable:  true,
		RetryAfter: getFutureTime(),
	}

	armClient := mockarmclient.NewMockInterface(ctrl)
	subnetClient := getTestSubnetClientWithRetryAfterReader(armClient)
	rerr := subnetClient.Delete(context.TODO(), "rg", "vnet", "subnet1")
	assert.NotNil(t, rerr)
	assert.Equal(t, subnetDeleteErr, rerr)
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

	subnet := getTestSubnet("subnet1")
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().DeleteResource(gomock.Any(), to.String(subnet.ID), "").Return(throttleErr).Times(1)

	subnetClient := getTestSubnetClient(armClient)
	rerr := subnetClient.Delete(context.TODO(), "rg", "vnet", "subnet1")
	assert.NotNil(t, rerr)
	assert.Equal(t, throttleErr, rerr)
}

func getTestSubnet(name string) network.Subnet {
	return network.Subnet{
		ID:   to.StringPtr(fmt.Sprintf("/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Network/virtualNetworks/vnet/subnets/%s", name)),
		Name: to.StringPtr(name),
	}
}

func getTestSubnetClient(armClient armclient.Interface) *Client {
	rateLimiterReader, rateLimiterWriter := azclients.NewRateLimiter(&azclients.RateLimitConfig{})
	return &Client{
		armClient:         armClient,
		subscriptionID:    "subscriptionID",
		rateLimiterReader: rateLimiterReader,
		rateLimiterWriter: rateLimiterWriter,
	}
}

func getTestSubnetClientWithNeverRateLimiter(armClient armclient.Interface) *Client {
	rateLimiterReader := flowcontrol.NewFakeNeverRateLimiter()
	rateLimiterWriter := flowcontrol.NewFakeNeverRateLimiter()
	return &Client{
		armClient:         armClient,
		subscriptionID:    "subscriptionID",
		rateLimiterReader: rateLimiterReader,
		rateLimiterWriter: rateLimiterWriter,
	}
}

func getTestSubnetClientWithRetryAfterReader(armClient armclient.Interface) *Client {
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

func getFutureTime() time.Time {
	return time.Unix(3000000000, 0)
}
