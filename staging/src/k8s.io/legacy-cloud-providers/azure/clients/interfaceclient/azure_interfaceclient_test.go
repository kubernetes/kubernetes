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

package interfaceclient

import (
	"bytes"
	"context"
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

	interfaceClient := New(config)
	assert.Equal(t, "sub", interfaceClient.subscriptionID)
	assert.NotEmpty(t, interfaceClient.rateLimiterReader)
	assert.NotEmpty(t, interfaceClient.rateLimiterWriter)
}

func TestGetNotFound(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Network/networkInterfaces/nic1"
	response := &http.Response{
		StatusCode: http.StatusNotFound,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
	}
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	nicClient := getTestInterfaceClient(armClient)
	expected := network.Interface{Response: autorest.Response{}}
	result, rerr := nicClient.Get(context.TODO(), "rg", "nic1", "")
	assert.Equal(t, expected, result)
	assert.NotNil(t, rerr)
	assert.Equal(t, http.StatusNotFound, rerr.HTTPStatusCode)
}

func TestGetInternalError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Network/networkInterfaces/nic1"
	response := &http.Response{
		StatusCode: http.StatusInternalServerError,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
	}
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	nicClient := getTestInterfaceClient(armClient)
	expected := network.Interface{Response: autorest.Response{}}
	result, rerr := nicClient.Get(context.TODO(), "rg", "nic1", "")
	assert.Equal(t, expected, result)
	assert.NotNil(t, rerr)
	assert.Equal(t, http.StatusInternalServerError, rerr.HTTPStatusCode)
}

func TestGetThrottle(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Network/networkInterfaces/nic1"
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

	nicClient := getTestInterfaceClient(armClient)
	result, rerr := nicClient.Get(context.TODO(), "rg", "nic1", "")
	assert.Empty(t, result)
	assert.Equal(t, throttleErr, rerr)
}

func TestGetVirtualMachineScaleSetNetworkInterface(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss/virtualMachines/0/networkInterfaces/nic1"
	testInterface := getTestVMSSInterface("nic1")
	networkInterface, err := testInterface.MarshalJSON()
	assert.Nil(t, err)
	response := &http.Response{
		StatusCode: http.StatusOK,
		Body:       ioutil.NopCloser(bytes.NewReader(networkInterface)),
	}
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().GetResourceWithDecorators(gomock.Any(), resourceID, gomock.Any()).Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	nicClient := getTestInterfaceClient(armClient)
	expected := getTestVMSSInterface("nic1")
	expected.Response = autorest.Response{Response: response}
	result, rerr := nicClient.GetVirtualMachineScaleSetNetworkInterface(context.TODO(), "rg", "vmss", "0", "nic1", "")
	assert.Equal(t, expected, result)
	assert.Nil(t, rerr)

	response = &http.Response{
		StatusCode: http.StatusTooManyRequests,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
	}
	throttleErr := &retry.Error{
		HTTPStatusCode: http.StatusTooManyRequests,
		RawError:       fmt.Errorf("error"),
		Retriable:      true,
		RetryAfter:     time.Unix(100, 0),
	}

	armClient.EXPECT().GetResourceWithDecorators(gomock.Any(), resourceID, gomock.Any()).Return(response, throttleErr).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)
	result, rerr = nicClient.GetVirtualMachineScaleSetNetworkInterface(context.TODO(), "rg", "vmss", "0", "nic1", "test")
	assert.Empty(t, result)
	assert.Equal(t, throttleErr, rerr)
}

func TestCreateOrUpdate(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	testInterface := getTestInterface("nic1")
	armClient := mockarmclient.NewMockInterface(ctrl)
	response := &http.Response{
		StatusCode: http.StatusOK,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte(""))),
	}
	armClient.EXPECT().PutResource(gomock.Any(), to.String(testInterface.ID), testInterface).Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	nicClient := getTestInterfaceClient(armClient)
	rerr := nicClient.CreateOrUpdate(context.TODO(), "rg", "nic1", testInterface)
	assert.Nil(t, rerr)

	response = &http.Response{
		StatusCode: http.StatusNoContent,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
	}
	noContentErr := &retry.Error{
		HTTPStatusCode: http.StatusNoContent,
		RawError:       fmt.Errorf("error"),
		Retriable:      true,
		RetryAfter:     time.Unix(100, 0),
	}

	armClient.EXPECT().PutResource(gomock.Any(), to.String(testInterface.ID), testInterface).Return(response, noContentErr).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)
	rerr = nicClient.CreateOrUpdate(context.TODO(), "rg", "nic1", testInterface)
	assert.Equal(t, noContentErr, rerr)
}

func TestDelete(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	r := getTestInterface("interface1")
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().DeleteResource(gomock.Any(), to.String(r.ID), "").Return(nil).Times(1)

	nicClient := getTestInterfaceClient(armClient)
	rerr := nicClient.Delete(context.TODO(), "rg", "interface1")
	assert.Nil(t, rerr)

	noContentErr := &retry.Error{
		HTTPStatusCode: http.StatusNoContent,
		RawError:       fmt.Errorf("error"),
		Retriable:      true,
		RetryAfter:     time.Unix(100, 0),
	}
	armClient.EXPECT().DeleteResource(gomock.Any(), to.String(r.ID), "").Return(noContentErr).Times(1)

	rerr = nicClient.Delete(context.TODO(), "rg", "interface1")
	assert.Equal(t, noContentErr, rerr)
}

func getTestInterface(name string) network.Interface {
	resourceID := fmt.Sprintf("/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Network/networkInterfaces/%s", name)
	return network.Interface{
		ID:       to.StringPtr(resourceID),
		Name:     to.StringPtr(name),
		Location: to.StringPtr("eastus"),
	}
}

func getTestVMSSInterface(name string) network.Interface {
	resourceID := fmt.Sprintf("/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss/virtualMachines/0/networkInterfaces/%s", name)
	return network.Interface{
		ID:       to.StringPtr(resourceID),
		Location: to.StringPtr("eastus"),
		InterfacePropertiesFormat: &network.InterfacePropertiesFormat{
			Primary: to.BoolPtr(true),
		},
	}
}

func getTestInterfaceClient(armClient armclient.Interface) *Client {
	rateLimiterReader, rateLimiterWriter := azclients.NewRateLimiter(&azclients.RateLimitConfig{})
	return &Client{
		armClient:         armClient,
		subscriptionID:    "subscriptionID",
		rateLimiterReader: rateLimiterReader,
		rateLimiterWriter: rateLimiterWriter,
	}
}
