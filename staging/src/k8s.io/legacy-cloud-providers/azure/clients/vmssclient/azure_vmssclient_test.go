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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"testing"
	"time"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-12-01/compute"
	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/azure"
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

	vmssClient := New(config)
	assert.Equal(t, "sub", vmssClient.subscriptionID)
	assert.NotEmpty(t, vmssClient.rateLimiterReader)
	assert.NotEmpty(t, vmssClient.rateLimiterWriter)
}

func TestGet(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss1"
	response := &http.Response{
		StatusCode: http.StatusOK,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
	}

	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	expected := compute.VirtualMachineScaleSet{Response: autorest.Response{Response: response}}
	vmssClient := getTestVMSSClient(armClient)
	result, rerr := vmssClient.Get(context.TODO(), "rg", "vmss1")
	assert.Equal(t, expected, result)
	assert.Nil(t, rerr)
}

func TestGetNeverRateLimiter(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	vmssGetErr := &retry.Error{
		RawError:  fmt.Errorf("azure cloud provider rate limited(%s) for operation %q", "read", "VMSSGet"),
		Retriable: true,
	}

	armClient := mockarmclient.NewMockInterface(ctrl)
	vmssClient := getTestVMSSClientWithNeverRateLimiter(armClient)
	expected := compute.VirtualMachineScaleSet{}
	result, rerr := vmssClient.Get(context.TODO(), "rg", "vmss1")
	assert.Equal(t, expected, result)
	assert.Equal(t, vmssGetErr, rerr)
}

func TestGetRetryAfterReader(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	vmssGetErr := &retry.Error{
		RawError:   fmt.Errorf("azure cloud provider throttled for operation %s with reason %q", "VMSSGet", "client throttled"),
		Retriable:  true,
		RetryAfter: getFutureTime(),
	}

	armClient := mockarmclient.NewMockInterface(ctrl)
	vmssClient := getTestVMSSClientWithRetryAfterReader(armClient)
	expected := compute.VirtualMachineScaleSet{}
	result, rerr := vmssClient.Get(context.TODO(), "rg", "vmss1")
	assert.Equal(t, expected, result)
	assert.Equal(t, vmssGetErr, rerr)
}

func TestGetNotFound(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss1"
	response := &http.Response{
		StatusCode: http.StatusNotFound,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
	}
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	vmssClient := getTestVMSSClient(armClient)
	expectedVMSS := compute.VirtualMachineScaleSet{Response: autorest.Response{}}
	result, rerr := vmssClient.Get(context.TODO(), "rg", "vmss1")
	assert.Equal(t, expectedVMSS, result)
	assert.NotNil(t, rerr)
	assert.Equal(t, http.StatusNotFound, rerr.HTTPStatusCode)
}

func TestGetInternalError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss1"
	response := &http.Response{
		StatusCode: http.StatusInternalServerError,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
	}
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	vmssClient := getTestVMSSClient(armClient)
	expectedVMSS := compute.VirtualMachineScaleSet{Response: autorest.Response{}}
	result, rerr := vmssClient.Get(context.TODO(), "rg", "vmss1")
	assert.Equal(t, expectedVMSS, result)
	assert.NotNil(t, rerr)
	assert.Equal(t, http.StatusInternalServerError, rerr.HTTPStatusCode)
}

func TestGetThrottle(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss1"
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

	vmssClient := getTestVMSSClient(armClient)
	result, rerr := vmssClient.Get(context.TODO(), "rg", "vmss1")
	assert.Empty(t, result)
	assert.Equal(t, throttleErr, rerr)
}

func TestList(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets"
	armClient := mockarmclient.NewMockInterface(ctrl)
	vmssList := []compute.VirtualMachineScaleSet{getTestVMSS("vmss1"), getTestVMSS("vmss2"), getTestVMSS("vmss3")}
	responseBody, err := json.Marshal(compute.VirtualMachineScaleSetListResult{Value: &vmssList})
	assert.NoError(t, err)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(
		&http.Response{
			StatusCode: http.StatusOK,
			Body:       ioutil.NopCloser(bytes.NewReader(responseBody)),
		}, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	vmssClient := getTestVMSSClient(armClient)
	result, rerr := vmssClient.List(context.TODO(), "rg")
	assert.Nil(t, rerr)
	assert.Equal(t, 3, len(result))
}

func TestListNotFound(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets"
	response := &http.Response{
		StatusCode: http.StatusNotFound,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
	}
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	vmssClient := getTestVMSSClient(armClient)
	expected := []compute.VirtualMachineScaleSet{}
	result, rerr := vmssClient.List(context.TODO(), "rg")
	assert.Equal(t, expected, result)
	assert.NotNil(t, rerr)
	assert.Equal(t, http.StatusNotFound, rerr.HTTPStatusCode)
}

func TestListInternalError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets"
	response := &http.Response{
		StatusCode: http.StatusInternalServerError,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
	}
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	vmssClient := getTestVMSSClient(armClient)
	expected := []compute.VirtualMachineScaleSet{}
	result, rerr := vmssClient.List(context.TODO(), "rg")
	assert.Equal(t, expected, result)
	assert.NotNil(t, rerr)
	assert.Equal(t, http.StatusInternalServerError, rerr.HTTPStatusCode)
}

func TestListThrottle(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets"
	armClient := mockarmclient.NewMockInterface(ctrl)
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
	vmssClient := getTestVMSSClient(armClient)
	result, rerr := vmssClient.List(context.TODO(), "rg")
	assert.Empty(t, result)
	assert.NotNil(t, rerr)
	assert.Equal(t, throttleErr, rerr)
}

func TestListWithListResponderError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets"
	armClient := mockarmclient.NewMockInterface(ctrl)
	vmssList := []compute.VirtualMachineScaleSet{getTestVMSS("vmss1"), getTestVMSS("vmss2"), getTestVMSS("vmss3")}
	responseBody, err := json.Marshal(compute.VirtualMachineScaleSetListResult{Value: &vmssList})
	assert.NoError(t, err)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(
		&http.Response{
			StatusCode: http.StatusNotFound,
			Body:       ioutil.NopCloser(bytes.NewReader(responseBody)),
		}, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)
	vmssClient := getTestVMSSClient(armClient)
	result, rerr := vmssClient.List(context.TODO(), "rg")
	assert.NotNil(t, rerr)
	assert.Equal(t, 0, len(result))
}

func TestListWithNextPage(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets"
	armClient := mockarmclient.NewMockInterface(ctrl)
	vmssList := []compute.VirtualMachineScaleSet{getTestVMSS("vmss1"), getTestVMSS("vmss2"), getTestVMSS("vmss3")}
	partialResponse, err := json.Marshal(compute.VirtualMachineScaleSetListResult{Value: &vmssList, NextLink: to.StringPtr("nextLink")})
	assert.NoError(t, err)
	pagedResponse, err := json.Marshal(compute.VirtualMachineScaleSetListResult{Value: &vmssList})
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
	vmssClient := getTestVMSSClient(armClient)
	result, rerr := vmssClient.List(context.TODO(), "rg")
	assert.Nil(t, rerr)
	assert.Equal(t, 6, len(result))
}

func TestListNeverRateLimiter(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	vmssListErr := &retry.Error{
		RawError:  fmt.Errorf("azure cloud provider rate limited(%s) for operation %q", "read", "VMSSList"),
		Retriable: true,
	}

	armClient := mockarmclient.NewMockInterface(ctrl)
	vmssClient := getTestVMSSClientWithNeverRateLimiter(armClient)
	result, rerr := vmssClient.List(context.TODO(), "rg")
	assert.Equal(t, 0, len(result))
	assert.NotNil(t, rerr)
	assert.Equal(t, vmssListErr, rerr)
}

func TestListRetryAfterReader(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	vmssListErr := &retry.Error{
		RawError:   fmt.Errorf("azure cloud provider throttled for operation %s with reason %q", "VMSSList", "client throttled"),
		Retriable:  true,
		RetryAfter: getFutureTime(),
	}

	armClient := mockarmclient.NewMockInterface(ctrl)
	vmssClient := getTestVMSSClientWithRetryAfterReader(armClient)
	result, rerr := vmssClient.List(context.TODO(), "rg")
	assert.Equal(t, 0, len(result))
	assert.NotNil(t, rerr)
	assert.Equal(t, vmssListErr, rerr)
}

func TestListNextResultsMultiPages(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	tests := []struct {
		name           string
		prepareErr     error
		sendErr        *retry.Error
		expectedErrMsg string
	}{
		{
			name:       "testlistNextResultsSuccessful",
			prepareErr: nil,
			sendErr:    nil,
		},
		{
			name:           "testPrepareGetRequestError",
			prepareErr:     fmt.Errorf("error"),
			expectedErrMsg: "Failure preparing next results request",
		},
		{
			name:           "testSendError",
			sendErr:        &retry.Error{RawError: fmt.Errorf("error")},
			expectedErrMsg: "Failure sending next results request",
		},
	}

	lastResult := compute.VirtualMachineScaleSetListResult{
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

		vmssClient := getTestVMSSClient(armClient)
		result, err := vmssClient.listNextResults(context.TODO(), lastResult)
		if err != nil {
			assert.Equal(t, err.(autorest.DetailedError).Message, test.expectedErrMsg)
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

	tests := []struct {
		name       string
		prepareErr error
		sendErr    *retry.Error
	}{
		{
			name:       "testListResponderError",
			prepareErr: nil,
			sendErr:    nil,
		},
		{
			name:    "testSendError",
			sendErr: &retry.Error{RawError: fmt.Errorf("error")},
		},
	}

	lastResult := compute.VirtualMachineScaleSetListResult{
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
				StatusCode: http.StatusNotFound,
				Body:       ioutil.NopCloser(bytes.NewReader([]byte(`{"foo":"bar"}`))),
			}, test.sendErr)
			armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any())
		}

		response := &http.Response{
			StatusCode: http.StatusNotFound,
			Body:       ioutil.NopCloser(bytes.NewBuffer([]byte(`{"foo":"bar"}`))),
		}
		expected := compute.VirtualMachineScaleSetListResult{}
		expected.Response = autorest.Response{Response: response}
		vmssClient := getTestVMSSClient(armClient)
		result, err := vmssClient.listNextResults(context.TODO(), lastResult)
		assert.Error(t, err)
		if test.sendErr != nil {
			assert.NotEqual(t, expected, result)
		} else {
			assert.Equal(t, expected, result)
		}
	}
}

func TestCreateOrUpdate(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	vmss := getTestVMSS("vmss1")
	armClient := mockarmclient.NewMockInterface(ctrl)
	response := &http.Response{
		StatusCode: http.StatusOK,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte(""))),
	}
	armClient.EXPECT().PutResource(gomock.Any(), to.String(vmss.ID), vmss).Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	vmssClient := getTestVMSSClient(armClient)
	rerr := vmssClient.CreateOrUpdate(context.TODO(), "rg", "vmss1", vmss)
	assert.Nil(t, rerr)
}

func TestCreateOrUpdateWithCreateOrUpdateResponderError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	vmss := getTestVMSS("vmss1")
	armClient := mockarmclient.NewMockInterface(ctrl)
	response := &http.Response{
		StatusCode: http.StatusNotFound,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte(""))),
	}
	armClient.EXPECT().PutResource(gomock.Any(), to.String(vmss.ID), vmss).Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	vmssClient := getTestVMSSClient(armClient)
	rerr := vmssClient.CreateOrUpdate(context.TODO(), "rg", "vmss1", vmss)
	assert.NotNil(t, rerr)
}

func TestCreateOrUpdateNeverRateLimiter(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	vmssCreateOrUpdateErr := &retry.Error{
		RawError:  fmt.Errorf("azure cloud provider rate limited(%s) for operation %q", "write", "VMSSCreateOrUpdate"),
		Retriable: true,
	}

	armClient := mockarmclient.NewMockInterface(ctrl)
	vmssClient := getTestVMSSClientWithNeverRateLimiter(armClient)
	vmss := getTestVMSS("vmss1")
	rerr := vmssClient.CreateOrUpdate(context.TODO(), "rg", "vmss1", vmss)
	assert.NotNil(t, rerr)
	assert.Equal(t, vmssCreateOrUpdateErr, rerr)
}

func TestCreateOrUpdateRetryAfterReader(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	vmssCreateOrUpdateErr := &retry.Error{
		RawError:   fmt.Errorf("azure cloud provider throttled for operation %s with reason %q", "VMSSCreateOrUpdate", "client throttled"),
		Retriable:  true,
		RetryAfter: getFutureTime(),
	}

	vmss := getTestVMSS("vmss1")
	armClient := mockarmclient.NewMockInterface(ctrl)
	vmssClient := getTestVMSSClientWithRetryAfterReader(armClient)
	rerr := vmssClient.CreateOrUpdate(context.TODO(), "rg", "vmss1", vmss)
	assert.NotNil(t, rerr)
	assert.Equal(t, vmssCreateOrUpdateErr, rerr)
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

	vmss := getTestVMSS("vmss1")
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().PutResource(gomock.Any(), to.String(vmss.ID), vmss).Return(response, throttleErr).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	vmssClient := getTestVMSSClient(armClient)
	rerr := vmssClient.CreateOrUpdate(context.TODO(), "rg", "vmss1", vmss)
	assert.NotNil(t, rerr)
	assert.Equal(t, throttleErr, rerr)
}

func TestCreateOrUpdateAsync(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	vmss := getTestVMSS("vmss1")
	armClient := mockarmclient.NewMockInterface(ctrl)
	future := &azure.Future{}

	armClient.EXPECT().PutResourceAsync(gomock.Any(), to.String(vmss.ID), vmss).Return(future, nil).Times(1)
	vmssClient := getTestVMSSClient(armClient)
	_, rerr := vmssClient.CreateOrUpdateAsync(context.TODO(), "rg", "vmss1", vmss)
	assert.Nil(t, rerr)

	retryErr := &retry.Error{RawError: fmt.Errorf("error")}
	armClient.EXPECT().PutResourceAsync(gomock.Any(), to.String(vmss.ID), vmss).Return(future, retryErr).Times(1)
	_, rerr = vmssClient.CreateOrUpdateAsync(context.TODO(), "rg", "vmss1", vmss)
	assert.Equal(t, retryErr, rerr)
}

func TestCreateOrUpdateAsyncNeverRateLimiter(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	vmssCreateOrUpdateAsyncErr := &retry.Error{
		RawError:  fmt.Errorf("azure cloud provider rate limited(%s) for operation %q", "write", "VMSSCreateOrUpdateAsync"),
		Retriable: true,
	}

	armClient := mockarmclient.NewMockInterface(ctrl)
	vmssClient := getTestVMSSClientWithNeverRateLimiter(armClient)
	vmss := getTestVMSS("vmss1")
	_, rerr := vmssClient.CreateOrUpdateAsync(context.TODO(), "rg", "vmss1", vmss)
	assert.NotNil(t, rerr)
	assert.Equal(t, vmssCreateOrUpdateAsyncErr, rerr)
}

func TestCreateOrUpdateAsyncRetryAfterReader(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	vmssCreateOrUpdateAsyncErr := &retry.Error{
		RawError:   fmt.Errorf("azure cloud provider throttled for operation %s with reason %q", "VMSSCreateOrUpdateAsync", "client throttled"),
		Retriable:  true,
		RetryAfter: getFutureTime(),
	}

	vmss := getTestVMSS("vmss1")
	armClient := mockarmclient.NewMockInterface(ctrl)
	vmssClient := getTestVMSSClientWithRetryAfterReader(armClient)
	_, rerr := vmssClient.CreateOrUpdateAsync(context.TODO(), "rg", "vmss1", vmss)
	assert.NotNil(t, rerr)
	assert.Equal(t, vmssCreateOrUpdateAsyncErr, rerr)
}

func TestCreateOrUpdateAsyncThrottle(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	throttleErr := &retry.Error{
		HTTPStatusCode: http.StatusTooManyRequests,
		RawError:       fmt.Errorf("error"),
		Retriable:      true,
		RetryAfter:     time.Unix(100, 0),
	}

	vmss := getTestVMSS("vmss1")
	future := &azure.Future{}
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().PutResourceAsync(gomock.Any(), to.String(vmss.ID), vmss).Return(future, throttleErr).Times(1)

	vmssClient := getTestVMSSClient(armClient)
	_, rerr := vmssClient.CreateOrUpdateAsync(context.TODO(), "rg", "vmss1", vmss)
	assert.NotNil(t, rerr)
	assert.Equal(t, throttleErr, rerr)
}

func TestWaitForAsyncOperationResult(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	armClient := mockarmclient.NewMockInterface(ctrl)
	response := &http.Response{
		StatusCode: http.StatusOK,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte(""))),
	}

	armClient.EXPECT().WaitForAsyncOperationResult(gomock.Any(), &azure.Future{}, "VMSSWaitForAsyncOperationResult").Return(response, nil)
	vmssClient := getTestVMSSClient(armClient)
	_, err := vmssClient.WaitForAsyncOperationResult(context.TODO(), &azure.Future{})
	assert.NoError(t, err)
}

func TestDeleteInstances(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	r := getTestVMSS("vmss1")
	vmInstanceIDs := compute.VirtualMachineScaleSetVMInstanceRequiredIDs{
		InstanceIds: &[]string{"0", "1", "2"},
	}
	response := &http.Response{
		StatusCode: http.StatusOK,
		Request:    &http.Request{Method: "POST"},
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
	}
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().PostResource(gomock.Any(), to.String(r.ID), "delete", vmInstanceIDs).Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)
	armClient.EXPECT().WaitForAsyncOperationCompletion(gomock.Any(), gomock.Any(), "vmssclient.DeleteInstances").Return(nil).Times(1)

	client := getTestVMSSClient(armClient)
	rerr := client.DeleteInstances(context.TODO(), "rg", "vmss1", vmInstanceIDs)
	assert.Nil(t, rerr)
}

func TestDeleteInstancesNeverRateLimiter(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	vmInstanceIDs := compute.VirtualMachineScaleSetVMInstanceRequiredIDs{
		InstanceIds: &[]string{"0", "1", "2"},
	}
	vmssDeleteInstancesErr := &retry.Error{
		RawError:  fmt.Errorf("azure cloud provider rate limited(%s) for operation %q", "write", "VMSSDeleteInstances"),
		Retriable: true,
	}

	armClient := mockarmclient.NewMockInterface(ctrl)
	vmssClient := getTestVMSSClientWithNeverRateLimiter(armClient)
	rerr := vmssClient.DeleteInstances(context.TODO(), "rg", "vmss1", vmInstanceIDs)
	assert.NotNil(t, rerr)
	assert.Equal(t, vmssDeleteInstancesErr, rerr)
}

func TestDeleteInstancesRetryAfterReader(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	vmInstanceIDs := compute.VirtualMachineScaleSetVMInstanceRequiredIDs{
		InstanceIds: &[]string{"0", "1", "2"},
	}
	vmssDeleteInstancesErr := &retry.Error{
		RawError:   fmt.Errorf("azure cloud provider throttled for operation %s with reason %q", "VMSSDeleteInstances", "client throttled"),
		Retriable:  true,
		RetryAfter: getFutureTime(),
	}

	armClient := mockarmclient.NewMockInterface(ctrl)
	vmssClient := getTestVMSSClientWithRetryAfterReader(armClient)
	rerr := vmssClient.DeleteInstances(context.TODO(), "rg", "vmss1", vmInstanceIDs)
	assert.NotNil(t, rerr)
	assert.Equal(t, vmssDeleteInstancesErr, rerr)
}

func TestDeleteInstancesThrottle(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	vmss := getTestVMSS("vmss1")
	vmInstanceIDs := compute.VirtualMachineScaleSetVMInstanceRequiredIDs{
		InstanceIds: &[]string{"0", "1", "2"},
	}
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
	armClient.EXPECT().PostResource(gomock.Any(), to.String(vmss.ID), "delete", vmInstanceIDs).Return(response, throttleErr).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	vmssClient := getTestVMSSClient(armClient)
	rerr := vmssClient.DeleteInstances(context.TODO(), "rg", "vmss1", vmInstanceIDs)
	assert.NotNil(t, rerr)
	assert.Equal(t, throttleErr, rerr)
}

func TestDeleteInstancesWaitError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	vmss := getTestVMSS("vmss1")
	vmInstanceIDs := compute.VirtualMachineScaleSetVMInstanceRequiredIDs{
		InstanceIds: &[]string{"0", "1", "2"},
	}
	response := &http.Response{
		StatusCode: http.StatusOK,
		Request:    &http.Request{Method: "POST"},
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
	}
	err := fmt.Errorf("%s", string("Wait error"))
	vmssDeleteInstancesErr := &retry.Error{
		RawError:  err,
		Retriable: false,
	}

	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().PostResource(gomock.Any(), to.String(vmss.ID), "delete", vmInstanceIDs).Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)
	armClient.EXPECT().WaitForAsyncOperationCompletion(gomock.Any(), gomock.Any(), "vmssclient.DeleteInstances").Return(err).Times(1)

	vmssClient := getTestVMSSClient(armClient)
	rerr := vmssClient.DeleteInstances(context.TODO(), "rg", "vmss1", vmInstanceIDs)
	assert.NotNil(t, rerr)
	assert.Equal(t, vmssDeleteInstancesErr, rerr)
}
func TestDeleteInstancesAsync(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	vmss := getTestVMSS("vmss1")
	vmInstanceIDs := compute.VirtualMachineScaleSetVMInstanceRequiredIDs{
		InstanceIds: &[]string{"0", "1", "2"},
	}
	response := &http.Response{
		StatusCode: http.StatusOK,
		Request:    &http.Request{Method: "POST"},
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
	}
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().PostResource(gomock.Any(), to.String(vmss.ID), "delete", vmInstanceIDs).Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	vmssClient := getTestVMSSClient(armClient)
	future, rerr := vmssClient.DeleteInstancesAsync(context.TODO(), "rg", "vmss1", vmInstanceIDs)
	assert.Nil(t, rerr)
	assert.Equal(t, future.Status(), "Succeeded")

	// on error
	retryErr := &retry.Error{RawError: fmt.Errorf("error")}
	armClient.EXPECT().PostResource(gomock.Any(), to.String(vmss.ID), "delete", vmInstanceIDs).Return(&http.Response{StatusCode: http.StatusBadRequest}, retryErr).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)
	_, rerr = vmssClient.DeleteInstancesAsync(context.TODO(), "rg", "vmss1", vmInstanceIDs)
	assert.Equal(t, retryErr, rerr)
}

func getTestVMSS(name string) compute.VirtualMachineScaleSet {
	return compute.VirtualMachineScaleSet{
		ID:       to.StringPtr("/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss1"),
		Name:     to.StringPtr(name),
		Location: to.StringPtr("eastus"),
		Sku: &compute.Sku{
			Name:     to.StringPtr("Standard"),
			Capacity: to.Int64Ptr(3),
		},
	}
}

func getTestVMSSClient(armClient armclient.Interface) *Client {
	rateLimiterReader, rateLimiterWriter := azclients.NewRateLimiter(&azclients.RateLimitConfig{})
	return &Client{
		armClient:         armClient,
		subscriptionID:    "subscriptionID",
		rateLimiterReader: rateLimiterReader,
		rateLimiterWriter: rateLimiterWriter,
	}
}

func getTestVMSSClientWithNeverRateLimiter(armClient armclient.Interface) *Client {
	rateLimiterReader := flowcontrol.NewFakeNeverRateLimiter()
	rateLimiterWriter := flowcontrol.NewFakeNeverRateLimiter()
	return &Client{
		armClient:         armClient,
		subscriptionID:    "subscriptionID",
		rateLimiterReader: rateLimiterReader,
		rateLimiterWriter: rateLimiterWriter,
	}
}

func getTestVMSSClientWithRetryAfterReader(armClient armclient.Interface) *Client {
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
