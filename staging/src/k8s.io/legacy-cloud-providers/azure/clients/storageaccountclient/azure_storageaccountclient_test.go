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

package storageaccountclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"testing"
	"time"

	"github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2019-06-01/storage"
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

	saClient := New(config)
	assert.Equal(t, "sub", saClient.subscriptionID)
	assert.NotEmpty(t, saClient.rateLimiterReader)
	assert.NotEmpty(t, saClient.rateLimiterWriter)
}

func TestGetProperties(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Storage/storageAccounts/sa1"
	response := &http.Response{
		StatusCode: http.StatusOK,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
	}

	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	saClient := getTestStorageAccountClient(armClient)
	expected := storage.Account{Response: autorest.Response{Response: response}}
	result, rerr := saClient.GetProperties(context.TODO(), "rg", "sa1")
	assert.Equal(t, expected, result)
	assert.Nil(t, rerr)
}

func TestAllNeverRateLimiter(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	saErr1 := &retry.Error{
		RawError:  fmt.Errorf("azure cloud provider rate limited(%s) for operation %q", "read", "StorageAccountGet"),
		Retriable: true,
	}

	saErr2 := &retry.Error{
		RawError:  fmt.Errorf("azure cloud provider rate limited(%s) for operation %q", "read", "StorageAccountListKeys"),
		Retriable: true,
	}

	saErr3 := &retry.Error{
		RawError:  fmt.Errorf("azure cloud provider rate limited(%s) for operation %q", "read", "StorageAccountListByResourceGroup"),
		Retriable: true,
	}

	saErr4 := &retry.Error{
		RawError:  fmt.Errorf("azure cloud provider rate limited(%s) for operation %q", "write", "StorageAccountCreate"),
		Retriable: true,
	}

	saErr5 := &retry.Error{
		RawError:  fmt.Errorf("azure cloud provider rate limited(%s) for operation %q", "write", "StorageAccountDelete"),
		Retriable: true,
	}

	armClient := mockarmclient.NewMockInterface(ctrl)

	saClient := getTestStorageAccountClientWithNeverRateLimiter(armClient)
	expected1 := storage.Account{}
	expected2 := storage.AccountListKeysResult{}
	expected3 := []storage.Account(nil)
	result1, rerr1 := saClient.GetProperties(context.TODO(), "rg", "sa1")
	assert.Equal(t, expected1, result1)
	assert.Equal(t, saErr1, rerr1)
	result2, rerr2 := saClient.ListKeys(context.TODO(), "rg", "sa1")
	assert.Equal(t, expected2, result2)
	assert.Equal(t, saErr2, rerr2)
	result3, rerr3 := saClient.ListByResourceGroup(context.TODO(), "rg")
	assert.Equal(t, expected3, result3)
	assert.Equal(t, saErr3, rerr3)

	sa := storage.AccountCreateParameters{
		Location: to.StringPtr("eastus"),
	}

	rerr4 := saClient.Create(context.TODO(), "rg", "sa1", sa)
	assert.Equal(t, saErr4, rerr4)
	rerr5 := saClient.Delete(context.TODO(), "rg", "sa1")
	assert.Equal(t, saErr5, rerr5)
}

func TestAllRetryAfterReader(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	saErr1 := &retry.Error{
		RawError:   fmt.Errorf("azure cloud provider throttled for operation %s with reason %q", "StorageAccountGet", "client throttled"),
		Retriable:  true,
		RetryAfter: getFutureTime(),
	}

	saErr2 := &retry.Error{
		RawError:   fmt.Errorf("azure cloud provider throttled for operation %s with reason %q", "StorageAccountListKeys", "client throttled"),
		Retriable:  true,
		RetryAfter: getFutureTime(),
	}

	saErr3 := &retry.Error{
		RawError:   fmt.Errorf("azure cloud provider throttled for operation %s with reason %q", "StorageAccountListByResourceGroup", "client throttled"),
		Retriable:  true,
		RetryAfter: getFutureTime(),
	}

	saErr4 := &retry.Error{
		RawError:   fmt.Errorf("azure cloud provider throttled for operation %s with reason %q", "StorageAccountCreate", "client throttled"),
		Retriable:  true,
		RetryAfter: getFutureTime(),
	}

	saErr5 := &retry.Error{
		RawError:   fmt.Errorf("azure cloud provider throttled for operation %s with reason %q", "StorageAccountDelete", "client throttled"),
		Retriable:  true,
		RetryAfter: getFutureTime(),
	}

	armClient := mockarmclient.NewMockInterface(ctrl)

	saClient := getTestStorageAccountClientWithRetryAfterReader(armClient)
	expected1 := storage.Account{}
	expected2 := storage.AccountListKeysResult{}
	expected3 := []storage.Account(nil)
	result1, rerr1 := saClient.GetProperties(context.TODO(), "rg", "sa1")
	assert.Equal(t, expected1, result1)
	assert.Equal(t, saErr1, rerr1)
	result2, rerr2 := saClient.ListKeys(context.TODO(), "rg", "sa1")
	assert.Equal(t, expected2, result2)
	assert.Equal(t, saErr2, rerr2)
	result3, rerr3 := saClient.ListByResourceGroup(context.TODO(), "rg")
	assert.Equal(t, expected3, result3)
	assert.Equal(t, saErr3, rerr3)

	sa := storage.AccountCreateParameters{
		Location: to.StringPtr("eastus"),
	}

	rerr4 := saClient.Create(context.TODO(), "rg", "sa1", sa)
	assert.Equal(t, saErr4, rerr4)
	rerr5 := saClient.Delete(context.TODO(), "rg", "sa1")
	assert.Equal(t, saErr5, rerr5)
}

func TestAllThrottle(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Storage/storageAccounts/sa1"
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

	sa := storage.AccountCreateParameters{
		Location: to.StringPtr("eastus"),
	}

	r := getTestStorageAccount("sa1")

	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().PostResource(gomock.Any(), resourceID, "listKeys", struct{}{}).Return(response, throttleErr).Times(1)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(response, throttleErr).Times(1)
	armClient.EXPECT().PutResource(gomock.Any(), resourceID, sa).Return(response, throttleErr).Times(1)
	armClient.EXPECT().DeleteResource(gomock.Any(), to.String(r.ID), "").Return(throttleErr).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(3)

	saClient := getTestStorageAccountClient(armClient)
	result1, rerr1 := saClient.GetProperties(context.TODO(), "rg", "sa1")
	assert.Empty(t, result1)
	assert.Equal(t, throttleErr, rerr1)
	result2, rerr2 := saClient.ListKeys(context.TODO(), "rg", "sa1")
	assert.Empty(t, result2)
	assert.Equal(t, throttleErr, rerr2)
	rerr3 := saClient.Create(context.TODO(), "rg", "sa1", sa)
	assert.Equal(t, throttleErr, rerr3)
	rerr4 := saClient.Delete(context.TODO(), "rg", "sa1")
	assert.Equal(t, throttleErr, rerr4)

	resourceID2 := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Storage/storageAccounts"
	armClient2 := mockarmclient.NewMockInterface(ctrl)
	armClient2.EXPECT().GetResource(gomock.Any(), resourceID2, "").Return(response, throttleErr).Times(1)
	armClient2.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	saClient2 := getTestStorageAccountClient(armClient2)
	result5, rerr5 := saClient2.ListByResourceGroup(context.TODO(), "rg")
	assert.Empty(t, result5)
	assert.Equal(t, throttleErr, rerr5)
}

func TestGetPropertiesNotFound(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Storage/storageAccounts/sa1"
	response := &http.Response{
		StatusCode: http.StatusNotFound,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
	}
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	saClient := getTestStorageAccountClient(armClient)
	expected := storage.Account{Response: autorest.Response{}}
	result, rerr := saClient.GetProperties(context.TODO(), "rg", "sa1")
	assert.Equal(t, expected, result)
	assert.NotNil(t, rerr)
	assert.Equal(t, http.StatusNotFound, rerr.HTTPStatusCode)
}

func TestGetPropertiesInternalError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Storage/storageAccounts/sa1"
	response := &http.Response{
		StatusCode: http.StatusInternalServerError,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
	}
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	saClient := getTestStorageAccountClient(armClient)
	expected := storage.Account{Response: autorest.Response{}}
	result, rerr := saClient.GetProperties(context.TODO(), "rg", "sa1")
	assert.Equal(t, expected, result)
	assert.NotNil(t, rerr)
	assert.Equal(t, http.StatusInternalServerError, rerr.HTTPStatusCode)
}

func TestListKeys(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Storage/storageAccounts/sa1"
	response := &http.Response{
		StatusCode: http.StatusOK,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
	}
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().PostResource(gomock.Any(), resourceID, "listKeys", struct{}{}).Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	saClient := getTestStorageAccountClient(armClient)
	expected := storage.AccountListKeysResult{Response: autorest.Response{Response: response}}
	result, rerr := saClient.ListKeys(context.TODO(), "rg", "sa1")
	assert.Nil(t, rerr)
	assert.Equal(t, expected, result)
}

func TestListKeysResponderError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Storage/storageAccounts/sa1"
	response := &http.Response{
		StatusCode: http.StatusNotFound,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
	}
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().PostResource(gomock.Any(), resourceID, "listKeys", struct{}{}).Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	saClient := getTestStorageAccountClient(armClient)
	expected := storage.AccountListKeysResult{Response: autorest.Response{}}
	result, rerr := saClient.ListKeys(context.TODO(), "rg", "sa1")
	assert.Equal(t, expected, result)
	assert.NotNil(t, rerr)
	assert.Equal(t, http.StatusNotFound, rerr.HTTPStatusCode)
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

	lastResult := storage.AccountListResult{
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

		saClient := getTestStorageAccountClient(armClient)
		result, err := saClient.listNextResults(context.TODO(), lastResult)
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

	lastResult := storage.AccountListResult{
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

	saClient := getTestStorageAccountClient(armClient)
	response := &http.Response{
		StatusCode: http.StatusNotFound,
		Body:       ioutil.NopCloser(bytes.NewBuffer([]byte(`{"foo":"bar"}`))),
	}
	expected := storage.AccountListResult{Response: autorest.Response{Response: response}}
	result, err := saClient.listNextResults(context.TODO(), lastResult)
	assert.Error(t, err)
	assert.Equal(t, expected, result)
}

func TestListByResourceGroup(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Storage/storageAccounts"
	armClient := mockarmclient.NewMockInterface(ctrl)
	snList := []storage.Account{getTestStorageAccount("sn1"), getTestStorageAccount("pip2"), getTestStorageAccount("pip3")}
	responseBody, err := json.Marshal(map[string]interface{}{"value": snList, "nextLink": ""})
	assert.NoError(t, err)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(
		&http.Response{
			StatusCode: http.StatusOK,
			Body:       ioutil.NopCloser(bytes.NewReader(responseBody)),
		}, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	saClient := getTestStorageAccountClient(armClient)
	result, rerr := saClient.ListByResourceGroup(context.TODO(), "rg")
	assert.Nil(t, rerr)
	assert.Equal(t, 3, len(result))
}

func TestListByResourceGroupResponderError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Storage/storageAccounts"
	armClient := mockarmclient.NewMockInterface(ctrl)
	snList := []storage.Account{getTestStorageAccount("sn1"), getTestStorageAccount("pip2"), getTestStorageAccount("pip3")}
	responseBody, err := json.Marshal(storage.AccountListResult{Value: &snList})
	assert.NoError(t, err)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "").Return(
		&http.Response{
			StatusCode: http.StatusNotFound,
			Body:       ioutil.NopCloser(bytes.NewReader(responseBody)),
		}, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	saClient := getTestStorageAccountClient(armClient)
	result, rerr := saClient.ListByResourceGroup(context.TODO(), "rg")
	assert.NotNil(t, rerr)
	assert.Equal(t, 0, len(result))
}

func TestCreate(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Storage/storageAccounts/sa1"
	sa := storage.AccountCreateParameters{
		Location: to.StringPtr("eastus"),
	}
	armClient := mockarmclient.NewMockInterface(ctrl)
	response := &http.Response{
		StatusCode: http.StatusOK,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte(""))),
	}
	armClient.EXPECT().PutResource(gomock.Any(), resourceID, sa).Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	saClient := getTestStorageAccountClient(armClient)
	rerr := saClient.Create(context.TODO(), "rg", "sa1", sa)
	assert.Nil(t, rerr)
}

func TestCreateResponderError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Storage/storageAccounts/sa1"
	sa := storage.AccountCreateParameters{
		Location: to.StringPtr("eastus"),
	}
	armClient := mockarmclient.NewMockInterface(ctrl)
	response := &http.Response{
		StatusCode: http.StatusNotFound,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte(""))),
	}
	armClient.EXPECT().PutResource(gomock.Any(), resourceID, sa).Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	saClient := getTestStorageAccountClient(armClient)
	rerr := saClient.Create(context.TODO(), "rg", "sa1", sa)
	assert.NotNil(t, rerr)
}

func TestDelete(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	r := getTestStorageAccount("sa1")
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().DeleteResource(gomock.Any(), to.String(r.ID), "").Return(nil).Times(1)

	rtClient := getTestStorageAccountClient(armClient)
	rerr := rtClient.Delete(context.TODO(), "rg", "sa1")
	assert.Nil(t, rerr)
}

func getTestStorageAccount(name string) storage.Account {
	return storage.Account{
		ID:       to.StringPtr(fmt.Sprintf("/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Storage/storageAccounts/%s", name)),
		Name:     to.StringPtr(name),
		Location: to.StringPtr("eastus"),
	}
}

func getTestStorageAccountClient(armClient armclient.Interface) *Client {
	rateLimiterReader, rateLimiterWriter := azclients.NewRateLimiter(&azclients.RateLimitConfig{})
	return &Client{
		armClient:         armClient,
		subscriptionID:    "subscriptionID",
		rateLimiterReader: rateLimiterReader,
		rateLimiterWriter: rateLimiterWriter,
	}
}

func getTestStorageAccountClientWithNeverRateLimiter(armClient armclient.Interface) *Client {
	rateLimiterReader := flowcontrol.NewFakeNeverRateLimiter()
	rateLimiterWriter := flowcontrol.NewFakeNeverRateLimiter()
	return &Client{
		armClient:         armClient,
		subscriptionID:    "subscriptionID",
		rateLimiterReader: rateLimiterReader,
		rateLimiterWriter: rateLimiterWriter,
	}
}

func getTestStorageAccountClientWithRetryAfterReader(armClient armclient.Interface) *Client {
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
