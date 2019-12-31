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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"testing"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-07-01/compute"
	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/to"
	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"

	azclients "k8s.io/legacy-cloud-providers/azure/clients"
	"k8s.io/legacy-cloud-providers/azure/clients/armclient"
	"k8s.io/legacy-cloud-providers/azure/clients/armclient/mockarmclient"
)

func TestGetNotFound(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss1/virtualMachines/0"
	response := &http.Response{
		StatusCode: http.StatusNotFound,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
	}
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "InstanceView").Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	vmssClient := getTestVMSSVMClient(armClient)
	expectedVM := compute.VirtualMachineScaleSetVM{Response: autorest.Response{}}
	result, rerr := vmssClient.Get(context.TODO(), "rg", "vmss1", "0", "InstanceView")
	assert.Equal(t, expectedVM, result)
	assert.NotNil(t, rerr)
	assert.Equal(t, http.StatusNotFound, rerr.HTTPStatusCode)
}

func TestGetInternalError(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss1/virtualMachines/1"
	response := &http.Response{
		StatusCode: http.StatusInternalServerError,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("{}"))),
	}
	armClient := mockarmclient.NewMockInterface(ctrl)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "InstanceView").Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	vmssClient := getTestVMSSVMClient(armClient)
	expectedVM := compute.VirtualMachineScaleSetVM{Response: autorest.Response{}}
	result, rerr := vmssClient.Get(context.TODO(), "rg", "vmss1", "1", "InstanceView")
	assert.Equal(t, expectedVM, result)
	assert.NotNil(t, rerr)
	assert.Equal(t, http.StatusInternalServerError, rerr.HTTPStatusCode)
}

func TestList(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	resourceID := "/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss1/virtualMachines"
	armClient := mockarmclient.NewMockInterface(ctrl)
	vmssList := []compute.VirtualMachineScaleSetVM{getTestVMSSVM("vmss1", "1"), getTestVMSSVM("vmss1", "2"), getTestVMSSVM("vmss1", "3")}
	responseBody, err := json.Marshal(compute.VirtualMachineScaleSetVMListResult{Value: &vmssList})
	assert.Nil(t, err)
	armClient.EXPECT().GetResource(gomock.Any(), resourceID, "InstanceView").Return(
		&http.Response{
			StatusCode: http.StatusOK,
			Body:       ioutil.NopCloser(bytes.NewReader(responseBody)),
		}, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	vmssClient := getTestVMSSVMClient(armClient)
	result, rerr := vmssClient.List(context.TODO(), "rg", "vmss1", "InstanceView")
	assert.Nil(t, rerr)
	assert.Equal(t, 3, len(result))
}

func TestUpdate(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	vmssVM := getTestVMSSVM("vmss1", "0")
	armClient := mockarmclient.NewMockInterface(ctrl)
	response := &http.Response{
		StatusCode: http.StatusOK,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte(""))),
	}
	armClient.EXPECT().PutResource(gomock.Any(), to.String(vmssVM.ID), vmssVM).Return(response, nil).Times(1)
	armClient.EXPECT().CloseResponse(gomock.Any(), gomock.Any()).Times(1)

	vmssClient := getTestVMSSVMClient(armClient)
	rerr := vmssClient.Update(context.TODO(), "rg", "vmss1", "0", vmssVM, "test")
	assert.Nil(t, rerr)
}

func getTestVMSSVM(vmssName, instanceID string) compute.VirtualMachineScaleSetVM {
	resourceID := fmt.Sprintf("/subscriptions/subscriptionID/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/%s/virtualMachines/%s", vmssName, instanceID)
	return compute.VirtualMachineScaleSetVM{
		ID:         to.StringPtr(resourceID),
		InstanceID: to.StringPtr(instanceID),
		Location:   to.StringPtr("eastus"),
	}
}

func getTestVMSSVMClient(armClient armclient.Interface) *Client {
	rateLimiterReader, rateLimiterWriter := azclients.NewRateLimiter(&azclients.RateLimitConfig{})
	return &Client{
		armClient:         armClient,
		subscriptionID:    "subscriptionID",
		rateLimiterReader: rateLimiterReader,
		rateLimiterWriter: rateLimiterWriter,
	}
}
