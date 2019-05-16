/*
Copyright 2017 The Kubernetes Authors.

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

package azure

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-03-01/compute"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	cloudprovider "k8s.io/cloud-provider"
)

func TestShouldRetryHTTPRequest(t *testing.T) {
	tests := []struct {
		code     int
		err      error
		expected bool
	}{
		{
			code:     http.StatusBadRequest,
			expected: true,
		},
		{
			code:     http.StatusInternalServerError,
			expected: true,
		},
		{
			code:     http.StatusOK,
			err:      fmt.Errorf("some error"),
			expected: true,
		},
		{
			code:     http.StatusOK,
			expected: false,
		},
		{
			code:     399,
			expected: false,
		},
	}
	for _, test := range tests {
		resp := &http.Response{
			StatusCode: test.code,
		}
		res := shouldRetryHTTPRequest(resp, test.err)
		if res != test.expected {
			t.Errorf("expected: %v, saw: %v", test.expected, res)
		}
	}
}

func TestIsSuccessResponse(t *testing.T) {
	tests := []struct {
		code     int
		expected bool
	}{
		{
			code:     http.StatusNotFound,
			expected: false,
		},
		{
			code:     http.StatusInternalServerError,
			expected: false,
		},
		{
			code:     http.StatusOK,
			expected: true,
		},
	}

	for _, test := range tests {
		resp := http.Response{
			StatusCode: test.code,
		}
		res := isSuccessHTTPResponse(&resp)
		if res != test.expected {
			t.Errorf("expected: %v, saw: %v", test.expected, res)
		}
	}
}

func TestProcessRetryResponse(t *testing.T) {
	az := &Cloud{}
	tests := []struct {
		code int
		err  error
		stop bool
	}{
		{
			code: http.StatusBadRequest,
			stop: false,
		},
		{
			code: http.StatusInternalServerError,
			stop: false,
		},
		{
			code: http.StatusSeeOther,
			err:  fmt.Errorf("some error"),
			stop: false,
		},
		{
			code: http.StatusSeeOther,
			stop: true,
		},
		{
			code: http.StatusOK,
			stop: true,
		},
		{
			code: 399,
			stop: true,
		},
	}

	for _, test := range tests {
		resp := &http.Response{
			StatusCode: test.code,
		}
		res, err := az.processHTTPRetryResponse(nil, "", resp, test.err)
		if res != test.stop {
			t.Errorf("expected: %v, saw: %v", test.stop, res)
		}
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	}
}

func TestRequestBackoff(t *testing.T) {
	var durationZeroValue time.Duration
	explicitDurationValue := time.Second
	explicitFactorValue := float64(2)
	explicitStepsValue := 6
	explicitJitterValue := float64(4)
	type expectedResponse struct {
		expectedDuration time.Duration
		expectedFactor   float64
		expectedSteps    int
		expectedJitter   float64
		expectedCap      time.Duration
	}
	expectedResponseBackoffDisabled := expectedResponse{
		expectedSteps: 1,
	}
	cases := []struct {
		cloud    *Cloud
		expected expectedResponse
	}{
		// Default (backoff not enabled) case
		{
			cloud:    &Cloud{},
			expected: expectedResponseBackoffDisabled,
		},
		// Backoff enabled with values
		{
			cloud: &Cloud{
				Config: Config{
					CloudProviderBackoff: true,
				},
				resourceRequestBackoff: wait.Backoff{
					Duration: explicitDurationValue,
					Factor:   explicitFactorValue,
					Steps:    explicitStepsValue,
					Jitter:   explicitJitterValue,
				},
			},
			expected: expectedResponse{
				expectedDuration: explicitDurationValue,
				expectedFactor:   explicitFactorValue,
				expectedSteps:    explicitStepsValue,
				expectedJitter:   explicitJitterValue,
				expectedCap:      durationZeroValue,
			},
		},
		// Values provided but not explicitly enabled
		{
			cloud: &Cloud{
				resourceRequestBackoff: wait.Backoff{
					Duration: explicitDurationValue,
					Factor:   explicitFactorValue,
					Steps:    explicitStepsValue,
					Jitter:   explicitJitterValue,
				},
			},
			expected: expectedResponseBackoffDisabled,
		},
	}

	for _, c := range cases {
		backoff := c.cloud.requestBackoff()
		if c.expected.expectedDuration != backoff.Duration {
			t.Fatalf("Expected backoff.Duration to be %s, instead got %s", c.expected.expectedDuration, backoff.Duration)
		}
		if c.expected.expectedFactor != backoff.Factor {
			t.Fatalf("Expected backoff.Factor to be %f, instead got %f", c.expected.expectedFactor, backoff.Factor)
		}
		if c.expected.expectedSteps != backoff.Steps {
			t.Fatalf("Expected backoff.Steps to be %d, instead got %d", c.expected.expectedSteps, backoff.Steps)
		}
		if c.expected.expectedJitter != backoff.Jitter {
			t.Fatalf("Expected backoff.Jitter to be %f, instead got %f", c.expected.expectedJitter, backoff.Jitter)
		}
		if c.expected.expectedCap != backoff.Cap {
			t.Fatalf("Expected backoff.Cap to be %s, instead got %s", c.expected.expectedCap, backoff.Cap)
		}
	}
}

func TestGetVirtualMachineWithRetry(t *testing.T) {
	type expectedResponse struct {
		expectedVM  compute.VirtualMachine
		expectedErr error
	}
	explicitVMNameValue := "bar"
	dataSource, fakeCache := newFakeCacheVirtualMachine(t)
	dataSource.set(map[string]*compute.VirtualMachine{explicitVMNameValue: &compute.VirtualMachine{Name: &explicitVMNameValue}})
	cases := []struct {
		cloud    *Cloud
		vmName   types.NodeName
		expected expectedResponse
	}{
		// Backoff not enabled, get non-existent vm
		{
			cloud: &Cloud{
				vmCache: fakeCache,
			},
			vmName: "foo",
			expected: expectedResponse{
				expectedErr: cloudprovider.InstanceNotFound,
			},
		},
		// Backoff not enabled, get existent vm
		{
			cloud: &Cloud{
				vmCache: fakeCache,
			},
			vmName: types.NodeName(explicitVMNameValue),
			expected: expectedResponse{
				expectedVM: compute.VirtualMachine{
					Name: &explicitVMNameValue,
				},
				expectedErr: nil,
			},
		},
	}

	for _, c := range cases {
		vm, err := c.cloud.GetVirtualMachineWithRetry(c.vmName)
		if c.expected.expectedErr != err {
			t.Fatalf("Expected err to be %s, instead got %s", c.expected.expectedErr, err)
		}
		if c.expected.expectedVM.Name != nil {
			if *c.expected.expectedVM.Name != *vm.Name {
				t.Fatalf("Expected vm Name to be %s, instead got %s", *c.expected.expectedVM.Name, *vm.Name)
			}
		}
	}
}
