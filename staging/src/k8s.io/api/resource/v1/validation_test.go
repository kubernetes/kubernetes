/*
Copyright The Kubernetes Authors.

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

package v1_test

import (
	"context"
	"fmt"
	"strings"
	"testing"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestValidateDeviceAttribute(t *testing.T) {
	testcases := map[string]struct {
		attribute *resourceapi.DeviceAttribute
		wantField string
	}{
		"valid": {
			attribute: &resourceapi.DeviceAttribute{
				StringValue: new("LATEST-GPU-MODEL"),
			},
		},
		"no value": {
			attribute: &resourceapi.DeviceAttribute{},
			wantField: "attribute",
		},
		"multiple values": {
			attribute: &resourceapi.DeviceAttribute{
				BoolValue:   new(true),
				StringValue: new("LATEST-GPU-MODEL"),
			},
			wantField: "attribute",
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			errs := resourceapi.Validate_DeviceAttribute(
				context.Background(),
				operation.Operation{},
				field.NewPath("attribute"),
				tc.attribute,
				nil,
			)
			if tc.wantField == "" {
				if len(errs) != 0 {
					t.Fatalf("expected no validation errors, got %v", errs)
				}
				return
			}
			if len(errs) != 1 {
				t.Fatalf("expected one validation error for %q, got %v", tc.wantField, errs)
			}
			if errs[0].Field != tc.wantField {
				t.Errorf("expected validation error for %q, got %q", tc.wantField, errs[0].Field)
			}
		})
	}
}

func TestValidateNetworkDeviceData(t *testing.T) {
	tooManyIPs := make([]string, 17)
	for i := range tooManyIPs {
		tooManyIPs[i] = fmt.Sprintf("192.0.2.%d/24", i)
	}

	testcases := map[string]struct {
		networkData *resourceapi.NetworkDeviceData
		wantField   string
	}{
		"valid": {
			networkData: &resourceapi.NetworkDeviceData{
				InterfaceName:   "eth0",
				IPs:             []string{"192.0.2.5/24"},
				HardwareAddress: "02:42:ac:11:00:02",
			},
		},
		"interface name too long": {
			networkData: &resourceapi.NetworkDeviceData{
				InterfaceName: strings.Repeat("a", 257),
			},
			wantField: "networkData.interfaceName",
		},
		"too many IPs": {
			networkData: &resourceapi.NetworkDeviceData{
				IPs: tooManyIPs,
			},
			wantField: "networkData.ips",
		},
		"duplicate IPs": {
			networkData: &resourceapi.NetworkDeviceData{
				IPs: []string{"192.0.2.5/24", "192.0.2.5/24"},
			},
			wantField: "networkData.ips[1]",
		},
		"hardware address too long": {
			networkData: &resourceapi.NetworkDeviceData{
				HardwareAddress: strings.Repeat("a", 129),
			},
			wantField: "networkData.hardwareAddress",
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			errs := resourceapi.Validate_NetworkDeviceData(
				context.Background(),
				operation.Operation{},
				field.NewPath("networkData"),
				tc.networkData,
				nil,
			)
			if tc.wantField == "" {
				if len(errs) != 0 {
					t.Fatalf("expected no validation errors, got %v", errs)
				}
				return
			}
			if len(errs) != 1 {
				t.Fatalf("expected one validation error for %q, got %v", tc.wantField, errs)
			}
			if errs[0].Field != tc.wantField {
				t.Errorf("expected validation error for %q, got %q", tc.wantField, errs[0].Field)
			}
		})
	}
}
