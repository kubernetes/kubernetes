/*
Copyright 2018 The Kubernetes Authors.

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

package options

import (
	"net"
	"strings"
	"testing"
	"time"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
)

func TestServerRunOptionsValidate(t *testing.T) {
	testCases := []struct {
		name        string
		testOptions *ServerRunOptions
		expectErr   string
	}{
		{
			name: "Test when TargetRAMMB is negative value",
			testOptions: &ServerRunOptions{
				AdvertiseAddress:            net.ParseIP("192.168.10.10"),
				CorsAllowedOriginList:       []string{"10.10.10.100", "10.10.10.200"},
				MaxRequestsInFlight:         400,
				MaxMutatingRequestsInFlight: 200,
				RequestTimeout:              time.Duration(2) * time.Minute,
				MinRequestTimeout:           1800,
				JSONPatchMaxCopyBytes:       10 * 1024 * 1024,
				MaxRequestBodyBytes:         10 * 1024 * 1024,
				TargetRAMMB:                 -65536,
			},
			expectErr: "--target-ram-mb can not be negative value",
		},
		{
			name: "Test when MaxRequestsInFlight is negative value",
			testOptions: &ServerRunOptions{
				AdvertiseAddress:            net.ParseIP("192.168.10.10"),
				CorsAllowedOriginList:       []string{"10.10.10.100", "10.10.10.200"},
				MaxRequestsInFlight:         -400,
				MaxMutatingRequestsInFlight: 200,
				RequestTimeout:              time.Duration(2) * time.Minute,
				MinRequestTimeout:           1800,
				JSONPatchMaxCopyBytes:       10 * 1024 * 1024,
				MaxRequestBodyBytes:         10 * 1024 * 1024,
				TargetRAMMB:                 65536,
			},
			expectErr: "--max-requests-inflight can not be negative value",
		},
		{
			name: "Test when MaxMutatingRequestsInFlight is negative value",
			testOptions: &ServerRunOptions{
				AdvertiseAddress:            net.ParseIP("192.168.10.10"),
				CorsAllowedOriginList:       []string{"10.10.10.100", "10.10.10.200"},
				MaxRequestsInFlight:         400,
				MaxMutatingRequestsInFlight: -200,
				RequestTimeout:              time.Duration(2) * time.Minute,
				MinRequestTimeout:           1800,
				JSONPatchMaxCopyBytes:       10 * 1024 * 1024,
				MaxRequestBodyBytes:         10 * 1024 * 1024,
				TargetRAMMB:                 65536,
			},
			expectErr: "--max-mutating-requests-inflight can not be negative value",
		},
		{
			name: "Test when RequestTimeout is negative value",
			testOptions: &ServerRunOptions{
				AdvertiseAddress:            net.ParseIP("192.168.10.10"),
				CorsAllowedOriginList:       []string{"10.10.10.100", "10.10.10.200"},
				MaxRequestsInFlight:         400,
				MaxMutatingRequestsInFlight: 200,
				RequestTimeout:              -time.Duration(2) * time.Minute,
				MinRequestTimeout:           1800,
				JSONPatchMaxCopyBytes:       10 * 1024 * 1024,
				MaxRequestBodyBytes:         10 * 1024 * 1024,
				TargetRAMMB:                 65536,
			},
			expectErr: "--request-timeout can not be negative value",
		},
		{
			name: "Test when MinRequestTimeout is negative value",
			testOptions: &ServerRunOptions{
				AdvertiseAddress:            net.ParseIP("192.168.10.10"),
				CorsAllowedOriginList:       []string{"10.10.10.100", "10.10.10.200"},
				MaxRequestsInFlight:         400,
				MaxMutatingRequestsInFlight: 200,
				RequestTimeout:              time.Duration(2) * time.Minute,
				MinRequestTimeout:           -1800,
				JSONPatchMaxCopyBytes:       10 * 1024 * 1024,
				MaxRequestBodyBytes:         10 * 1024 * 1024,
				TargetRAMMB:                 65536,
			},
			expectErr: "--min-request-timeout can not be negative value",
		},
		{
			name: "Test when JSONPatchMaxCopyBytes is negative value",
			testOptions: &ServerRunOptions{
				AdvertiseAddress:            net.ParseIP("192.168.10.10"),
				CorsAllowedOriginList:       []string{"10.10.10.100", "10.10.10.200"},
				MaxRequestsInFlight:         400,
				MaxMutatingRequestsInFlight: 200,
				RequestTimeout:              time.Duration(2) * time.Minute,
				MinRequestTimeout:           1800,
				JSONPatchMaxCopyBytes:       -10 * 1024 * 1024,
				MaxRequestBodyBytes:         10 * 1024 * 1024,
				TargetRAMMB:                 65536,
			},
			expectErr: "--json-patch-max-copy-bytes can not be negative value",
		},
		{
			name: "Test when MaxRequestBodyBytes is negative value",
			testOptions: &ServerRunOptions{
				AdvertiseAddress:            net.ParseIP("192.168.10.10"),
				CorsAllowedOriginList:       []string{"10.10.10.100", "10.10.10.200"},
				MaxRequestsInFlight:         400,
				MaxMutatingRequestsInFlight: 200,
				RequestTimeout:              time.Duration(2) * time.Minute,
				MinRequestTimeout:           1800,
				JSONPatchMaxCopyBytes:       10 * 1024 * 1024,
				MaxRequestBodyBytes:         -10 * 1024 * 1024,
				TargetRAMMB:                 65536,
			},
			expectErr: "--max-resource-write-bytes can not be negative value",
		},
		{
			name: "Test when MaxStartupSequenceDuration is negative value",
			testOptions: &ServerRunOptions{
				AdvertiseAddress:            net.ParseIP("192.168.10.10"),
				CorsAllowedOriginList:       []string{"10.10.10.100", "10.10.10.200"},
				MaxRequestsInFlight:         400,
				MaxMutatingRequestsInFlight: 200,
				RequestTimeout:              time.Duration(2) * time.Minute,
				MinRequestTimeout:           1800,
				JSONPatchMaxCopyBytes:       10 * 1024 * 1024,
				MaxRequestBodyBytes:         10 * 1024 * 1024,
				TargetRAMMB:                 65536,
				MaxStartupSequenceDuration:  -time.Second,
			},
			expectErr: "--maximum-startup-sequence-duration can not be a negative value",
		},
		{
			name: "Test when MinimalShutdownDuration is negative value",
			testOptions: &ServerRunOptions{
				AdvertiseAddress:            net.ParseIP("192.168.10.10"),
				CorsAllowedOriginList:       []string{"10.10.10.100", "10.10.10.200"},
				MaxRequestsInFlight:         400,
				MaxMutatingRequestsInFlight: 200,
				RequestTimeout:              time.Duration(2) * time.Minute,
				MinRequestTimeout:           1800,
				JSONPatchMaxCopyBytes:       10 * 1024 * 1024,
				MaxRequestBodyBytes:         10 * 1024 * 1024,
				TargetRAMMB:                 65536,
				ShutdownDelayDuration:       -time.Second,
			},
			expectErr: "--shutdown-delay-duration can not be negative value",
		},
		{
			name: "Test when ServerRunOptions is valid",
			testOptions: &ServerRunOptions{
				AdvertiseAddress:            net.ParseIP("192.168.10.10"),
				CorsAllowedOriginList:       []string{"10.10.10.100", "10.10.10.200"},
				MaxRequestsInFlight:         400,
				MaxMutatingRequestsInFlight: 200,
				RequestTimeout:              time.Duration(2) * time.Minute,
				MinRequestTimeout:           1800,
				JSONPatchMaxCopyBytes:       10 * 1024 * 1024,
				MaxRequestBodyBytes:         10 * 1024 * 1024,
				TargetRAMMB:                 65536,
			},
		},
	}

	for _, testcase := range testCases {
		t.Run(testcase.name, func(t *testing.T) {
			errs := testcase.testOptions.Validate()
			if len(testcase.expectErr) != 0 && !strings.Contains(utilerrors.NewAggregate(errs).Error(), testcase.expectErr) {
				t.Errorf("got err: %v, expected err: %s", errs, testcase.expectErr)
			}

			if len(testcase.expectErr) == 0 && len(errs) != 0 {
				t.Errorf("got err: %s, expected err nil", errs)
			}
		})
	}
}
