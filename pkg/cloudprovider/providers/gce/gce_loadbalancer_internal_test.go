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

package gce

import (
	"testing"

	compute "google.golang.org/api/compute/v1"
	"k8s.io/apimachinery/pkg/types"
)

func TestMergeHealthChecks(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		desc                   string
		checkIntervalSec       int64
		timeoutSec             int64
		healthyThreshold       int64
		unhealthyThreshold     int64
		wantCheckIntervalSec   int64
		wantTimeoutSec         int64
		wantHealthyThreshold   int64
		wantUnhealthyThreshold int64
	}{
		{"unchanged", gceHcCheckIntervalSeconds, gceHcTimeoutSeconds, gceHcHealthyThreshold, gceHcUnhealthyThreshold, gceHcCheckIntervalSeconds, gceHcTimeoutSeconds, gceHcHealthyThreshold, gceHcUnhealthyThreshold},
		{"interval - too small - should reconcile", gceHcCheckIntervalSeconds - 1, gceHcTimeoutSeconds, gceHcHealthyThreshold, gceHcUnhealthyThreshold, gceHcCheckIntervalSeconds, gceHcTimeoutSeconds, gceHcHealthyThreshold, gceHcUnhealthyThreshold},
		{"timeout - too small - should reconcile", gceHcCheckIntervalSeconds, gceHcTimeoutSeconds - 1, gceHcHealthyThreshold, gceHcUnhealthyThreshold, gceHcCheckIntervalSeconds, gceHcTimeoutSeconds, gceHcHealthyThreshold, gceHcUnhealthyThreshold},
		{"healthy threshold - too small - should reconcile", gceHcCheckIntervalSeconds, gceHcTimeoutSeconds, gceHcHealthyThreshold - 1, gceHcUnhealthyThreshold, gceHcCheckIntervalSeconds, gceHcTimeoutSeconds, gceHcHealthyThreshold, gceHcUnhealthyThreshold},
		{"unhealthy threshold - too small - should reconcile", gceHcCheckIntervalSeconds, gceHcTimeoutSeconds, gceHcHealthyThreshold, gceHcUnhealthyThreshold - 1, gceHcCheckIntervalSeconds, gceHcTimeoutSeconds, gceHcHealthyThreshold, gceHcUnhealthyThreshold},
		{"interval - user configured - should keep", gceHcCheckIntervalSeconds + 1, gceHcTimeoutSeconds, gceHcHealthyThreshold, gceHcUnhealthyThreshold, gceHcCheckIntervalSeconds + 1, gceHcTimeoutSeconds, gceHcHealthyThreshold, gceHcUnhealthyThreshold},
		{"timeout - user configured - should keep", gceHcCheckIntervalSeconds, gceHcTimeoutSeconds + 1, gceHcHealthyThreshold, gceHcUnhealthyThreshold, gceHcCheckIntervalSeconds, gceHcTimeoutSeconds + 1, gceHcHealthyThreshold, gceHcUnhealthyThreshold},
		{"healthy threshold - user configured - should keep", gceHcCheckIntervalSeconds, gceHcTimeoutSeconds, gceHcHealthyThreshold + 1, gceHcUnhealthyThreshold, gceHcCheckIntervalSeconds, gceHcTimeoutSeconds, gceHcHealthyThreshold + 1, gceHcUnhealthyThreshold},
		{"unhealthy threshold - user configured - should keep", gceHcCheckIntervalSeconds, gceHcTimeoutSeconds, gceHcHealthyThreshold, gceHcUnhealthyThreshold + 1, gceHcCheckIntervalSeconds, gceHcTimeoutSeconds, gceHcHealthyThreshold, gceHcUnhealthyThreshold + 1},
	} {
		t.Run(tc.desc, func(t *testing.T) {
			wantHC := newInternalLBHealthCheck("hc", types.NamespacedName{Name: "svc", Namespace: "default"}, false, "/", 12345)
			hc := &compute.HealthCheck{
				CheckIntervalSec:   tc.checkIntervalSec,
				TimeoutSec:         tc.timeoutSec,
				HealthyThreshold:   tc.healthyThreshold,
				UnhealthyThreshold: tc.unhealthyThreshold,
			}
			mergeHealthChecks(hc, wantHC)
			if wantHC.CheckIntervalSec != tc.wantCheckIntervalSec {
				t.Errorf("wantHC.CheckIntervalSec = %d; want %d", wantHC.CheckIntervalSec, tc.checkIntervalSec)
			}
			if wantHC.TimeoutSec != tc.wantTimeoutSec {
				t.Errorf("wantHC.TimeoutSec = %d; want %d", wantHC.TimeoutSec, tc.timeoutSec)
			}
			if wantHC.HealthyThreshold != tc.wantHealthyThreshold {
				t.Errorf("wantHC.HealthyThreshold = %d; want %d", wantHC.HealthyThreshold, tc.healthyThreshold)
			}
			if wantHC.UnhealthyThreshold != tc.wantUnhealthyThreshold {
				t.Errorf("wantHC.UnhealthyThreshold = %d; want %d", wantHC.UnhealthyThreshold, tc.unhealthyThreshold)
			}
		})
	}
}

func TestCompareHealthChecks(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		desc        string
		modifier    func(*compute.HealthCheck)
		wantChanged bool
	}{
		{"unchanged", nil, false},
		{"nil HttpHealthCheck", func(hc *compute.HealthCheck) { hc.HttpHealthCheck = nil }, true},
		{"desc does not match", func(hc *compute.HealthCheck) { hc.Description = "bad-desc" }, true},
		{"port does not match", func(hc *compute.HealthCheck) { hc.HttpHealthCheck.Port = 54321 }, true},
		{"requestPath does not match", func(hc *compute.HealthCheck) { hc.HttpHealthCheck.RequestPath = "/anotherone" }, true},
		{"interval needs update", func(hc *compute.HealthCheck) { hc.CheckIntervalSec = gceHcCheckIntervalSeconds - 1 }, true},
		{"timeout needs update", func(hc *compute.HealthCheck) { hc.TimeoutSec = gceHcTimeoutSeconds - 1 }, true},
		{"healthy threshold needs update", func(hc *compute.HealthCheck) { hc.HealthyThreshold = gceHcHealthyThreshold - 1 }, true},
		{"unhealthy threshold needs update", func(hc *compute.HealthCheck) { hc.UnhealthyThreshold = gceHcUnhealthyThreshold - 1 }, true},
		{"interval does not need update", func(hc *compute.HealthCheck) { hc.CheckIntervalSec = gceHcCheckIntervalSeconds + 1 }, false},
		{"timeout does not need update", func(hc *compute.HealthCheck) { hc.TimeoutSec = gceHcTimeoutSeconds + 1 }, false},
		{"healthy threshold does not need update", func(hc *compute.HealthCheck) { hc.HealthyThreshold = gceHcHealthyThreshold + 1 }, false},
		{"unhealthy threshold does not need update", func(hc *compute.HealthCheck) { hc.UnhealthyThreshold = gceHcUnhealthyThreshold + 1 }, false},
	} {
		t.Run(tc.desc, func(t *testing.T) {
			hc := newInternalLBHealthCheck("hc", types.NamespacedName{Name: "svc", Namespace: "default"}, false, "/", 12345)
			wantHC := newInternalLBHealthCheck("hc", types.NamespacedName{Name: "svc", Namespace: "default"}, false, "/", 12345)
			if tc.modifier != nil {
				tc.modifier(hc)
			}
			if gotChanged := needToUpdateHealthChecks(hc, wantHC); gotChanged != tc.wantChanged {
				t.Errorf("needToUpdateHealthChecks(%#v, %#v) = %t; want changed = %t", hc, wantHC, gotChanged, tc.wantChanged)
			}
		})
	}
}
