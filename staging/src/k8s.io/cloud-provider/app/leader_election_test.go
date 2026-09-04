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

package app

import (
	"testing"
	"time"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/leaderelection"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	cmfeatures "k8s.io/controller-manager/pkg/features"
	_ "k8s.io/controller-manager/pkg/features/register"
)

func TestNewLeaderElectionConfig(t *testing.T) {
	const (
		leaseDuration = 15 * time.Second
		renewDeadline = 10 * time.Second
		retryPeriod   = 2 * time.Second
		leaseName     = "cloud-controller-manager"
	)
	callbacks := leaderelection.LeaderCallbacks{}

	for _, tc := range []struct {
		name                string
		featureEnabled      bool
		wantReleaseOnCancel bool
	}{
		{name: "feature gate disabled (default)", featureEnabled: false, wantReleaseOnCancel: false},
		{name: "feature gate enabled", featureEnabled: true, wantReleaseOnCancel: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, cmfeatures.ControllerManagerReleaseLeaderElectionLockOnExit, tc.featureEnabled)

			got := newLeaderElectionConfig(nil, leaseDuration, renewDeadline, retryPeriod, callbacks, nil, leaseName)

			if got.ReleaseOnCancel != tc.wantReleaseOnCancel {
				t.Errorf("ReleaseOnCancel = %v, want %v", got.ReleaseOnCancel, tc.wantReleaseOnCancel)
			}
			if got.Name != leaseName {
				t.Errorf("Name = %q, want %q", got.Name, leaseName)
			}
			if got.LeaseDuration != leaseDuration {
				t.Errorf("LeaseDuration = %v, want %v", got.LeaseDuration, leaseDuration)
			}
			if got.RenewDeadline != renewDeadline {
				t.Errorf("RenewDeadline = %v, want %v", got.RenewDeadline, renewDeadline)
			}
			if got.RetryPeriod != retryPeriod {
				t.Errorf("RetryPeriod = %v, want %v", got.RetryPeriod, retryPeriod)
			}
		})
	}
}
