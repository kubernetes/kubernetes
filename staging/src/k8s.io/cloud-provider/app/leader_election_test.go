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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/leaderelection"
	cloudcontrollerconfig "k8s.io/cloud-provider/app/config"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	cmfeatures "k8s.io/controller-manager/pkg/features"
	_ "k8s.io/controller-manager/pkg/features/register"
)

func TestLeaderElectionConfigFor(t *testing.T) {
	const (
		leaseDuration = 15 * time.Second
		renewDeadline = 10 * time.Second
		retryPeriod   = 2 * time.Second
		leaseName     = "cloud-controller-manager"
	)
	config := &cloudcontrollerconfig.Config{}
	config.ComponentConfig.Generic.LeaderElection.LeaseDuration = metav1.Duration{Duration: leaseDuration}
	config.ComponentConfig.Generic.LeaderElection.RenewDeadline = metav1.Duration{Duration: renewDeadline}
	config.ComponentConfig.Generic.LeaderElection.RetryPeriod = metav1.Duration{Duration: retryPeriod}
	c := config.Complete()

	for _, tc := range []struct {
		name                string
		featureEnabled      bool
		wantReleaseOnCancel bool
	}{
		{name: "feature gate disabled", featureEnabled: false, wantReleaseOnCancel: false},
		{name: "feature gate enabled", featureEnabled: true, wantReleaseOnCancel: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, cmfeatures.ControllerManagerReleaseLeaderElectionLockOnExit, tc.featureEnabled)

			got := leaderElectionConfigFor(c, nil, leaseName, leaderelection.LeaderCallbacks{}, nil)

			if got.ReleaseOnCancel != tc.wantReleaseOnCancel {
				t.Errorf("ReleaseOnCancel = %v, want %v", got.ReleaseOnCancel, tc.wantReleaseOnCancel)
			}
			if got.Name != leaseName {
				t.Errorf("Name = %q, want %q", got.Name, leaseName)
			}
			if got.LeaseDuration != leaseDuration || got.RenewDeadline != renewDeadline || got.RetryPeriod != retryPeriod {
				t.Errorf("durations = %v/%v/%v, want %v/%v/%v", got.LeaseDuration, got.RenewDeadline, got.RetryPeriod, leaseDuration, renewDeadline, retryPeriod)
			}
		})
	}
}
