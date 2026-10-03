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
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/legacyregistry"
	metricstestutil "k8s.io/component-base/metrics/testutil"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	critesting "k8s.io/cri-api/pkg/apis/testing"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/cmd/kubelet/app/options"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubeletmetrics "k8s.io/kubernetes/pkg/kubelet/metrics"
)

func TestGetCgroupDriverFromCRI(t *testing.T) {
	for _, tc := range []struct {
		name             string
		allowFallback    bool
		configuredDriver string
		runtimeConfig    *runtimeapi.LinuxRuntimeConfiguration
		runtimeError     error
		expectedDriver   string
		expectedError    string
	}{
		{
			name:             "unsupported runtime is rejected by default",
			configuredDriver: "systemd",
			runtimeError:     status.Error(codes.Unimplemented, "RuntimeConfig is not implemented"),
			expectedDriver:   "systemd",
			expectedError:    "DisableCgroupDriverFallback=false",
		},
		{
			name:             "unsupported runtime can fall back to systemd",
			allowFallback:    true,
			configuredDriver: "systemd",
			runtimeError:     status.Error(codes.Unimplemented, "RuntimeConfig is not implemented"),
			expectedDriver:   "systemd",
		},
		{
			name:             "unsupported runtime can fall back to cgroupfs",
			allowFallback:    true,
			configuredDriver: "cgroupfs",
			runtimeError:     status.Error(codes.Unimplemented, "RuntimeConfig is not implemented"),
			expectedDriver:   "cgroupfs",
		},
		{
			name:             "CRI systemd overrides the configured driver",
			configuredDriver: "cgroupfs",
			runtimeConfig:    &runtimeapi.LinuxRuntimeConfiguration{CgroupDriver: runtimeapi.CgroupDriver_SYSTEMD},
			expectedDriver:   "systemd",
		},
		{
			name:             "CRI cgroupfs overrides the configured driver even with fallback enabled",
			allowFallback:    true,
			configuredDriver: "systemd",
			runtimeConfig:    &runtimeapi.LinuxRuntimeConfiguration{CgroupDriver: runtimeapi.CgroupDriver_CGROUPFS},
			expectedDriver:   "cgroupfs",
		},
		{
			name:             "other runtime errors are not masked by fallback",
			allowFallback:    true,
			configuredDriver: "systemd",
			runtimeError:     status.Error(codes.Unavailable, "runtime unavailable"),
			expectedDriver:   "systemd",
			expectedError:    "runtime unavailable",
		},
		{
			name:             "unknown driver is not masked by fallback",
			allowFallback:    true,
			configuredDriver: "systemd",
			runtimeConfig:    &runtimeapi.LinuxRuntimeConfiguration{CgroupDriver: runtimeapi.CgroupDriver(100)},
			expectedDriver:   "systemd",
			expectedError:    "runtime returned an unknown cgroup driver 100",
		},
		{
			name:             "response without Linux config preserves the configured driver",
			configuredDriver: "cgroupfs",
			expectedDriver:   "cgroupfs",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.allowFallback {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DisableCgroupDriverFallback, false)
			} else {
				require.True(t, utilfeature.DefaultFeatureGate.Enabled(features.DisableCgroupDriverFallback))
			}
			runtime := critesting.NewFakeRuntimeService()
			runtime.FakeLinuxConfiguration = tc.runtimeConfig
			if tc.runtimeError != nil {
				for range 3 {
					runtime.InjectError("RuntimeConfig", tc.runtimeError)
				}
			}
			server := &options.KubeletServer{
				KubeletConfiguration: kubeletconfig.KubeletConfiguration{CgroupDriver: tc.configuredDriver},
			}
			_, ctx := ktesting.NewTestContext(t)
			err := getCgroupDriverFromCRI(ctx, server, &kubelet.Dependencies{RemoteRuntimeService: runtime})
			if err == nil && status.Code(tc.runtimeError) == codes.Unimplemented {
				// Startup normally runs once; each fallback subtest registers the metric again.
				t.Cleanup(func() {
					legacyregistry.Registerer().Unregister(kubeletmetrics.CRILosingSupport)
					kubeletmetrics.CRILosingSupport.Reset()
				})
			}
			if tc.expectedError != "" {
				require.ErrorContains(t, err, tc.expectedError)
				if tc.runtimeError != nil {
					require.Equal(t, status.Code(tc.runtimeError), status.Code(err))
				}
			} else {
				require.NoError(t, err)
			}
			require.Equal(t, tc.expectedDriver, server.CgroupDriver)
			if status.Code(tc.runtimeError) == codes.Unimplemented {
				require.Equal(t, []string{"RuntimeConfig"}, runtime.GetCalls(), "unsupported RPCs should not be retried")
			}
			expectedMetrics := ""
			if tc.allowFallback && status.Code(tc.runtimeError) == codes.Unimplemented {
				expectedMetrics = `
# HELP kubelet_cri_losing_support [ALPHA] the Kubernetes version that the currently running CRI implementation will lose support on if not upgraded.
# TYPE kubelet_cri_losing_support gauge
kubelet_cri_losing_support{version="1.38.0"} 1
`
			}
			require.NoError(t, metricstestutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expectedMetrics), kubeletmetrics.KubeletSubsystem+"_"+kubeletmetrics.CRILosingSupportKey))
		})
	}
}
