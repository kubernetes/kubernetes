/*
Copyright 2024 The Kubernetes Authors.

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

package v1alpha2

import (
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	componentbaseconfig "k8s.io/component-base/config/v1alpha1"
	logsapi "k8s.io/component-base/logs/api/v1"
	kubeproxyconfigv1alpha2 "k8s.io/kube-proxy/config/v1alpha2"
	"k8s.io/utils/ptr"
)

func TestDefaultsKubeProxyConfiguration(t *testing.T) {
	oomScore := int32(-999)
	ctMaxPerCore := int32(32768)
	ctMin := int32(131072)
	testCases := []struct {
		name     string
		original *kubeproxyconfigv1alpha2.KubeProxyConfiguration
		expected *kubeproxyconfigv1alpha2.KubeProxyConfiguration
	}{
		{
			name:     "empty-config",
			original: &kubeproxyconfigv1alpha2.KubeProxyConfiguration{},
			expected: &kubeproxyconfigv1alpha2.KubeProxyConfiguration{
				FeatureGates:         map[string]bool{},
				HealthzBindAddresses: []string{"0.0.0.0/0", "::/0"},
				HealthzBindPort:      10256,
				MetricsBindAddresses: []string{"127.0.0.0/8", "::1/128"},
				MetricsBindPort:      10249,
				ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
					ContentType: "application/vnd.kubernetes.protobuf",
					QPS:         5,
					Burst:       10,
				},
				IPTables: kubeproxyconfigv1alpha2.KubeProxyIPTablesConfiguration{
					MasqueradeBit:      ptr.To[int32](14),
					LocalhostNodePorts: ptr.To(false),
				},
				IPVS: kubeproxyconfigv1alpha2.KubeProxyIPVSConfiguration{
					MasqueradeBit: ptr.To[int32](14),
				},
				NFTables: kubeproxyconfigv1alpha2.KubeProxyNFTablesConfiguration{
					MasqueradeBit: ptr.To[int32](14),
				},
				Linux: kubeproxyconfigv1alpha2.KubeProxyLinuxConfiguration{
					MasqueradeAll: false,
					OOMScoreAdj:   &oomScore,
					Conntrack: kubeproxyconfigv1alpha2.KubeProxyConntrackConfiguration{
						MaxPerCore:            &ctMaxPerCore,
						Min:                   &ctMin,
						TCPEstablishedTimeout: &metav1.Duration{Duration: 24 * time.Hour},
						TCPCloseWaitTimeout:   &metav1.Duration{Duration: 1 * time.Hour},
					},
				},
				ConfigHardFail:   ptr.To(true),
				SyncPeriod:       metav1.Duration{Duration: 30 * time.Second},
				MinSyncPeriod:    metav1.Duration{Duration: 1 * time.Second},
				ConfigSyncPeriod: metav1.Duration{Duration: 15 * time.Minute},
				Logging: logsapi.LoggingConfiguration{
					Format:         "text",
					FlushFrequency: logsapi.TimeOrMetaDuration{Duration: metav1.Duration{Duration: 5 * time.Second}, SerializeAsString: true},
				},
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			SetDefaults_KubeProxyConfiguration(tc.original)
			if diff := cmp.Diff(tc.expected, tc.original); diff != "" {
				t.Errorf("Got unexpected defaults (-want, +got):\n%s", diff)
			}
		})
	}
}
