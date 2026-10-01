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

package validation

import (
	"runtime"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeproxyconfigv1alpha1 "k8s.io/kube-proxy/config/v1alpha1"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
	"k8s.io/kubernetes/pkg/proxy/apis/config/scheme"
	"k8s.io/utils/ptr"
)

func TestValidateV1Alpha1FieldPaths(t *testing.T) {
	negativeDuration := metav1.Duration{Duration: -time.Second}
	for _, tc := range []struct {
		name           string
		mutate         func(*kubeproxyconfigv1alpha1.KubeProxyConfiguration)
		expectedFields []string
		linuxOnly      bool
	}{
		{
			name: "addresses and iptables",
			mutate: func(config *kubeproxyconfigv1alpha1.KubeProxyConfiguration) {
				config.BindAddress = "10.10.12.11:2000"
				config.HealthzBindAddress = "0.0.0.0"
				config.MetricsBindAddress = "127.0.0.1"
				config.IPTables.MasqueradeBit = ptr.To[int32](32)
			},
			expectedFields: []string{"bindAddress", "healthzBindAddress", "metricsBindAddress", "iptables.masqueradeBit"},
		},
		{
			name: "flattened linux fields",
			mutate: func(config *kubeproxyconfigv1alpha1.KubeProxyConfiguration) {
				config.OOMScoreAdj = ptr.To[int32](1001)
				config.Conntrack.MaxPerCore = ptr.To[int32](-1)
				config.Conntrack.Min = ptr.To[int32](-1)
				config.Conntrack.TCPEstablishedTimeout = &negativeDuration
				config.Conntrack.TCPCloseWaitTimeout = &negativeDuration
				config.Conntrack.UDPTimeout = negativeDuration
				config.Conntrack.UDPStreamTimeout = negativeDuration
			},
			expectedFields: []string{"oomScoreAdj", "conntrack.maxPerCore", "conntrack.min", "conntrack.tcpEstablishedTimeout", "conntrack.tcpCloseWaitTimeout", "conntrack.udpTimeout", "conntrack.udpStreamTimeout"},
		},
		{
			name: "ipvs",
			mutate: func(config *kubeproxyconfigv1alpha1.KubeProxyConfiguration) {
				config.Mode = "ipvs"
				config.IPVS.TCPTimeout = negativeDuration
				config.IPVS.TCPFinTimeout = negativeDuration
				config.IPVS.UDPTimeout = negativeDuration
				config.IPVS.ExcludeCIDRs = []string{"192.168.0.0/16", "invalid"}
			},
			expectedFields: []string{"ipvs.tcpTimeout", "ipvs.tcpFinTimeout", "ipvs.udpTimeout", "ipvs.excludeCIDRs[1]"},
			linuxOnly:      true,
		},
		{
			name: "nftables",
			mutate: func(config *kubeproxyconfigv1alpha1.KubeProxyConfiguration) {
				config.Mode = "nftables"
				config.NFTables.MasqueradeBit = ptr.To[int32](32)
			},
			expectedFields: []string{"nftables.masqueradeBit"},
			linuxOnly:      true,
		},
		{
			name: "other top level fields",
			mutate: func(config *kubeproxyconfigv1alpha1.KubeProxyConfiguration) {
				config.FeatureGates = map[string]bool{"UnknownFeature": true}
				config.ClientConnection.Burst = -1
				config.ConfigSyncPeriod = negativeDuration
				config.Mode = "invalid"
				config.DetectLocalMode = "invalid"
				config.NodePortAddresses = []string{"192.168.0.0/16", "invalid"}
				config.ShowHiddenMetricsForVersion = "invalid"
				config.Logging.Format = "invalid"
			},
			expectedFields: []string{"featureGates", "clientConnection.burst", "configSyncPeriod", "mode", "detectLocalMode", "nodePortAddresses[1]", "showHiddenMetricsForVersion", "logging.format"},
		},
		{
			name: "bridge interface",
			mutate: func(config *kubeproxyconfigv1alpha1.KubeProxyConfiguration) {
				config.DetectLocalMode = "BridgeInterface"
			},
			expectedFields: []string{"detectLocal.bridgeInterface"},
		},
		{
			name: "interface name prefix",
			mutate: func(config *kubeproxyconfigv1alpha1.KubeProxyConfiguration) {
				config.DetectLocalMode = "InterfaceNamePrefix"
			},
			expectedFields: []string{"detectLocal.interfaceNamePrefix"},
		},
		{
			name: "malformed cluster CIDR",
			mutate: func(config *kubeproxyconfigv1alpha1.KubeProxyConfiguration) {
				config.DetectLocalMode = "ClusterCIDR"
				config.ClusterCIDR = "192.168.0.0/16,invalid"
			},
			expectedFields: []string{"clusterCIDR"},
		},
		{
			name: "single stack cluster CIDR pair",
			mutate: func(config *kubeproxyconfigv1alpha1.KubeProxyConfiguration) {
				config.DetectLocalMode = "ClusterCIDR"
				config.ClusterCIDR = "192.168.0.0/16,10.0.0.0/8"
			},
			expectedFields: []string{"clusterCIDR"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.linuxOnly && runtime.GOOS == "windows" {
				t.Skip("Proxy mode is not supported on Windows")
			}
			config := &kubeproxyconfigv1alpha1.KubeProxyConfiguration{}
			scheme.Scheme.Default(config)
			tc.mutate(config)
			internal := &kubeproxyconfig.KubeProxyConfiguration{}
			require.NoError(t, scheme.Scheme.Convert(config, internal, nil))

			var fields []string
			for _, err := range Validate(internal) {
				fields = append(fields, err.Field)
			}
			assert.ElementsMatch(t, tc.expectedFields, fields)
		})
	}
}

func TestValidateV1Alpha1SyncPeriodPaths(t *testing.T) {
	for _, mode := range []kubeproxyconfigv1alpha1.ProxyMode{"", "iptables", "ipvs", "nftables", "kernelspace"} {
		t.Run("mode="+string(mode), func(t *testing.T) {
			if mode != "" && (runtime.GOOS == "windows") != (mode == "kernelspace") {
				t.Skip("Proxy mode is not supported on this platform")
			}
			for _, tc := range []struct {
				name           string
				syncPeriod     time.Duration
				minSyncPeriod  time.Duration
				expectedField  string
				expectedDetail string
			}{
				{"nonpositive sync period", 0, 0, "syncPeriod", "must be greater than 0"},
				{"negative minimum sync period", time.Second, -time.Second, "minSyncPeriod", "must be greater than or equal to 0"},
				{"sync period below minimum", time.Second, 2 * time.Second, "syncPeriod", ""},
			} {
				t.Run(tc.name, func(t *testing.T) {
					config := &kubeproxyconfigv1alpha1.KubeProxyConfiguration{Mode: mode}
					scheme.Scheme.Default(config)
					syncPeriod, minSyncPeriod := &config.IPTables.SyncPeriod, &config.IPTables.MinSyncPeriod
					prefix := "iptables"
					switch mode {
					case "ipvs":
						syncPeriod, minSyncPeriod = &config.IPVS.SyncPeriod, &config.IPVS.MinSyncPeriod
						prefix = "ipvs"
					case "nftables":
						syncPeriod, minSyncPeriod = &config.NFTables.SyncPeriod, &config.NFTables.MinSyncPeriod
						prefix = "nftables"
					}
					syncPeriod.Duration, minSyncPeriod.Duration = tc.syncPeriod, tc.minSyncPeriod
					internal := &kubeproxyconfig.KubeProxyConfiguration{}
					require.NoError(t, scheme.Scheme.Convert(config, internal, nil))

					errs := Validate(internal)
					require.Len(t, errs, 1)
					assert.Equal(t, prefix+"."+tc.expectedField, errs[0].Field)
					expectedDetail := tc.expectedDetail
					if expectedDetail == "" {
						expectedDetail = "must be greater than or equal to " + prefix + ".minSyncPeriod"
					}
					assert.Equal(t, expectedDetail, errs[0].Detail)
				})
			}
		})
	}
}
