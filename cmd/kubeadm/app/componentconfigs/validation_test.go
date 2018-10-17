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

package componentconfigs

import (
	"strings"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
	utilpointer "k8s.io/utils/pointer"
)

func TestValidateKubeProxyConfiguration(t *testing.T) {
	var tests = []struct {
		clusterConfig *kubeadm.ClusterConfiguration
		msg           string
		expectErr     bool
	}{
		{
			clusterConfig: &kubeadm.ClusterConfiguration{
				ComponentConfigs: kubeadm.ComponentConfigs{
					KubeProxy: &kubeproxyconfig.KubeProxyConfiguration{
						BindAddress:        "192.168.59.103",
						HealthzBindAddress: "0.0.0.0:10256",
						MetricsBindAddress: "127.0.0.1:10249",
						ClusterCIDR:        "192.168.59.0/24",
						UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
						ConfigSyncPeriod:   metav1.Duration{Duration: 1 * time.Second},
						IPTables: kubeproxyconfig.KubeProxyIPTablesConfiguration{
							MasqueradeAll: true,
							SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
						},
						IPVS: kubeproxyconfig.KubeProxyIPVSConfiguration{
							SyncPeriod:    metav1.Duration{Duration: 10 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 5 * time.Second},
						},
						Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
							Max:                   utilpointer.Int32Ptr(2),
							MaxPerCore:            utilpointer.Int32Ptr(1),
							Min:                   utilpointer.Int32Ptr(1),
							TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
							TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
						},
					},
				},
			},
			expectErr: false,
		},
		{
			clusterConfig: &kubeadm.ClusterConfiguration{
				ComponentConfigs: kubeadm.ComponentConfigs{
					KubeProxy: &kubeproxyconfig.KubeProxyConfiguration{
						// only BindAddress is invalid
						BindAddress:        "10.10.12.11:2000",
						HealthzBindAddress: "0.0.0.0:10256",
						MetricsBindAddress: "127.0.0.1:10249",
						ClusterCIDR:        "192.168.59.0/24",
						UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
						ConfigSyncPeriod:   metav1.Duration{Duration: 1 * time.Second},
						IPTables: kubeproxyconfig.KubeProxyIPTablesConfiguration{
							MasqueradeAll: true,
							SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
						},
						IPVS: kubeproxyconfig.KubeProxyIPVSConfiguration{
							SyncPeriod:    metav1.Duration{Duration: 10 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 5 * time.Second},
						},
						Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
							Max:                   utilpointer.Int32Ptr(2),
							MaxPerCore:            utilpointer.Int32Ptr(1),
							Min:                   utilpointer.Int32Ptr(1),
							TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
							TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
						},
					},
				},
			},
			msg:       "not a valid textual representation of an IP address",
			expectErr: true,
		},
		{
			clusterConfig: &kubeadm.ClusterConfiguration{
				ComponentConfigs: kubeadm.ComponentConfigs{
					KubeProxy: &kubeproxyconfig.KubeProxyConfiguration{
						BindAddress: "10.10.12.11",
						// only HealthzBindAddress is invalid
						HealthzBindAddress: "0.0.0.0",
						MetricsBindAddress: "127.0.0.1:10249",
						ClusterCIDR:        "192.168.59.0/24",
						UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
						ConfigSyncPeriod:   metav1.Duration{Duration: 1 * time.Second},
						IPTables: kubeproxyconfig.KubeProxyIPTablesConfiguration{
							MasqueradeAll: true,
							SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
						},
						IPVS: kubeproxyconfig.KubeProxyIPVSConfiguration{
							SyncPeriod:    metav1.Duration{Duration: 10 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 5 * time.Second},
						},
						Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
							Max:                   utilpointer.Int32Ptr(2),
							MaxPerCore:            utilpointer.Int32Ptr(1),
							Min:                   utilpointer.Int32Ptr(1),
							TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
							TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
						},
					},
				},
			},
			msg:       "must be IP:port",
			expectErr: true,
		},
		{
			clusterConfig: &kubeadm.ClusterConfiguration{
				ComponentConfigs: kubeadm.ComponentConfigs{
					KubeProxy: &kubeproxyconfig.KubeProxyConfiguration{
						BindAddress:        "10.10.12.11",
						HealthzBindAddress: "0.0.0.0:12345",
						// only MetricsBindAddress is invalid
						MetricsBindAddress: "127.0.0.1",
						ClusterCIDR:        "192.168.59.0/24",
						UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
						ConfigSyncPeriod:   metav1.Duration{Duration: 1 * time.Second},
						IPTables: kubeproxyconfig.KubeProxyIPTablesConfiguration{
							MasqueradeAll: true,
							SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
						},
						IPVS: kubeproxyconfig.KubeProxyIPVSConfiguration{
							SyncPeriod:    metav1.Duration{Duration: 10 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 5 * time.Second},
						},
						Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
							Max:                   utilpointer.Int32Ptr(2),
							MaxPerCore:            utilpointer.Int32Ptr(1),
							Min:                   utilpointer.Int32Ptr(1),
							TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
							TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
						},
					},
				},
			},
			msg:       "must be IP:port",
			expectErr: true,
		},
		{
			clusterConfig: &kubeadm.ClusterConfiguration{
				ComponentConfigs: kubeadm.ComponentConfigs{
					KubeProxy: &kubeproxyconfig.KubeProxyConfiguration{
						BindAddress:        "10.10.12.11",
						HealthzBindAddress: "0.0.0.0:12345",
						MetricsBindAddress: "127.0.0.1:10249",
						// only ClusterCIDR is invalid
						ClusterCIDR:      "192.168.59.0",
						UDPIdleTimeout:   metav1.Duration{Duration: 1 * time.Second},
						ConfigSyncPeriod: metav1.Duration{Duration: 1 * time.Second},
						IPTables: kubeproxyconfig.KubeProxyIPTablesConfiguration{
							MasqueradeAll: true,
							SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
						},
						IPVS: kubeproxyconfig.KubeProxyIPVSConfiguration{
							SyncPeriod:    metav1.Duration{Duration: 10 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 5 * time.Second},
						},
						Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
							Max:                   utilpointer.Int32Ptr(2),
							MaxPerCore:            utilpointer.Int32Ptr(1),
							Min:                   utilpointer.Int32Ptr(1),
							TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
							TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
						},
					},
				},
			},
			msg:       "must be a valid CIDR block (e.g. 10.100.0.0/16)",
			expectErr: true,
		},
		{
			clusterConfig: &kubeadm.ClusterConfiguration{
				ComponentConfigs: kubeadm.ComponentConfigs{
					KubeProxy: &kubeproxyconfig.KubeProxyConfiguration{
						BindAddress:        "10.10.12.11",
						HealthzBindAddress: "0.0.0.0:12345",
						MetricsBindAddress: "127.0.0.1:10249",
						ClusterCIDR:        "192.168.59.0/24",
						// only UDPIdleTimeout is invalid
						UDPIdleTimeout:   metav1.Duration{Duration: -1 * time.Second},
						ConfigSyncPeriod: metav1.Duration{Duration: 1 * time.Second},
						IPTables: kubeproxyconfig.KubeProxyIPTablesConfiguration{
							MasqueradeAll: true,
							SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
						},
						IPVS: kubeproxyconfig.KubeProxyIPVSConfiguration{
							SyncPeriod:    metav1.Duration{Duration: 10 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 5 * time.Second},
						},
						Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
							Max:                   utilpointer.Int32Ptr(2),
							MaxPerCore:            utilpointer.Int32Ptr(1),
							Min:                   utilpointer.Int32Ptr(1),
							TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
							TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
						},
					},
				},
			},
			msg:       "must be greater than 0",
			expectErr: true,
		},
		{
			clusterConfig: &kubeadm.ClusterConfiguration{
				ComponentConfigs: kubeadm.ComponentConfigs{
					KubeProxy: &kubeproxyconfig.KubeProxyConfiguration{
						BindAddress:        "10.10.12.11",
						HealthzBindAddress: "0.0.0.0:12345",
						MetricsBindAddress: "127.0.0.1:10249",
						ClusterCIDR:        "192.168.59.0/24",
						UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
						// only ConfigSyncPeriod is invalid
						ConfigSyncPeriod: metav1.Duration{Duration: -1 * time.Second},
						IPTables: kubeproxyconfig.KubeProxyIPTablesConfiguration{
							MasqueradeAll: true,
							SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
						},
						IPVS: kubeproxyconfig.KubeProxyIPVSConfiguration{
							SyncPeriod:    metav1.Duration{Duration: 10 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 5 * time.Second},
						},
						Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
							Max:                   utilpointer.Int32Ptr(2),
							MaxPerCore:            utilpointer.Int32Ptr(1),
							Min:                   utilpointer.Int32Ptr(1),
							TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
							TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
						},
					},
				},
			},
			msg:       "must be greater than 0",
			expectErr: true,
		},
	}
	for i, rt := range tests {
		err := ValidateKubeProxyConfiguration(rt.clusterConfig, nil).ToAggregate()
		if (err != nil) != rt.expectErr {
			t.Errorf("%d failed ValidateKubeProxyConfiguration: expected error %t, got error %t", i, rt.expectErr, err != nil)
		}
		if err != nil && !strings.Contains(err.Error(), rt.msg) {
			t.Errorf("%d failed ValidateKubeProxyConfiguration: unexpected error: %v, expected: %s", i, err, rt.msg)
		}
	}
}

func TestValidateKubeletConfiguration(t *testing.T) {
	var tests = []struct {
		clusterConfig *kubeadm.ClusterConfiguration
		expectErr     bool
	}{
		{
			clusterConfig: &kubeadm.ClusterConfiguration{
				ComponentConfigs: kubeadm.ComponentConfigs{
					Kubelet: &kubeletconfig.KubeletConfiguration{
						CgroupsPerQOS:               true,
						EnforceNodeAllocatable:      []string{"pods", "system-reserved", "kube-reserved"},
						SystemCgroups:               "",
						CgroupRoot:                  "",
						EventBurst:                  10,
						EventRecordQPS:              5,
						HealthzPort:                 kubeadmconstants.KubeletHealthzPort,
						ImageGCHighThresholdPercent: 85,
						ImageGCLowThresholdPercent:  80,
						IPTablesDropBit:             15,
						IPTablesMasqueradeBit:       14,
						KubeAPIBurst:                10,
						KubeAPIQPS:                  5,
						MaxOpenFiles:                1000000,
						MaxPods:                     110,
						OOMScoreAdj:                 -999,
						PodsPerCore:                 100,
						Port:                        65535,
						ReadOnlyPort:                0,
						RegistryBurst:               10,
						RegistryPullQPS:             5,
						HairpinMode:                 "promiscuous-bridge",
						NodeLeaseDurationSeconds:    40,
					},
				},
			},
			expectErr: false,
		},
		{
			clusterConfig: &kubeadm.ClusterConfiguration{
				ComponentConfigs: kubeadm.ComponentConfigs{
					Kubelet: &kubeletconfig.KubeletConfiguration{
						CgroupsPerQOS:               false,
						EnforceNodeAllocatable:      []string{"pods", "system-reserved", "kube-reserved", "illegal-key"},
						SystemCgroups:               "/",
						CgroupRoot:                  "",
						EventBurst:                  -10,
						EventRecordQPS:              -10,
						HealthzPort:                 -10,
						ImageGCHighThresholdPercent: 101,
						ImageGCLowThresholdPercent:  101,
						IPTablesDropBit:             -10,
						IPTablesMasqueradeBit:       -10,
						KubeAPIBurst:                -10,
						KubeAPIQPS:                  -10,
						MaxOpenFiles:                -10,
						MaxPods:                     -10,
						OOMScoreAdj:                 -1001,
						PodsPerCore:                 -10,
						Port:                        0,
						ReadOnlyPort:                -10,
						RegistryBurst:               -10,
						RegistryPullQPS:             -10,
					},
				},
			},
			expectErr: true,
		},
	}
	for i, rt := range tests {
		err := ValidateKubeletConfiguration(rt.clusterConfig, nil).ToAggregate()
		if (err != nil) != rt.expectErr {
			t.Errorf("%d failed ValidateKubeletConfiguration: expected error %t, got error %t", i, rt.expectErr, err != nil)
		}
	}
}
