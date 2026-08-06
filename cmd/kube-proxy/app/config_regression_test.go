package app

import (
	"os"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/spf13/pflag"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
)

func TestConfigRegression(t *testing.T) {
	testCases := []struct {
		name     string
		config   string
		flags    []string
		expected func() *kubeproxyconfig.KubeProxyConfiguration
	}{
		{
			name: "kubeadm configuration with iptables mode",
			config: `apiVersion: kubeproxy.config.k8s.io/v1alpha1
bindAddress: 0.0.0.0
clientConnection:
  acceptContentTypes: ""
  burst: 0
  contentType: ""
  kubeconfig: /var/lib/kube-proxy/kubeconfig.conf
  qps: 0
clusterCIDR: 10.244.0.0/16
configSyncPeriod: 0s
conntrack:
  maxPerCore: null
  min: null
  tcpCloseWaitTimeout: null
  tcpEstablishedTimeout: null
detectLocalMode: ""
enableProfiling: false
healthzBindAddress: ""
hostnameOverride: ""
iptables:
  masqueradeAll: false
  masqueradeBit: null
  minSyncPeriod: 0s
  syncPeriod: 0s
ipvs:
  excludeCIDRs: null
  minSyncPeriod: 0s
  scheduler: ""
  strictARP: false
  syncPeriod: 0s
  tcpFinTimeout: 0s
  tcpTimeout: 0s
  udpTimeout: 0s
kind: KubeProxyConfiguration
metricsBindAddress: ""
mode: "iptables"
nodePortAddresses: null
oomScoreAdj: null
portRange: ""
showHiddenMetricsForVersion: ""
`,
			flags: []string{
				"--hostname-override=node1",
			},
			expected: func() *kubeproxyconfig.KubeProxyConfiguration {
				cfg := newKubeProxyConfiguration()
				cfg.BindAddress = "0.0.0.0"
				cfg.ClientConnection.Kubeconfig = "/var/lib/kube-proxy/kubeconfig.conf"
				cfg.DetectLocal.ClusterCIDRs = []string{"10.244.0.0/16"}
				cfg.DetectLocalMode = kubeproxyconfig.LocalModeClusterCIDR
				cfg.Mode = "iptables"
				cfg.HostnameOverride = "node1"
				return cfg
			},
		},
		{
			name: "legacy flag-based e2e configuration",
			config: "",
			flags: []string{
				"--kubeconfig=/var/lib/kube-proxy/kubeconfig.conf",
				"--cluster-cidr=10.244.0.0/16",
				"--proxy-mode=ipvs",
				"--hostname-override=node1",
				"--ipvs-scheduler=rr",
				"--ipvs-strict-arp=true",
			},
			expected: func() *kubeproxyconfig.KubeProxyConfiguration {
				cfg := newKubeProxyConfiguration()
				cfg.ClientConnection.Kubeconfig = "/var/lib/kube-proxy/kubeconfig.conf"
				cfg.DetectLocal.ClusterCIDRs = []string{"10.244.0.0/16"}
				cfg.DetectLocalMode = kubeproxyconfig.LocalModeClusterCIDR
				cfg.Mode = "ipvs"
				cfg.HostnameOverride = "node1"
				cfg.IPVS.Scheduler = "rr"
				cfg.IPVS.StrictARP = true
				return cfg
			},
		},
		{
			name: "IPv6 configuration",
			config: `apiVersion: kubeproxy.config.k8s.io/v1alpha1
bindAddress: "::"
clientConnection:
  kubeconfig: /var/lib/kube-proxy/kubeconfig.conf
clusterCIDR: fd00:1::/64
kind: KubeProxyConfiguration
metricsBindAddress: "[::1]:10249"
mode: "iptables"
`,
			flags: []string{
				"--hostname-override=node1",
			},
			expected: func() *kubeproxyconfig.KubeProxyConfiguration {
				cfg := newKubeProxyConfiguration()
				cfg.BindAddress = "::"
				cfg.ClientConnection.Kubeconfig = "/var/lib/kube-proxy/kubeconfig.conf"
				cfg.DetectLocal.ClusterCIDRs = []string{"fd00:1::/64"}
				cfg.DetectLocalMode = kubeproxyconfig.LocalModeClusterCIDR
				cfg.MetricsBindAddress = "[::1]:10249"
				cfg.HealthzBindAddress = "[::]:10256"
				cfg.Mode = "iptables"
				cfg.HostnameOverride = "node1"
				return cfg
			},
		},
		{
			name: "Dual-stack configuration",
			config: `apiVersion: kubeproxy.config.k8s.io/v1alpha1
bindAddress: 0.0.0.0
clientConnection:
  kubeconfig: /var/lib/kube-proxy/kubeconfig.conf
clusterCIDR: 10.244.0.0/16,fd00:1::/64
kind: KubeProxyConfiguration
mode: "iptables"
`,
			flags: []string{
				"--hostname-override=node1",
			},
			expected: func() *kubeproxyconfig.KubeProxyConfiguration {
				cfg := newKubeProxyConfiguration()
				cfg.BindAddress = "0.0.0.0"
				cfg.ClientConnection.Kubeconfig = "/var/lib/kube-proxy/kubeconfig.conf"
				cfg.DetectLocal.ClusterCIDRs = []string{"10.244.0.0/16", "fd00:1::/64"}
				cfg.DetectLocalMode = kubeproxyconfig.LocalModeClusterCIDR
				cfg.Mode = "iptables"
				cfg.HostnameOverride = "node1"
				return cfg
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			opts := NewOptions()
			fs := pflag.NewFlagSet("test", pflag.ContinueOnError)
			opts.AddFlags(fs)

			if tc.config != "" {
				file, err := os.CreateTemp("", "kube-proxy-config-*.yaml")
				if err != nil {
					t.Fatalf("failed to create temp file: %v", err)
				}
				defer os.Remove(file.Name())

				if _, err := file.WriteString(tc.config); err != nil {
					t.Fatalf("failed to write config to temp file: %v", err)
				}
				file.Close()

				tc.flags = append(tc.flags, "--config="+file.Name())
			}

			if err := fs.Parse(tc.flags); err != nil {
				t.Fatalf("failed to parse flags: %v", err)
			}

			if err := opts.Complete(fs); err != nil {
				t.Fatalf("failed to complete options: %v", err)
			}

			expectedConfig := tc.expected()
			
			if diff := cmp.Diff(expectedConfig, opts.config); diff != "" {
				t.Errorf("Resulting config differs from expected:\n%s", diff)
			}
		})
	}
}
