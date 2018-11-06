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

package options

import (
	"net"
	"reflect"
	"testing"
	"time"

	"github.com/spf13/pflag"

	apimachineryconfig "k8s.io/apimachinery/pkg/apis/config"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/diff"
	apiserverconfig "k8s.io/apiserver/pkg/apis/config"
	apiserveroptions "k8s.io/apiserver/pkg/server/options"
	cmoptions "k8s.io/kubernetes/cmd/controller-manager/app/options"
)

func TestDefaultFlags(t *testing.T) {
	s, _ := NewCloudControllerManagerOptions()

	expected := &CloudControllerManagerOptions{
		Generic: &cmoptions.GenericControllerManagerConfigurationOptions{
			Port:            DefaultInsecureCloudControllerManagerPort, // Note: InsecureServingOptions.ApplyTo will write the flag value back into the component config
			Address:         "0.0.0.0",                                 // Note: InsecureServingOptions.ApplyTo will write the flag value back into the component config
			MinResyncPeriod: metav1.Duration{Duration: 12 * time.Hour},
			ClientConnection: apimachineryconfig.ClientConnectionConfiguration{
				ContentType: "application/vnd.kubernetes.protobuf",
				QPS:         20.0,
				Burst:       30,
			},
			ControllerStartInterval: metav1.Duration{Duration: 0},
			LeaderElection: apiserverconfig.LeaderElectionConfiguration{
				ResourceLock:  "endpoints",
				LeaderElect:   true,
				LeaseDuration: metav1.Duration{Duration: 15 * time.Second},
				RenewDeadline: metav1.Duration{Duration: 10 * time.Second},
				RetryPeriod:   metav1.Duration{Duration: 2 * time.Second},
			},
			Debugging: &cmoptions.DebuggingOptions{
				EnableContentionProfiling: false,
			},
			Controllers: []string{"*"},
		},
		KubeCloudShared: &cmoptions.KubeCloudSharedOptions{
			RouteReconciliationPeriod: metav1.Duration{Duration: 10 * time.Second},
			NodeMonitorPeriod:         metav1.Duration{Duration: 5 * time.Second},
			ClusterName:               "kubernetes",
			ClusterCIDR:               "",
			AllocateNodeCIDRs:         false,
			CIDRAllocatorType:         "",
			ConfigureCloudRoutes:      true,
			CloudProvider: &cmoptions.CloudProviderOptions{
				Name:            "",
				CloudConfigFile: "",
			},
		},
		ServiceController: &cmoptions.ServiceControllerOptions{
			ConcurrentServiceSyncs: 1,
		},
		SecureServing: (&apiserveroptions.SecureServingOptions{
			BindPort:    10258,
			BindAddress: net.ParseIP("0.0.0.0"),
			ServerCert: apiserveroptions.GeneratableKeyCert{
				CertDirectory: "",
				PairName:      "cloud-controller-manager",
			},
			HTTP2MaxStreamsPerConnection: 0,
		}).WithLoopback(),
		InsecureServing: (&apiserveroptions.DeprecatedInsecureServingOptions{
			BindAddress: net.ParseIP("0.0.0.0"),
			BindPort:    int(0),
			BindNetwork: "tcp",
		}).WithLoopback(),
		Authentication: &apiserveroptions.DelegatingAuthenticationOptions{
			CacheTTL:   10 * time.Second,
			ClientCert: apiserveroptions.ClientCertAuthenticationOptions{},
			RequestHeader: apiserveroptions.RequestHeaderAuthenticationOptions{
				UsernameHeaders:     []string{"x-remote-user"},
				GroupHeaders:        []string{"x-remote-group"},
				ExtraHeaderPrefixes: []string{"x-remote-extra-"},
			},
			RemoteKubeConfigFileOptional: true,
		},
		Authorization: &apiserveroptions.DelegatingAuthorizationOptions{
			AllowCacheTTL:                10 * time.Second,
			DenyCacheTTL:                 10 * time.Second,
			RemoteKubeConfigFileOptional: true,
			AlwaysAllowPaths:             []string{"/healthz"}, // note: this does not match /healthz/ or
		},
		Kubeconfig:                "",
		Master:                    "",
		NodeStatusUpdateFrequency: metav1.Duration{Duration: 5 * time.Minute},
	}
	if !reflect.DeepEqual(expected, s) {
		t.Errorf("Got different run options than expected.\nDifference detected on:\n%s", diff.ObjectReflectDiff(expected, s))
	}
}

func TestAddFlags(t *testing.T) {
	fs := pflag.NewFlagSet("addflagstest", pflag.ContinueOnError)
	s, _ := NewCloudControllerManagerOptions()
	for _, f := range s.Flags().FlagSets {
		fs.AddFlagSet(f)
	}

	args := []string{
		"--address=192.168.4.10",
		"--allocate-node-cidrs=true",
		"--bind-address=192.168.4.21",
		"--cert-dir=/a/b/c",
		"--cloud-config=/cloud-config",
		"--cloud-provider=gce",
		"--cluster-cidr=1.2.3.4/24",
		"--cluster-name=k8s",
		"--configure-cloud-routes=false",
		"--contention-profiling=true",
		"--controller-start-interval=2m",
		"--http2-max-streams-per-connection=47",
		"--kube-api-burst=100",
		"--kube-api-content-type=application/vnd.kubernetes.protobuf",
		"--kube-api-qps=50.0",
		"--kubeconfig=/kubeconfig",
		"--leader-elect=false",
		"--leader-elect-lease-duration=30s",
		"--leader-elect-renew-deadline=15s",
		"--leader-elect-resource-lock=configmap",
		"--leader-elect-retry-period=5s",
		"--master=192.168.4.20",
		"--min-resync-period=100m",
		"--node-status-update-frequency=10m",
		"--port=10000",
		"--profiling=false",
		"--route-reconciliation-period=30s",
		"--secure-port=10001",
		"--use-service-account-credentials=false",
	}
	fs.Parse(args)

	expected := &CloudControllerManagerOptions{
		Generic: &cmoptions.GenericControllerManagerConfigurationOptions{
			Port:            DefaultInsecureCloudControllerManagerPort, // Note: InsecureServingOptions.ApplyTo will write the flag value back into the component config
			Address:         "0.0.0.0",                                 // Note: InsecureServingOptions.ApplyTo will write the flag value back into the component config
			MinResyncPeriod: metav1.Duration{Duration: 100 * time.Minute},
			ClientConnection: apimachineryconfig.ClientConnectionConfiguration{
				ContentType: "application/vnd.kubernetes.protobuf",
				QPS:         50.0,
				Burst:       100,
			},
			ControllerStartInterval: metav1.Duration{Duration: 2 * time.Minute},
			LeaderElection: apiserverconfig.LeaderElectionConfiguration{
				ResourceLock:  "configmap",
				LeaderElect:   false,
				LeaseDuration: metav1.Duration{Duration: 30 * time.Second},
				RenewDeadline: metav1.Duration{Duration: 15 * time.Second},
				RetryPeriod:   metav1.Duration{Duration: 5 * time.Second},
			},
			Debugging: &cmoptions.DebuggingOptions{
				EnableContentionProfiling: true,
			},
			Controllers: []string{"*"},
		},
		KubeCloudShared: &cmoptions.KubeCloudSharedOptions{
			CloudProvider: &cmoptions.CloudProviderOptions{
				Name:            "gce",
				CloudConfigFile: "/cloud-config",
			},
			RouteReconciliationPeriod: metav1.Duration{Duration: 30 * time.Second},
			NodeMonitorPeriod:         metav1.Duration{Duration: 5 * time.Second},
			ClusterName:               "k8s",
			ClusterCIDR:               "1.2.3.4/24",
			AllocateNodeCIDRs:         true,
			CIDRAllocatorType:         "RangeAllocator",
			ConfigureCloudRoutes:      false,
		},
		ServiceController: &cmoptions.ServiceControllerOptions{
			ConcurrentServiceSyncs: 1,
		},
		SecureServing: (&apiserveroptions.SecureServingOptions{
			BindPort:    10001,
			BindAddress: net.ParseIP("192.168.4.21"),
			ServerCert: apiserveroptions.GeneratableKeyCert{
				CertDirectory: "/a/b/c",
				PairName:      "cloud-controller-manager",
			},
			HTTP2MaxStreamsPerConnection: 47,
		}).WithLoopback(),
		InsecureServing: (&apiserveroptions.DeprecatedInsecureServingOptions{
			BindAddress: net.ParseIP("192.168.4.10"),
			BindPort:    int(10000),
			BindNetwork: "tcp",
		}).WithLoopback(),
		Authentication: &apiserveroptions.DelegatingAuthenticationOptions{
			CacheTTL:   10 * time.Second,
			ClientCert: apiserveroptions.ClientCertAuthenticationOptions{},
			RequestHeader: apiserveroptions.RequestHeaderAuthenticationOptions{
				UsernameHeaders:     []string{"x-remote-user"},
				GroupHeaders:        []string{"x-remote-group"},
				ExtraHeaderPrefixes: []string{"x-remote-extra-"},
			},
			RemoteKubeConfigFileOptional: true,
		},
		Authorization: &apiserveroptions.DelegatingAuthorizationOptions{
			AllowCacheTTL:                10 * time.Second,
			DenyCacheTTL:                 10 * time.Second,
			RemoteKubeConfigFileOptional: true,
			AlwaysAllowPaths:             []string{"/healthz"}, // note: this does not match /healthz/ or
		},
		Kubeconfig:                "/kubeconfig",
		Master:                    "192.168.4.20",
		NodeStatusUpdateFrequency: metav1.Duration{Duration: 10 * time.Minute},
	}
	if !reflect.DeepEqual(expected, s) {
		t.Errorf("Got different run options than expected.\nDifference detected on:\n%s", diff.ObjectReflectDiff(expected, s))
	}
}
