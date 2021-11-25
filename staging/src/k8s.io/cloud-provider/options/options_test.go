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
	"reflect"
	"testing"
	"time"

	"github.com/spf13/pflag"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/diff"
	apiserveroptions "k8s.io/apiserver/pkg/server/options"
	cpconfig "k8s.io/cloud-provider/config"
	serviceconfig "k8s.io/cloud-provider/controllers/service/config"
	componentbaseconfig "k8s.io/component-base/config"
	cmconfig "k8s.io/controller-manager/config"
	cmoptions "k8s.io/controller-manager/options"
	migration "k8s.io/controller-manager/pkg/leadermigration/options"
	netutils "k8s.io/utils/net"
)

func TestDefaultFlags(t *testing.T) {
	s, _ := NewCloudControllerManagerOptions()

	expected := &CloudControllerManagerOptions{
		Generic: &cmoptions.GenericControllerManagerConfigurationOptions{
			GenericControllerManagerConfiguration: &cmconfig.GenericControllerManagerConfiguration{
				Port:            DefaultInsecureCloudControllerManagerPort, // Note: InsecureServingOptions.ApplyTo will write the flag value back into the component config
				Address:         "0.0.0.0",                                 // Note: InsecureServingOptions.ApplyTo will write the flag value back into the component config
				MinResyncPeriod: metav1.Duration{Duration: 12 * time.Hour},
				ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
					ContentType: "application/vnd.kubernetes.protobuf",
					QPS:         20.0,
					Burst:       30,
				},
				ControllerStartInterval: metav1.Duration{Duration: 0},
				LeaderElection: componentbaseconfig.LeaderElectionConfiguration{
					ResourceLock:      "leases",
					LeaderElect:       true,
					LeaseDuration:     metav1.Duration{Duration: 15 * time.Second},
					RenewDeadline:     metav1.Duration{Duration: 10 * time.Second},
					RetryPeriod:       metav1.Duration{Duration: 2 * time.Second},
					ResourceName:      "cloud-controller-manager",
					ResourceNamespace: "kube-system",
				},
				Controllers: []string{"*"},
			},
			Debugging: &cmoptions.DebuggingOptions{
				DebuggingConfiguration: &componentbaseconfig.DebuggingConfiguration{
					EnableProfiling:           true,
					EnableContentionProfiling: false,
				},
			},
			LeaderMigration: &migration.LeaderMigrationOptions{},
		},
		KubeCloudShared: &KubeCloudSharedOptions{
			KubeCloudSharedConfiguration: &cpconfig.KubeCloudSharedConfiguration{
				RouteReconciliationPeriod: metav1.Duration{Duration: 10 * time.Second},
				NodeMonitorPeriod:         metav1.Duration{Duration: 5 * time.Second},
				ClusterName:               "kubernetes",
				ClusterCIDR:               "",
				AllocateNodeCIDRs:         false,
				CIDRAllocatorType:         "",
				ConfigureCloudRoutes:      true,
			},
			CloudProvider: &CloudProviderOptions{
				CloudProviderConfiguration: &cpconfig.CloudProviderConfiguration{
					Name:            "",
					CloudConfigFile: "",
				},
			},
		},
		ServiceController: &ServiceControllerOptions{
			ServiceControllerConfiguration: &serviceconfig.ServiceControllerConfiguration{
				ConcurrentServiceSyncs: 1,
			},
		},
		SecureServing: (&apiserveroptions.SecureServingOptions{
			BindPort:    10258,
			BindAddress: netutils.ParseIPSloppy("0.0.0.0"),
			ServerCert: apiserveroptions.GeneratableKeyCert{
				CertDirectory: "",
				PairName:      "cloud-controller-manager",
			},
			HTTP2MaxStreamsPerConnection: 0,
		}).WithLoopback(),
		InsecureServing: (&apiserveroptions.DeprecatedInsecureServingOptions{
			BindAddress: netutils.ParseIPSloppy("0.0.0.0"),
			BindPort:    int(0),
			BindNetwork: "tcp",
		}).WithLoopback(),
		Authentication: &apiserveroptions.DelegatingAuthenticationOptions{
			CacheTTL:            10 * time.Second,
			TokenRequestTimeout: 10 * time.Second,
			WebhookRetryBackoff: apiserveroptions.DefaultAuthWebhookRetryBackoff(),
			ClientCert:          apiserveroptions.ClientCertAuthenticationOptions{},
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
			ClientTimeout:                10 * time.Second,
			WebhookRetryBackoff:          apiserveroptions.DefaultAuthWebhookRetryBackoff(),
			RemoteKubeConfigFileOptional: true,
			AlwaysAllowPaths:             []string{"/healthz", "/readyz", "/livez"}, // note: this does not match /healthz/ or /healthz/*
			AlwaysAllowGroups:            []string{"system:masters"},
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
	for _, f := range s.Flags([]string{""}, []string{""}).FlagSets {
		fs.AddFlagSet(f)
	}

	args := []string{
		"--address=192.168.4.10",
		"--allocate-node-cidrs=true",
		"--authorization-always-allow-paths=", // this proves that we can clear the default
		"--bind-address=192.168.4.21",
		"--cert-dir=/a/b/c",
		"--cloud-config=/cloud-config",
		"--cloud-provider=gce",
		"--cluster-cidr=1.2.3.4/24",
		"--cluster-name=k8s",
		"--configure-cloud-routes=false",
		"--contention-profiling=true",
		"--controller-start-interval=2m",
		"--controllers=foo,bar",
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
			GenericControllerManagerConfiguration: &cmconfig.GenericControllerManagerConfiguration{
				Port:            DefaultInsecureCloudControllerManagerPort, // Note: InsecureServingOptions.ApplyTo will write the flag value back into the component config
				Address:         "0.0.0.0",                                 // Note: InsecureServingOptions.ApplyTo will write the flag value back into the component config
				MinResyncPeriod: metav1.Duration{Duration: 100 * time.Minute},
				ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
					ContentType: "application/vnd.kubernetes.protobuf",
					QPS:         50.0,
					Burst:       100,
				},
				ControllerStartInterval: metav1.Duration{Duration: 2 * time.Minute},
				LeaderElection: componentbaseconfig.LeaderElectionConfiguration{
					ResourceLock:      "configmap",
					LeaderElect:       false,
					LeaseDuration:     metav1.Duration{Duration: 30 * time.Second},
					RenewDeadline:     metav1.Duration{Duration: 15 * time.Second},
					RetryPeriod:       metav1.Duration{Duration: 5 * time.Second},
					ResourceName:      "cloud-controller-manager",
					ResourceNamespace: "kube-system",
				},
				Controllers: []string{"foo", "bar"},
			},
			Debugging: &cmoptions.DebuggingOptions{
				DebuggingConfiguration: &componentbaseconfig.DebuggingConfiguration{
					EnableProfiling:           false,
					EnableContentionProfiling: true,
				},
			},
			LeaderMigration: &migration.LeaderMigrationOptions{},
		},
		KubeCloudShared: &KubeCloudSharedOptions{
			KubeCloudSharedConfiguration: &cpconfig.KubeCloudSharedConfiguration{
				RouteReconciliationPeriod: metav1.Duration{Duration: 30 * time.Second},
				NodeMonitorPeriod:         metav1.Duration{Duration: 5 * time.Second},
				ClusterName:               "k8s",
				ClusterCIDR:               "1.2.3.4/24",
				AllocateNodeCIDRs:         true,
				CIDRAllocatorType:         "RangeAllocator",
				ConfigureCloudRoutes:      false,
			},
			CloudProvider: &CloudProviderOptions{
				CloudProviderConfiguration: &cpconfig.CloudProviderConfiguration{
					Name:            "gce",
					CloudConfigFile: "/cloud-config",
				},
			},
		},
		ServiceController: &ServiceControllerOptions{
			ServiceControllerConfiguration: &serviceconfig.ServiceControllerConfiguration{
				ConcurrentServiceSyncs: 1,
			},
		},
		SecureServing: (&apiserveroptions.SecureServingOptions{
			BindPort:    10001,
			BindAddress: netutils.ParseIPSloppy("192.168.4.21"),
			ServerCert: apiserveroptions.GeneratableKeyCert{
				CertDirectory: "/a/b/c",
				PairName:      "cloud-controller-manager",
			},
			HTTP2MaxStreamsPerConnection: 47,
		}).WithLoopback(),
		InsecureServing: (&apiserveroptions.DeprecatedInsecureServingOptions{
			BindAddress: netutils.ParseIPSloppy("192.168.4.10"),
			BindPort:    int(10000),
			BindNetwork: "tcp",
		}).WithLoopback(),
		Authentication: &apiserveroptions.DelegatingAuthenticationOptions{
			CacheTTL:            10 * time.Second,
			TokenRequestTimeout: 10 * time.Second,
			WebhookRetryBackoff: apiserveroptions.DefaultAuthWebhookRetryBackoff(),
			ClientCert:          apiserveroptions.ClientCertAuthenticationOptions{},
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
			ClientTimeout:                10 * time.Second,
			WebhookRetryBackoff:          apiserveroptions.DefaultAuthWebhookRetryBackoff(),
			RemoteKubeConfigFileOptional: true,
			AlwaysAllowPaths:             []string{},
			AlwaysAllowGroups:            []string{"system:masters"},
		},
		Kubeconfig:                "/kubeconfig",
		Master:                    "192.168.4.20",
		NodeStatusUpdateFrequency: metav1.Duration{Duration: 10 * time.Minute},
	}
	if !reflect.DeepEqual(expected, s) {
		t.Errorf("Got different run options than expected.\nDifference detected on:\n%s", diff.ObjectReflectDiff(expected, s))
	}
}
