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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/diff"
	apiserveroptions "k8s.io/apiserver/pkg/server/options"
	cmoptions "k8s.io/kubernetes/cmd/controller-manager/app/options"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
)

func TestDefaultFlags(t *testing.T) {
	s, _ := NewCloudControllerManagerOptions()

	expected := &CloudControllerManagerOptions{
		CloudProvider: &cmoptions.CloudProviderOptions{
			Name:            "",
			CloudConfigFile: "",
		},
		Debugging: &cmoptions.DebuggingOptions{
			EnableContentionProfiling: false,
		},
		GenericComponent: &cmoptions.GenericComponentConfigOptions{
			MinResyncPeriod:         metav1.Duration{Duration: 12 * time.Hour},
			ContentType:             "application/vnd.kubernetes.protobuf",
			KubeAPIQPS:              20.0,
			KubeAPIBurst:            30,
			ControllerStartInterval: metav1.Duration{Duration: 0},
			LeaderElection: componentconfig.LeaderElectionConfiguration{
				ResourceLock:  "endpoints",
				LeaderElect:   true,
				LeaseDuration: metav1.Duration{Duration: 15 * time.Second},
				RenewDeadline: metav1.Duration{Duration: 10 * time.Second},
				RetryPeriod:   metav1.Duration{Duration: 2 * time.Second},
			},
		},
		KubeCloudShared: &cmoptions.KubeCloudSharedOptions{
			Port:                      10253,     // Note: InsecureServingOptions.ApplyTo will write the flag value back into the component config
			Address:                   "0.0.0.0", // Note: InsecureServingOptions.ApplyTo will write the flag value back into the component config
			RouteReconciliationPeriod: metav1.Duration{Duration: 10 * time.Second},
			NodeMonitorPeriod:         metav1.Duration{Duration: 5 * time.Second},
			ClusterName:               "kubernetes",
			ClusterCIDR:               "",
			AllocateNodeCIDRs:         false,
			CIDRAllocatorType:         "",
			ConfigureCloudRoutes:      true,
		},
		ServiceController: &cmoptions.ServiceControllerOptions{
			ConcurrentServiceSyncs: 1,
		},
		SecureServing: &apiserveroptions.SecureServingOptions{
			BindPort:    0,
			BindAddress: net.ParseIP("0.0.0.0"),
			ServerCert: apiserveroptions.GeneratableKeyCert{
				CertDirectory: "/var/run/kubernetes",
				PairName:      "cloud-controller-manager",
			},
			HTTP2MaxStreamsPerConnection: 0,
		},
		InsecureServing: &cmoptions.InsecureServingOptions{
			BindAddress: net.ParseIP("0.0.0.0"),
			BindPort:    int(10253),
			BindNetwork: "tcp",
		},
		Kubeconfig: "",
		Master:     "",
		NodeStatusUpdateFrequency: metav1.Duration{Duration: 5 * time.Minute},
	}
	if !reflect.DeepEqual(expected, s) {
		t.Errorf("Got different run options than expected.\nDifference detected on:\n%s", diff.ObjectReflectDiff(expected, s))
	}
}

func TestAddFlags(t *testing.T) {
	f := pflag.NewFlagSet("addflagstest", pflag.ContinueOnError)
	s, _ := NewCloudControllerManagerOptions()
	s.AddFlags(f)

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
	f.Parse(args)

	expected := &CloudControllerManagerOptions{
		CloudProvider: &cmoptions.CloudProviderOptions{
			Name:            "gce",
			CloudConfigFile: "/cloud-config",
		},
		Debugging: &cmoptions.DebuggingOptions{
			EnableContentionProfiling: true,
		},
		GenericComponent: &cmoptions.GenericComponentConfigOptions{
			MinResyncPeriod:         metav1.Duration{Duration: 100 * time.Minute},
			ContentType:             "application/vnd.kubernetes.protobuf",
			KubeAPIQPS:              50.0,
			KubeAPIBurst:            100,
			ControllerStartInterval: metav1.Duration{Duration: 2 * time.Minute},
			LeaderElection: componentconfig.LeaderElectionConfiguration{
				ResourceLock:  "configmap",
				LeaderElect:   false,
				LeaseDuration: metav1.Duration{Duration: 30 * time.Second},
				RenewDeadline: metav1.Duration{Duration: 15 * time.Second},
				RetryPeriod:   metav1.Duration{Duration: 5 * time.Second},
			},
		},
		KubeCloudShared: &cmoptions.KubeCloudSharedOptions{
			Port:                      10253,     // Note: InsecureServingOptions.ApplyTo will write the flag value back into the component config
			Address:                   "0.0.0.0", // Note: InsecureServingOptions.ApplyTo will write the flag value back into the component config
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
		SecureServing: &apiserveroptions.SecureServingOptions{
			BindPort:    10001,
			BindAddress: net.ParseIP("192.168.4.21"),
			ServerCert: apiserveroptions.GeneratableKeyCert{
				CertDirectory: "/a/b/c",
				PairName:      "cloud-controller-manager",
			},
			HTTP2MaxStreamsPerConnection: 47,
		},
		InsecureServing: &cmoptions.InsecureServingOptions{
			BindAddress: net.ParseIP("192.168.4.10"),
			BindPort:    int(10000),
			BindNetwork: "tcp",
		},
		Kubeconfig: "/kubeconfig",
		Master:     "192.168.4.20",
		NodeStatusUpdateFrequency: metav1.Duration{Duration: 10 * time.Minute},
	}
	if !reflect.DeepEqual(expected, s) {
		t.Errorf("Got different run options than expected.\nDifference detected on:\n%s", diff.ObjectReflectDiff(expected, s))
	}
}
