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
	"k8s.io/kubernetes/pkg/apis/componentconfig"
)

func TestAddFlags(t *testing.T) {
	f := pflag.NewFlagSet("addflagstest", pflag.ContinueOnError)
	s := NewCloudControllerManagerServer()
	s.AddFlags(f)

	args := []string{
		"--address=192.168.4.10",
		"--allocate-node-cidrs=true",
		"--cloud-config=/cloud-config",
		"--cloud-provider=gce",
		"--cluster-cidr=1.2.3.4/24",
		"--cluster-name=k8s",
		"--configure-cloud-routes=false",
		"--contention-profiling=true",
		"--controller-start-interval=2m",
		"--min-resync-period=5m",
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
		"--min-resync-period=8h",
		"--port=10000",
		"--profiling=false",
		"--node-status-update-frequency=10m",
		"--route-reconciliation-period=30s",
		"--min-resync-period=100m",
		"--use-service-account-credentials=false",
	}
	f.Parse(args)

	expected := &CloudControllerManagerServer{
		KubeControllerManagerConfiguration: componentconfig.KubeControllerManagerConfiguration{
			CloudProvider:             "gce",
			CloudConfigFile:           "/cloud-config",
			Port:                      10000,
			Address:                   "192.168.4.10",
			ConcurrentServiceSyncs:    1,
			MinResyncPeriod:           metav1.Duration{Duration: 100 * time.Minute},
			NodeMonitorPeriod:         metav1.Duration{Duration: 5 * time.Second},
			ClusterName:               "k8s",
			ConfigureCloudRoutes:      false,
			AllocateNodeCIDRs:         true,
			ContentType:               "application/vnd.kubernetes.protobuf",
			EnableContentionProfiling: true,
			KubeAPIQPS:                50.0,
			KubeAPIBurst:              100,
			LeaderElection: componentconfig.LeaderElectionConfiguration{
				ResourceLock:  "configmap",
				LeaderElect:   false,
				LeaseDuration: metav1.Duration{Duration: 30 * time.Second},
				RenewDeadline: metav1.Duration{Duration: 15 * time.Second},
				RetryPeriod:   metav1.Duration{Duration: 5 * time.Second},
			},
			ControllerStartInterval:   metav1.Duration{Duration: 2 * time.Minute},
			RouteReconciliationPeriod: metav1.Duration{Duration: 30 * time.Second},
			ClusterCIDR:               "1.2.3.4/24",
		},
		Kubeconfig: "/kubeconfig",
		Master:     "192.168.4.20",
		NodeStatusUpdateFrequency: metav1.Duration{Duration: 10 * time.Minute},
	}
	if !reflect.DeepEqual(expected, s) {
		t.Errorf("Got different run options than expected.\nDifference detected on:\n%s", diff.ObjectReflectDiff(expected, s))
	}
}
