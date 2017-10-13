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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/kubernetes/pkg/apis/componentconfig"

	"github.com/spf13/pflag"
)

func TestAddFlags(t *testing.T) {
	f := pflag.NewFlagSet("addflagstest", pflag.ContinueOnError)
	s := NewSchedulerServer()
	s.AddFlags(f)

	args := []string{
		"--address=192.168.4.20",
		"--algorithm-provider=FooProvider",
		"--contention-profiling=true",
		"--failure-domains=kubernetes.io/hostname",
		"--hard-pod-affinity-symmetric-weight=0",
		"--kube-api-burst=80",
		"--kube-api-content-type=application/vnd.kubernetes.protobuf",
		"--kube-api-qps=40.0",
		"--kubeconfig=/foo/bar/kubeconfig",
		"--leader-elect=true",
		"--leader-elect-lease-duration=20s",
		"--leader-elect-renew-deadline=15s",
		"--leader-elect-resource-lock=endpoints",
		"--leader-elect-retry-period=3s",
		"--lock-object-name=test-lock-object-name",
		"--lock-object-namespace=test-lock-object-ns",
		"--master=192.168.4.20",
		"--policy-config-file=/foo/bar/policyconfig",
		"--policy-configmap=test-policy-configmap",
		"--policy-configmap-namespace=test-policy-configmap-ns",
		"--port=10000",
		"--profiling=false",
		"--scheduler-name=test-scheduler-name",
		"--use-legacy-policy-config=true",
	}

	f.Parse(args)

	expected := &SchedulerServer{
		KubeSchedulerConfiguration: componentconfig.KubeSchedulerConfiguration{
			Port:                      10000,
			Address:                   "192.168.4.20",
			AlgorithmProvider:         "FooProvider",
			PolicyConfigFile:          "/foo/bar/policyconfig",
			EnableContentionProfiling: true,
			EnableProfiling:           false,

			ContentType:   "application/vnd.kubernetes.protobuf",
			KubeAPIQPS:    40.0,
			KubeAPIBurst:  80,
			SchedulerName: "test-scheduler-name",
			LeaderElection: componentconfig.LeaderElectionConfiguration{
				ResourceLock:  "endpoints",
				LeaderElect:   true,
				LeaseDuration: metav1.Duration{Duration: 20 * time.Second},
				RenewDeadline: metav1.Duration{Duration: 15 * time.Second},
				RetryPeriod:   metav1.Duration{Duration: 3 * time.Second},
			},

			LockObjectNamespace: "test-lock-object-ns",
			LockObjectName:      "test-lock-object-name",

			PolicyConfigMapName:      "test-policy-configmap",
			PolicyConfigMapNamespace: "test-policy-configmap-ns",
			UseLegacyPolicyConfig:    true,

			HardPodAffinitySymmetricWeight: 0,
			FailureDomains:                 "kubernetes.io/hostname",
		},
		Kubeconfig: "/foo/bar/kubeconfig",
		Master:     "192.168.4.20",
	}

	if !reflect.DeepEqual(expected, s) {
		t.Errorf("Got different run options than expected.\nDifference detected on:\n%s", diff.ObjectReflectDiff(expected, s))
	}
}
