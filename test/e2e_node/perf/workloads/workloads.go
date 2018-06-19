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

package workloads

import (
	"time"

	corev1 "k8s.io/api/core/v1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
)

// NodePerfWorkload provides the necessary information to run a workload for
// node performance testing.
type NodePerfWorkload interface {
	// Name of the workload.
	Name() string
	// PodSpec used to run this workload.
	PodSpec() corev1.PodSpec
	// Timeout provides the expected time to completion
	// for this workload.
	Timeout() time.Duration
	// KubeletConfig specifies the Kubelet configuration
	// required for this workload.
	KubeletConfig(old *kubeletconfig.KubeletConfiguration) (new *kubeletconfig.KubeletConfiguration, err error)
	// PreTestExec is used for defining logic that needs
	// to be run before restarting the Kubelet with the new Kubelet
	// configuration required for the workload.
	PreTestExec() error
	// PostTestExec is used for defining logic that needs
	// to be run after the workload has completed.
	PostTestExec() error
	// ExtractPerformanceFromLogs is used get the performance of the workload
	// from pod logs. Currently, we support only performance reported in
	// time.Duration format.
	ExtractPerformanceFromLogs(logs string) (perf time.Duration, err error)
}

// NodePerfWorkloads is the collection of all node performance testing workloads.
var NodePerfWorkloads = []NodePerfWorkload{npbISWorkload{}, npbEPWorkload{}, tfWideDeepWorkload{}}
