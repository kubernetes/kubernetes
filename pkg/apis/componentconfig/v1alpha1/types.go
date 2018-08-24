/*
Copyright 2015 The Kubernetes Authors.

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

package v1alpha1

import (
	apimachineryconfigv1alpha1 "k8s.io/apimachinery/pkg/apis/config/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiserverconfigv1alpha1 "k8s.io/apiserver/pkg/apis/config/v1alpha1"
)

const (
	// "kube-system" is the default scheduler lock object namespace
	SchedulerDefaultLockObjectNamespace string = metav1.NamespaceSystem
	// "kube-scheduler" is the default scheduler lock object name
	SchedulerDefaultLockObjectName = "kube-scheduler"

	SchedulerDefaultProviderName = "DefaultProvider"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

type KubeSchedulerConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// ClientConnection specifies the kubeconfig file and client connection
	// settings for the proxy server to use when communicating with the apiserver.
	ClientConnection apimachineryconfigv1alpha1.ClientConnectionConfiguration `json:"clientConnection"`
	// DebuggingConfiguration holds profiling- and debugging-related fields
	// TODO: DebuggingConfiguration is inlined because it's been like that earlier.
	// We might consider making it a "real" sub-struct.
	apiserverconfigv1alpha1.DebuggingConfiguration `json:",inline"`
	// LeaderElection defines the configuration of leader election client.
	// TODO: Migrate the kube-scheduler-specific stuff into the generic LeaderElectionConfig?
	LeaderElection KubeSchedulerLeaderElectionConfiguration `json:"leaderElection"`

	// SchedulerName is name of the scheduler, used to select which pods
	// will be processed by this scheduler, based on pod's "spec.SchedulerName".
	SchedulerName string `json:"schedulerName"`
	// AlgorithmSource specifies the scheduler algorithm source.
	AlgorithmSource SchedulerAlgorithmSource `json:"algorithmSource"`
	// RequiredDuringScheduling affinity is not symmetric, but there is an implicit PreferredDuringScheduling affinity rule
	// corresponding to every RequiredDuringScheduling affinity rule.
	// HardPodAffinitySymmetricWeight represents the weight of implicit PreferredDuringScheduling affinity rule, in the range 0-100.
	HardPodAffinitySymmetricWeight int32 `json:"hardPodAffinitySymmetricWeight"`
	// HealthzBindAddress is the IP address and port for the health check server to serve on,
	// defaulting to 0.0.0.0:10251
	HealthzBindAddress string `json:"healthzBindAddress"`
	// MetricsBindAddress is the IP address and port for the metrics server to
	// serve on, defaulting to 0.0.0.0:10251.
	MetricsBindAddress string `json:"metricsBindAddress"`
	// Indicate the "all topologies" set for empty topologyKey when it's used for PreferredDuringScheduling pod anti-affinity.
	// DEPRECATED: This is no longer used.
	FailureDomains string `json:"failureDomains"`

	// DisablePreemption disables the pod preemption feature.
	DisablePreemption bool `json:"disablePreemption"`

	// PercentageOfNodeToScore specifies what percentage of all nodes should be scored in each
	// scheduling cycle. This helps improve scheduler's performance. Scheduler always tries to find
	// at least "minFeasibleNodesToFind" feasible nodes no matter what the value of this flag is.
	// When this value is below 100%, the scheduler stops finding feasible nodes for running a pod
	// once it finds that percentage of feasible nodes of the whole cluster size. For example, if the
	// cluster size is 500 nodes and the value of this flag is 30, then scheduler stops finding
	// feasible nodes once it finds 150 feasible nodes.
	// When the value is 0, default percentage (50%) of the nodes will be scored.
	PercentageOfNodesToScore int32 `json:"percentageOfNodesToScore"`
}

// SchedulerAlgorithmSource is the source of a scheduler algorithm. One source
// field must be specified, and source fields are mutually exclusive.
type SchedulerAlgorithmSource struct {
	// Policy is a policy based algorithm source.
	Policy *SchedulerPolicySource `json:"policy,omitempty"`
	// Provider is the name of a scheduling algorithm provider to use.
	Provider *string `json:"provider,omitempty"`
}

// SchedulerPolicySource configures a means to obtain a scheduler Policy. One
// source field must be specified, and source fields are mutually exclusive.
type SchedulerPolicySource struct {
	// File is a file policy source.
	File *SchedulerPolicyFileSource `json:"file,omitempty"`
	// ConfigMap is a config map policy source.
	ConfigMap *SchedulerPolicyConfigMapSource `json:"configMap,omitempty"`
}

// SchedulerPolicyFileSource is a policy serialized to disk and accessed via
// path.
type SchedulerPolicyFileSource struct {
	// Path is the location of a serialized policy.
	Path string `json:"path"`
}

// SchedulerPolicyConfigMapSource is a policy serialized into a config map value
// under the SchedulerPolicyConfigMapKey key.
type SchedulerPolicyConfigMapSource struct {
	// Namespace is the namespace of the policy config map.
	Namespace string `json:"namespace"`
	// Name is the name of hte policy config map.
	Name string `json:"name"`
}

// KubeSchedulerLeaderElectionConfiguration expands LeaderElectionConfiguration
// to include scheduler specific configuration.
type KubeSchedulerLeaderElectionConfiguration struct {
	apiserverconfigv1alpha1.LeaderElectionConfiguration `json:",inline"`

	// LockObjectNamespace defines the namespace of the lock object
	LockObjectNamespace string `json:"lockObjectNamespace"`
	// LockObjectName defines the lock object name
	LockObjectName string `json:"lockObjectName"`
}
