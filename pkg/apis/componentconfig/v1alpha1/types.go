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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// ClientConnectionConfiguration contains details for constructing a client.
type ClientConnectionConfiguration struct {
	// kubeConfigFile is the path to a kubeconfig file.
	KubeConfigFile string `json:"kubeconfig"`
	// acceptContentTypes defines the Accept header sent by clients when connecting to a server, overriding the
	// default value of 'application/json'. This field will control all connections to the server used by a particular
	// client.
	AcceptContentTypes string `json:"acceptContentTypes"`
	// contentType is the content type used when sending data to the server from this client.
	ContentType string `json:"contentType"`
	// cps controls the number of queries per second allowed for this connection.
	QPS float32 `json:"qps"`
	// burst allows extra queries to accumulate when a client is exceeding its rate.
	Burst int32 `json:"burst"`
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

// SchedulerAlgorithmSource is the source of a scheduler algorithm. One source
// field must be specified, and source fields are mutually exclusive.
type SchedulerAlgorithmSource struct {
	// Policy is a policy based algorithm source.
	Policy *SchedulerPolicySource `json:"policy,omitempty"`
	// Provider is the name of a scheduling algorithm provider to use.
	Provider *string `json:"provider,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

type KubeSchedulerConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// SchedulerName is name of the scheduler, used to select which pods
	// will be processed by this scheduler, based on pod's "spec.SchedulerName".
	SchedulerName string `json:"schedulerName"`
	// AlgorithmSource specifies the scheduler algorithm source.
	AlgorithmSource SchedulerAlgorithmSource `json:"algorithmSource"`
	// RequiredDuringScheduling affinity is not symmetric, but there is an implicit PreferredDuringScheduling affinity rule
	// corresponding to every RequiredDuringScheduling affinity rule.
	// HardPodAffinitySymmetricWeight represents the weight of implicit PreferredDuringScheduling affinity rule, in the range 0-100.
	HardPodAffinitySymmetricWeight int32 `json:"hardPodAffinitySymmetricWeight"`

	// LeaderElection defines the configuration of leader election client.
	LeaderElection KubeSchedulerLeaderElectionConfiguration `json:"leaderElection"`

	// ClientConnection specifies the kubeconfig file and client connection
	// settings for the proxy server to use when communicating with the apiserver.
	ClientConnection ClientConnectionConfiguration `json:"clientConnection"`
	// HealthzBindAddress is the IP address and port for the health check server to serve on,
	// defaulting to 0.0.0.0:10251
	HealthzBindAddress string `json:"healthzBindAddress"`
	// MetricsBindAddress is the IP address and port for the metrics server to
	// serve on, defaulting to 0.0.0.0:10251.
	MetricsBindAddress string `json:"metricsBindAddress"`
	// EnableProfiling enables profiling via web interface on /debug/pprof
	// handler. Profiling handlers will be handled by metrics server.
	EnableProfiling bool `json:"enableProfiling"`
	// EnableContentionProfiling enables lock contention profiling, if
	// EnableProfiling is true.
	EnableContentionProfiling bool `json:"enableContentionProfiling"`

	// Indicate the "all topologies" set for empty topologyKey when it's used for PreferredDuringScheduling pod anti-affinity.
	FailureDomains string `json:"failureDomains"`
}

// LeaderElectionConfiguration defines the configuration of leader election
// clients for components that can run with leader election enabled.
type LeaderElectionConfiguration struct {
	// leaderElect enables a leader election client to gain leadership
	// before executing the main loop. Enable this when running replicated
	// components for high availability.
	LeaderElect *bool `json:"leaderElect"`
	// leaseDuration is the duration that non-leader candidates will wait
	// after observing a leadership renewal until attempting to acquire
	// leadership of a led but unrenewed leader slot. This is effectively the
	// maximum duration that a leader can be stopped before it is replaced
	// by another candidate. This is only applicable if leader election is
	// enabled.
	LeaseDuration metav1.Duration `json:"leaseDuration"`
	// renewDeadline is the interval between attempts by the acting master to
	// renew a leadership slot before it stops leading. This must be less
	// than or equal to the lease duration. This is only applicable if leader
	// election is enabled.
	RenewDeadline metav1.Duration `json:"renewDeadline"`
	// retryPeriod is the duration the clients should wait between attempting
	// acquisition and renewal of a leadership. This is only applicable if
	// leader election is enabled.
	RetryPeriod metav1.Duration `json:"retryPeriod"`
	// resourceLock indicates the resource object type that will be used to lock
	// during leader election cycles.
	ResourceLock string `json:"resourceLock"`
}

// KubeSchedulerLeaderElectionConfiguration expands LeaderElectionConfiguration
// to include scheduler specific configuration.
type KubeSchedulerLeaderElectionConfiguration struct {
	LeaderElectionConfiguration `json:",inline"`
	// LockObjectNamespace defines the namespace of the lock object
	LockObjectNamespace string `json:"lockObjectNamespace"`
	// LockObjectName defines the lock object name
	LockObjectName string `json:"lockObjectName"`
}

const (
	// "kube-system" is the default scheduler lock object namespace
	SchedulerDefaultLockObjectNamespace string = "kube-system"

	// "kube-scheduler" is the default scheduler lock object name
	SchedulerDefaultLockObjectName = "kube-scheduler"

	SchedulerDefaultProviderName = "DefaultProvider"
)
