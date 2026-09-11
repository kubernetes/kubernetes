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

package config

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// ClientConnectionConfiguration contains details for constructing a client.
type ClientConnectionConfiguration struct {
	// kubeconfig is the path to a KubeConfig file.
	Kubeconfig string
	// acceptContentTypes defines the Accept header sent by clients when connecting to a server, overriding the
	// default value of 'application/json'. This field will control all connections to the server used by a particular
	// client.
	AcceptContentTypes string
	// contentType is the content type used when sending data to the server from this client.
	ContentType string
	// qps controls the number of queries per second allowed for this connection.
	QPS float32
	// burst allows extra queries to accumulate when a client is exceeding its rate.
	Burst int32
}

// LeaderElectionConfiguration defines the configuration of leader election
// clients for components that can run with leader election enabled.
type LeaderElectionConfiguration struct {
	// leaderElect enables a leader election client to gain leadership
	// before executing the main loop. Enable this when running replicated
	// components for high availability.
	LeaderElect bool
	// leaseDuration is the duration that non-leader candidates will wait
	// after observing a leadership renewal until attempting to acquire
	// leadership of a led but unrenewed leader slot. This is effectively the
	// maximum duration that a leader can be stopped before it is replaced
	// by another candidate. This is only applicable if leader election is
	// enabled.
	LeaseDuration metav1.Duration
	// renewDeadline is the interval between attempts by the acting master to
	// renew a leadership slot before it stops leading. This must be less
	// than or equal to the lease duration. This is only applicable if leader
	// election is enabled.
	RenewDeadline metav1.Duration
	// retryPeriod is the duration the clients should wait between attempting
	// acquisition and renewal of a leadership. This is only applicable if
	// leader election is enabled.
	RetryPeriod metav1.Duration
	// resourceLock indicates the resource object type that will be used to lock
	// during leader election cycles.
	ResourceLock string
	// resourceName indicates the name of resource object that will be used to lock
	// during leader election cycles.
	ResourceName string
	// resourceNamespace indicates the namespace of resource object that will be used to lock
	// during leader election cycles.
	ResourceNamespace string
}

// EncryptionAlgorithmType defines the type of key algorithm used for certificate signing requests.
type EncryptionAlgorithmType string

const (
	// EncryptionAlgorithmECDSAP256 defines the ECDSA encryption algorithm type with curve P256.
	EncryptionAlgorithmECDSAP256 EncryptionAlgorithmType = "ECDSA-P256"
	// EncryptionAlgorithmECDSAP384 defines the ECDSA encryption algorithm type with curve P384.
	EncryptionAlgorithmECDSAP384 EncryptionAlgorithmType = "ECDSA-P384"
	// EncryptionAlgorithmRSA2048 defines the RSA encryption algorithm type with key size 2048 bits.
	EncryptionAlgorithmRSA2048 EncryptionAlgorithmType = "RSA-2048"
	// EncryptionAlgorithmRSA3072 defines the RSA encryption algorithm type with key size 3072 bits.
	EncryptionAlgorithmRSA3072 EncryptionAlgorithmType = "RSA-3072"
	// EncryptionAlgorithmRSA4096 defines the RSA encryption algorithm type with key size 4096 bits.
	EncryptionAlgorithmRSA4096 EncryptionAlgorithmType = "RSA-4096"
	// EncryptionAlgorithmMLDSA44 defines the ML-DSA-44 encryption algorithm variant.
	EncryptionAlgorithmMLDSA44 EncryptionAlgorithmType = "ML-DSA-44"
	// EncryptionAlgorithmMLDSA65 defines the ML-DSA-65 encryption algorithm variant.
	EncryptionAlgorithmMLDSA65 EncryptionAlgorithmType = "ML-DSA-65"
	// EncryptionAlgorithmMLDSA87 defines the ML-DSA-87 encryption algorithm variant.
	EncryptionAlgorithmMLDSA87 EncryptionAlgorithmType = "ML-DSA-87"
	// EncryptionAlgorithmDefault is the default encryption algorithm (ECDSA P-256).
	EncryptionAlgorithmDefault = EncryptionAlgorithmECDSAP256
)

// DebuggingConfiguration holds configuration for Debugging related features.
type DebuggingConfiguration struct {
	// enableProfiling enables profiling via web interface host:port/debug/pprof/
	EnableProfiling bool
	// enableContentionProfiling enables block profiling, if
	// enableProfiling is true.
	EnableContentionProfiling bool
}
