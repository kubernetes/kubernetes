/*
Copyright 2019 The Kubernetes Authors.

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

package service

import (
	"time"
)

const (
	// RespondingTimeout is how long to wait for a service to be responding.
	RespondingTimeout = 2 * time.Minute

	// MaxNodesForEndpointsTests is the max number for testing endpoints.
	// Don't test with more than 3 nodes.
	// Many tests create an endpoint per node, in large clusters, this is
	// resource and time intensive.
	MaxNodesForEndpointsTests = 3
)

const (
	// KubeProxyLagTimeout is the maximum time a kube-proxy daemon on a node is allowed
	// to not notice a Service update, such as type=NodePort.
	// TODO: This timeout should be O(10s), observed values are O(1m), 5m is very
	// liberal. Fix tracked in #20567.
	KubeProxyLagTimeout = 5 * time.Minute

	// KubeProxyEndpointLagTimeout is the maximum time a kube-proxy daemon on a node is allowed
	// to not notice an Endpoint update.
	KubeProxyEndpointLagTimeout = 30 * time.Second

	// LoadBalancerLagTimeoutDefault is the maximum time a load balancer is allowed to
	// not respond after creation.
	LoadBalancerLagTimeoutDefault = 2 * time.Minute

	// LoadBalancerLagTimeoutAWS is the delay between ELB creation and serving traffic
	// on AWS. A few minutes is typical, so use 10m.
	LoadBalancerLagTimeoutAWS = 10 * time.Minute

	// LoadBalancerCreateTimeoutDefault is the default time to wait for a load balancer to be created/modified.
	// TODO: once support ticket 21807001 is resolved, reduce this timeout back to something reasonable
	LoadBalancerCreateTimeoutDefault = 20 * time.Minute
	// LoadBalancerCreateTimeoutLarge is the maximum time to wait for a load balancer to be created/modified.
	LoadBalancerCreateTimeoutLarge = 2 * time.Hour

	// LoadBalancerPropagationTimeoutDefault is the default time to wait for pods to
	// be targeted by load balancers.
	LoadBalancerPropagationTimeoutDefault = 10 * time.Minute

	// LoadBalancerCleanupTimeout is the time required by the loadbalancer to cleanup, proportional to numApps/Ing.
	// Bring the cleanup timeout back down to 5m once b/33588344 is resolved.
	LoadBalancerCleanupTimeout = 15 * time.Minute

	// LoadBalancerPollTimeout is the time required by the loadbalancer to poll.
	// On average it takes ~6 minutes for a single backend to come online in GCE.
	LoadBalancerPollTimeout = 22 * time.Minute
	// LoadBalancerPollInterval is the interval value in which the loadbalancer polls.
	LoadBalancerPollInterval = 30 * time.Second

	// LargeClusterMinNodesNumber is the number of nodes which a large cluster consists of.
	LargeClusterMinNodesNumber = 100

	// TestTimeout is used for most polling/waiting activities
	TestTimeout = 60 * time.Second

	// ServiceEndpointsTimeout is the maximum time in which endpoints for the service should be created.
	ServiceEndpointsTimeout = 2 * time.Minute

	// ServiceReachabilityShortPollTimeout is the maximum time in which service must be reachable during polling.
	ServiceReachabilityShortPollTimeout = 2 * time.Minute
)
