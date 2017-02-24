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

package bandwidth

import "k8s.io/apimachinery/pkg/api/resource"

type BandwidthShaper interface {
	// Limit the bandwidth for a particular CIDR on a particular interface
	//   * ingress and egress are in bits/second
	//   * cidr is expected to be a valid network CIDR (e.g. '1.2.3.4/32' or '10.20.0.1/16')
	// 'egress' bandwidth limit applies to all packets on the interface whose source matches 'cidr'
	// 'ingress' bandwidth limit applies to all packets on the interface whose destination matches 'cidr'
	// Limits are aggregate limits for the CIDR, not per IP address.  CIDRs must be unique, but can be overlapping, traffic
	// that matches multiple CIDRs counts against all limits.
	Limit(cidr string, egress, ingress *resource.Quantity) error
	// Remove a bandwidth limit for a particular CIDR on a particular network interface
	Reset(cidr string) error
	// Reconcile the interface managed by this shaper with the state on the ground.
	ReconcileInterface() error
	// Reconcile a CIDR managed by this shaper with the state on the ground
	ReconcileCIDR(cidr string, egress, ingress *resource.Quantity) error
	// GetCIDRs returns the set of CIDRs that are being managed by this shaper
	GetCIDRs() ([]string, error)
}
