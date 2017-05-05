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

package gce

import (
	"time"

	"k8s.io/kubernetes/pkg/api/v1"
	netsets "k8s.io/kubernetes/pkg/util/net/sets"

	compute "google.golang.org/api/compute/v1"
)

func newFirewallMetricContext(request string, region string) *metricContext {
	return &metricContext{
		start:      time.Now(),
		attributes: []string{"firewall_" + request, region, unusedMetricLabel},
	}
}

// GetFirewall returns the Firewall by name.
func (gce *GCECloud) GetFirewall(name string) (*compute.Firewall, error) {
	mc := newFirewallMetricContext("get", "")
	v, err := gce.service.Firewalls.Get(gce.projectID, name).Do()
	return v, mc.Observe(err)
}

// CreateFirewall creates the given firewall rule.
func (gce *GCECloud) CreateFirewall(name, desc string, sourceRanges netsets.IPNet, ports []int64, hostNames []string) error {
	region, err := GetGCERegion(gce.localZone)
	if err != nil {
		return err
	}

	mc := newFirewallMetricContext("create", region)
	// TODO: This completely breaks modularity in the cloudprovider but
	// the methods shared with the TCPLoadBalancer take v1.ServicePorts.
	svcPorts := []v1.ServicePort{}
	// TODO: Currently the only consumer of this method is the GCE L7
	// loadbalancer controller, which never needs a protocol other than
	// TCP.  We should pipe through a mapping of port:protocol and
	// default to TCP if UDP ports are required. This means the method
	// signature will change forcing downstream clients to refactor
	// interfaces.
	for _, p := range ports {
		svcPorts = append(svcPorts, v1.ServicePort{Port: int32(p), Protocol: v1.ProtocolTCP})
	}

	hosts, err := gce.getInstancesByNames(hostNames)
	if err != nil {
		mc.Observe(err)
		return err
	}

	return mc.Observe(gce.createFirewall(name, region, desc, sourceRanges, svcPorts, hosts))
}

// DeleteFirewall deletes the given firewall rule.
func (gce *GCECloud) DeleteFirewall(name string) error {
	region, err := GetGCERegion(gce.localZone)
	if err != nil {
		return err
	}

	mc := newFirewallMetricContext("delete", region)

	return mc.Observe(gce.deleteFirewall(name, region))
}

// UpdateFirewall applies the given firewall rule as an update to an
// existing firewall rule with the same name.
func (gce *GCECloud) UpdateFirewall(name, desc string, sourceRanges netsets.IPNet, ports []int64, hostNames []string) error {

	region, err := GetGCERegion(gce.localZone)
	if err != nil {
		return err
	}

	mc := newFirewallMetricContext("update", region)
	// TODO: This completely breaks modularity in the cloudprovider but
	// the methods shared with the TCPLoadBalancer take v1.ServicePorts.
	svcPorts := []v1.ServicePort{}
	// TODO: Currently the only consumer of this method is the GCE L7
	// loadbalancer controller, which never needs a protocol other than
	// TCP.  We should pipe through a mapping of port:protocol and
	// default to TCP if UDP ports are required. This means the method
	// signature will change, forcing downstream clients to refactor
	// interfaces.
	for _, p := range ports {
		svcPorts = append(svcPorts, v1.ServicePort{Port: int32(p), Protocol: v1.ProtocolTCP})
	}

	hosts, err := gce.getInstancesByNames(hostNames)
	if err != nil {
		mc.Observe(err)
		return err
	}

	return mc.Observe(gce.updateFirewall(name, region, desc, sourceRanges, svcPorts, hosts))
}
