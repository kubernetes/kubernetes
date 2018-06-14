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
	compute "google.golang.org/api/compute/v1"

	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/meta"
)

func newFirewallMetricContext(request string) *metricContext {
	return newGenericMetricContext("firewall", request, unusedMetricLabel, unusedMetricLabel, computeV1Version)
}

// GetFirewall returns the Firewall by name.
func (gce *GCECloud) GetFirewall(name string) (*compute.Firewall, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newFirewallMetricContext("get")
	v, err := gce.c.Firewalls().Get(ctx, meta.GlobalKey(name))
	return v, mc.Observe(err)
}

// CreateFirewall creates the passed firewall
func (gce *GCECloud) CreateFirewall(f *compute.Firewall) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newFirewallMetricContext("create")
	return mc.Observe(gce.c.Firewalls().Insert(ctx, meta.GlobalKey(f.Name), f))
}

// DeleteFirewall deletes the given firewall rule.
func (gce *GCECloud) DeleteFirewall(name string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newFirewallMetricContext("delete")
	return mc.Observe(gce.c.Firewalls().Delete(ctx, meta.GlobalKey(name)))
}

// UpdateFirewall applies the given firewall as an update to an existing service.
func (gce *GCECloud) UpdateFirewall(f *compute.Firewall) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newFirewallMetricContext("update")
	return mc.Observe(gce.c.Firewalls().Update(ctx, meta.GlobalKey(f.Name), f))
}
