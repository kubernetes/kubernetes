/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package lb

import (
	"fmt"

	compute "google.golang.org/api/compute/v1"
	gce "k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
)

const (
	urlMapPort             = 8082
	defaultPort            = 80
	defaultPortRange       = "80"
	defaultHttpHealthCheck = "default-health-check"

	// A single target proxy/urlmap/forwarding rule is created per loadbalancer.
	// Tagged with the name of the IngressPoint.
	targetProxyPrefix    = "k8-tp"
	forwardingRulePrefix = "k8-fw"
	urlMapPrefix         = "k8-um"

	// A backend is created per nodePort, tagged with the nodeport.
	// This allows sharing of backends across loadbalancers.
	backendPrefix = "k8-bg"

	// A single instance-group is created per cluster manager.
	// Tagged with the name of the controller.
	instanceGroupPrefix = "k8-ig"

	// The gce api uses the name of a path rule to match a host rule.
	// In the current implementation,
	hostRulePrefix = "host"

	// State string required by gce library to list all instances.
	allInstances = "ALL"

	// Used in the test RunServer method to denote a delete request.
	deleteType = "del"
)

// ClusterManager manages L7s at a cluster level.
type ClusterManager struct {
	ClusterName    string
	cloud          *gce.GCECloud
	backendPool    *Backends
	l7Pool         *L7s
	instancePool   *Instances
	defaultIg      *compute.InstanceGroup
	defaultBackend *compute.BackendService
	// TODO: Include default health check
}

// NewClusterManager creates a cluster manager for shared resources.
func NewClusterManager(name string) (*ClusterManager, error) {
	cloud, err := gce.NewGCECloud(nil)
	if err != nil {
		return nil, err
	}
	cluster := ClusterManager{
		ClusterName: name,
		cloud:       cloud,
	}
	cluster.backendPool = NewBackendPool(&cluster)
	cluster.l7Pool = NewL7Pool(&cluster)
	cluster.instancePool = NewInstancePool(&cluster)

	ig, err := cluster.instancePool.create(
		fmt.Sprintf("%v-%v", instanceGroupPrefix, name))
	if err != nil {
		return nil, err
	}
	cluster.defaultIg = ig

	// The backend pool is engineered for the 99% case, to easily create
	// backends that consume a port on the instance group. The default
	// backend does not, so we need to hand craft it.
	def := fmt.Sprintf("%v-%v", backendPrefix, "default")
	be, _ := cloud.GetBackend(def)
	if be == nil {
		be, err = cluster.backendPool.create(
			ig, &compute.NamedPort{Port: defaultPort, Name: "default"}, def)
		if err != nil {
			return nil, err
		}
	}
	cluster.defaultBackend = be

	return &cluster, nil
}

func (c *ClusterManager) AddBackend(port int64) error {
	return c.backendPool.Add(port)
}

func (c *ClusterManager) GetBackend(port int64) (*compute.BackendService, error) {
	return c.backendPool.Get(port)
}

func (c *ClusterManager) DeleteBackend(port int64) error {
	return c.backendPool.Delete(port)
}

func (c *ClusterManager) SyncBackends(ports []int64) error {
	return c.backendPool.Sync(ports)
}

func (c *ClusterManager) AddL7(name string) error {
	return c.l7Pool.Add(name)
}

func (c *ClusterManager) GetL7(name string) (*L7, error) {
	return c.l7Pool.Get(name)
}

func (c *ClusterManager) DeleteL7(name string) error {
	return c.l7Pool.Delete(name)
}

func (c *ClusterManager) SyncL7s(names []string) error {
	return c.l7Pool.Sync(names)
}

func (c *ClusterManager) AddNodes(nodeNames []string) error {
	return c.instancePool.Add(nodeNames)
}

func (c *ClusterManager) RemoveNodes(nodeNames []string) error {
	return c.instancePool.Remove(nodeNames)
}

func (c *ClusterManager) SyncNodes(nodeNames []string) error {
	return c.instancePool.Sync(nodeNames)
}
