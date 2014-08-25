/*
Copyright 2014 Google Inc. All rights reserved.

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

/*
Package etcd_cloud implements the cloudprovider interface. It aims to
leverage the etcd data store for retrieving k8s minions.

Each minion must have an entry under the /minions etcd directory and
the value must be the IP address of the minion.
*/
package etcd_cloud

import (
	"net"
	"regexp"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"

	"github.com/coreos/go-etcd/etcd"
)

var etcdServerList = []string{
	"http://127.0.0.1:4001",
}

// EtcdClient is an injectable interface for testing.
type EtcdClient interface {
	Get(key string, sort, recursive bool) (*etcd.Response, error)
}

func init() {
	cloudprovider.RegisterCloudProvider("etcd", func() (cloudprovider.Interface, error) { return newEtcdCloud() })
}

// EtcdCloud is an implementation of Interface, TCPLoadBalancer and Instances
// for etcd.
type EtcdCloud struct {
	client EtcdClient
}

// newEtcdCloud creates a new instance of EtcdCloud.
func newEtcdCloud() (*EtcdCloud, error) {
	return &EtcdCloud{
		client: etcd.NewClient(etcdServerList),
	}, nil
}

// Instances returns an implementation of Instances for etcd.
func (c *EtcdCloud) Instances() (cloudprovider.Instances, bool) {
	return c, true
}

// IPAddress returns an net.IP for the given instance.
func (c *EtcdCloud) IPAddress(instance string) (net.IP, error) {
	return net.ParseIP(instance), nil
}

// List returns a list of minions.
// List searches the etcd /minions directory for registred minions. Each
// minion is represented by a /minions/<minion> key and the minion's IP
// address as the value.
func (c *EtcdCloud) List(filter string) ([]string, error) {
	instances := make([]string, 0)
	matcher, err := regexp.Compile(filter)
	if err != nil {
		return instances, err
	}
	response, err := c.client.Get("/minions", false, true)
	if err != nil {
		return instances, err
	}
	for _, node := range response.Node.Nodes {
		if matcher.MatchString(node.Value) {
			instances = append(instances, node.Value)
		}
	}
	return instances, nil
}

// TCPLoadBalancer returns an implementation of TCPLoadBalancer for etcd.
func (c *EtcdCloud) TCPLoadBalancer() (cloudprovider.TCPLoadBalancer, bool) {
	return nil, false
}

// Zones returns an implementation of Zones for etcd.
func (c *EtcdCloud) Zones() (cloudprovider.Zones, bool) {
	return nil, false
}
