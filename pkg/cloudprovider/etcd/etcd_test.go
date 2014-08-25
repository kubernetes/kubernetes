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

package etcd_cloud

import (
	"net"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"

	"github.com/coreos/go-etcd/etcd"
)

func TestList(t *testing.T) {
	c, err := newEtcdCloud()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	etcdClient := tools.NewFakeEtcdClient(t)
	etcdClient.Data["/minions"] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{Value: "203.0.113.10", ModifiedIndex: 1},
					{Value: "203.0.113.11", ModifiedIndex: 2},
				},
			},
		},
	}
	c.client = etcdClient
	instances, _ := c.Instances()
	got, err := instances.List(".*")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	want := []string{
		"203.0.113.10",
		"203.0.113.11",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Expected: %v, got %v", want, got)
	}
}

func TestIPAddress(t *testing.T) {
	c, err := newEtcdCloud()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	instances, _ := c.Instances()
	want := net.ParseIP("203.0.113.10")
	got, err := instances.IPAddress("203.0.113.10")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Expected: %v, got %v", want, got)
	}
}

func TestInstances(t *testing.T) {
	c, err := newEtcdCloud()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	instances, implemented := c.Instances()
	if !implemented {
		t.Errorf("Exected Instances to be implemented.")
	}
	if !reflect.DeepEqual(c, instances) {
		t.Errorf("Expected: %v, got %v", c, instances)
	}
}

func TestTCPLoadBalancer(t *testing.T) {
	c, err := newEtcdCloud()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	_, implemented := c.TCPLoadBalancer()
	if implemented {
		t.Errorf("Exected TCPLoadBalancer not to be implemented.")
	}
}

func TestZones(t *testing.T) {
	c, err := newEtcdCloud()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	_, implemented := c.Zones()
	if implemented {
		t.Errorf("Exected Zones not to be implemented")
	}
}
