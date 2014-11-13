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

package kubelet

import (
	"net"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/fake"
)

func TestJoinCluster(t *testing.T) {
	hostname := "host1"
	masterName := "1.2.3.4"
	client := &client.Fake{}
	cloud := &fake_cloud.FakeCloud{
		NodeResources: &api.NodeResources{},
		IP:            net.ParseIP("3.4.5.6"),
	}

	err := joinCluster(hostname, client, cloud, masterName)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(client.Actions) != 1 ||
		client.Actions[0].Action != "create-minion" {
		t.Errorf("unexpected action list: %v", client.Actions)
	}
	minion, ok := client.Actions[0].Value.(*api.Minion)
	if !ok {
		t.Fatalf("unexpected object: %#v", client.Actions[0].Value)
	}
	if minion.Name != hostname || minion.HostIP != cloud.IP.String() {
		t.Errorf("unexpected minion created: %v", minion)
	}
}

func TestRegisterKubelet(t *testing.T) {
	hostname := "host1"
	clusterName := "cluster2"
	client := &client.Fake{}
	cloud := &fake_cloud.FakeCloud{
		NodeResources: &api.NodeResources{},
		IP:            net.ParseIP("3.4.5.6"),
		ClusterList:   []string{"cluster1", "cluster2"},
		MasterName:    "master1",
	}

	err := registerKubelet(hostname, client, cloud, clusterName)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(client.Actions) != 1 ||
		client.Actions[0].Action != "create-minion" {
		t.Errorf("unexpected action list: %v", client.Actions)
	}
	minion, ok := client.Actions[0].Value.(*api.Minion)
	if !ok {
		t.Fatalf("unexpected object: %#v", client.Actions[0].Value)
	}
	if minion.Name != hostname || minion.HostIP != cloud.IP.String() {
		t.Errorf("unexpected minion created: %v", minion)
	}
}
