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

package master

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools/etcdtest"
)

func TestGetServersToValidate(t *testing.T) {
	master := Master{}
	config := Config{}
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.Machines = []string{"http://machine1:4001", "http://machine2", "http://machine3:4003"}
	config.EtcdHelper = tools.NewEtcdHelper(fakeClient, latest.Codec, etcdtest.PathPrefix())
	config.EtcdHelper.Versioner = nil

	master.nodeRegistry = registrytest.NewMinionRegistry([]string{"node1", "node2"}, api.NodeResources{})

	servers := master.getServersToValidate(&config)

	if len(servers) != 5 {
		t.Errorf("unexpected server list: %#v", servers)
	}
	for _, server := range []string{"scheduler", "controller-manager", "etcd-0", "etcd-1", "etcd-2"} {
		if _, ok := servers[server]; !ok {
			t.Errorf("server list missing: %s", server)
		}
	}
}
