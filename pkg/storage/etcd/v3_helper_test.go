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

package etcd

import (
	"reflect"
	"testing"

	"golang.org/x/net/context"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/storage/etcd/etcdtest"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/integration"
)

func TestETCD3(t *testing.T) {
	cluster := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 1})
	defer cluster.Terminate(t)

	// reuse cluster for all tests
	testSet(t, cluster)
}

func testSet(t *testing.T, clus *integration.ClusterV3) {
	tests := []struct {
		key    string
		setPod *api.Pod
	}{{ // test single set
		key:    "/some/key",
		setPod: &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}},
	}}

	etcdClient := clus.RandClient()
	defer etcdClient.KV.Delete(context.TODO(), "\x00", clientv3.WithFromKey())

	helper := newV3Helper(etcdClient, testapi.Default.Codec(), etcdtest.PathPrefix())
	ctx := context.Background()
	for i, tt := range tests {
		setOutput := &api.Pod{}
		err := helper.Set(ctx, tt.key, tt.setPod, setOutput, 0)
		if err != nil {
			t.Fatalf("Set failed: %#v", err)
		}
		if tt.setPod.ObjectMeta.Name != setOutput.ObjectMeta.Name {
			t.Errorf("#%d: pod name want=%s, get=%s", i, tt.setPod.ObjectMeta.Name, setOutput.ObjectMeta.Name)
		}
		if setOutput.ResourceVersion == "" {
			t.Errorf("#%d: output should have none empty resource version")
		}

		getOutput := &api.Pod{} // reset
		err = helper.Get(ctx, tt.key, getOutput, false)
		if err != nil {
			t.Fatalf("Get failed: %#v", err)
		}
		if !reflect.DeepEqual(setOutput, getOutput) {
			t.Errorf("#%d: pod want=%#v, get=%#v", i, setOutput, getOutput)
		}
	}
}
