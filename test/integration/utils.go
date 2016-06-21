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

package integration

import (
	"fmt"
	"math/rand"
	"testing"

	etcd "github.com/coreos/etcd/client"
	"github.com/golang/glog"
	"golang.org/x/net/context"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/test/integration/framework"
)

func newEtcdClient() etcd.Client {
	cfg := etcd.Config{
		Endpoints: []string{framework.GetEtcdURLFromEnv()},
	}
	client, err := etcd.New(cfg)
	if err != nil {
		glog.Fatalf("unable to connect to etcd for testing: %v", err)
	}
	return client
}

func requireEtcd() {
	if _, err := etcd.NewKeysAPI(newEtcdClient()).Get(context.TODO(), "/", nil); err != nil {
		glog.Fatalf("unable to connect to etcd for integration testing: %v", err)
	}
}

func withEtcdKey(f func(string)) {
	prefix := fmt.Sprintf("/test-%d", rand.Int63())
	defer etcd.NewKeysAPI(newEtcdClient()).Delete(context.TODO(), prefix, &etcd.DeleteOptions{Recursive: true})
	f(prefix)
}

func deleteAllEtcdKeys() {
	keysAPI := etcd.NewKeysAPI(newEtcdClient())
	keys, err := keysAPI.Get(context.TODO(), "/", nil)
	if err != nil {
		glog.Fatalf("Unable to list root etcd keys: %v", err)
	}
	for _, node := range keys.Node.Nodes {
		if _, err := keysAPI.Delete(context.TODO(), node.Key, &etcd.DeleteOptions{Recursive: true}); err != nil {
			glog.Fatalf("Unable delete key: %v", err)
		}
	}

}

func deletePodOrErrorf(t *testing.T, c *client.Client, ns, name string) {
	if err := c.Pods(ns).Delete(name, nil); err != nil {
		t.Errorf("unable to delete pod %v: %v", name, err)
	}
}
