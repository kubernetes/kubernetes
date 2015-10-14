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

package framework

import (
	"fmt"
	"math/rand"
	"os"

	"github.com/coreos/go-etcd/etcd"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/tools/etcdtest"
)

// If you need to start an etcd instance by hand, you also need to insert a key
// for this check to pass (*any* key will do, eg:
//curl -L http://127.0.0.1:4001/v2/keys/message -XPUT -d value="Hello world").
func init() {
	RequireEtcd()
}

// NewEtcdClient creates a new etcd client, defaulting to server http://127.0.0.1:4001.
// If ETCD_HOST and ETCD_PORT are set in the environment, these are used as the address for the etcd server instead.
func NewEtcdClient() *etcd.Client {
	servers := []string{}
	if os.Getenv("ETCD_HOST") != "" && os.Getenv("ETCD_PORT") != "" {
		servers = append(servers, fmt.Sprintf("http://%s:%s", os.Getenv("ETCD_HOST"), os.Getenv("ETCD_PORT")))
	}
	return etcd.NewClient(servers)
}

func NewEtcdStorage() (storage.Interface, error) {
	return master.NewEtcdStorage(NewEtcdClient(), latest.GroupOrDie("").InterfacesFor, testapi.Default.Version(), etcdtest.PathPrefix())
}

func NewExtensionsEtcdStorage(client *etcd.Client) (storage.Interface, error) {
	if client == nil {
		client = NewEtcdClient()
	}
	return master.NewEtcdStorage(client, latest.GroupOrDie("extensions").InterfacesFor, testapi.Extensions.GroupAndVersion(), etcdtest.PathPrefix())
}

func RequireEtcd() {
	if _, err := NewEtcdClient().Get("/", false, false); err != nil {
		glog.Fatalf("unable to connect to etcd for testing: %v", err)
	}
}

func WithEtcdKey(f func(string)) {
	prefix := fmt.Sprintf("/test-%d", rand.Int63())
	defer NewEtcdClient().Delete(prefix, true)
	f(prefix)
}

// DeleteAllEtcdKeys deletes all keys from etcd.
// TODO: Instead of sprinkling calls to this throughout the code, adjust the
// prefix in etcdtest package; then just delete everything once at the end
// of the test run.
func DeleteAllEtcdKeys() {
	glog.Infof("Deleting all etcd keys")
	client := NewEtcdClient()
	keys, err := client.Get("/", false, false)
	if err != nil {
		glog.Fatalf("Unable to list root etcd keys: %v", err)
	}
	for _, node := range keys.Node.Nodes {
		if _, err := client.Delete(node.Key, true); err != nil {
			glog.Fatalf("Unable delete key: %v", err)
		}
	}

}
