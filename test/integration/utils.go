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
	"net/http"
	"net/http/httptest"
	"testing"

	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apiserver"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/pkg/tools/etcdtest"
	"k8s.io/kubernetes/plugin/pkg/admission/admit"

	etcd "github.com/coreos/etcd/client"
	"github.com/golang/glog"
	"golang.org/x/net/context"
)

func newEtcdClient() etcd.Client {
	cfg := etcd.Config {
		Endpoints: []string{"http://127.0.0.1:4001"},
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
	defer etcd.NewKeysAPI(newEtcdClient()).Delete(context.TODO(),prefix, &etcd.DeleteOptions{Recursive: true})
	f(prefix)
}

func deleteAllEtcdKeys() {
	client := newEtcdClient()
	kAPI := etcd.NewKeysAPI(client)
	keys, err := kAPI.Get(context.TODO(),"/", nil)
	if err != nil {
		glog.Fatalf("Unable to list root etcd keys: %v", err)
	}
	for _, node := range keys.Node.Nodes {
		if _, err := kAPI.Delete(context.TODO(), node.Key, &etcd.DeleteOptions{Recursive: true}); err != nil {
			glog.Fatalf("Unable delete key: %v", err)
		}
	}

}

func runAMaster(t *testing.T) (*master.Master, *httptest.Server) {
	etcdStorage, err := master.NewEtcdStorage(newEtcdClient(), latest.GroupOrDie("").InterfacesFor, testapi.Default.Version(), etcdtest.PathPrefix())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	m := master.New(&master.Config{
		DatabaseStorage:       etcdStorage,
		KubeletClient:         client.FakeKubeletClient{},
		EnableCoreControllers: true,
		EnableLogsSupport:     false,
		EnableProfiling:       true,
		EnableUISupport:       false,
		APIPrefix:             "/api",
		Authorizer:            apiserver.NewAlwaysAllowAuthorizer(),
		AdmissionControl:      admit.NewAlwaysAdmit(),
		StorageVersions:       map[string]string{"": testapi.Default.Version()},
	})

	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))

	return m, s
}
