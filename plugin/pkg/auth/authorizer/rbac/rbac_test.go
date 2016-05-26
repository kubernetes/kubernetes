/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package rbac

import (
	"testing"

	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/registry/clusterrole"
	clusterroleetcd "k8s.io/kubernetes/pkg/registry/clusterrole/etcd"
	"k8s.io/kubernetes/pkg/registry/clusterrolebinding"
	clusterrolebindingetcd "k8s.io/kubernetes/pkg/registry/clusterrolebinding/etcd"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/role"
	roleetcd "k8s.io/kubernetes/pkg/registry/role/etcd"
	"k8s.io/kubernetes/pkg/registry/rolebinding"
	rolebindingetcd "k8s.io/kubernetes/pkg/registry/rolebinding/etcd"
	"k8s.io/kubernetes/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/storage/etcd/etcdtest"
	etcdtesting "k8s.io/kubernetes/pkg/storage/etcd/testing"
)

func TestNew(t *testing.T) {
	// NOTE(ericchiang): Can't get this strategy to do reads. Get cryptic "client: etcd cluster is unavailable or misconfigured"
	// Writes work fine, so use to test storing initial data.
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)

	codec := testapi.Groups[rbac.GroupName].StorageCodec()
	getRESTOptions := func(resource string) generic.RESTOptions {
		cacheSize := etcdtest.DeserializationCacheSize
		storage := etcd.NewEtcdStorage(server.Client, codec, resource, false, cacheSize)
		return generic.RESTOptions{Storage: storage, Decorator: generic.UndecoratedStorage}
	}

	roleRegistry := role.NewRegistry(roleetcd.NewREST(getRESTOptions("roles")))
	roleBindingRegistry := rolebinding.NewRegistry(rolebindingetcd.NewREST(getRESTOptions("rolebindings")))
	clusterRoleRegistry := clusterrole.NewRegistry(clusterroleetcd.NewREST(getRESTOptions("clusterroles")))
	clusterRoleBindingRegistry := clusterrolebinding.NewRegistry(clusterrolebindingetcd.NewREST(getRESTOptions("clusterrolebindings")))
	_, err := New(roleRegistry, roleBindingRegistry, clusterRoleRegistry, clusterRoleBindingRegistry, "")
	if err != nil {
		t.Fatalf("failed to create authorizer: %v", err)
	}
}
