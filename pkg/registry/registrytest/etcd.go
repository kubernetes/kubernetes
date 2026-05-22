/*
Copyright 2015 The Kubernetes Authors.

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

package registrytest

import (
	"testing"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/server/options"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	legacyscheme "k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/kubeapiserver"
)

// NewEtcdStorage is for testing.  It configures the etcd storage for a bogus resource; the test must not care.
func NewEtcdStorage(t *testing.T, group string) (*storagebackend.ConfigForResource, *etcd3testing.EtcdTestServer) {
	return NewEtcdStorageForResource(t, schema.GroupResource{Group: group, Resource: "any"})
}

func NewEtcdStorageForResource(t *testing.T, resource schema.GroupResource) (*storagebackend.ConfigForResource, *etcd3testing.EtcdTestServer) {
	t.Helper()

	server, config := etcd3testing.NewUnsecuredEtcd3TestClientServer(t)

	options := options.NewEtcdOptions(config)
	completedConfig := kubeapiserver.NewStorageFactoryConfig().Complete(options)
	completedConfig.APIResourceConfig = serverstorage.NewResourceConfig()
	factory, err := completedConfig.New()
	if err != nil {
		t.Fatalf("Error while making storage factory: %v", err)
	}
	example := exampleForResource(legacyscheme.Scheme, resource)
	resourceConfig, err := factory.NewConfig(resource, example)
	if err != nil {
		t.Fatalf("Error while finding storage destination: %v", err)
	}
	return resourceConfig, server
}

// exampleForResource finds an example object for the given resource by searching
// the scheme for a kind whose plural resource name matches. This lets the storage
// factory auto-calculate the correct storage version for resources that don't
// exist in their group's top priority version (e.g. ClusterTrustBundle is in
// v1alpha1/v1beta1 but not v1).
func exampleForResource(scheme *runtime.Scheme, resource schema.GroupResource) runtime.Object {
	for gvk := range scheme.AllKnownTypes() {
		if gvk.Group != resource.Group || gvk.Version == runtime.APIVersionInternal {
			continue
		}
		plural, _ := meta.UnsafeGuessKindToResource(gvk)
		if plural.Resource == resource.Resource {
			// The version doesn't matter: emulatedStorageVersion only
			// extracts the Kind name, then looks up all versions independently.
			obj, err := scheme.New(gvk)
			if err == nil {
				return obj
			}
		}
	}
	return nil
}
