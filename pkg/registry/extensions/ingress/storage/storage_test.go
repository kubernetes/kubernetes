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

package storage

import (
	"testing"

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	etcdtesting "k8s.io/kubernetes/pkg/storage/etcd/testing"
	"k8s.io/kubernetes/pkg/util/intstr"
)

func newStorage(t *testing.T) (*REST, *StatusREST, *etcdtesting.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, extensions.GroupName)
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "ingresses",
	}
	ingressStorage, statusStorage := NewREST(restOptions)
	return ingressStorage, statusStorage, server
}

var (
	namespace           = api.NamespaceNone
	name                = "foo-ingress"
	defaultHostname     = "foo.bar.com"
	defaultBackendName  = "default-backend"
	defaultBackendPort  = intstr.FromInt(80)
	defaultLoadBalancer = "127.0.0.1"
	defaultPath         = "/foo"
	defaultPathMap      = map[string]string{defaultPath: defaultBackendName}
	defaultTLS          = []extensions.IngressTLS{
		{Hosts: []string{"foo.bar.com", "*.bar.com"}, SecretName: "fooSecret"},
	}
)

type IngressRuleValues map[string]string

func toHTTPIngressPaths(pathMap map[string]string) []extensions.HTTPIngressPath {
	httpPaths := []extensions.HTTPIngressPath{}
	for path, backend := range pathMap {
		httpPaths = append(httpPaths, extensions.HTTPIngressPath{
			Path: path,
			Backend: extensions.IngressBackend{
				ServiceName: backend,
				ServicePort: defaultBackendPort,
			},
		})
	}
	return httpPaths
}

func toIngressRules(hostRules map[string]IngressRuleValues) []extensions.IngressRule {
	rules := []extensions.IngressRule{}
	for host, pathMap := range hostRules {
		rules = append(rules, extensions.IngressRule{
			Host: host,
			IngressRuleValue: extensions.IngressRuleValue{
				HTTP: &extensions.HTTPIngressRuleValue{
					Paths: toHTTPIngressPaths(pathMap),
				},
			},
		})
	}
	return rules
}

func newIngress(pathMap map[string]string) *extensions.Ingress {
	return &extensions.Ingress{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: extensions.IngressSpec{
			Backend: &extensions.IngressBackend{
				ServiceName: defaultBackendName,
				ServicePort: defaultBackendPort,
			},
			Rules: toIngressRules(map[string]IngressRuleValues{
				defaultHostname: pathMap,
			}),
			TLS: defaultTLS,
		},
		Status: extensions.IngressStatus{
			LoadBalancer: api.LoadBalancerStatus{
				Ingress: []api.LoadBalancerIngress{
					{IP: defaultLoadBalancer},
				},
			},
		},
	}
}

func validIngress() *extensions.Ingress {
	return newIngress(defaultPathMap)
}

func TestCreate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := registrytest.New(t, storage.Store)
	ingress := validIngress()
	noDefaultBackendAndRules := validIngress()
	noDefaultBackendAndRules.Spec.Backend = &extensions.IngressBackend{}
	noDefaultBackendAndRules.Spec.Rules = []extensions.IngressRule{}
	badPath := validIngress()
	badPath.Spec.Rules = toIngressRules(map[string]IngressRuleValues{
		"foo.bar.com": {"/invalid[": "svc"}})
	test.TestCreate(
		// valid
		ingress,
		noDefaultBackendAndRules,
		badPath,
	)
}

func TestUpdate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := registrytest.New(t, storage.Store)
	test.TestUpdate(
		// valid
		validIngress(),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.Ingress)
			object.Spec.Rules = toIngressRules(map[string]IngressRuleValues{
				"bar.foo.com": {"/bar": defaultBackendName},
			})
			object.Spec.TLS = append(object.Spec.TLS, extensions.IngressTLS{
				Hosts:      []string{"*.google.com"},
				SecretName: "googleSecret",
			})
			return object
		},
		// invalid updateFunc: ObjeceMeta is not to be tampered with.
		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.Ingress)
			object.Name = ""
			return object
		},

		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.Ingress)
			object.Spec.Rules = toIngressRules(map[string]IngressRuleValues{
				"foo.bar.com": {"/invalid[": "svc"}})
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := registrytest.New(t, storage.Store)
	test.TestDelete(validIngress())
}

func TestGet(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := registrytest.New(t, storage.Store)
	test.TestGet(validIngress())
}

func TestList(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := registrytest.New(t, storage.Store)
	test.TestList(validIngress())
}

func TestWatch(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := registrytest.New(t, storage.Store)
	test.TestWatch(
		validIngress(),
		// matching labels
		[]labels.Set{},
		// not matching labels
		[]labels.Set{
			{"a": "c"},
			{"foo": "bar"},
		},
		// matching fields
		[]fields.Set{
			{"metadata.name": name},
		},
		// not matching fields
		[]fields.Set{
			{"metadata.name": "bar"},
			{"name": name},
		},
	)
}

// TODO TestUpdateStatus
