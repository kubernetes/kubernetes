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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	_ "k8s.io/kubernetes/pkg/apis/extensions/install"
	"k8s.io/kubernetes/pkg/apis/networking"
	_ "k8s.io/kubernetes/pkg/apis/networking/install"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func newStorage(t *testing.T) (*REST, *StatusREST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorageForResource(t, networking.Resource("ingresses"))
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "ingresses",
	}
	ingressStorage, statusStorage, err := NewREST(restOptions)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return ingressStorage, statusStorage, server
}

var (
	namespace           = metav1.NamespaceNone
	name                = "foo-ingress"
	defaultHostname     = "foo.bar.com"
	defaultBackendName  = "default-backend"
	defaultBackendPort  = intstr.FromInt(80)
	defaultLoadBalancer = "127.0.0.1"
	defaultPath         = "/foo"
	defaultPathType     = networking.PathTypeImplementationSpecific
	defaultPathMap      = map[string]string{defaultPath: defaultBackendName}
	defaultTLS          = []networking.IngressTLS{
		{Hosts: []string{"foo.bar.com", "*.bar.com"}, SecretName: "foosecret"},
	}
	serviceBackend = &networking.IngressServiceBackend{
		Name: "defaultbackend",
		Port: networking.ServiceBackendPort{
			Name:   "",
			Number: 80,
		},
	}
)

type IngressRuleValues map[string]string

func toHTTPIngressPaths(pathMap map[string]string) []networking.HTTPIngressPath {
	httpPaths := []networking.HTTPIngressPath{}
	for path, backend := range pathMap {
		httpPaths = append(httpPaths, networking.HTTPIngressPath{
			Path:     path,
			PathType: &defaultPathType,
			Backend: networking.IngressBackend{
				Service: &networking.IngressServiceBackend{
					Name: backend,
					Port: networking.ServiceBackendPort{
						Name:   defaultBackendPort.StrVal,
						Number: defaultBackendPort.IntVal,
					},
				},
			},
		})
	}
	return httpPaths
}

func toIngressRules(hostRules map[string]IngressRuleValues) []networking.IngressRule {
	rules := []networking.IngressRule{}
	for host, pathMap := range hostRules {
		rules = append(rules, networking.IngressRule{
			Host: host,
			IngressRuleValue: networking.IngressRuleValue{
				HTTP: &networking.HTTPIngressRuleValue{
					Paths: toHTTPIngressPaths(pathMap),
				},
			},
		})
	}
	return rules
}

func newIngress(pathMap map[string]string) *networking.Ingress {
	return &networking.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: networking.IngressSpec{
			DefaultBackend: &networking.IngressBackend{
				Service: serviceBackend.DeepCopy(),
			},
			Rules: toIngressRules(map[string]IngressRuleValues{
				defaultHostname: pathMap,
			}),
			TLS: defaultTLS,
		},
		Status: networking.IngressStatus{
			LoadBalancer: api.LoadBalancerStatus{
				Ingress: []api.LoadBalancerIngress{
					{IP: defaultLoadBalancer},
				},
			},
		},
	}
}

func validIngress() *networking.Ingress {
	return newIngress(defaultPathMap)
}

func TestCreate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	ingress := validIngress()
	noDefaultBackendAndRules := validIngress()
	noDefaultBackendAndRules.Spec.DefaultBackend.Service = &networking.IngressServiceBackend{}
	noDefaultBackendAndRules.Spec.Rules = []networking.IngressRule{}
	badPath := validIngress()
	badPath.Spec.Rules = toIngressRules(map[string]IngressRuleValues{
		"foo.bar.com": {"invalid-no-leading-slash": "svc"}})
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
	test := genericregistrytest.New(t, storage.Store)
	test.TestUpdate(
		// valid
		validIngress(),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*networking.Ingress)
			object.Spec.Rules = toIngressRules(map[string]IngressRuleValues{
				"bar.foo.com": {"/bar": defaultBackendName},
			})
			object.Spec.TLS = append(object.Spec.TLS, networking.IngressTLS{
				Hosts:      []string{"*.google.com"},
				SecretName: "googlesecret",
			})
			return object
		},
		// invalid updateFunc: ObjectMeta is not to be tampered with.
		func(obj runtime.Object) runtime.Object {
			object := obj.(*networking.Ingress)
			object.Name = ""
			return object
		},

		func(obj runtime.Object) runtime.Object {
			object := obj.(*networking.Ingress)
			object.Spec.Rules = toIngressRules(map[string]IngressRuleValues{
				"foo.bar.com": {"invalid-no-leading-slash": "svc"}})
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestDelete(validIngress())
}

func TestGet(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestGet(validIngress())
}

func TestList(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestList(validIngress())
}

func TestWatch(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
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

func TestShortNames(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	expected := []string{"ing"}
	registrytest.AssertShortNames(t, storage, expected)
}

// TODO TestUpdateStatus
