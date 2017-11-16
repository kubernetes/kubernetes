/*
Copyright 2016 The Kubernetes Authors.

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
	"os"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/apimachinery/announced"
	"k8s.io/apimachinery/pkg/apimachinery/registered"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apiserver/pkg/apis/example"
	exampleinstall "k8s.io/apiserver/pkg/apis/example/install"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	"k8s.io/apiserver/pkg/storage/storagebackend"
)

var (
	v1GroupVersion = schema.GroupVersion{Group: "", Version: "v1"}

	registry       = registered.NewOrDie(os.Getenv("KUBE_API_VERSIONS"))
	announce       = make(announced.APIGroupFactoryRegistry)
	scheme         = runtime.NewScheme()
	codecs         = serializer.NewCodecFactory(scheme)
	parameterCodec = runtime.NewParameterCodec(scheme)
)

func init() {
	metav1.AddToGroupVersion(scheme, metav1.SchemeGroupVersion)
	scheme.AddUnversionedTypes(v1GroupVersion,
		&metav1.Status{},
		&metav1.APIVersions{},
		&metav1.APIGroupList{},
		&metav1.APIGroup{},
		&metav1.APIResourceList{},
	)

	exampleinstall.Install(announce, registry, scheme)
}

type fakeNegotiater struct {
	serializer, streamSerializer runtime.Serializer
	framer                       runtime.Framer
	types, streamTypes           []string
}

func (n *fakeNegotiater) SupportedMediaTypes() []runtime.SerializerInfo {
	var out []runtime.SerializerInfo
	for _, s := range n.types {
		info := runtime.SerializerInfo{Serializer: n.serializer, MediaType: s, EncodesAsText: true}
		for _, t := range n.streamTypes {
			if t == s {
				info.StreamSerializer = &runtime.StreamSerializerInfo{
					EncodesAsText: true,
					Framer:        n.framer,
					Serializer:    n.streamSerializer,
				}
			}
		}
		out = append(out, info)
	}
	return out
}

func (n *fakeNegotiater) UniversalDeserializer() runtime.Decoder {
	return n.serializer
}

func (n *fakeNegotiater) EncoderForVersion(serializer runtime.Encoder, gv runtime.GroupVersioner) runtime.Encoder {
	return n.serializer
}

func (n *fakeNegotiater) DecoderToVersion(serializer runtime.Decoder, gv runtime.GroupVersioner) runtime.Decoder {
	return n.serializer
}

func TestConfigurableStorageFactory(t *testing.T) {
	ns := &fakeNegotiater{types: []string{"test/test"}}
	f := NewDefaultStorageFactory(storagebackend.Config{}, "test/test", ns, NewDefaultResourceEncodingConfig(registry), NewResourceConfig(), nil)
	f.AddCohabitatingResources(example.Resource("test"), schema.GroupResource{Resource: "test2", Group: "2"})
	called := false
	testEncoderChain := func(e runtime.Encoder) runtime.Encoder {
		called = true
		return e
	}
	f.AddSerializationChains(testEncoderChain, nil, example.Resource("test"))
	f.SetEtcdLocation(example.Resource("*"), []string{"/server2"})
	f.SetEtcdPrefix(example.Resource("test"), "/prefix_for_test")

	config, err := f.NewConfig(example.Resource("test"))
	if err != nil {
		t.Fatal(err)
	}
	if config.Prefix != "/prefix_for_test" || !reflect.DeepEqual(config.ServerList, []string{"/server2"}) {
		t.Errorf("unexpected config %#v", config)
	}
	if !called {
		t.Errorf("expected encoder chain to be called")
	}
}

func TestUpdateEtcdOverrides(t *testing.T) {
	registry := registered.NewOrDie(os.Getenv("KUBE_API_VERSIONS"))
	announced := make(announced.APIGroupFactoryRegistry)
	exampleinstall.Install(announced, registry, scheme)

	testCases := []struct {
		resource schema.GroupResource
		servers  []string
	}{
		{
			resource: schema.GroupResource{Group: example.GroupName, Resource: "resource"},
			servers:  []string{"http://127.0.0.1:10000"},
		},
		{
			resource: schema.GroupResource{Group: example.GroupName, Resource: "resource"},
			servers:  []string{"http://127.0.0.1:10000", "http://127.0.0.1:20000"},
		},
		{
			resource: schema.GroupResource{Group: example.GroupName, Resource: "resource"},
			servers:  []string{"http://127.0.0.1:10000"},
		},
	}

	defaultEtcdLocation := []string{"http://127.0.0.1"}
	for i, test := range testCases {
		defaultConfig := storagebackend.Config{
			Prefix:     "/registry",
			ServerList: defaultEtcdLocation,
		}
		storageFactory := NewDefaultStorageFactory(defaultConfig, "", codecs, NewDefaultResourceEncodingConfig(registry), NewResourceConfig(), nil)
		storageFactory.SetEtcdLocation(test.resource, test.servers)

		var err error
		config, err := storageFactory.NewConfig(test.resource)
		if err != nil {
			t.Errorf("%d: unexpected error %v", i, err)
			continue
		}
		if !reflect.DeepEqual(config.ServerList, test.servers) {
			t.Errorf("%d: expected %v, got %v", i, test.servers, config.ServerList)
			continue
		}

		config, err = storageFactory.NewConfig(schema.GroupResource{Group: examplev1.GroupName, Resource: "unlikely"})
		if err != nil {
			t.Errorf("%d: unexpected error %v", i, err)
			continue
		}
		if !reflect.DeepEqual(config.ServerList, defaultEtcdLocation) {
			t.Errorf("%d: expected %v, got %v", i, defaultEtcdLocation, config.ServerList)
			continue
		}

	}
}
