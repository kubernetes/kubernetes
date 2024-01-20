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
	"mime"
	"reflect"
	"strings"
	"testing"

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

	scheme = runtime.NewScheme()
	codecs = serializer.NewCodecFactory(scheme)
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

	exampleinstall.Install(scheme)
}

type fakeNegotiater struct {
	serializer, streamSerializer runtime.Serializer
	framer                       runtime.Framer
	types, streamTypes           []string
}

func (n *fakeNegotiater) SupportedMediaTypes() []runtime.SerializerInfo {
	var out []runtime.SerializerInfo
	for _, s := range n.types {
		mediaType, _, err := mime.ParseMediaType(s)
		if err != nil {
			panic(err)
		}
		parts := strings.SplitN(mediaType, "/", 2)
		if len(parts) == 1 {
			// this is an error on the server side
			parts = append(parts, "")
		}

		info := runtime.SerializerInfo{
			Serializer:       n.serializer,
			MediaType:        s,
			MediaTypeType:    parts[0],
			MediaTypeSubType: parts[1],
			EncodesAsText:    true,
		}

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
	f := NewDefaultStorageFactory(storagebackend.Config{}, "test/test", ns, NewDefaultResourceEncodingConfig(scheme), NewResourceConfig(), nil)
	f.AddCohabitatingResources(example.Resource("test"), schema.GroupResource{Resource: "test2", Group: "2"})
	called := false
	testEncoderChain := func(e runtime.Encoder) runtime.Encoder {
		called = true
		return e
	}
	f.AddSerializationChains(testEncoderChain, nil, example.Resource("test"))
	f.SetEtcdLocation(example.Resource("*"), []string{"/server2"})
	f.SetEtcdPrefix(example.Resource("test"), "/prefix_for_test")

	config, err := f.NewConfig(example.Resource("test"), nil)
	if err != nil {
		t.Fatal(err)
	}
	if config.Prefix != "/prefix_for_test" || !reflect.DeepEqual(config.Transport.ServerList, []string{"/server2"}) {
		t.Errorf("unexpected config %#v", config)
	}
	if !called {
		t.Errorf("expected encoder chain to be called")
	}
}

func TestUpdateEtcdOverrides(t *testing.T) {
	exampleinstall.Install(scheme)

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
			Prefix: "/registry",
			Transport: storagebackend.TransportConfig{
				ServerList: defaultEtcdLocation,
			},
		}
		storageFactory := NewDefaultStorageFactory(defaultConfig, "", codecs, NewDefaultResourceEncodingConfig(scheme), NewResourceConfig(), nil)
		storageFactory.SetEtcdLocation(test.resource, test.servers)

		var err error
		config, err := storageFactory.NewConfig(test.resource, nil)
		if err != nil {
			t.Errorf("%d: unexpected error %v", i, err)
			continue
		}
		if !reflect.DeepEqual(config.Transport.ServerList, test.servers) {
			t.Errorf("%d: expected %v, got %v", i, test.servers, config.Transport.ServerList)
			continue
		}

		config, err = storageFactory.NewConfig(schema.GroupResource{Group: examplev1.GroupName, Resource: "unlikely"}, nil)
		if err != nil {
			t.Errorf("%d: unexpected error %v", i, err)
			continue
		}
		if !reflect.DeepEqual(config.Transport.ServerList, defaultEtcdLocation) {
			t.Errorf("%d: expected %v, got %v", i, defaultEtcdLocation, config.Transport.ServerList)
			continue
		}

	}
}

func TestConfigs(t *testing.T) {
	exampleinstall.Install(scheme)
	defaultEtcdLocations := []string{"http://127.0.0.1", "http://127.0.0.2"}

	testCases := []struct {
		resource    *schema.GroupResource
		servers     []string
		wantConfigs []storagebackend.Config
	}{
		{
			wantConfigs: []storagebackend.Config{
				{Transport: storagebackend.TransportConfig{ServerList: defaultEtcdLocations}, Prefix: "/registry"},
			},
		},
		{
			resource: &schema.GroupResource{Group: example.GroupName, Resource: "resource"},
			servers:  []string{},
			wantConfigs: []storagebackend.Config{
				{Transport: storagebackend.TransportConfig{ServerList: defaultEtcdLocations}, Prefix: "/registry"},
			},
		},
		{
			resource: &schema.GroupResource{Group: example.GroupName, Resource: "resource"},
			servers:  []string{"http://127.0.0.1:10000"},
			wantConfigs: []storagebackend.Config{
				{Transport: storagebackend.TransportConfig{ServerList: defaultEtcdLocations}, Prefix: "/registry"},
				{Transport: storagebackend.TransportConfig{ServerList: []string{"http://127.0.0.1:10000"}}, Prefix: "/registry"},
			},
		},
		{
			resource: &schema.GroupResource{Group: example.GroupName, Resource: "resource"},
			servers:  []string{"http://127.0.0.1:10000", "https://127.0.0.1", "http://127.0.0.2"},
			wantConfigs: []storagebackend.Config{
				{Transport: storagebackend.TransportConfig{ServerList: defaultEtcdLocations}, Prefix: "/registry"},
				{Transport: storagebackend.TransportConfig{ServerList: []string{"http://127.0.0.1:10000", "https://127.0.0.1", "http://127.0.0.2"}}, Prefix: "/registry"},
			},
		},
	}

	for i, test := range testCases {
		defaultConfig := storagebackend.Config{
			Prefix: "/registry",
			Transport: storagebackend.TransportConfig{
				ServerList: defaultEtcdLocations,
			},
		}
		storageFactory := NewDefaultStorageFactory(defaultConfig, "", codecs, NewDefaultResourceEncodingConfig(scheme), NewResourceConfig(), nil)
		if test.resource != nil {
			storageFactory.SetEtcdLocation(*test.resource, test.servers)
		}

		got := storageFactory.Configs()
		if !reflect.DeepEqual(test.wantConfigs, got) {
			t.Errorf("%d: expected %v, got %v", i, test.wantConfigs, got)
			continue
		}
	}
}

// func TestStorageFactoryCompatibilityVersion(t *testing.T) {
// 	sch := runtime.NewScheme()
// 	installflowcontrol.Install(sch)
// 	installadmissionregistration.Install(sch)
// 	installbatch.Install(sch)
// 	installcore.Install(sch)

// 	// FlowSchema
// 	//   - v1beta1: 1.20.0 - 1.23.0
// 	//   - v1beta2: 1.23.0 - 1.26.0
// 	//   - v1beta3: 1.26.0 - 1.30.0
// 	//   - v1: 1.30.0+
// 	// CronJob
// 	//	 - v1beta1: 1.8.0 - 1.21.0
// 	//	 - v1: 1.21.0+
// 	// ValidatingAdmissionPolicy
// 	//	 - v1beta1: 1.29.0 - 1.31.0
// 	//	 - v1: 1.31.0+

// 	testcases := []struct {
// 		effectiveVersion string
// 		example          runtime.Object
// 		resource         schema.GroupResource
// 		expectedVersion  schema.GroupVersion
// 	}{
// 		{
// 			// Basic case. Beta version for long time
// 			effectiveVersion: "1.14.0",
// 			example:          &batch.CronJob{},
// 			resource:         batch.Resource("cronjobs"),
// 			expectedVersion:  schema.GroupVersion{Group: "batch", Version: "v1beta1"},
// 		},
// 		{
// 			// Basic case. Beta version for long time
// 			effectiveVersion: "1.20.0",
// 			example:          &batch.CronJob{},
// 			resource:         batch.Resource("cronjobs"),
// 			expectedVersion:  schema.GroupVersion{Group: "batch", Version: "v1beta1"},
// 		},
// 		{
// 			// Basic case. Beta version for long time
// 			effectiveVersion: "1.20.0",
// 			example:          &batch.CronJob{},
// 			resource:         batch.Resource("cronjobs"),
// 			expectedVersion:  schema.GroupVersion{Group: "batch", Version: "v1beta1"},
// 		},
// 		{
// 			// Basic case. GA version for long time
// 			effectiveVersion: "1.28.0",
// 			example:          &batch.CronJob{},
// 			resource:         batch.Resource("cronjobs"),
// 			expectedVersion:  schema.GroupVersion{Group: "batch", Version: "v1"},
// 		},
// 		{
// 			// Basic core/v1
// 			effectiveVersion: "1.31.0",
// 			example:          &core.Pod{},
// 			resource:         core.Resource("pods"),
// 			expectedVersion:  schema.GroupVersion{Group: "", Version: "v1"},
// 		},
// 		{
// 			// Corner case: 1.1.0 has no flowcontrol. Options are to error
// 			// out or to use the latest version. This test assumes the latter.
// 			effectiveVersion: "1.1.0",
// 			example:          &flowcontrol.FlowSchema{},
// 			resource:         flowcontrol.Resource("flowschemas"),
// 			expectedVersion:  schema.GroupVersion{Group: "flowcontrol.apiserver.k8s.io", Version: "v1"},
// 		},
// 		{
// 			effectiveVersion: "1.21.0",
// 			example:          &flowcontrol.FlowSchema{},
// 			resource:         flowcontrol.Resource("flowschemas"),
// 			expectedVersion:  schema.GroupVersion{Group: "flowcontrol.apiserver.k8s.io", Version: "v1beta1"},
// 		},
// 		{
// 			// v2Beta1 introduced this version, but minCompatibility should
// 			// force v1beta1
// 			effectiveVersion: "1.23.0",
// 			example:          &flowcontrol.FlowSchema{},
// 			resource:         flowcontrol.Resource("flowschemas"),
// 			expectedVersion:  schema.GroupVersion{Group: "flowcontrol.apiserver.k8s.io", Version: "v1beta1"},
// 		},
// 		{
// 			effectiveVersion: "1.24.0",
// 			example:          &flowcontrol.FlowSchema{},
// 			resource:         flowcontrol.Resource("flowschemas"),
// 			expectedVersion:  schema.GroupVersion{Group: "flowcontrol.apiserver.k8s.io", Version: "v1beta2"},
// 		},
// 		{
// 			effectiveVersion: "1.26.0",
// 			example:          &flowcontrol.FlowSchema{},
// 			resource:         flowcontrol.Resource("flowschemas"),
// 			expectedVersion:  schema.GroupVersion{Group: "flowcontrol.apiserver.k8s.io", Version: "v1beta2"},
// 		},
// 		{
// 			effectiveVersion: "1.27.0",
// 			example:          &flowcontrol.FlowSchema{},
// 			resource:         flowcontrol.Resource("flowschemas"),
// 			expectedVersion:  schema.GroupVersion{Group: "flowcontrol.apiserver.k8s.io", Version: "v1beta3"},
// 		},
// 		{
// 			// GA API introduced 1.29 but must keep storing in v1beta3 for downgrades
// 			effectiveVersion: "1.29.0",
// 			example:          &flowcontrol.FlowSchema{},
// 			resource:         flowcontrol.Resource("flowschemas"),
// 			expectedVersion:  schema.GroupVersion{Group: "flowcontrol.apiserver.k8s.io", Version: "v1beta3"},
// 		},
// 		{
// 			// Version after GA api is introduced
// 			effectiveVersion: "1.30.0",
// 			example:          &flowcontrol.FlowSchema{},
// 			resource:         flowcontrol.Resource("flowschemas"),
// 			expectedVersion:  schema.GroupVersion{Group: "flowcontrol.apiserver.k8s.io", Version: "v1"},
// 		},
// 		{
// 			effectiveVersion: "1.30.0",
// 			example:          &admissionregistration.ValidatingAdmissionPolicy{},
// 			resource:         admissionregistration.Resource("validatingadmissionpolicies"),
// 			expectedVersion:  schema.GroupVersion{Group: "admissionregistration.k8s.io", Version: "v1beta1"},
// 		},
// 		{
// 			effectiveVersion: "1.31.0",
// 			example:          &admissionregistration.ValidatingAdmissionPolicy{},
// 			resource:         admissionregistration.Resource("validatingadmissionpolicies"),
// 			expectedVersion:  schema.GroupVersion{Group: "admissionregistration.k8s.io", Version: "v1"},
// 		},
// 		{
// 			effectiveVersion: "1.29.0",
// 			example:          &admissionregistration.ValidatingAdmissionPolicy{},
// 			resource:         admissionregistration.Resource("validatingadmissionpolicies"),
// 			expectedVersion:  schema.GroupVersion{Group: "admissionregistration.k8s.io", Version: "v1beta1"},
// 		},
// 	}

// 	for _, tc := range testcases {
// 		config := NewDefaultResourceEncodingConfig(legacyscheme.Scheme)
// 		config.SetEffectiveVersion(version.NewEffectiveVersion(tc.effectiveVersion))
// 		f := NewDefaultStorageFactory(
// 			storagebackend.Config{},
// 			"",
// 			legacyscheme.Codecs,
// 			config,
// 			NewResourceConfig(),
// 			nil)

// 		cfg, err := f.NewConfig(tc.resource, tc.example)
// 		if err != nil {
// 			t.Fatalf("unexpected error: %v", err)
// 		}

// 		gvks, _, err := legacyscheme.Scheme.ObjectKinds(tc.example)
// 		if err != nil {
// 			t.Fatalf("unexpected error: %v", err)
// 		}
// 		expectEncodeVersioner := runtime.NewMultiGroupVersioner(tc.expectedVersion,
// 			// One for memory version and one for storage version
// 			// Test assumes storage & memory are same for all cases
// 			schema.GroupKind{
// 				Group: gvks[0].Group,
// 			}, schema.GroupKind{
// 				Group: gvks[0].Group,
// 			})
// 		if cfg.EncodeVersioner.Identifier() != expectEncodeVersioner.Identifier() {
// 			t.Errorf("expected %v, got %v", expectEncodeVersioner, cfg.EncodeVersioner)
// 		}
// 	}
// }
