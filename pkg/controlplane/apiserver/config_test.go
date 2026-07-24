/*
Copyright 2024 The Kubernetes Authors.

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

package apiserver

import (
	"context"
	"fmt"
	"net"
	"testing"

	extensionsapiserver "k8s.io/apiextensions-apiserver/pkg/apiserver"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	apiserveroptions "k8s.io/apiserver/pkg/server/options"
	aggregatorscheme "k8s.io/kube-aggregator/pkg/apiserver/scheme"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/controlplane/apiserver/options"
	generatedopenapi "k8s.io/kubernetes/pkg/generated/openapi"
	"k8s.io/kubernetes/pkg/kubeapiserver/admission/exclusion"
	netutils "k8s.io/utils/net"
)

func TestBuildGenericConfig(t *testing.T) {
	opts := options.NewOptions()
	s := (&apiserveroptions.SecureServingOptions{
		BindAddress: netutils.ParseIPSloppy("127.0.0.1"),
	}).WithLoopback()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to listen on 127.0.0.1:0")
	}
	defer ln.Close()
	s.Listener = ln
	s.BindPort = ln.Addr().(*net.TCPAddr).Port
	opts.SecureServing = s

	completedOptions, err := opts.Complete(context.TODO(), nil, nil)
	if err != nil {
		t.Fatalf("Failed to complete apiserver options: %v", err)
	}

	genericConfig, _, storageFactory, err := BuildGenericConfig(
		completedOptions,
		[]*runtime.Scheme{legacyscheme.Scheme, extensionsapiserver.Scheme, aggregatorscheme.Scheme},
		nil,
		generatedopenapi.GetOpenAPIDefinitions,
		nil,
	)
	if err != nil {
		t.Fatalf("Failed to build generic config: %v", err)
	}
	if genericConfig.StorageObjectCountTracker == nil {
		t.Errorf("genericConfig StorageObjectCountTracker is absent")
	}
	if genericConfig.StorageObjectCountTracker != storageFactory.StorageConfig.StorageObjectCountTracker {
		t.Errorf("There are different StorageObjectCountTracker in genericConfig and storageFactory")
	}

	restOptions, err := genericConfig.RESTOptionsGetter.GetRESTOptions(schema.GroupResource{Group: "", Resource: ""}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if restOptions.StorageConfig.StorageObjectCountTracker != genericConfig.StorageObjectCountTracker {
		t.Errorf("There are different StorageObjectCountTracker in restOptions and serverConfig")
	}
}

func TestConditionalRequestClassifier(t *testing.T) {
	// Pick one excluded resource to use in tests.
	excludedResources := exclusion.Excluded()
	if len(excludedResources) == 0 {
		t.Fatal("expected at least one excluded resource")
	}
	excludedGR := excludedResources[0]

	tests := []struct {
		name     string
		attrs    authorizer.AttributesRecord
		expected bool
	}{
		{
			name: "empty resource returns false",
			attrs: authorizer.AttributesRecord{
				APIGroup:   "",
				APIVersion: "v1",
				Resource:   "",
				Verb:       "create",
			},
			expected: false,
		},
		{
			name: "wildcard resource returns false",
			attrs: authorizer.AttributesRecord{
				APIGroup:   "",
				APIVersion: "v1",
				Resource:   "*",
				Verb:       "create",
			},
			expected: false,
		},
		{
			name: "wildcard API group returns false",
			attrs: authorizer.AttributesRecord{
				APIGroup:   "*",
				APIVersion: "v1",
				Resource:   "pods",
				Verb:       "create",
			},
			expected: false,
		},
		{
			name: "empty API version returns false",
			attrs: authorizer.AttributesRecord{
				APIGroup:   "",
				APIVersion: "",
				Resource:   "pods",
				Verb:       "create",
			},
			expected: false,
		},
		{
			name: "wildcard API version returns false",
			attrs: authorizer.AttributesRecord{
				APIGroup:   "",
				APIVersion: "*",
				Resource:   "pods",
				Verb:       "create",
			},
			expected: false,
		},
		{
			name: "excluded resource returns false",
			attrs: authorizer.AttributesRecord{
				APIGroup:   excludedGR.Group,
				APIVersion: "v1",
				Resource:   excludedGR.Resource,
				Verb:       "create",
			},
			expected: false,
		},
		{
			name: "non-admission verb get returns false",
			attrs: authorizer.AttributesRecord{
				APIGroup:   "",
				APIVersion: "v1",
				Resource:   "pods",
				Verb:       "get",
			},
			expected: false,
		},
		{
			name: "non-admission verb list returns false",
			attrs: authorizer.AttributesRecord{
				APIGroup:   "",
				APIVersion: "v1",
				Resource:   "pods",
				Verb:       "list",
			},
			expected: false,
		},
		{
			name: "non-admission verb watch returns false",
			attrs: authorizer.AttributesRecord{
				APIGroup:   "",
				APIVersion: "v1",
				Resource:   "pods",
				Verb:       "watch",
			},
			expected: false,
		},
		{
			name: "admission verb create returns true",
			attrs: authorizer.AttributesRecord{
				APIGroup:   "",
				APIVersion: "v1",
				Resource:   "pods",
				Verb:       "create",
			},
			expected: true,
		},
		{
			name: "admission verb update returns true",
			attrs: authorizer.AttributesRecord{
				APIGroup:   "",
				APIVersion: "v1",
				Resource:   "pods",
				Verb:       "update",
			},
			expected: true,
		},
		{
			name: "admission verb patch returns true",
			attrs: authorizer.AttributesRecord{
				APIGroup:   "",
				APIVersion: "v1",
				Resource:   "pods",
				Verb:       "patch",
			},
			expected: true,
		},
		{
			name: "admission verb delete returns true",
			attrs: authorizer.AttributesRecord{
				APIGroup:   "",
				APIVersion: "v1",
				Resource:   "pods",
				Verb:       "delete",
			},
			expected: true,
		},
		{
			name: "admission verb deletecollection returns true",
			attrs: authorizer.AttributesRecord{
				APIGroup:   "",
				APIVersion: "v1",
				Resource:   "pods",
				Verb:       "deletecollection",
			},
			expected: true,
		},
		{
			name: "non-core group admission verb returns true",
			attrs: authorizer.AttributesRecord{
				APIGroup:   "apps",
				APIVersion: "v1",
				Resource:   "deployments",
				Verb:       "create",
			},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := conditionalRequestClassifier(&tt.attrs)
			if got != tt.expected {
				t.Errorf("conditionalRequestClassifier(%s) = %v, want %v",
					fmt.Sprintf("{group=%q, version=%q, resource=%q, verb=%q}",
						tt.attrs.APIGroup, tt.attrs.APIVersion, tt.attrs.Resource, tt.attrs.Verb),
					got, tt.expected)
			}
		})
	}
}
