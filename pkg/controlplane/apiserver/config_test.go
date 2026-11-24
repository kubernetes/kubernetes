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
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	extensionsapiserver "k8s.io/apiextensions-apiserver/pkg/apiserver"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/version"
	genericapifilters "k8s.io/apiserver/pkg/endpoints/filters"
	"k8s.io/apiserver/pkg/endpoints/request"
	genericregistry "k8s.io/apiserver/pkg/registry/generic"
	genericapiserver "k8s.io/apiserver/pkg/server"
	apiserveroptions "k8s.io/apiserver/pkg/server/options"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/apiserver/pkg/storage/value/compression"
	"k8s.io/apiserver/pkg/util/compatibility"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	basecompatibility "k8s.io/component-base/compatibility"
	aggregatorscheme "k8s.io/kube-aggregator/pkg/apiserver/scheme"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/controlplane/apiserver/options"
	generatedopenapi "k8s.io/kubernetes/pkg/generated/openapi"
	netutils "k8s.io/utils/net"
)

// newCompletedOptions completes the default options, serving on a free
// loopback port. A non-nil configure edits the options before they're completed.
func newCompletedOptions(t *testing.T, configure func(*options.Options)) options.CompletedOptions {
	t.Helper()
	opts := options.NewOptions()
	s := (&apiserveroptions.SecureServingOptions{
		BindAddress: netutils.ParseIPSloppy("127.0.0.1"),
	}).WithLoopback()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to listen on 127.0.0.1:0")
	}
	t.Cleanup(func() { _ = ln.Close() })
	s.Listener = ln
	s.BindPort = ln.Addr().(*net.TCPAddr).Port
	opts.SecureServing = s
	if configure != nil {
		configure(opts)
	}

	completedOptions, err := opts.Complete(context.TODO(), nil, nil)
	if err != nil {
		t.Fatalf("Failed to complete apiserver options: %v", err)
	}
	return completedOptions
}

func TestBuildGenericConfig(t *testing.T) {
	completedOptions := newCompletedOptions(t, nil)
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

// TestStorageCompressionReachesEveryConfig checks that built-in, custom resource and
// APIService stores get the file's Rule and a raw storage factory store doesn't. It
// deletes the file after BuildGenericConfig, so the other configs can't reread it.
func TestStorageCompressionReachesEveryConfig(t *testing.T) {
	deflate := compression.Rule{Algorithm: compression.AlgorithmDeflate, MinSize: 1024}
	tests := []struct {
		name             string
		setFlag          bool
		minCompatibility *version.Version
		wantErr          string
	}{
		{name: "unset", minCompatibility: version.MajorMinor(1, 37)},
		{name: "Deflate at 1.37", setFlag: true, minCompatibility: version.MajorMinor(1, 37), wantErr: "the server's is 1.37"},
		{name: "Deflate at 1.38", setFlag: true, minCompatibility: version.MajorMinor(1, 38)},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var path string
			if tc.setFlag {
				path = filepath.Join(t.TempDir(), "compression.yaml")
				content := `apiVersion: apiserver.config.k8s.io/v1alpha1
kind: StorageCompressionConfiguration
compressionResources:
- resources: ["*.*"]
  algorithm: Deflate
`
				if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			completedOptions := newCompletedOptions(t, func(opts *options.Options) {
				opts.Etcd.StorageCompressionConfigFilepath = path
				// A private registry sets the kube min compatibility version without changing the
				// process-wide one.
				effectiveVersion := compatibility.DefaultKubeEffectiveVersionForTest()
				effectiveVersion.SetMinCompatibilityVersion(tc.minCompatibility)
				registry := basecompatibility.NewComponentGlobalsRegistry()
				if err := registry.Register(basecompatibility.DefaultKubeComponent, effectiveVersion, utilfeature.DefaultFeatureGate.DeepCopy()); err != nil {
					t.Fatal(err)
				}
				opts.GenericServerRunOptions.ComponentGlobalsRegistry = registry
			})

			genericConfig, _, storageFactory, err := BuildGenericConfig(
				completedOptions,
				[]*runtime.Scheme{legacyscheme.Scheme, extensionsapiserver.Scheme, aggregatorscheme.Scheme},
				nil,
				generatedopenapi.GetOpenAPIDefinitions,
				nil,
			)
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) || !strings.Contains(err.Error(), path) {
					t.Fatalf("got %v, want an error naming %s and containing %q", err, path, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("Failed to build generic config: %v", err)
			}
			if tc.setFlag {
				if err := os.Remove(path); err != nil {
					t.Fatal(err)
				}
			}
			apiExtensionsConfig, err := CreateAPIExtensionsConfig(*genericConfig, nil, nil, completedOptions, 1, nil, nil)
			if err != nil {
				t.Fatalf("Failed to create the apiextensions config: %v", err)
			}
			aggregatorConfig, err := CreateAggregatorConfig(*genericConfig, completedOptions, nil, nil, nil, nil, nil)
			if err != nil {
				t.Fatalf("Failed to create the aggregator config: %v", err)
			}
			storageConfig := func(getter genericregistry.RESTOptionsGetter, gr schema.GroupResource) *storagebackend.ConfigForResource {
				t.Helper()
				restOptions, err := getter.GetRESTOptions(gr, nil)
				if err != nil {
					t.Fatal(err)
				}
				return restOptions.StorageConfig
			}
			want := compression.Rule{}
			if tc.setFlag {
				want = deflate
			}
			for name, config := range map[string]*storagebackend.ConfigForResource{
				"configmaps":  storageConfig(genericConfig.RESTOptionsGetter, schema.GroupResource{Resource: "configmaps"}),
				"widgets":     storageConfig(apiExtensionsConfig.ExtraConfig.CRDRESTOptionsGetter, schema.GroupResource{Group: "example.com", Resource: "widgets"}),
				"apiservices": storageConfig(aggregatorConfig.GenericConfig.RESTOptionsGetter, schema.GroupResource{Group: "apiregistration.k8s.io", Resource: "apiservices"}),
			} {
				if config.CompressionRule != want {
					t.Errorf("%s: got Rule %+v, want %+v", name, config.CompressionRule, want)
				}
			}
			// Stores built straight from the storage factory never see the rules.
			apiServerIPInfo, err := storageFactory.NewConfig(api.Resource("apiServerIPInfo"), &api.Endpoints{})
			if err != nil {
				t.Fatal(err)
			}
			if apiServerIPInfo.CompressionRule != (compression.Rule{}) {
				t.Errorf("apiServerIPInfo: got Rule %+v, want the zero Rule", apiServerIPInfo.CompressionRule)
			}
		})
	}
}

func TestConditionalRequestClassifier(t *testing.T) {
	// Build attributes the same way the authorization filter does: parse the
	// URL into a RequestInfo, then derive authorizer.Attributes from it. This
	// keeps the test honest about what the classifier actually sees in the
	// handler chain, instead of hand-crafting attribute combinations that the
	// RequestInfo parser would never produce.
	resolver := genericapiserver.NewRequestInfoResolver(&genericapiserver.Config{
		LegacyAPIGroupPrefixes: sets.NewString(genericapiserver.DefaultLegacyAPIPrefix),
	})

	tests := []struct {
		method   string
		url      string
		expected bool
	}{
		// Resource requests subject to admission are conditions-aware.
		{method: http.MethodPost, url: "/api/v1/namespaces/default/pods", expected: true},
		{method: http.MethodPut, url: "/api/v1/namespaces/default/pods/foo", expected: true},
		{method: http.MethodPatch, url: "/api/v1/namespaces/default/pods/foo", expected: true},
		{method: http.MethodDelete, url: "/api/v1/namespaces/default/pods/foo", expected: true},
		{method: http.MethodDelete, url: "/api/v1/namespaces/default/pods", expected: true},
		{method: http.MethodPost, url: "/api/v1/namespaces", expected: true},
		{method: http.MethodPut, url: "/api/v1/namespaces/ns-1/status", expected: true},
		{method: http.MethodPost, url: "/apis/apps/v1/namespaces/default/deployments", expected: true},
		{method: http.MethodPut, url: "/apis/example.com/v1/namespaces/default/widgets/foo", expected: true},
		// A literal "*" path segment is parsed as the resource (or name) "*".
		// It is still a resource request that goes through admission, so it is
		// classified like any other resource.
		{method: http.MethodPost, url: "/api/v1/*", expected: true},
		{method: http.MethodPut, url: "/api/v1/namespaces/*", expected: true},

		// Read-only verbs do not go through admission.
		{method: http.MethodGet, url: "/api/v1/namespaces/default/pods/foo"},
		{method: http.MethodGet, url: "/api/v1/namespaces/default/pods"},
		{method: http.MethodGet, url: "/api/v1/namespaces/default/pods?watch=true"},
		{method: http.MethodGet, url: "/api/v1/*"},

		// Resources excluded from admission cannot have conditions enforced.
		{method: http.MethodPost, url: "/apis/authorization.k8s.io/v1/subjectaccessreviews"},
		{method: http.MethodPost, url: "/apis/authorization.k8s.io/v1/namespaces/default/localsubjectaccessreviews"},
		{method: http.MethodPost, url: "/apis/authentication.k8s.io/v1/tokenreviews"},

		// Non-resource requests are never conditions-aware, even for verbs
		// that would be subject to admission on a resource.
		{method: http.MethodPost, url: "/"},
		{method: http.MethodPost, url: "/healthz"},
		{method: http.MethodPost, url: "/metrics"},
		{method: http.MethodPost, url: "/version"},
		{method: http.MethodPost, url: "/openapi/v3"},
		{method: http.MethodPost, url: "/api"},
		{method: http.MethodPost, url: "/api/v1"},
		{method: http.MethodPost, url: "/apis"},
		{method: http.MethodPost, url: "/apis/apps"},
		{method: http.MethodPost, url: "/apis/apps/v1"},
		{method: http.MethodPut, url: "/foo/bar"},
		{method: http.MethodDelete, url: "/logs/foo"},
	}

	for _, tt := range tests {
		t.Run(tt.method+" "+tt.url, func(t *testing.T) {
			req := httptest.NewRequest(tt.method, tt.url, nil)
			info, err := resolver.NewRequestInfo(req)
			if err != nil {
				t.Fatalf("failed to parse request info: %v", err)
			}
			attrs, err := genericapifilters.GetAuthorizerAttributes(request.WithRequestInfo(req.Context(), info))
			if err != nil {
				t.Fatalf("failed to get authorizer attributes: %v", err)
			}
			if got := conditionalRequestClassifier(attrs); got != tt.expected {
				t.Errorf("conditionalRequestClassifier() = %v, want %v; attributes: resourceRequest=%v group=%q version=%q resource=%q subresource=%q verb=%q",
					got, tt.expected, attrs.IsResourceRequest(), attrs.GetAPIGroup(), attrs.GetAPIVersion(), attrs.GetResource(), attrs.GetSubresource(), attrs.GetVerb())
			}
		})
	}
}
