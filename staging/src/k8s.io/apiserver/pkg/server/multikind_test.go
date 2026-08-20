package server

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/watch"
	openapinamer "k8s.io/apiserver/pkg/endpoints/openapi"
	"k8s.io/apiserver/pkg/registry/rest"
	kubeopenapi "k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

type multiKindType struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`
}

const multiKindTypeModelName = "io.k8s.apiserver.pkg.server.multiKindType"

func (multiKindType) OpenAPIModelName() string { return multiKindTypeModelName }

func (m *multiKindType) DeepCopyObject() runtime.Object {
	out := *m
	out.ObjectMeta = *m.ObjectMeta.DeepCopy()
	return &out
}

type multiKindTypeList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []multiKindType `json:"items"`
}

func (m *multiKindTypeList) DeepCopyObject() runtime.Object {
	out := multiKindTypeList{
		TypeMeta: m.TypeMeta,
		ListMeta: *m.ListMeta.DeepCopy(),
		Items:    make([]multiKindType, len(m.Items)),
	}
	for i := range m.Items {
		out.Items[i] = *m.Items[i].DeepCopyObject().(*multiKindType)
	}
	return &out
}

type multiKindStorage struct {
	kind string
	gv   schema.GroupVersion
}

var _ rest.Getter = &multiKindStorage{}
var _ rest.Lister = &multiKindStorage{}
var _ rest.Watcher = &multiKindStorage{}

var _ rest.GroupVersionKindProvider = &multiKindStorage{}
var _ rest.GroupVersionListKindProvider = &multiKindStorage{}

func (s *multiKindStorage) GroupVersionKind(containingGV schema.GroupVersion) schema.GroupVersionKind {
	return containingGV.WithKind(s.kind)
}

func (s *multiKindStorage) GroupVersionListKind(containingGV schema.GroupVersion) schema.GroupVersionKind {
	return containingGV.WithKind(s.kind + "List")
}

func (s *multiKindStorage) New() runtime.Object     { return &multiKindType{} }
func (s *multiKindStorage) NewList() runtime.Object { return &multiKindTypeList{} }
func (s *multiKindStorage) Destroy()                {}
func (s *multiKindStorage) NamespaceScoped() bool   { return true }
func (s *multiKindStorage) GetSingularName() string { return strings.ToLower(s.kind) }
func (s *multiKindStorage) ConvertToTable(ctx context.Context, object runtime.Object, tableOptions runtime.Object) (*metav1.Table, error) {
	return nil, nil
}

func (s *multiKindStorage) Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	return &multiKindType{
		TypeMeta: metav1.TypeMeta{
			Kind:       s.kind,
			APIVersion: s.gv.String(),
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
	}, nil
}

func (s *multiKindStorage) List(ctx context.Context, options *metainternalversion.ListOptions) (runtime.Object, error) {
	return &multiKindTypeList{}, nil
}

func (s *multiKindStorage) Watch(ctx context.Context, options *metainternalversion.ListOptions) (watch.Interface, error) {
	w := watch.NewFake()
	go func() {
		w.Add(&multiKindType{
			TypeMeta: metav1.TypeMeta{
				Kind:       s.kind,
				APIVersion: s.gv.String(),
			},
			ObjectMeta: metav1.ObjectMeta{
				Name: "watchobj",
			},
		})
		w.Stop()
	}()
	return w, nil
}

func multiKindOpenAPIDefinitions(ref kubeopenapi.ReferenceCallback) map[string]kubeopenapi.OpenAPIDefinition {
	defs := testGetOpenAPIDefinitions(ref)
	emptyDef := kubeopenapi.OpenAPIDefinition{
		Schema: spec.Schema{
			SchemaProps: spec.SchemaProps{
				Description: "Test type registered under multiple Kind names",
				Properties:  map[string]spec.Schema{},
			},
		},
	}
	defs[multiKindTypeModelName] = emptyDef
	defs["k8s.io/apiserver/pkg/server.multiKindTypeList"] = emptyDef
	defs["io.k8s.apimachinery.pkg.apis.meta.v1.WatchEvent"] = emptyDef
	return defs
}

func newMultiKindTestServer(t *testing.T) (*httptest.Server, schema.GroupVersion) {
	t.Helper()

	config, _ := setUp(t)

	gv := schema.GroupVersion{Group: "testgroup", Version: "v1"}
	testScheme := runtime.NewScheme()
	// Register the same Go type under two different Kind names. The
	// registration order matters: without the fix, the first-registered
	// Kind ("Foo") would be returned for both resources.
	testScheme.AddKnownTypeWithName(gv.WithKind("Foo"), &multiKindType{})
	testScheme.AddKnownTypeWithName(gv.WithKind("Bar"), &multiKindType{})
	testScheme.AddKnownTypeWithName(gv.WithKind("FooList"), &multiKindTypeList{})
	testScheme.AddKnownTypeWithName(gv.WithKind("BarList"), &multiKindTypeList{})
	testScheme.AddKnownTypes(v1GroupVersion, &metav1.Status{})
	metav1.AddToGroupVersion(testScheme, v1GroupVersion)
	metav1.AddToGroupVersion(testScheme, gv)

	config.OpenAPIConfig = DefaultOpenAPIConfig(multiKindOpenAPIDefinitions, openapinamer.NewDefinitionNamer(testScheme))
	config.OpenAPIConfig.Info.Version = "unversioned"
	config.OpenAPIV3Config = DefaultOpenAPIV3Config(multiKindOpenAPIDefinitions, openapinamer.NewDefinitionNamer(testScheme))
	config.OpenAPIV3Config.Info.Version = "unversioned"

	s, err := config.Complete(nil).New("test", NewEmptyDelegate())
	if err != nil {
		t.Fatalf("Error creating server: %v", err)
	}

	testCodecs := serializer.NewCodecFactory(testScheme)

	apiGroupInfo := APIGroupInfo{
		PrioritizedVersions: []schema.GroupVersion{gv},
		VersionedResourcesStorageMap: map[string]map[string]rest.Storage{
			gv.Version: {
				"foos": &multiKindStorage{kind: "Foo", gv: gv},
				"bars": &multiKindStorage{kind: "Bar", gv: gv},
			},
		},
		OptionsExternalVersion: &schema.GroupVersion{Version: "v1"},
		ParameterCodec:         runtime.NewParameterCodec(testScheme),
		NegotiatedSerializer:   testCodecs,
		Scheme:                 testScheme,
	}

	if err := s.InstallAPIGroup(&apiGroupInfo); err != nil {
		t.Fatalf("Error installing API group: %v", err)
	}

	s.PrepareRun()

	server := httptest.NewServer(s.Handler)
	t.Cleanup(server.Close)

	return server, gv
}

// TestMultiKindResponseEncoding verifies that when a single Go type is
// registered under multiple Kind names in the same GroupVersion, each API
// resource endpoint serializes responses with the correct Kind.
func TestMultiKindResponseEncoding(t *testing.T) {
	server, gv := newMultiKindTestServer(t)

	for _, tt := range []struct {
		resource string
		wantKind string
		list     bool
	}{
		{"foos", "Foo", false},
		{"bars", "Bar", false},
		{"foos", "FooList", true},
		{"bars", "BarList", true},
	} {
		t.Run(tt.wantKind, func(t *testing.T) {
			var url string
			if tt.list {
				url = server.URL + "/apis/testgroup/v1/namespaces/default/" + tt.resource
			} else {
				url = server.URL + "/apis/testgroup/v1/namespaces/default/" + tt.resource + "/testobj"
			}
			resp, err := http.Get(url)
			if err != nil {
				t.Fatalf("GET %s: %v", url, err)
			}
			defer resp.Body.Close()

			body, err := io.ReadAll(resp.Body)
			if err != nil {
				t.Fatalf("reading response body: %v", err)
			}

			if resp.StatusCode != http.StatusOK {
				t.Fatalf("GET %s: status %d, body: %s", url, resp.StatusCode, body)
			}

			var result struct {
				Kind       string `json:"kind"`
				APIVersion string `json:"apiVersion"`
			}
			if err := json.Unmarshal(body, &result); err != nil {
				t.Fatalf("unmarshalling response JSON: %v\nbody: %s", err, body)
			}

			if result.Kind != tt.wantKind {
				t.Errorf("kind: got %q, want %q", result.Kind, tt.wantKind)
			}
			if result.APIVersion != gv.String() {
				t.Errorf("apiVersion: got %q, want %q", result.APIVersion, gv.String())
			}
		})
	}
}

// TestMultiKindDiscovery verifies that the discovery document served at
// /apis/<group>/<version> reports the correct Kind for each resource when a
// single Go type is registered under multiple Kind names.
func TestMultiKindDiscovery(t *testing.T) {
	server, _ := newMultiKindTestServer(t)

	resp, err := http.Get(server.URL + "/apis/testgroup/v1")
	if err != nil {
		t.Fatalf("GET discovery: %v", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("reading response body: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET discovery: status %d, body: %s", resp.StatusCode, body)
	}

	var resourceList metav1.APIResourceList
	if err := json.Unmarshal(body, &resourceList); err != nil {
		t.Fatalf("unmarshalling APIResourceList: %v\nbody: %s", err, body)
	}

	kindByResource := map[string]string{}
	for _, r := range resourceList.APIResources {
		kindByResource[r.Name] = r.Kind
	}

	for _, tt := range []struct {
		resource string
		wantKind string
	}{
		{"foos", "Foo"},
		{"bars", "Bar"},
	} {
		t.Run(tt.resource, func(t *testing.T) {
			got, ok := kindByResource[tt.resource]
			if !ok {
				t.Fatalf("resource %q not found in discovery", tt.resource)
			}
			if got != tt.wantKind {
				t.Errorf("discovery kind for %q: got %q, want %q", tt.resource, got, tt.wantKind)
			}
		})
	}
}

// TestMultiKindOpenAPI verifies that the OpenAPI v2 spec served at
// /openapi/v2 contains x-kubernetes-group-version-kind entries with the
// correct Kind for each resource when a single Go type is registered under
// multiple Kind names.
func TestMultiKindOpenAPI(t *testing.T) {
	server, _ := newMultiKindTestServer(t)

	resp, err := http.Get(server.URL + "/openapi/v2")
	if err != nil {
		t.Fatalf("GET /openapi/v2: %v", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("reading response body: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /openapi/v2: status %d, body: %s", resp.StatusCode, body)
	}

	var openAPIDoc struct {
		Definitions map[string]struct {
			XKubernetesGroupVersionKind []struct {
				Group   string `json:"group"`
				Version string `json:"version"`
				Kind    string `json:"kind"`
			} `json:"x-kubernetes-group-version-kind"`
		} `json:"definitions"`
	}
	if err := json.Unmarshal(body, &openAPIDoc); err != nil {
		t.Fatalf("unmarshalling OpenAPI doc: %v", err)
	}

	// Find the definition for our test type and check that it has both
	// Kind names in x-kubernetes-group-version-kind.
	def, ok := openAPIDoc.Definitions[multiKindTypeModelName]
	if !ok {
		t.Fatalf("definition %q not found in OpenAPI spec", multiKindTypeModelName)
	}

	foundKinds := map[string]bool{}
	for _, gvk := range def.XKubernetesGroupVersionKind {
		foundKinds[gvk.Kind] = true
	}

	for _, wantKind := range []string{"Foo", "Bar"} {
		t.Run(wantKind, func(t *testing.T) {
			if !foundKinds[wantKind] {
				t.Errorf("x-kubernetes-group-version-kind missing Kind %q; found: %v", wantKind, def.XKubernetesGroupVersionKind)
			}
		})
	}
}

// TestMultiKindWatch verifies that objects embedded in watch events have the
// correct Kind when a single Go type is registered under multiple Kind names.
func TestMultiKindWatch(t *testing.T) {
	server, gv := newMultiKindTestServer(t)

	for _, tt := range []struct {
		resource string
		wantKind string
	}{
		{"foos", "Foo"},
		{"bars", "Bar"},
	} {
		t.Run(tt.wantKind, func(t *testing.T) {
			url := server.URL + "/apis/testgroup/v1/namespaces/default/" + tt.resource + "?watch=true"
			resp, err := http.Get(url)
			if err != nil {
				t.Fatalf("GET %s: %v", url, err)
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusOK {
				body, _ := io.ReadAll(resp.Body)
				t.Fatalf("GET %s: status %d, body: %s", url, resp.StatusCode, body)
			}

			scanner := bufio.NewScanner(resp.Body)
			if !scanner.Scan() {
				t.Fatalf("no watch event received")
			}

			var event struct {
				Type   string `json:"type"`
				Object struct {
					Kind       string `json:"kind"`
					APIVersion string `json:"apiVersion"`
				} `json:"object"`
			}
			if err := json.Unmarshal(scanner.Bytes(), &event); err != nil {
				t.Fatalf("unmarshalling watch event: %v\nraw: %s", err, scanner.Bytes())
			}

			if event.Object.Kind != tt.wantKind {
				t.Errorf("embedded object kind: got %q, want %q", event.Object.Kind, tt.wantKind)
			}
			if event.Object.APIVersion != gv.String() {
				t.Errorf("embedded object apiVersion: got %q, want %q", event.Object.APIVersion, gv.String())
			}
		})
	}
}
