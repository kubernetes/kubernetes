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

package testing

import (
	"errors"
	"fmt"
	"path/filepath"
	"time"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/categories"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi"
	openapitesting "k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi/testing"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
	"k8s.io/kubernetes/pkg/kubectl/validation"
	"k8s.io/kubernetes/pkg/printers"
)

// +k8s:deepcopy-gen=true
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type InternalType struct {
	Kind       string
	APIVersion string

	Name string
}

// +k8s:deepcopy-gen=true
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type ExternalType struct {
	Kind       string `json:"kind"`
	APIVersion string `json:"apiVersion"`

	Name string `json:"name"`
}

// +k8s:deepcopy-gen=true
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type ExternalType2 struct {
	Kind       string `json:"kind"`
	APIVersion string `json:"apiVersion"`

	Name string `json:"name"`
}

func (obj *InternalType) GetObjectKind() schema.ObjectKind { return obj }
func (obj *InternalType) SetGroupVersionKind(gvk schema.GroupVersionKind) {
	obj.APIVersion, obj.Kind = gvk.ToAPIVersionAndKind()
}
func (obj *InternalType) GroupVersionKind() schema.GroupVersionKind {
	return schema.FromAPIVersionAndKind(obj.APIVersion, obj.Kind)
}
func (obj *ExternalType) GetObjectKind() schema.ObjectKind { return obj }
func (obj *ExternalType) SetGroupVersionKind(gvk schema.GroupVersionKind) {
	obj.APIVersion, obj.Kind = gvk.ToAPIVersionAndKind()
}
func (obj *ExternalType) GroupVersionKind() schema.GroupVersionKind {
	return schema.FromAPIVersionAndKind(obj.APIVersion, obj.Kind)
}
func (obj *ExternalType2) GetObjectKind() schema.ObjectKind { return obj }
func (obj *ExternalType2) SetGroupVersionKind(gvk schema.GroupVersionKind) {
	obj.APIVersion, obj.Kind = gvk.ToAPIVersionAndKind()
}
func (obj *ExternalType2) GroupVersionKind() schema.GroupVersionKind {
	return schema.FromAPIVersionAndKind(obj.APIVersion, obj.Kind)
}

func NewInternalType(kind, apiversion, name string) *InternalType {
	item := InternalType{Kind: kind,
		APIVersion: apiversion,
		Name:       name}
	return &item
}

// +k8s:deepcopy-gen=true
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type InternalNamespacedType struct {
	Kind       string
	APIVersion string

	Name      string
	Namespace string
}

// +k8s:deepcopy-gen=true
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type ExternalNamespacedType struct {
	Kind       string `json:"kind"`
	APIVersion string `json:"apiVersion"`

	Name      string `json:"name"`
	Namespace string `json:"namespace"`
}

// +k8s:deepcopy-gen=true
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type ExternalNamespacedType2 struct {
	Kind       string `json:"kind"`
	APIVersion string `json:"apiVersion"`

	Name      string `json:"name"`
	Namespace string `json:"namespace"`
}

func (obj *InternalNamespacedType) GetObjectKind() schema.ObjectKind { return obj }
func (obj *InternalNamespacedType) SetGroupVersionKind(gvk schema.GroupVersionKind) {
	obj.APIVersion, obj.Kind = gvk.ToAPIVersionAndKind()
}
func (obj *InternalNamespacedType) GroupVersionKind() schema.GroupVersionKind {
	return schema.FromAPIVersionAndKind(obj.APIVersion, obj.Kind)
}
func (obj *ExternalNamespacedType) GetObjectKind() schema.ObjectKind { return obj }
func (obj *ExternalNamespacedType) SetGroupVersionKind(gvk schema.GroupVersionKind) {
	obj.APIVersion, obj.Kind = gvk.ToAPIVersionAndKind()
}
func (obj *ExternalNamespacedType) GroupVersionKind() schema.GroupVersionKind {
	return schema.FromAPIVersionAndKind(obj.APIVersion, obj.Kind)
}
func (obj *ExternalNamespacedType2) GetObjectKind() schema.ObjectKind { return obj }
func (obj *ExternalNamespacedType2) SetGroupVersionKind(gvk schema.GroupVersionKind) {
	obj.APIVersion, obj.Kind = gvk.ToAPIVersionAndKind()
}
func (obj *ExternalNamespacedType2) GroupVersionKind() schema.GroupVersionKind {
	return schema.FromAPIVersionAndKind(obj.APIVersion, obj.Kind)
}

func NewInternalNamespacedType(kind, apiversion, name, namespace string) *InternalNamespacedType {
	item := InternalNamespacedType{Kind: kind,
		APIVersion: apiversion,
		Name:       name,
		Namespace:  namespace}
	return &item
}

var versionErr = errors.New("not a version")

func versionErrIfFalse(b bool) error {
	if b {
		return nil
	}
	return versionErr
}

var ValidVersion = legacyscheme.Registry.GroupOrDie(api.GroupName).GroupVersion.Version
var InternalGV = schema.GroupVersion{Group: "apitest", Version: runtime.APIVersionInternal}
var UnlikelyGV = schema.GroupVersion{Group: "apitest", Version: "unlikelyversion"}
var ValidVersionGV = schema.GroupVersion{Group: "apitest", Version: ValidVersion}

func NewExternalScheme() (*runtime.Scheme, meta.RESTMapper, runtime.Codec) {
	scheme := runtime.NewScheme()
	mapper, codec := AddToScheme(scheme)
	return scheme, mapper, codec
}

func AddToScheme(scheme *runtime.Scheme) (meta.RESTMapper, runtime.Codec) {
	scheme.AddKnownTypeWithName(InternalGV.WithKind("Type"), &InternalType{})
	scheme.AddKnownTypeWithName(UnlikelyGV.WithKind("Type"), &ExternalType{})
	//This tests that kubectl will not confuse the external scheme with the internal scheme, even when they accidentally have versions of the same name.
	scheme.AddKnownTypeWithName(ValidVersionGV.WithKind("Type"), &ExternalType2{})

	scheme.AddKnownTypeWithName(InternalGV.WithKind("NamespacedType"), &InternalNamespacedType{})
	scheme.AddKnownTypeWithName(UnlikelyGV.WithKind("NamespacedType"), &ExternalNamespacedType{})
	//This tests that kubectl will not confuse the external scheme with the internal scheme, even when they accidentally have versions of the same name.
	scheme.AddKnownTypeWithName(ValidVersionGV.WithKind("NamespacedType"), &ExternalNamespacedType2{})

	codecs := serializer.NewCodecFactory(scheme)
	codec := codecs.LegacyCodec(UnlikelyGV)
	mapper := meta.NewDefaultRESTMapper([]schema.GroupVersion{UnlikelyGV, ValidVersionGV}, func(version schema.GroupVersion) (*meta.VersionInterfaces, error) {
		return &meta.VersionInterfaces{
			ObjectConvertor:  scheme,
			MetadataAccessor: meta.NewAccessor(),
		}, versionErrIfFalse(version == ValidVersionGV || version == UnlikelyGV)
	})
	for _, gv := range []schema.GroupVersion{UnlikelyGV, ValidVersionGV} {
		for kind := range scheme.KnownTypes(gv) {
			gvk := gv.WithKind(kind)

			scope := meta.RESTScopeNamespace
			mapper.Add(gvk, scope)
		}
	}

	return mapper, codec
}

type fakeCachedDiscoveryClient struct {
	discovery.DiscoveryInterface
}

func (d *fakeCachedDiscoveryClient) Fresh() bool {
	return true
}

func (d *fakeCachedDiscoveryClient) Invalidate() {
}

func (d *fakeCachedDiscoveryClient) ServerResources() ([]*metav1.APIResourceList, error) {
	return []*metav1.APIResourceList{}, nil
}

type TestFactory struct {
	cmdutil.Factory

	Client             kubectl.RESTClient
	UnstructuredClient kubectl.RESTClient
	DescriberVal       printers.Describer
	Namespace          string
	ClientConfigVal    *restclient.Config
	CommandVal         string

	UnstructuredClientForMappingFunc func(mapping *meta.RESTMapping) (resource.RESTClient, error)
	OpenAPISchemaFunc                func() (openapi.Resources, error)
}

func NewTestFactory() *TestFactory {
	return &TestFactory{
		Factory: cmdutil.NewFactory(nil),
	}
}

func (f *TestFactory) CategoryExpander() categories.CategoryExpander {
	return categories.LegacyCategoryExpander
}

func (f *TestFactory) ClientConfig() (*restclient.Config, error) {
	return f.ClientConfigVal, nil
}

func (f *TestFactory) BareClientConfig() (*restclient.Config, error) {
	return f.ClientConfigVal, nil
}

func (f *TestFactory) ClientForMapping(mapping *meta.RESTMapping) (resource.RESTClient, error) {
	return f.Client, nil
}

func (f *TestFactory) UnstructuredClientForMapping(mapping *meta.RESTMapping) (resource.RESTClient, error) {
	if f.UnstructuredClientForMappingFunc != nil {
		return f.UnstructuredClientForMappingFunc(mapping)
	}
	return f.UnstructuredClient, nil
}

func (f *TestFactory) Describer(*meta.RESTMapping) (printers.Describer, error) {
	return f.DescriberVal, nil
}

func (f *TestFactory) Validator(validate bool) (validation.Schema, error) {
	return validation.NullSchema{}, nil
}

func (f *TestFactory) DefaultNamespace() (string, bool, error) {
	return f.Namespace, false, nil
}

func (f *TestFactory) OpenAPISchema() (openapi.Resources, error) {
	if f.OpenAPISchemaFunc != nil {
		return f.OpenAPISchemaFunc()
	}
	return openapitesting.EmptyResources{}, nil
}

func (f *TestFactory) Command(*cobra.Command, bool) string {
	return f.CommandVal
}

func (f *TestFactory) NewBuilder() *resource.Builder {
	mapper, typer := f.Object()

	return resource.NewBuilder(
		&resource.Mapper{
			RESTMapper:   mapper,
			ObjectTyper:  typer,
			ClientMapper: resource.ClientMapperFunc(f.ClientForMapping),
			Decoder:      cmdutil.InternalVersionDecoder(),
		},
		&resource.Mapper{
			RESTMapper:   mapper,
			ObjectTyper:  typer,
			ClientMapper: resource.ClientMapperFunc(f.UnstructuredClientForMapping),
			Decoder:      unstructured.UnstructuredJSONScheme,
		},
		f.CategoryExpander(),
	)
}

func (f *TestFactory) KubernetesClientSet() (*kubernetes.Clientset, error) {
	fakeClient := f.Client.(*fake.RESTClient)
	clientset := kubernetes.NewForConfigOrDie(f.ClientConfigVal)

	clientset.CoreV1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.AuthorizationV1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.AuthorizationV1beta1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.AuthorizationV1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.AuthorizationV1beta1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.AutoscalingV1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.AutoscalingV2beta1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.BatchV1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.BatchV2alpha1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.CertificatesV1beta1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.ExtensionsV1beta1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.RbacV1alpha1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.RbacV1beta1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.StorageV1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.StorageV1beta1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.AppsV1beta1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.AppsV1beta2().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.PolicyV1beta1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.DiscoveryClient.RESTClient().(*restclient.RESTClient).Client = fakeClient.Client

	return clientset, nil
}

func (f *TestFactory) ClientSet() (internalclientset.Interface, error) {
	// Swap the HTTP client out of the REST client with the fake
	// version.
	fakeClient := f.Client.(*fake.RESTClient)
	clientset := internalclientset.NewForConfigOrDie(f.ClientConfigVal)
	clientset.Core().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.Authentication().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.Authorization().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.Autoscaling().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.Batch().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.Certificates().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.Extensions().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.Rbac().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.Storage().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.Apps().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.Policy().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.DiscoveryClient.RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	return clientset, nil
}

func (f *TestFactory) RESTClient() (*restclient.RESTClient, error) {
	// Swap out the HTTP client out of the client with the fake's version.
	fakeClient := f.Client.(*fake.RESTClient)
	restClient, err := restclient.RESTClientFor(f.ClientConfigVal)
	if err != nil {
		panic(err)
	}
	restClient.Client = fakeClient.Client
	return restClient, nil
}

func (f *TestFactory) DiscoveryClient() (discovery.CachedDiscoveryInterface, error) {
	fakeClient := f.Client.(*fake.RESTClient)
	discoveryClient := discovery.NewDiscoveryClientForConfigOrDie(f.ClientConfigVal)
	discoveryClient.RESTClient().(*restclient.RESTClient).Client = fakeClient.Client

	cacheDir := filepath.Join("", ".kube", "cache", "discovery")
	return cmdutil.NewCachedDiscoveryClient(discoveryClient, cacheDir, time.Duration(10*time.Minute)), nil
}

func (f *TestFactory) ClientSetForVersion(requiredVersion *schema.GroupVersion) (internalclientset.Interface, error) {
	return f.ClientSet()
}

func (f *TestFactory) Object() (meta.RESTMapper, runtime.ObjectTyper) {
	groupResources := testDynamicResources()
	mapper := discovery.NewRESTMapper(
		groupResources,
		meta.InterfacesForUnstructuredConversion(func(version schema.GroupVersion) (*meta.VersionInterfaces, error) {
			switch version {
			// provide typed objects for these two versions
			case ValidVersionGV, UnlikelyGV:
				return &meta.VersionInterfaces{
					ObjectConvertor:  scheme.Scheme,
					MetadataAccessor: meta.NewAccessor(),
				}, nil
				// otherwise fall back to the legacy scheme
			default:
				return legacyscheme.Registry.InterfacesFor(version)
			}
		}),
	)
	// for backwards compatibility with existing tests, allow rest mappings from the scheme to show up
	// TODO: make this opt-in?
	mapper = meta.FirstHitRESTMapper{
		MultiRESTMapper: meta.MultiRESTMapper{
			mapper,
			legacyscheme.Registry.RESTMapper(),
		},
	}

	// TODO: should probably be the external scheme
	typer := discovery.NewUnstructuredObjectTyper(groupResources, legacyscheme.Scheme)
	fakeDs := &fakeCachedDiscoveryClient{}
	expander := cmdutil.NewShortcutExpander(mapper, fakeDs)
	return expander, typer
}

func (f *TestFactory) LogsForObject(object, options runtime.Object, timeout time.Duration) (*restclient.Request, error) {
	c, err := f.ClientSet()
	if err != nil {
		panic(err)
	}

	switch t := object.(type) {
	case *api.Pod:
		opts, ok := options.(*api.PodLogOptions)
		if !ok {
			return nil, errors.New("provided options object is not a PodLogOptions")
		}
		return c.Core().Pods(f.Namespace).GetLogs(t.Name, opts), nil
	default:
		return nil, fmt.Errorf("cannot get the logs from %T", object)
	}
}

func testDynamicResources() []*discovery.APIGroupResources {
	return []*discovery.APIGroupResources{
		{
			Group: metav1.APIGroup{
				Versions: []metav1.GroupVersionForDiscovery{
					{Version: "v1"},
				},
				PreferredVersion: metav1.GroupVersionForDiscovery{Version: "v1"},
			},
			VersionedResources: map[string][]metav1.APIResource{
				"v1": {
					{Name: "pods", Namespaced: true, Kind: "Pod"},
					{Name: "services", Namespaced: true, Kind: "Service"},
					{Name: "replicationcontrollers", Namespaced: true, Kind: "ReplicationController"},
					{Name: "componentstatuses", Namespaced: false, Kind: "ComponentStatus"},
					{Name: "nodes", Namespaced: false, Kind: "Node"},
					{Name: "secrets", Namespaced: true, Kind: "Secret"},
					{Name: "configmaps", Namespaced: true, Kind: "ConfigMap"},
					{Name: "namespacedtype", Namespaced: true, Kind: "NamespacedType"},
					{Name: "namespaces", Namespaced: false, Kind: "Namespace"},
					{Name: "resourcequotas", Namespaced: true, Kind: "ResourceQuota"},
				},
			},
		},
		{
			Group: metav1.APIGroup{
				Name: "extensions",
				Versions: []metav1.GroupVersionForDiscovery{
					{Version: "v1beta1"},
				},
				PreferredVersion: metav1.GroupVersionForDiscovery{Version: "v1beta1"},
			},
			VersionedResources: map[string][]metav1.APIResource{
				"v1beta1": {
					{Name: "deployments", Namespaced: true, Kind: "Deployment"},
					{Name: "replicasets", Namespaced: true, Kind: "ReplicaSet"},
				},
			},
		},
		{
			Group: metav1.APIGroup{
				Name: "apps",
				Versions: []metav1.GroupVersionForDiscovery{
					{Version: "v1beta1"},
					{Version: "v1beta2"},
					{Version: "v1"},
				},
				PreferredVersion: metav1.GroupVersionForDiscovery{Version: "v1"},
			},
			VersionedResources: map[string][]metav1.APIResource{
				"v1beta1": {
					{Name: "deployments", Namespaced: true, Kind: "Deployment"},
					{Name: "replicasets", Namespaced: true, Kind: "ReplicaSet"},
				},
				"v1beta2": {
					{Name: "deployments", Namespaced: true, Kind: "Deployment"},
				},
				"v1": {
					{Name: "deployments", Namespaced: true, Kind: "Deployment"},
					{Name: "replicasets", Namespaced: true, Kind: "ReplicaSet"},
				},
			},
		},
		{
			Group: metav1.APIGroup{
				Name: "autoscaling",
				Versions: []metav1.GroupVersionForDiscovery{
					{Version: "v1"},
					{Version: "v2beta1"},
				},
				PreferredVersion: metav1.GroupVersionForDiscovery{Version: "v2beta1"},
			},
			VersionedResources: map[string][]metav1.APIResource{
				"v1": {
					{Name: "horizontalpodautoscalers", Namespaced: true, Kind: "HorizontalPodAutoscaler"},
				},
				"v2beta1": {
					{Name: "horizontalpodautoscalers", Namespaced: true, Kind: "HorizontalPodAutoscaler"},
				},
			},
		},
		{
			Group: metav1.APIGroup{
				Name: "storage.k8s.io",
				Versions: []metav1.GroupVersionForDiscovery{
					{Version: "v1beta1"},
					{Version: "v0"},
				},
				PreferredVersion: metav1.GroupVersionForDiscovery{Version: "v1beta1"},
			},
			VersionedResources: map[string][]metav1.APIResource{
				"v1beta1": {
					{Name: "storageclasses", Namespaced: false, Kind: "StorageClass"},
				},
				// bogus version of a known group/version/resource to make sure kubectl falls back to generic object mode
				"v0": {
					{Name: "storageclasses", Namespaced: false, Kind: "StorageClass"},
				},
			},
		},
		{
			Group: metav1.APIGroup{
				Name: "rbac.authorization.k8s.io",
				Versions: []metav1.GroupVersionForDiscovery{
					{Version: "v1beta1"},
					{Version: "v1"},
				},
				PreferredVersion: metav1.GroupVersionForDiscovery{Version: "v1"},
			},
			VersionedResources: map[string][]metav1.APIResource{
				"v1": {
					{Name: "clusterroles", Namespaced: false, Kind: "ClusterRole"},
				},
				"v1beta1": {
					{Name: "clusterrolebindings", Namespaced: false, Kind: "ClusterRoleBinding"},
				},
			},
		},
		{
			Group: metav1.APIGroup{
				Name: "company.com",
				Versions: []metav1.GroupVersionForDiscovery{
					{Version: "v1"},
				},
				PreferredVersion: metav1.GroupVersionForDiscovery{Version: "v1"},
			},
			VersionedResources: map[string][]metav1.APIResource{
				"v1": {
					{Name: "bars", Namespaced: true, Kind: "Bar"},
				},
			},
		},
		{
			Group: metav1.APIGroup{
				Name: "unit-test.test.com",
				Versions: []metav1.GroupVersionForDiscovery{
					{GroupVersion: "unit-test.test.com/v1", Version: "v1"},
				},
				PreferredVersion: metav1.GroupVersionForDiscovery{
					GroupVersion: "unit-test.test.com/v1",
					Version:      "v1"},
			},
			VersionedResources: map[string][]metav1.APIResource{
				"v1": {
					{Name: "widgets", Namespaced: true, Kind: "Widget"},
				},
			},
		},
		{
			Group: metav1.APIGroup{
				Name: "apitest",
				Versions: []metav1.GroupVersionForDiscovery{
					{GroupVersion: "apitest/unlikelyversion", Version: "unlikelyversion"},
				},
				PreferredVersion: metav1.GroupVersionForDiscovery{
					GroupVersion: "apitest/unlikelyversion",
					Version:      "unlikelyversion"},
			},
			VersionedResources: map[string][]metav1.APIResource{
				"unlikelyversion": {
					{Name: "types", SingularName: "type", Namespaced: false, Kind: "Type"},
				},
			},
		},
	}
}
