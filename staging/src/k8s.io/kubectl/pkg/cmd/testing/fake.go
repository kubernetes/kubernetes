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
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/meta/testrestmapper"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/discovery"
	diskcached "k8s.io/client-go/discovery/cached/disk"
	"k8s.io/client-go/dynamic"
	fakedynamic "k8s.io/client-go/dynamic/fake"
	"k8s.io/client-go/kubernetes"
	openapiclient "k8s.io/client-go/openapi"
	"k8s.io/client-go/openapi/openapitest"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	"k8s.io/client-go/restmapper"
	scaleclient "k8s.io/client-go/scale"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util/openapi"
	openapitesting "k8s.io/kubectl/pkg/util/openapi/testing"
	"k8s.io/kubectl/pkg/validation"
)

// InternalType is the schema for internal type
// +k8s:deepcopy-gen=true
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type InternalType struct {
	Kind       string
	APIVersion string

	Name string
}

// ExternalType is the schema for external type
// +k8s:deepcopy-gen=true
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type ExternalType struct {
	Kind       string `json:"kind"`
	APIVersion string `json:"apiVersion"`

	Name string `json:"name"`
}

// ExternalType2 is another schema for external type
// +k8s:deepcopy-gen=true
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type ExternalType2 struct {
	Kind       string `json:"kind"`
	APIVersion string `json:"apiVersion"`

	Name string `json:"name"`
}

// GetObjectKind returns the ObjectKind schema
func (obj *InternalType) GetObjectKind() schema.ObjectKind { return obj }

// SetGroupVersionKind sets the version and kind
func (obj *InternalType) SetGroupVersionKind(gvk schema.GroupVersionKind) {
	obj.APIVersion, obj.Kind = gvk.ToAPIVersionAndKind()
}

// GroupVersionKind returns GroupVersionKind schema
func (obj *InternalType) GroupVersionKind() schema.GroupVersionKind {
	return schema.FromAPIVersionAndKind(obj.APIVersion, obj.Kind)
}

// GetObjectKind returns the ObjectKind schema
func (obj *ExternalType) GetObjectKind() schema.ObjectKind { return obj }

// SetGroupVersionKind returns the GroupVersionKind schema
func (obj *ExternalType) SetGroupVersionKind(gvk schema.GroupVersionKind) {
	obj.APIVersion, obj.Kind = gvk.ToAPIVersionAndKind()
}

// GroupVersionKind returns the GroupVersionKind schema
func (obj *ExternalType) GroupVersionKind() schema.GroupVersionKind {
	return schema.FromAPIVersionAndKind(obj.APIVersion, obj.Kind)
}

// GetObjectKind returns the ObjectKind schema
func (obj *ExternalType2) GetObjectKind() schema.ObjectKind { return obj }

// SetGroupVersionKind sets the API version and obj kind from schema
func (obj *ExternalType2) SetGroupVersionKind(gvk schema.GroupVersionKind) {
	obj.APIVersion, obj.Kind = gvk.ToAPIVersionAndKind()
}

// GroupVersionKind returns the FromAPIVersionAndKind schema
func (obj *ExternalType2) GroupVersionKind() schema.GroupVersionKind {
	return schema.FromAPIVersionAndKind(obj.APIVersion, obj.Kind)
}

// NewInternalType returns an initialized InternalType instance
func NewInternalType(kind, apiversion, name string) *InternalType {
	item := InternalType{Kind: kind,
		APIVersion: apiversion,
		Name:       name}
	return &item
}

func convertInternalTypeToExternalType(in *InternalType, out *ExternalType, s conversion.Scope) error {
	out.Kind = in.Kind
	out.APIVersion = in.APIVersion
	out.Name = in.Name
	return nil
}

func convertInternalTypeToExternalType2(in *InternalType, out *ExternalType2, s conversion.Scope) error {
	out.Kind = in.Kind
	out.APIVersion = in.APIVersion
	out.Name = in.Name
	return nil
}

func convertExternalTypeToInternalType(in *ExternalType, out *InternalType, s conversion.Scope) error {
	out.Kind = in.Kind
	out.APIVersion = in.APIVersion
	out.Name = in.Name
	return nil
}

func convertExternalType2ToInternalType(in *ExternalType2, out *InternalType, s conversion.Scope) error {
	out.Kind = in.Kind
	out.APIVersion = in.APIVersion
	out.Name = in.Name
	return nil
}

// InternalNamespacedType schema for internal namespaced types
// +k8s:deepcopy-gen=true
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type InternalNamespacedType struct {
	Kind       string
	APIVersion string

	Name      string
	Namespace string
}

// ExternalNamespacedType schema for external namespaced types
// +k8s:deepcopy-gen=true
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type ExternalNamespacedType struct {
	Kind       string `json:"kind"`
	APIVersion string `json:"apiVersion"`

	Name      string `json:"name"`
	Namespace string `json:"namespace"`
}

// ExternalNamespacedType2 schema for external namespaced types
// +k8s:deepcopy-gen=true
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type ExternalNamespacedType2 struct {
	Kind       string `json:"kind"`
	APIVersion string `json:"apiVersion"`

	Name      string `json:"name"`
	Namespace string `json:"namespace"`
}

// GetObjectKind returns the ObjectKind schema
func (obj *InternalNamespacedType) GetObjectKind() schema.ObjectKind { return obj }

// SetGroupVersionKind sets the API group and kind from schema
func (obj *InternalNamespacedType) SetGroupVersionKind(gvk schema.GroupVersionKind) {
	obj.APIVersion, obj.Kind = gvk.ToAPIVersionAndKind()
}

// GroupVersionKind returns the GroupVersionKind schema
func (obj *InternalNamespacedType) GroupVersionKind() schema.GroupVersionKind {
	return schema.FromAPIVersionAndKind(obj.APIVersion, obj.Kind)
}

// GetObjectKind returns the ObjectKind schema
func (obj *ExternalNamespacedType) GetObjectKind() schema.ObjectKind { return obj }

// SetGroupVersionKind sets the API version and kind from schema
func (obj *ExternalNamespacedType) SetGroupVersionKind(gvk schema.GroupVersionKind) {
	obj.APIVersion, obj.Kind = gvk.ToAPIVersionAndKind()
}

// GroupVersionKind returns the GroupVersionKind schema
func (obj *ExternalNamespacedType) GroupVersionKind() schema.GroupVersionKind {
	return schema.FromAPIVersionAndKind(obj.APIVersion, obj.Kind)
}

// GetObjectKind returns the ObjectKind schema
func (obj *ExternalNamespacedType2) GetObjectKind() schema.ObjectKind { return obj }

// SetGroupVersionKind sets the API version and kind from schema
func (obj *ExternalNamespacedType2) SetGroupVersionKind(gvk schema.GroupVersionKind) {
	obj.APIVersion, obj.Kind = gvk.ToAPIVersionAndKind()
}

// GroupVersionKind returns the GroupVersionKind schema
func (obj *ExternalNamespacedType2) GroupVersionKind() schema.GroupVersionKind {
	return schema.FromAPIVersionAndKind(obj.APIVersion, obj.Kind)
}

// NewInternalNamespacedType returns an initialized instance of InternalNamespacedType
func NewInternalNamespacedType(kind, apiversion, name, namespace string) *InternalNamespacedType {
	item := InternalNamespacedType{Kind: kind,
		APIVersion: apiversion,
		Name:       name,
		Namespace:  namespace}
	return &item
}

func convertInternalNamespacedTypeToExternalNamespacedType(in *InternalNamespacedType, out *ExternalNamespacedType, s conversion.Scope) error {
	out.Kind = in.Kind
	out.APIVersion = in.APIVersion
	out.Name = in.Name
	out.Namespace = in.Namespace
	return nil
}

func convertInternalNamespacedTypeToExternalNamespacedType2(in *InternalNamespacedType, out *ExternalNamespacedType2, s conversion.Scope) error {
	out.Kind = in.Kind
	out.APIVersion = in.APIVersion
	out.Name = in.Name
	out.Namespace = in.Namespace
	return nil
}

func convertExternalNamespacedTypeToInternalNamespacedType(in *ExternalNamespacedType, out *InternalNamespacedType, s conversion.Scope) error {
	out.Kind = in.Kind
	out.APIVersion = in.APIVersion
	out.Name = in.Name
	out.Namespace = in.Namespace
	return nil
}

func convertExternalNamespacedType2ToInternalNamespacedType(in *ExternalNamespacedType2, out *InternalNamespacedType, s conversion.Scope) error {
	out.Kind = in.Kind
	out.APIVersion = in.APIVersion
	out.Name = in.Name
	out.Namespace = in.Namespace
	return nil
}

// ValidVersion of API
var ValidVersion = "v1"

// InternalGV is the internal group version object
var InternalGV = schema.GroupVersion{Group: "apitest", Version: runtime.APIVersionInternal}

// UnlikelyGV is a group version object for unrecognised version
var UnlikelyGV = schema.GroupVersion{Group: "apitest", Version: "unlikelyversion"}

// ValidVersionGV is the valid group version object
var ValidVersionGV = schema.GroupVersion{Group: "apitest", Version: ValidVersion}

// NewExternalScheme returns required objects for ExternalScheme
func NewExternalScheme() (*runtime.Scheme, meta.RESTMapper, runtime.Codec) {
	scheme := runtime.NewScheme()
	mapper, codec := AddToScheme(scheme)
	return scheme, mapper, codec
}

func registerConversions(s *runtime.Scheme) error {
	if err := s.AddConversionFunc((*InternalType)(nil), (*ExternalType)(nil), func(a, b interface{}, scope conversion.Scope) error {
		return convertInternalTypeToExternalType(a.(*InternalType), b.(*ExternalType), scope)
	}); err != nil {
		return err
	}
	if err := s.AddConversionFunc((*InternalType)(nil), (*ExternalType2)(nil), func(a, b interface{}, scope conversion.Scope) error {
		return convertInternalTypeToExternalType2(a.(*InternalType), b.(*ExternalType2), scope)
	}); err != nil {
		return err
	}
	if err := s.AddConversionFunc((*ExternalType)(nil), (*InternalType)(nil), func(a, b interface{}, scope conversion.Scope) error {
		return convertExternalTypeToInternalType(a.(*ExternalType), b.(*InternalType), scope)
	}); err != nil {
		return err
	}
	if err := s.AddConversionFunc((*ExternalType2)(nil), (*InternalType)(nil), func(a, b interface{}, scope conversion.Scope) error {
		return convertExternalType2ToInternalType(a.(*ExternalType2), b.(*InternalType), scope)
	}); err != nil {
		return err
	}
	if err := s.AddConversionFunc((*InternalNamespacedType)(nil), (*ExternalNamespacedType)(nil), func(a, b interface{}, scope conversion.Scope) error {
		return convertInternalNamespacedTypeToExternalNamespacedType(a.(*InternalNamespacedType), b.(*ExternalNamespacedType), scope)
	}); err != nil {
		return err
	}
	if err := s.AddConversionFunc((*InternalNamespacedType)(nil), (*ExternalNamespacedType2)(nil), func(a, b interface{}, scope conversion.Scope) error {
		return convertInternalNamespacedTypeToExternalNamespacedType2(a.(*InternalNamespacedType), b.(*ExternalNamespacedType2), scope)
	}); err != nil {
		return err
	}
	if err := s.AddConversionFunc((*ExternalNamespacedType)(nil), (*InternalNamespacedType)(nil), func(a, b interface{}, scope conversion.Scope) error {
		return convertExternalNamespacedTypeToInternalNamespacedType(a.(*ExternalNamespacedType), b.(*InternalNamespacedType), scope)
	}); err != nil {
		return err
	}
	if err := s.AddConversionFunc((*ExternalNamespacedType2)(nil), (*InternalNamespacedType)(nil), func(a, b interface{}, scope conversion.Scope) error {
		return convertExternalNamespacedType2ToInternalNamespacedType(a.(*ExternalNamespacedType2), b.(*InternalNamespacedType), scope)
	}); err != nil {
		return err
	}
	return nil
}

// AddToScheme adds required objects into scheme
func AddToScheme(scheme *runtime.Scheme) (meta.RESTMapper, runtime.Codec) {
	scheme.AddKnownTypeWithName(InternalGV.WithKind("Type"), &InternalType{})
	scheme.AddKnownTypeWithName(UnlikelyGV.WithKind("Type"), &ExternalType{})
	// This tests that kubectl will not confuse the external scheme with the internal scheme, even when they accidentally have versions of the same name.
	scheme.AddKnownTypeWithName(ValidVersionGV.WithKind("Type"), &ExternalType2{})

	scheme.AddKnownTypeWithName(InternalGV.WithKind("NamespacedType"), &InternalNamespacedType{})
	scheme.AddKnownTypeWithName(UnlikelyGV.WithKind("NamespacedType"), &ExternalNamespacedType{})
	// This tests that kubectl will not confuse the external scheme with the internal scheme, even when they accidentally have versions of the same name.
	scheme.AddKnownTypeWithName(ValidVersionGV.WithKind("NamespacedType"), &ExternalNamespacedType2{})

	utilruntime.Must(registerConversions(scheme))

	codecs := serializer.NewCodecFactory(scheme)
	codec := codecs.LegacyCodec(UnlikelyGV)
	mapper := meta.NewDefaultRESTMapper([]schema.GroupVersion{UnlikelyGV, ValidVersionGV})
	for _, gv := range []schema.GroupVersion{UnlikelyGV, ValidVersionGV} {
		for kind := range scheme.KnownTypes(gv) {
			gvk := gv.WithKind(kind)

			scope := meta.RESTScopeNamespace
			mapper.Add(gvk, scope)
		}
	}

	return mapper, codec
}

type FakeCachedDiscoveryClient struct {
	discovery.DiscoveryInterface
	Groups             []*metav1.APIGroup
	Resources          []*metav1.APIResourceList
	PreferredResources []*metav1.APIResourceList
	Invalidations      int
}

func NewFakeCachedDiscoveryClient() *FakeCachedDiscoveryClient {
	return &FakeCachedDiscoveryClient{
		Groups:             []*metav1.APIGroup{},
		Resources:          []*metav1.APIResourceList{},
		PreferredResources: []*metav1.APIResourceList{},
		Invalidations:      0,
	}
}

func (d *FakeCachedDiscoveryClient) Fresh() bool {
	return true
}

func (d *FakeCachedDiscoveryClient) Invalidate() {
	d.Invalidations++
}

func (d *FakeCachedDiscoveryClient) ServerGroupsAndResources() ([]*metav1.APIGroup, []*metav1.APIResourceList, error) {
	return d.Groups, d.Resources, nil
}

func (d *FakeCachedDiscoveryClient) ServerGroups() (*metav1.APIGroupList, error) {
	groupList := &metav1.APIGroupList{Groups: []metav1.APIGroup{}}
	for _, g := range d.Groups {
		groupList.Groups = append(groupList.Groups, *g)
	}
	return groupList, nil
}

func (d *FakeCachedDiscoveryClient) ServerPreferredResources() ([]*metav1.APIResourceList, error) {
	return d.PreferredResources, nil
}

// TestFactory extends cmdutil.Factory
type TestFactory struct {
	cmdutil.Factory

	kubeConfigFlags *genericclioptions.TestConfigFlags

	Client             RESTClient
	ScaleGetter        scaleclient.ScalesGetter
	UnstructuredClient RESTClient
	ClientConfigVal    *restclient.Config
	FakeDynamicClient  *fakedynamic.FakeDynamicClient

	tempConfigFile *os.File

	UnstructuredClientForMappingFunc resource.FakeClientFunc
	OpenAPISchemaFunc                func() (openapi.Resources, error)
	OpenAPIV3ClientFunc              func() (openapiclient.Client, error)
}

// NewTestFactory returns an initialized TestFactory instance
func NewTestFactory() *TestFactory {
	// specify an optionalClientConfig to explicitly use in testing
	// to avoid polluting an existing user config.
	tmpFile, err := os.CreateTemp(os.TempDir(), "cmdtests_temp")
	if err != nil {
		panic(fmt.Sprintf("unable to create a fake client config: %v", err))
	}

	loadingRules := &clientcmd.ClientConfigLoadingRules{
		Precedence:     []string{tmpFile.Name()},
		MigrationRules: map[string]string{},
	}

	overrides := &clientcmd.ConfigOverrides{ClusterDefaults: clientcmdapi.Cluster{Server: "http://localhost:8080"}}
	fallbackReader := bytes.NewBuffer([]byte{})
	clientConfig := clientcmd.NewInteractiveDeferredLoadingClientConfig(loadingRules, overrides, fallbackReader)

	configFlags := genericclioptions.NewTestConfigFlags().
		WithClientConfig(clientConfig).
		WithRESTMapper(testRESTMapper())

	restConfig, err := clientConfig.ClientConfig()
	if err != nil {
		panic(fmt.Sprintf("unable to create a fake restclient config: %v", err))
	}

	return &TestFactory{
		Factory:           cmdutil.NewFactory(configFlags),
		kubeConfigFlags:   configFlags,
		FakeDynamicClient: fakedynamic.NewSimpleDynamicClient(scheme.Scheme),
		tempConfigFile:    tmpFile,

		ClientConfigVal: restConfig,
	}
}

// WithNamespace is used to mention namespace reactively
func (f *TestFactory) WithNamespace(ns string) *TestFactory {
	f.kubeConfigFlags.WithNamespace(ns)
	return f
}

// WithClientConfig sets the client config to use
func (f *TestFactory) WithClientConfig(clientConfig clientcmd.ClientConfig) *TestFactory {
	f.kubeConfigFlags.WithClientConfig(clientConfig)
	return f
}

func (f *TestFactory) WithDiscoveryClient(discoveryClient discovery.CachedDiscoveryInterface) *TestFactory {
	f.kubeConfigFlags.WithDiscoveryClient(discoveryClient)
	return f
}

// Cleanup cleans up TestFactory temp config file
func (f *TestFactory) Cleanup() {
	if f.tempConfigFile == nil {
		return
	}

	f.tempConfigFile.Close()
	os.Remove(f.tempConfigFile.Name())
}

// ToRESTConfig is used to get ClientConfigVal from a TestFactory
func (f *TestFactory) ToRESTConfig() (*restclient.Config, error) {
	return f.ClientConfigVal, nil
}

// ClientForMapping is used to Client from a TestFactory
func (f *TestFactory) ClientForMapping(mapping *meta.RESTMapping) (resource.RESTClient, error) {
	return f.Client, nil
}

// PathOptions returns a new PathOptions with a temp file
func (f *TestFactory) PathOptions() *clientcmd.PathOptions {
	pathOptions := clientcmd.NewDefaultPathOptions()
	pathOptions.GlobalFile = f.tempConfigFile.Name()
	pathOptions.EnvVar = ""
	return pathOptions
}

// PathOptionsWithConfig writes a config to a temp file and returns PathOptions
func (f *TestFactory) PathOptionsWithConfig(config clientcmdapi.Config) (*clientcmd.PathOptions, error) {
	pathOptions := f.PathOptions()
	err := clientcmd.WriteToFile(config, pathOptions.GlobalFile)
	if err != nil {
		return nil, err
	}

	return pathOptions, nil
}

// UnstructuredClientForMapping is used to get UnstructuredClient from a TestFactory
func (f *TestFactory) UnstructuredClientForMapping(mapping *meta.RESTMapping) (resource.RESTClient, error) {
	if f.UnstructuredClientForMappingFunc != nil {
		return f.UnstructuredClientForMappingFunc(mapping.GroupVersionKind.GroupVersion())
	}
	return f.UnstructuredClient, nil
}

// Validator returns a validation schema
func (f *TestFactory) Validator(validateDirective string) (validation.Schema, error) {
	return validation.NullSchema{}, nil
}

// OpenAPISchema returns openapi resources
func (f *TestFactory) OpenAPISchema() (openapi.Resources, error) {
	if f.OpenAPISchemaFunc != nil {
		return f.OpenAPISchemaFunc()
	}
	return openapitesting.EmptyResources{}, nil
}

func (f *TestFactory) OpenAPIV3Client() (openapiclient.Client, error) {
	if f.OpenAPIV3ClientFunc != nil {
		return f.OpenAPIV3ClientFunc()
	}
	return openapitest.NewFakeClient(), nil
}

// NewBuilder returns an initialized resource.Builder instance
func (f *TestFactory) NewBuilder() *resource.Builder {
	return resource.NewFakeBuilder(
		func(version schema.GroupVersion) (resource.RESTClient, error) {
			if f.UnstructuredClientForMappingFunc != nil {
				return f.UnstructuredClientForMappingFunc(version)
			}
			if f.UnstructuredClient != nil {
				return f.UnstructuredClient, nil
			}
			return f.Client, nil
		},
		f.ToRESTMapper,
		func() (restmapper.CategoryExpander, error) {
			return resource.FakeCategoryExpander, nil
		},
	)
}

// KubernetesClientSet initializes and returns the Clientset using TestFactory
func (f *TestFactory) KubernetesClientSet() (*kubernetes.Clientset, error) {
	fakeClient := f.Client.(*fake.RESTClient)
	clientset := kubernetes.NewForConfigOrDie(f.ClientConfigVal)

	clientset.CoreV1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.AuthorizationV1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.AuthorizationV1beta1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.AuthorizationV1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.AuthorizationV1beta1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.AuthenticationV1alpha1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.AutoscalingV1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.AutoscalingV2().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.BatchV1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.CertificatesV1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.CertificatesV1beta1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.ExtensionsV1beta1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.RbacV1alpha1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.RbacV1beta1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.StorageV1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.StorageV1beta1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.AppsV1beta1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.AppsV1beta2().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.AppsV1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.PolicyV1beta1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.PolicyV1().RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.DiscoveryClient.RESTClient().(*restclient.RESTClient).Client = fakeClient.Client

	return clientset, nil
}

// DynamicClient returns a dynamic client from TestFactory
func (f *TestFactory) DynamicClient() (dynamic.Interface, error) {
	if f.FakeDynamicClient != nil {
		return f.FakeDynamicClient, nil
	}
	return f.Factory.DynamicClient()
}

// RESTClient returns a REST client from TestFactory
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

// DiscoveryClient returns a discovery client from TestFactory
func (f *TestFactory) DiscoveryClient() (discovery.CachedDiscoveryInterface, error) {
	fakeClient := f.Client.(*fake.RESTClient)

	cacheDir := filepath.Join("", ".kube", "cache", "discovery")
	cachedClient, err := diskcached.NewCachedDiscoveryClientForConfig(f.ClientConfigVal, cacheDir, "", time.Duration(10*time.Minute))
	if err != nil {
		return nil, err
	}
	cachedClient.RESTClient().(*restclient.RESTClient).Client = fakeClient.Client

	return cachedClient, nil
}

func testRESTMapper() meta.RESTMapper {
	groupResources := testDynamicResources()
	mapper := restmapper.NewDiscoveryRESTMapper(groupResources)
	// for backwards compatibility with existing tests, allow rest mappings from the scheme to show up
	// TODO: make this opt-in?
	mapper = meta.FirstHitRESTMapper{
		MultiRESTMapper: meta.MultiRESTMapper{
			mapper,
			testrestmapper.TestOnlyStaticRESTMapper(scheme.Scheme),
		},
	}

	fakeDs := NewFakeCachedDiscoveryClient()
	expander := restmapper.NewShortcutExpander(mapper, fakeDs, nil)
	return expander
}

// ScaleClient returns the ScalesGetter from a TestFactory
func (f *TestFactory) ScaleClient() (scaleclient.ScalesGetter, error) {
	return f.ScaleGetter, nil
}

func testDynamicResources() []*restmapper.APIGroupResources {
	return []*restmapper.APIGroupResources{
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
				Name: "batch",
				Versions: []metav1.GroupVersionForDiscovery{
					{Version: "v1beta1"},
					{Version: "v1"},
				},
				PreferredVersion: metav1.GroupVersionForDiscovery{Version: "v1"},
			},
			VersionedResources: map[string][]metav1.APIResource{
				"v1beta1": {
					{Name: "cronjobs", Namespaced: true, Kind: "CronJob"},
				},
				"v1": {
					{Name: "jobs", Namespaced: true, Kind: "Job"},
				},
			},
		},
		{
			Group: metav1.APIGroup{
				Name: "autoscaling",
				Versions: []metav1.GroupVersionForDiscovery{
					{Version: "v1"},
					{Version: "v2"},
				},
				PreferredVersion: metav1.GroupVersionForDiscovery{Version: "v2"},
			},
			VersionedResources: map[string][]metav1.APIResource{
				"v1": {
					{Name: "horizontalpodautoscalers", Namespaced: true, Kind: "HorizontalPodAutoscaler"},
				},
				"v2": {
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
					{Name: "applysets", Namespaced: false, Kind: "ApplySet"},
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
