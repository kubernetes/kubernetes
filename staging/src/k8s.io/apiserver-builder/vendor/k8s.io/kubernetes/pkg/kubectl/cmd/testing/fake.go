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
	"io"
	"path/filepath"
	"time"

	"github.com/emicklei/go-restful/swagger"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/client-go/discovery"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/printers"
)

type InternalType struct {
	Kind       string
	APIVersion string

	Name string
}

type ExternalType struct {
	Kind       string `json:"kind"`
	APIVersion string `json:"apiVersion"`

	Name string `json:"name"`
}

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

type InternalNamespacedType struct {
	Kind       string
	APIVersion string

	Name      string
	Namespace string
}

type ExternalNamespacedType struct {
	Kind       string `json:"kind"`
	APIVersion string `json:"apiVersion"`

	Name      string `json:"name"`
	Namespace string `json:"namespace"`
}

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

var ValidVersion = api.Registry.GroupOrDie(api.GroupName).GroupVersion.Version
var InternalGV = schema.GroupVersion{Group: "apitest", Version: runtime.APIVersionInternal}
var UnlikelyGV = schema.GroupVersion{Group: "apitest", Version: "unlikelyversion"}
var ValidVersionGV = schema.GroupVersion{Group: "apitest", Version: ValidVersion}

func newExternalScheme() (*runtime.Scheme, meta.RESTMapper, runtime.Codec) {
	scheme := runtime.NewScheme()

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

	return scheme, mapper, codec
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
	Mapper             meta.RESTMapper
	Typer              runtime.ObjectTyper
	Client             kubectl.RESTClient
	UnstructuredClient kubectl.RESTClient
	Describer          printers.Describer
	Printer            printers.ResourcePrinter
	CommandPrinter     printers.ResourcePrinter
	Validator          validation.Schema
	Namespace          string
	ClientConfig       *restclient.Config
	Err                error
	Command            string
	GenericPrinter     bool
	TmpDir             string

	ClientForMappingFunc             func(mapping *meta.RESTMapping) (resource.RESTClient, error)
	UnstructuredClientForMappingFunc func(mapping *meta.RESTMapping) (resource.RESTClient, error)
}

type FakeFactory struct {
	tf    *TestFactory
	Codec runtime.Codec
}

func NewTestFactory() (cmdutil.Factory, *TestFactory, runtime.Codec, runtime.NegotiatedSerializer) {
	scheme, mapper, codec := newExternalScheme()
	t := &TestFactory{
		Validator: validation.NullSchema{},
		Mapper:    mapper,
		Typer:     scheme,
	}
	negotiatedSerializer := serializer.NegotiatedSerializerWrapper(runtime.SerializerInfo{Serializer: codec})
	return &FakeFactory{
		tf:    t,
		Codec: codec,
	}, t, codec, negotiatedSerializer
}

func (f *FakeFactory) DiscoveryClient() (discovery.CachedDiscoveryInterface, error) {
	discoveryClient, err := discovery.NewDiscoveryClientForConfig(f.tf.ClientConfig)
	if err != nil {
		return nil, err
	}
	return &fakeCachedDiscoveryClient{DiscoveryInterface: discoveryClient}, nil
}

func (f *FakeFactory) FlagSet() *pflag.FlagSet {
	return nil
}

func (f *FakeFactory) Object() (meta.RESTMapper, runtime.ObjectTyper) {
	return api.Registry.RESTMapper(), f.tf.Typer
}

func (f *FakeFactory) UnstructuredObject() (meta.RESTMapper, runtime.ObjectTyper, error) {
	groupResources := testDynamicResources()
	mapper := discovery.NewRESTMapper(groupResources, meta.InterfacesForUnstructured)
	typer := discovery.NewUnstructuredObjectTyper(groupResources)

	fakeDs := &fakeCachedDiscoveryClient{}
	expander, err := cmdutil.NewShortcutExpander(mapper, fakeDs)
	return expander, typer, err
}

func (f *FakeFactory) CategoryExpander() resource.CategoryExpander {
	return resource.LegacyCategoryExpander
}

func (f *FakeFactory) Decoder(bool) runtime.Decoder {
	return f.Codec
}

func (f *FakeFactory) JSONEncoder() runtime.Encoder {
	return f.Codec
}

func (f *FakeFactory) RESTClient() (*restclient.RESTClient, error) {
	return nil, nil
}

func (f *FakeFactory) ClientSet() (internalclientset.Interface, error) {
	return nil, nil
}

func (f *FakeFactory) ClientConfig() (*restclient.Config, error) {
	return f.tf.ClientConfig, f.tf.Err
}

func (f *FakeFactory) BareClientConfig() (*restclient.Config, error) {
	return f.tf.ClientConfig, f.tf.Err
}

func (f *FakeFactory) ClientForMapping(mapping *meta.RESTMapping) (resource.RESTClient, error) {
	if f.tf.ClientForMappingFunc != nil {
		return f.tf.ClientForMappingFunc(mapping)
	}
	return f.tf.Client, f.tf.Err
}

func (f *FakeFactory) FederationClientSetForVersion(version *schema.GroupVersion) (fedclientset.Interface, error) {
	return nil, nil
}
func (f *FakeFactory) FederationClientForVersion(version *schema.GroupVersion) (*restclient.RESTClient, error) {
	return nil, nil
}
func (f *FakeFactory) ClientSetForVersion(requiredVersion *schema.GroupVersion) (internalclientset.Interface, error) {
	return nil, nil
}
func (f *FakeFactory) ClientConfigForVersion(requiredVersion *schema.GroupVersion) (*restclient.Config, error) {
	return nil, nil
}

func (f *FakeFactory) UnstructuredClientForMapping(mapping *meta.RESTMapping) (resource.RESTClient, error) {
	if f.tf.UnstructuredClientForMappingFunc != nil {
		return f.tf.UnstructuredClientForMappingFunc(mapping)
	}
	return f.tf.UnstructuredClient, f.tf.Err
}

func (f *FakeFactory) Describer(*meta.RESTMapping) (printers.Describer, error) {
	return f.tf.Describer, f.tf.Err
}

func (f *FakeFactory) PrinterForCommand(cmd *cobra.Command) (printers.ResourcePrinter, bool, error) {
	return f.tf.CommandPrinter, f.tf.GenericPrinter, f.tf.Err
}

func (f *FakeFactory) Printer(mapping *meta.RESTMapping, options printers.PrintOptions) (printers.ResourcePrinter, error) {
	return f.tf.Printer, f.tf.Err
}

func (f *FakeFactory) Scaler(*meta.RESTMapping) (kubectl.Scaler, error) {
	return nil, nil
}

func (f *FakeFactory) Reaper(*meta.RESTMapping) (kubectl.Reaper, error) {
	return nil, nil
}

func (f *FakeFactory) HistoryViewer(*meta.RESTMapping) (kubectl.HistoryViewer, error) {
	return nil, nil
}

func (f *FakeFactory) Rollbacker(*meta.RESTMapping) (kubectl.Rollbacker, error) {
	return nil, nil
}

func (f *FakeFactory) StatusViewer(*meta.RESTMapping) (kubectl.StatusViewer, error) {
	return nil, nil
}

func (f *FakeFactory) MapBasedSelectorForObject(runtime.Object) (string, error) {
	return "", nil
}

func (f *FakeFactory) PortsForObject(runtime.Object) ([]string, error) {
	return nil, nil
}

func (f *FakeFactory) ProtocolsForObject(runtime.Object) (map[string]string, error) {
	return nil, nil
}

func (f *FakeFactory) LabelsForObject(runtime.Object) (map[string]string, error) {
	return nil, nil
}

func (f *FakeFactory) LogsForObject(object, options runtime.Object, timeout time.Duration) (*restclient.Request, error) {
	return nil, nil
}

func (f *FakeFactory) Pauser(info *resource.Info) ([]byte, error) {
	return nil, nil
}

func (f *FakeFactory) Resumer(info *resource.Info) ([]byte, error) {
	return nil, nil
}

func (f *FakeFactory) ResolveImage(name string) (string, error) {
	return name, nil
}

func (f *FakeFactory) Validator(validate bool, cacheDir string) (validation.Schema, error) {
	return f.tf.Validator, f.tf.Err
}

func (f *FakeFactory) SwaggerSchema(schema.GroupVersionKind) (*swagger.ApiDeclaration, error) {
	return nil, nil
}

func (f *FakeFactory) DefaultNamespace() (string, bool, error) {
	return f.tf.Namespace, false, f.tf.Err
}

func (f *FakeFactory) Generators(cmdName string) map[string]kubectl.Generator {
	var generator map[string]kubectl.Generator
	switch cmdName {
	case "run":
		generator = map[string]kubectl.Generator{
			cmdutil.DeploymentV1Beta1GeneratorName: kubectl.DeploymentV1Beta1{},
		}
	}
	return generator
}

func (f *FakeFactory) CanBeExposed(schema.GroupKind) error {
	return nil
}

func (f *FakeFactory) CanBeAutoscaled(schema.GroupKind) error {
	return nil
}

func (f *FakeFactory) AttachablePodForObject(ob runtime.Object, timeout time.Duration) (*api.Pod, error) {
	return nil, nil
}

func (f *FakeFactory) UpdatePodSpecForObject(obj runtime.Object, fn func(*api.PodSpec) error) (bool, error) {
	return false, nil
}

func (f *FakeFactory) EditorEnvs() []string {
	return nil
}

func (f *FakeFactory) PrintObjectSpecificMessage(obj runtime.Object, out io.Writer) {
}

func (f *FakeFactory) Command(*cobra.Command, bool) string {
	return f.tf.Command
}

func (f *FakeFactory) BindFlags(flags *pflag.FlagSet) {
}

func (f *FakeFactory) BindExternalFlags(flags *pflag.FlagSet) {
}

func (f *FakeFactory) PrintObject(cmd *cobra.Command, mapper meta.RESTMapper, obj runtime.Object, out io.Writer) error {
	return nil
}

func (f *FakeFactory) PrinterForMapping(cmd *cobra.Command, mapping *meta.RESTMapping, withNamespace bool) (printers.ResourcePrinter, error) {
	return f.tf.Printer, f.tf.Err
}

func (f *FakeFactory) NewBuilder() *resource.Builder {
	return nil
}

func (f *FakeFactory) DefaultResourceFilterOptions(cmd *cobra.Command, withNamespace bool) *printers.PrintOptions {
	return &printers.PrintOptions{}
}

func (f *FakeFactory) DefaultResourceFilterFunc() kubectl.Filters {
	return nil
}

func (f *FakeFactory) SuggestedPodTemplateResources() []schema.GroupResource {
	return []schema.GroupResource{}
}

type fakeMixedFactory struct {
	cmdutil.Factory
	tf        *TestFactory
	apiClient resource.RESTClient
}

func (f *fakeMixedFactory) Object() (meta.RESTMapper, runtime.ObjectTyper) {
	var multiRESTMapper meta.MultiRESTMapper
	multiRESTMapper = append(multiRESTMapper, f.tf.Mapper)
	multiRESTMapper = append(multiRESTMapper, testapi.Default.RESTMapper())
	priorityRESTMapper := meta.PriorityRESTMapper{
		Delegate: multiRESTMapper,
		ResourcePriority: []schema.GroupVersionResource{
			{Group: meta.AnyGroup, Version: "v1", Resource: meta.AnyResource},
		},
		KindPriority: []schema.GroupVersionKind{
			{Group: meta.AnyGroup, Version: "v1", Kind: meta.AnyKind},
		},
	}
	return priorityRESTMapper, runtime.MultiObjectTyper{f.tf.Typer, api.Scheme}
}

func (f *fakeMixedFactory) ClientForMapping(m *meta.RESTMapping) (resource.RESTClient, error) {
	if m.ObjectConvertor == api.Scheme {
		return f.apiClient, f.tf.Err
	}
	if f.tf.ClientForMappingFunc != nil {
		return f.tf.ClientForMappingFunc(m)
	}
	return f.tf.Client, f.tf.Err
}

func NewMixedFactory(apiClient resource.RESTClient) (cmdutil.Factory, *TestFactory, runtime.Codec) {
	f, t, c, _ := NewAPIFactory()
	return &fakeMixedFactory{
		Factory:   f,
		tf:        t,
		apiClient: apiClient,
	}, t, c
}

type fakeAPIFactory struct {
	cmdutil.Factory
	tf *TestFactory
}

func (f *fakeAPIFactory) Object() (meta.RESTMapper, runtime.ObjectTyper) {
	return testapi.Default.RESTMapper(), api.Scheme
}

func (f *fakeAPIFactory) UnstructuredObject() (meta.RESTMapper, runtime.ObjectTyper, error) {
	groupResources := testDynamicResources()
	mapper := discovery.NewRESTMapper(groupResources, meta.InterfacesForUnstructured)
	typer := discovery.NewUnstructuredObjectTyper(groupResources)
	fakeDs := &fakeCachedDiscoveryClient{}
	expander, err := cmdutil.NewShortcutExpander(mapper, fakeDs)
	return expander, typer, err
}

func (f *fakeAPIFactory) Decoder(bool) runtime.Decoder {
	return testapi.Default.Codec()
}

func (f *fakeAPIFactory) JSONEncoder() runtime.Encoder {
	return testapi.Default.Codec()
}

func (f *fakeAPIFactory) ClientSet() (internalclientset.Interface, error) {
	// Swap the HTTP client out of the REST client with the fake
	// version.
	fakeClient := f.tf.Client.(*fake.RESTClient)
	clientset := internalclientset.NewForConfigOrDie(f.tf.ClientConfig)
	clientset.CoreClient.RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.AuthenticationClient.RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.AuthorizationClient.RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.AutoscalingClient.RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.BatchClient.RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.CertificatesClient.RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.ExtensionsClient.RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.RbacClient.RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.StorageClient.RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.AppsClient.RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.PolicyClient.RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	clientset.DiscoveryClient.RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
	return clientset, f.tf.Err
}

func (f *fakeAPIFactory) RESTClient() (*restclient.RESTClient, error) {
	// Swap out the HTTP client out of the client with the fake's version.
	fakeClient := f.tf.Client.(*fake.RESTClient)
	restClient, err := restclient.RESTClientFor(f.tf.ClientConfig)
	if err != nil {
		panic(err)
	}
	restClient.Client = fakeClient.Client
	return restClient, f.tf.Err
}

func (f *fakeAPIFactory) DiscoveryClient() (discovery.CachedDiscoveryInterface, error) {
	fakeClient := f.tf.Client.(*fake.RESTClient)
	discoveryClient := discovery.NewDiscoveryClientForConfigOrDie(f.tf.ClientConfig)
	discoveryClient.RESTClient().(*restclient.RESTClient).Client = fakeClient.Client

	cacheDir := filepath.Join(f.tf.TmpDir, ".kube", "cache", "discovery")
	return cmdutil.NewCachedDiscoveryClient(discoveryClient, cacheDir, time.Duration(10*time.Minute)), nil
}

func (f *fakeAPIFactory) ClientSetForVersion(requiredVersion *schema.GroupVersion) (internalclientset.Interface, error) {
	return f.ClientSet()
}

func (f *fakeAPIFactory) ClientConfig() (*restclient.Config, error) {
	return f.tf.ClientConfig, f.tf.Err
}

func (f *fakeAPIFactory) ClientForMapping(m *meta.RESTMapping) (resource.RESTClient, error) {
	if f.tf.ClientForMappingFunc != nil {
		return f.tf.ClientForMappingFunc(m)
	}
	return f.tf.Client, f.tf.Err
}

func (f *fakeAPIFactory) UnstructuredClientForMapping(m *meta.RESTMapping) (resource.RESTClient, error) {
	if f.tf.UnstructuredClientForMappingFunc != nil {
		return f.tf.UnstructuredClientForMappingFunc(m)
	}
	return f.tf.UnstructuredClient, f.tf.Err
}

func (f *fakeAPIFactory) PrinterForCommand(cmd *cobra.Command) (printers.ResourcePrinter, bool, error) {
	return f.tf.CommandPrinter, f.tf.GenericPrinter, f.tf.Err
}

func (f *fakeAPIFactory) Describer(*meta.RESTMapping) (printers.Describer, error) {
	return f.tf.Describer, f.tf.Err
}

func (f *fakeAPIFactory) Printer(mapping *meta.RESTMapping, options printers.PrintOptions) (printers.ResourcePrinter, error) {
	return f.tf.Printer, f.tf.Err
}

func (f *fakeAPIFactory) LogsForObject(object, options runtime.Object, timeout time.Duration) (*restclient.Request, error) {
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
		return c.Core().Pods(f.tf.Namespace).GetLogs(t.Name, opts), nil
	default:
		fqKinds, _, err := api.Scheme.ObjectKinds(object)
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("cannot get the logs from %v", fqKinds[0])
	}
}

func (f *fakeAPIFactory) AttachablePodForObject(object runtime.Object, timeout time.Duration) (*api.Pod, error) {
	switch t := object.(type) {
	case *api.Pod:
		return t, nil
	default:
		gvks, _, err := api.Scheme.ObjectKinds(object)
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("cannot attach to %v: not implemented", gvks[0])
	}
}

func (f *fakeAPIFactory) Validator(validate bool, cacheDir string) (validation.Schema, error) {
	return f.tf.Validator, f.tf.Err
}

func (f *fakeAPIFactory) DefaultNamespace() (string, bool, error) {
	return f.tf.Namespace, false, f.tf.Err
}

func (f *fakeAPIFactory) Command(*cobra.Command, bool) string {
	return f.tf.Command
}

func (f *fakeAPIFactory) Generators(cmdName string) map[string]kubectl.Generator {
	return cmdutil.DefaultGenerators(cmdName)
}

func (f *fakeAPIFactory) PrintObject(cmd *cobra.Command, mapper meta.RESTMapper, obj runtime.Object, out io.Writer) error {
	gvks, _, err := api.Scheme.ObjectKinds(obj)
	if err != nil {
		return err
	}

	mapping, err := mapper.RESTMapping(gvks[0].GroupKind())
	if err != nil {
		return err
	}

	printer, err := f.PrinterForMapping(cmd, mapping, false)
	if err != nil {
		return err
	}
	return printer.PrintObj(obj, out)
}

func (f *fakeAPIFactory) PrinterForMapping(cmd *cobra.Command, mapping *meta.RESTMapping, withNamespace bool) (printers.ResourcePrinter, error) {
	return f.tf.Printer, f.tf.Err
}

func (f *fakeAPIFactory) NewBuilder() *resource.Builder {
	mapper, typer := f.Object()

	return resource.NewBuilder(mapper, f.CategoryExpander(), typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true))
}

func (f *fakeAPIFactory) SuggestedPodTemplateResources() []schema.GroupResource {
	return []schema.GroupResource{}
}

func NewAPIFactory() (cmdutil.Factory, *TestFactory, runtime.Codec, runtime.NegotiatedSerializer) {
	t := &TestFactory{
		Validator: validation.NullSchema{},
	}
	rf := cmdutil.NewFactory(nil)
	return &fakeAPIFactory{
		Factory: rf,
		tf:      t,
	}, t, testapi.Default.Codec(), testapi.Default.NegotiatedSerializer()
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
					{Name: "type", Namespaced: false, Kind: "Type"},
					{Name: "namespacedtype", Namespaced: true, Kind: "NamespacedType"},
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
	}
}
