/*
Copyright 2014 The Kubernetes Authors.

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

package cmd

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/emicklei/go-restful/swagger"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/typed/discovery"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/fake"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/serializer"
	"k8s.io/kubernetes/pkg/util/strings"
)

func initTestErrorHandler(t *testing.T) {
	cmdutil.BehaviorOnFatal(func(str string, code int) {
		t.Errorf("Error running command (exit code %d): %s", code, str)
	})
}

func defaultHeader() http.Header {
	header := http.Header{}
	header.Set("Content-Type", runtime.ContentTypeJSON)
	return header
}

func defaultClientConfig() *restclient.Config {
	return &restclient.Config{
		APIPath: "/api",
		ContentConfig: restclient.ContentConfig{
			NegotiatedSerializer: api.Codecs,
			ContentType:          runtime.ContentTypeJSON,
			GroupVersion:         &registered.GroupOrDie(api.GroupName).GroupVersion,
		},
	}
}

func defaultClientConfigForVersion(version *unversioned.GroupVersion) *restclient.Config {
	return &restclient.Config{
		APIPath: "/api",
		ContentConfig: restclient.ContentConfig{
			NegotiatedSerializer: api.Codecs,
			ContentType:          runtime.ContentTypeJSON,
			GroupVersion:         version,
		},
	}
}

type internalType struct {
	Kind       string
	APIVersion string

	Name string
}

type externalType struct {
	Kind       string `json:"kind"`
	APIVersion string `json:"apiVersion"`

	Name string `json:"name"`
}

type ExternalType2 struct {
	Kind       string `json:"kind"`
	APIVersion string `json:"apiVersion"`

	Name string `json:"name"`
}

func (obj *internalType) GetObjectKind() unversioned.ObjectKind { return obj }
func (obj *internalType) SetGroupVersionKind(gvk unversioned.GroupVersionKind) {
	obj.APIVersion, obj.Kind = gvk.ToAPIVersionAndKind()
}
func (obj *internalType) GroupVersionKind() unversioned.GroupVersionKind {
	return unversioned.FromAPIVersionAndKind(obj.APIVersion, obj.Kind)
}
func (obj *externalType) GetObjectKind() unversioned.ObjectKind { return obj }
func (obj *externalType) SetGroupVersionKind(gvk unversioned.GroupVersionKind) {
	obj.APIVersion, obj.Kind = gvk.ToAPIVersionAndKind()
}
func (obj *externalType) GroupVersionKind() unversioned.GroupVersionKind {
	return unversioned.FromAPIVersionAndKind(obj.APIVersion, obj.Kind)
}
func (obj *ExternalType2) GetObjectKind() unversioned.ObjectKind { return obj }
func (obj *ExternalType2) SetGroupVersionKind(gvk unversioned.GroupVersionKind) {
	obj.APIVersion, obj.Kind = gvk.ToAPIVersionAndKind()
}
func (obj *ExternalType2) GroupVersionKind() unversioned.GroupVersionKind {
	return unversioned.FromAPIVersionAndKind(obj.APIVersion, obj.Kind)
}

var versionErr = errors.New("not a version")

func versionErrIfFalse(b bool) error {
	if b {
		return nil
	}
	return versionErr
}

var validVersion = registered.GroupOrDie(api.GroupName).GroupVersion.Version
var internalGV = unversioned.GroupVersion{Group: "apitest", Version: runtime.APIVersionInternal}
var unlikelyGV = unversioned.GroupVersion{Group: "apitest", Version: "unlikelyversion"}
var validVersionGV = unversioned.GroupVersion{Group: "apitest", Version: validVersion}

func newExternalScheme() (*runtime.Scheme, meta.RESTMapper, runtime.Codec) {
	scheme := runtime.NewScheme()
	scheme.AddKnownTypeWithName(internalGV.WithKind("Type"), &internalType{})
	scheme.AddKnownTypeWithName(unlikelyGV.WithKind("Type"), &externalType{})
	//This tests that kubectl will not confuse the external scheme with the internal scheme, even when they accidentally have versions of the same name.
	scheme.AddKnownTypeWithName(validVersionGV.WithKind("Type"), &ExternalType2{})

	codecs := serializer.NewCodecFactory(scheme)
	codec := codecs.LegacyCodec(unlikelyGV)
	mapper := meta.NewDefaultRESTMapper([]unversioned.GroupVersion{unlikelyGV, validVersionGV}, func(version unversioned.GroupVersion) (*meta.VersionInterfaces, error) {
		return &meta.VersionInterfaces{
			ObjectConvertor:  scheme,
			MetadataAccessor: meta.NewAccessor(),
		}, versionErrIfFalse(version == validVersionGV || version == unlikelyGV)
	})
	for _, gv := range []unversioned.GroupVersion{unlikelyGV, validVersionGV} {
		for kind := range scheme.KnownTypes(gv) {
			gvk := gv.WithKind(kind)

			scope := meta.RESTScopeNamespace
			mapper.Add(gvk, scope)
		}
	}

	return scheme, mapper, codec
}

type testPrinter struct {
	Objects []runtime.Object
	Err     error
}

func (t *testPrinter) PrintObj(obj runtime.Object, out io.Writer) error {
	t.Objects = append(t.Objects, obj)
	fmt.Fprintf(out, "%#v", obj)
	return t.Err
}

// TODO: implement HandledResources()
func (t *testPrinter) HandledResources() []string {
	return []string{}
}

func (t *testPrinter) AfterPrint(output io.Writer, res string) error {
	return nil
}

type testDescriber struct {
	Name, Namespace string
	Settings        kubectl.DescriberSettings
	Output          string
	Err             error
}

func (t *testDescriber) Describe(namespace, name string, describerSettings kubectl.DescriberSettings) (output string, err error) {
	t.Namespace, t.Name = namespace, name
	t.Settings = describerSettings
	return t.Output, t.Err
}

type testFactory struct {
	Mapper       meta.RESTMapper
	Typer        runtime.ObjectTyper
	Client       kubectl.RESTClient
	Describer    kubectl.Describer
	Printer      kubectl.ResourcePrinter
	Validator    validation.Schema
	Namespace    string
	ClientConfig *restclient.Config
	Err          error
}

type fakeFactory struct {
	tf    *testFactory
	Codec runtime.Codec
}

func NewTestFactory() (cmdutil.Factory, *testFactory, runtime.Codec, runtime.NegotiatedSerializer) {
	scheme, mapper, codec := newExternalScheme()
	t := &testFactory{
		Validator: validation.NullSchema{},
		Mapper:    mapper,
		Typer:     scheme,
	}
	negotiatedSerializer := serializer.NegotiatedSerializerWrapper(
		runtime.SerializerInfo{Serializer: codec},
		runtime.StreamSerializerInfo{})
	return &fakeFactory{
		tf:    t,
		Codec: codec,
	}, t, codec, negotiatedSerializer
}

func (f *fakeFactory) FlagSet() *pflag.FlagSet {
	return nil
}

func (f *fakeFactory) Object() (meta.RESTMapper, runtime.ObjectTyper) {
	priorityRESTMapper := meta.PriorityRESTMapper{
		Delegate: f.tf.Mapper,
		ResourcePriority: []unversioned.GroupVersionResource{
			{Group: meta.AnyGroup, Version: "v1", Resource: meta.AnyResource},
		},
		KindPriority: []unversioned.GroupVersionKind{
			{Group: meta.AnyGroup, Version: "v1", Kind: meta.AnyKind},
		},
	}
	return priorityRESTMapper, f.tf.Typer
}

func (f *fakeFactory) UnstructuredObject() (meta.RESTMapper, runtime.ObjectTyper, error) {
	return nil, nil, nil
}

func (f *fakeFactory) Decoder(bool) runtime.Decoder {
	return f.Codec
}

func (f *fakeFactory) JSONEncoder() runtime.Encoder {
	return f.Codec
}

func (f *fakeFactory) RESTClient() (*restclient.RESTClient, error) {
	return nil, nil
}

func (f *fakeFactory) ClientSet() (*internalclientset.Clientset, error) {
	return nil, nil
}

func (f *fakeFactory) ClientConfig() (*restclient.Config, error) {
	return f.tf.ClientConfig, f.tf.Err
}

func (f *fakeFactory) ClientForMapping(*meta.RESTMapping) (resource.RESTClient, error) {
	return f.tf.Client, f.tf.Err
}

func (f *fakeFactory) UnstructuredClientForMapping(*meta.RESTMapping) (resource.RESTClient, error) {
	return nil, nil
}

func (f *fakeFactory) Describer(*meta.RESTMapping) (kubectl.Describer, error) {
	return f.tf.Describer, f.tf.Err
}

func (f *fakeFactory) Printer(mapping *meta.RESTMapping, options kubectl.PrintOptions) (kubectl.ResourcePrinter, error) {
	return f.tf.Printer, f.tf.Err
}

func (f *fakeFactory) Scaler(*meta.RESTMapping) (kubectl.Scaler, error) {
	return nil, nil
}

func (f *fakeFactory) Reaper(*meta.RESTMapping) (kubectl.Reaper, error) {
	return nil, nil
}

func (f *fakeFactory) HistoryViewer(*meta.RESTMapping) (kubectl.HistoryViewer, error) {
	return nil, nil
}

func (f *fakeFactory) Rollbacker(*meta.RESTMapping) (kubectl.Rollbacker, error) {
	return nil, nil
}

func (f *fakeFactory) StatusViewer(*meta.RESTMapping) (kubectl.StatusViewer, error) {
	return nil, nil
}

func (f *fakeFactory) MapBasedSelectorForObject(runtime.Object) (string, error) {
	return "", nil
}

func (f *fakeFactory) PortsForObject(runtime.Object) ([]string, error) {
	return nil, nil
}

func (f *fakeFactory) ProtocolsForObject(runtime.Object) (map[string]string, error) {
	return nil, nil
}

func (f *fakeFactory) LabelsForObject(runtime.Object) (map[string]string, error) {
	return nil, nil
}

func (f *fakeFactory) LogsForObject(object, options runtime.Object) (*restclient.Request, error) {
	return nil, nil
}

func (f *fakeFactory) PauseObject(runtime.Object) (bool, error) {
	return false, nil
}

func (f *fakeFactory) ResumeObject(runtime.Object) (bool, error) {
	return false, nil
}

func (f *fakeFactory) Validator(validate bool, cacheDir string) (validation.Schema, error) {
	return f.tf.Validator, f.tf.Err
}

func (f *fakeFactory) SwaggerSchema(unversioned.GroupVersionKind) (*swagger.ApiDeclaration, error) {
	return nil, nil
}

func (f *fakeFactory) DefaultNamespace() (string, bool, error) {
	return f.tf.Namespace, false, f.tf.Err
}

func (f *fakeFactory) Generators(string) map[string]kubectl.Generator {
	return nil
}

func (f *fakeFactory) CanBeExposed(unversioned.GroupKind) error {
	return nil
}

func (f *fakeFactory) CanBeAutoscaled(unversioned.GroupKind) error {
	return nil
}

func (f *fakeFactory) AttachablePodForObject(ob runtime.Object) (*api.Pod, error) {
	return nil, nil
}

func (f *fakeFactory) UpdatePodSpecForObject(obj runtime.Object, fn func(*api.PodSpec) error) (bool, error) {
	return false, nil
}

func (f *fakeFactory) EditorEnvs() []string {
	return nil
}

func (f *fakeFactory) PrintObjectSpecificMessage(obj runtime.Object, out io.Writer) {
}

func (f *fakeFactory) Command() string {
	return ""
}

func (f *fakeFactory) BindFlags(flags *pflag.FlagSet) {
}

func (f *fakeFactory) BindExternalFlags(flags *pflag.FlagSet) {
}

func (f *fakeFactory) PrintObject(cmd *cobra.Command, mapper meta.RESTMapper, obj runtime.Object, out io.Writer) error {
	return nil
}

func (f *fakeFactory) PrinterForMapping(cmd *cobra.Command, mapping *meta.RESTMapping, withNamespace bool) (kubectl.ResourcePrinter, error) {
	return f.tf.Printer, f.tf.Err
}

func (f *fakeFactory) NewBuilder() *resource.Builder {
	return nil
}

func (f *fakeFactory) DefaultResourceFilterOptions(cmd *cobra.Command, withNamespace bool) *kubectl.PrintOptions {
	return &kubectl.PrintOptions{}
}

func (f *fakeFactory) DefaultResourceFilterFunc() kubectl.Filters {
	return nil
}

type fakeMixedFactory struct {
	cmdutil.Factory
	tf        *testFactory
	apiClient resource.RESTClient
}

func (f *fakeMixedFactory) Object() (meta.RESTMapper, runtime.ObjectTyper) {
	var multiRESTMapper meta.MultiRESTMapper
	multiRESTMapper = append(multiRESTMapper, f.tf.Mapper)
	multiRESTMapper = append(multiRESTMapper, testapi.Default.RESTMapper())
	priorityRESTMapper := meta.PriorityRESTMapper{
		Delegate: multiRESTMapper,
		ResourcePriority: []unversioned.GroupVersionResource{
			{Group: meta.AnyGroup, Version: "v1", Resource: meta.AnyResource},
		},
		KindPriority: []unversioned.GroupVersionKind{
			{Group: meta.AnyGroup, Version: "v1", Kind: meta.AnyKind},
		},
	}
	return priorityRESTMapper, runtime.MultiObjectTyper{f.tf.Typer, api.Scheme}
}

func (f *fakeMixedFactory) ClientForMapping(m *meta.RESTMapping) (resource.RESTClient, error) {
	if m.ObjectConvertor == api.Scheme {
		return f.apiClient, f.tf.Err
	}
	return f.tf.Client, f.tf.Err
}

func NewMixedFactory(apiClient resource.RESTClient) (cmdutil.Factory, *testFactory, runtime.Codec) {
	f, t, c, _ := NewTestFactory()
	return &fakeMixedFactory{
		Factory:   f,
		tf:        t,
		apiClient: apiClient,
	}, t, c
}

type fakeAPIFactory struct {
	cmdutil.Factory
	tf *testFactory
}

func (f *fakeAPIFactory) Object() (meta.RESTMapper, runtime.ObjectTyper) {
	return testapi.Default.RESTMapper(), api.Scheme
}

func (f *fakeAPIFactory) UnstructuredObject() (meta.RESTMapper, runtime.ObjectTyper, error) {
	groupResources := testDynamicResources()
	mapper := discovery.NewRESTMapper(groupResources, meta.InterfacesForUnstructured)
	typer := discovery.NewUnstructuredObjectTyper(groupResources)

	return cmdutil.NewShortcutExpander(mapper, nil), typer, nil
}

func (f *fakeAPIFactory) Decoder(bool) runtime.Decoder {
	return testapi.Default.Codec()
}

func (f *fakeAPIFactory) JSONEncoder() runtime.Encoder {
	return testapi.Default.Codec()
}

func (f *fakeAPIFactory) ClientSet() (*internalclientset.Clientset, error) {
	// Swap out the HTTP client out of the client with the fake's version.
	fakeClient := f.tf.Client.(*fake.RESTClient)
	restClient, err := restclient.RESTClientFor(f.tf.ClientConfig)
	if err != nil {
		panic(err)
	}
	restClient.Client = fakeClient.Client
	return internalclientset.New(restClient), f.tf.Err
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

func (f *fakeAPIFactory) ClientConfig() (*restclient.Config, error) {
	return f.tf.ClientConfig, f.tf.Err
}

func (f *fakeAPIFactory) ClientForMapping(*meta.RESTMapping) (resource.RESTClient, error) {
	return f.tf.Client, f.tf.Err
}

func (f *fakeAPIFactory) UnstructuredClientForMapping(*meta.RESTMapping) (resource.RESTClient, error) {
	return f.tf.Client, f.tf.Err
}

func (f *fakeAPIFactory) Describer(*meta.RESTMapping) (kubectl.Describer, error) {
	return f.tf.Describer, f.tf.Err
}

func (f *fakeAPIFactory) Printer(mapping *meta.RESTMapping, options kubectl.PrintOptions) (kubectl.ResourcePrinter, error) {
	return f.tf.Printer, f.tf.Err
}

func (f *fakeAPIFactory) LogsForObject(object, options runtime.Object) (*restclient.Request, error) {
	fakeClient := f.tf.Client.(*fake.RESTClient)
	c := client.NewOrDie(f.tf.ClientConfig)
	c.Client = fakeClient.Client

	switch t := object.(type) {
	case *api.Pod:
		opts, ok := options.(*api.PodLogOptions)
		if !ok {
			return nil, errors.New("provided options object is not a PodLogOptions")
		}
		return c.Pods(f.tf.Namespace).GetLogs(t.Name, opts), nil
	default:
		fqKinds, _, err := api.Scheme.ObjectKinds(object)
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("cannot get the logs from %v", fqKinds[0])
	}
}

func (f *fakeAPIFactory) Validator(validate bool, cacheDir string) (validation.Schema, error) {
	return f.tf.Validator, f.tf.Err
}

func (f *fakeAPIFactory) DefaultNamespace() (string, bool, error) {
	return f.tf.Namespace, false, f.tf.Err
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

func (f *fakeAPIFactory) PrinterForMapping(cmd *cobra.Command, mapping *meta.RESTMapping, withNamespace bool) (kubectl.ResourcePrinter, error) {
	return f.tf.Printer, f.tf.Err
}

func (f *fakeAPIFactory) NewBuilder() *resource.Builder {
	mapper, typer := f.Object()

	return resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true))
}

func NewAPIFactory() (cmdutil.Factory, *testFactory, runtime.Codec, runtime.NegotiatedSerializer) {
	t := &testFactory{
		Validator: validation.NullSchema{},
	}
	rf := cmdutil.NewFactory(nil)
	return &fakeAPIFactory{
		Factory: rf,
		tf:      t,
	}, t, testapi.Default.Codec(), testapi.Default.NegotiatedSerializer()
}

func objBody(codec runtime.Codec, obj runtime.Object) io.ReadCloser {
	return ioutil.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(codec, obj))))
}

func stringBody(body string) io.ReadCloser {
	return ioutil.NopCloser(bytes.NewReader([]byte(body)))
}

// TODO(jlowdermilk): refactor the Factory so we can test client versions properly,
// with different client/server version skew scenarios.
// Verify that resource.RESTClients constructed from a factory respect mapping.APIVersion
//func TestClientVersions(t *testing.T) {
//	f := cmdutil.NewFactory(nil)
//
//	version := testapi.Default.Version()
//	mapping := &meta.RESTMapping{
//		APIVersion: version,
//	}
//	c, err := f.ClientForMapping(mapping)
//	if err != nil {
//		t.Errorf("unexpected error: %v", err)
//	}
//	client := c.(*client.RESTClient)
//	if client.APIVersion() != version {
//		t.Errorf("unexpected Client APIVersion: %s %v", client.APIVersion, client)
//	}
//}

func Example_printReplicationControllerWithNamespace() {
	f, tf, _, ns := NewAPIFactory()
	tf.Printer = kubectl.NewHumanReadablePrinter(kubectl.PrintOptions{
		WithNamespace: true,
		ColumnLabels:  []string{},
	})
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client:               nil,
	}
	cmd := NewCmdRun(f, os.Stdin, os.Stdout, os.Stderr)
	ctrl := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name:              "foo",
			Namespace:         "beep",
			Labels:            map[string]string{"foo": "bar"},
			CreationTimestamp: unversioned.Time{Time: time.Now().AddDate(-10, 0, 0)},
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: 1,
			Selector: map[string]string{"foo": "bar"},
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{"foo": "bar"},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  "foo",
							Image: "someimage",
						},
					},
				},
			},
		},
		Status: api.ReplicationControllerStatus{
			Replicas:      1,
			ReadyReplicas: 1,
		},
	}
	mapper, _ := f.Object()
	err := f.PrintObject(cmd, mapper, ctrl, os.Stdout)
	if err != nil {
		fmt.Printf("Unexpected error: %v", err)
	}
	// Output:
	// NAMESPACE   NAME      DESIRED   CURRENT   READY     AGE
	// beep        foo       1         1         1         10y
}

func Example_printMultiContainersReplicationControllerWithWide() {
	f, tf, _, ns := NewAPIFactory()
	tf.Printer = kubectl.NewHumanReadablePrinter(kubectl.PrintOptions{
		Wide:         true,
		ColumnLabels: []string{},
	})
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client:               nil,
	}
	cmd := NewCmdRun(f, os.Stdin, os.Stdout, os.Stderr)
	ctrl := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name:              "foo",
			Labels:            map[string]string{"foo": "bar"},
			CreationTimestamp: unversioned.Time{Time: time.Now().AddDate(-10, 0, 0)},
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: 1,
			Selector: map[string]string{"foo": "bar"},
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{"foo": "bar"},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  "foo",
							Image: "someimage",
						},
						{
							Name:  "foo2",
							Image: "someimage2",
						},
					},
				},
			},
		},
		Status: api.ReplicationControllerStatus{
			Replicas: 1,
		},
	}
	mapper, _ := f.Object()
	err := f.PrintObject(cmd, mapper, ctrl, os.Stdout)
	if err != nil {
		fmt.Printf("Unexpected error: %v", err)
	}
	// Output:
	// NAME      DESIRED   CURRENT   READY     AGE       CONTAINER(S)   IMAGE(S)               SELECTOR
	// foo       1         1         0         10y       foo,foo2       someimage,someimage2   foo=bar
}

func Example_printReplicationController() {
	f, tf, _, ns := NewAPIFactory()
	tf.Printer = kubectl.NewHumanReadablePrinter(kubectl.PrintOptions{
		ColumnLabels: []string{},
	})
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client:               nil,
	}
	cmd := NewCmdRun(f, os.Stdin, os.Stdout, os.Stderr)
	ctrl := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name:              "foo",
			Labels:            map[string]string{"foo": "bar"},
			CreationTimestamp: unversioned.Time{Time: time.Now().AddDate(-10, 0, 0)},
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: 1,
			Selector: map[string]string{"foo": "bar"},
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{"foo": "bar"},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  "foo",
							Image: "someimage",
						},
						{
							Name:  "foo2",
							Image: "someimage",
						},
					},
				},
			},
		},
		Status: api.ReplicationControllerStatus{
			Replicas: 1,
		},
	}
	mapper, _ := f.Object()
	err := f.PrintObject(cmd, mapper, ctrl, os.Stdout)
	if err != nil {
		fmt.Printf("Unexpected error: %v", err)
	}
	// Output:
	// NAME      DESIRED   CURRENT   READY     AGE
	// foo       1         1         0         10y
}

func Example_printPodWithWideFormat() {
	f, tf, _, ns := NewAPIFactory()
	tf.Printer = kubectl.NewHumanReadablePrinter(kubectl.PrintOptions{
		Wide:         true,
		ColumnLabels: []string{},
	})
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client:               nil,
	}
	nodeName := "kubernetes-node-abcd"
	cmd := NewCmdRun(f, os.Stdin, os.Stdout, os.Stderr)
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:              "test1",
			CreationTimestamp: unversioned.Time{Time: time.Now().AddDate(-10, 0, 0)},
		},
		Spec: api.PodSpec{
			Containers: make([]api.Container, 2),
			NodeName:   nodeName,
		},
		Status: api.PodStatus{
			Phase: "podPhase",
			ContainerStatuses: []api.ContainerStatus{
				{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
				{RestartCount: 3},
			},
			PodIP: "10.1.1.3",
		},
	}
	mapper, _ := f.Object()
	err := f.PrintObject(cmd, mapper, pod, os.Stdout)
	if err != nil {
		fmt.Printf("Unexpected error: %v", err)
	}
	// Output:
	// NAME      READY     STATUS     RESTARTS   AGE       IP         NODE
	// test1     1/2       podPhase   6          10y       10.1.1.3   kubernetes-node-abcd
}

func Example_printPodWithShowLabels() {
	f, tf, _, ns := NewAPIFactory()
	tf.Printer = kubectl.NewHumanReadablePrinter(kubectl.PrintOptions{
		ShowLabels:   true,
		ColumnLabels: []string{},
	})
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client:               nil,
	}
	nodeName := "kubernetes-node-abcd"
	cmd := NewCmdRun(f, os.Stdin, os.Stdout, os.Stderr)
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:              "test1",
			CreationTimestamp: unversioned.Time{Time: time.Now().AddDate(-10, 0, 0)},
			Labels: map[string]string{
				"l1": "key",
				"l2": "value",
			},
		},
		Spec: api.PodSpec{
			Containers: make([]api.Container, 2),
			NodeName:   nodeName,
		},
		Status: api.PodStatus{
			Phase: "podPhase",
			ContainerStatuses: []api.ContainerStatus{
				{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
				{RestartCount: 3},
			},
		},
	}
	mapper, _ := f.Object()
	err := f.PrintObject(cmd, mapper, pod, os.Stdout)
	if err != nil {
		fmt.Printf("Unexpected error: %v", err)
	}
	// Output:
	// NAME      READY     STATUS     RESTARTS   AGE       LABELS
	// test1     1/2       podPhase   6          10y       l1=key,l2=value
}

func newAllPhasePodList() *api.PodList {
	nodeName := "kubernetes-node-abcd"
	return &api.PodList{
		Items: []api.Pod{
			{
				ObjectMeta: api.ObjectMeta{
					Name:              "test1",
					CreationTimestamp: unversioned.Time{Time: time.Now().AddDate(-10, 0, 0)},
				},
				Spec: api.PodSpec{
					Containers: make([]api.Container, 2),
					NodeName:   nodeName,
				},
				Status: api.PodStatus{
					Phase: api.PodPending,
					ContainerStatuses: []api.ContainerStatus{
						{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
						{RestartCount: 3},
					},
				},
			},
			{
				ObjectMeta: api.ObjectMeta{
					Name:              "test2",
					CreationTimestamp: unversioned.Time{Time: time.Now().AddDate(-10, 0, 0)},
				},
				Spec: api.PodSpec{
					Containers: make([]api.Container, 2),
					NodeName:   nodeName,
				},
				Status: api.PodStatus{
					Phase: api.PodRunning,
					ContainerStatuses: []api.ContainerStatus{
						{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
						{RestartCount: 3},
					},
				},
			},
			{
				ObjectMeta: api.ObjectMeta{
					Name:              "test3",
					CreationTimestamp: unversioned.Time{Time: time.Now().AddDate(-10, 0, 0)},
				},
				Spec: api.PodSpec{
					Containers: make([]api.Container, 2),
					NodeName:   nodeName,
				},
				Status: api.PodStatus{
					Phase: api.PodSucceeded,
					ContainerStatuses: []api.ContainerStatus{
						{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
						{RestartCount: 3},
					},
				},
			},
			{
				ObjectMeta: api.ObjectMeta{
					Name:              "test4",
					CreationTimestamp: unversioned.Time{Time: time.Now().AddDate(-10, 0, 0)},
				},
				Spec: api.PodSpec{
					Containers: make([]api.Container, 2),
					NodeName:   nodeName,
				},
				Status: api.PodStatus{
					Phase: api.PodFailed,
					ContainerStatuses: []api.ContainerStatus{
						{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
						{RestartCount: 3},
					},
				},
			},
			{
				ObjectMeta: api.ObjectMeta{
					Name:              "test5",
					CreationTimestamp: unversioned.Time{Time: time.Now().AddDate(-10, 0, 0)},
				},
				Spec: api.PodSpec{
					Containers: make([]api.Container, 2),
					NodeName:   nodeName,
				},
				Status: api.PodStatus{
					Phase: api.PodUnknown,
					ContainerStatuses: []api.ContainerStatus{
						{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
						{RestartCount: 3},
					},
				},
			}},
	}
}

func Example_printPodHideTerminated() {
	f, tf, _, ns := NewAPIFactory()
	tf.Printer = kubectl.NewHumanReadablePrinter(kubectl.PrintOptions{
		ColumnLabels: []string{},
	})
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client:               nil,
	}
	cmd := NewCmdRun(f, os.Stdin, os.Stdout, os.Stderr)
	podList := newAllPhasePodList()
	// filter pods
	filterFuncs := f.DefaultResourceFilterFunc()
	filterOpts := f.DefaultResourceFilterOptions(cmd, false)
	_, filteredPodList, errs := cmdutil.FilterResourceList(podList, filterFuncs, filterOpts)
	if errs != nil {
		fmt.Printf("Unexpected filter error: %v\n", errs)
	}
	for _, pod := range filteredPodList {
		mapper, _ := f.Object()
		err := f.PrintObject(cmd, mapper, pod, os.Stdout)
		if err != nil {
			fmt.Printf("Unexpected error: %v", err)
		}
	}
	// Output:
	// NAME      READY     STATUS    RESTARTS   AGE
	// test1     1/2       Pending   6          10y
	// test2     1/2       Running   6         10y
	// test5     1/2       Unknown   6         10y
}

func Example_printPodShowAll() {
	f, tf, _, ns := NewAPIFactory()
	tf.Printer = kubectl.NewHumanReadablePrinter(kubectl.PrintOptions{
		ShowAll:      true,
		ColumnLabels: []string{},
	})
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client:               nil,
	}
	cmd := NewCmdRun(f, os.Stdin, os.Stdout, os.Stderr)
	podList := newAllPhasePodList()
	mapper, _ := f.Object()
	err := f.PrintObject(cmd, mapper, podList, os.Stdout)
	if err != nil {
		fmt.Printf("Unexpected error: %v", err)
	}
	// Output:
	// NAME      READY     STATUS      RESTARTS   AGE
	// test1     1/2       Pending     6          10y
	// test2     1/2       Running     6          10y
	// test3     1/2       Succeeded   6          10y
	// test4     1/2       Failed      6          10y
	// test5     1/2       Unknown     6          10y
}

func Example_printServiceWithNamespacesAndLabels() {
	f, tf, _, ns := NewAPIFactory()
	tf.Printer = kubectl.NewHumanReadablePrinter(kubectl.PrintOptions{
		WithNamespace: true,
		ColumnLabels:  []string{"l1"},
	})
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client:               nil,
	}
	cmd := NewCmdRun(f, os.Stdin, os.Stdout, os.Stderr)
	svc := &api.ServiceList{
		Items: []api.Service{
			{
				ObjectMeta: api.ObjectMeta{
					Name:              "svc1",
					Namespace:         "ns1",
					CreationTimestamp: unversioned.Time{Time: time.Now().AddDate(-10, 0, 0)},
					Labels: map[string]string{
						"l1": "value",
					},
				},
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{Protocol: "UDP", Port: 53},
						{Protocol: "TCP", Port: 53},
					},
					Selector: map[string]string{
						"s": "magic",
					},
					ClusterIP: "10.1.1.1",
				},
				Status: api.ServiceStatus{},
			},
			{
				ObjectMeta: api.ObjectMeta{
					Name:              "svc2",
					Namespace:         "ns2",
					CreationTimestamp: unversioned.Time{Time: time.Now().AddDate(-10, 0, 0)},
					Labels: map[string]string{
						"l1": "dolla-bill-yall",
					},
				},
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{Protocol: "TCP", Port: 80},
						{Protocol: "TCP", Port: 8080},
					},
					Selector: map[string]string{
						"s": "kazam",
					},
					ClusterIP: "10.1.1.2",
				},
				Status: api.ServiceStatus{},
			}},
	}
	ld := strings.NewLineDelimiter(os.Stdout, "|")
	defer ld.Flush()

	mapper, _ := f.Object()
	err := f.PrintObject(cmd, mapper, svc, ld)
	if err != nil {
		fmt.Printf("Unexpected error: %v", err)
	}
	// Output:
	// |NAMESPACE   NAME      CLUSTER-IP   EXTERNAL-IP   PORT(S)           AGE       L1|
	// |ns1         svc1      10.1.1.1     <unknown>     53/UDP,53/TCP     10y       value|
	// |ns2         svc2      10.1.1.2     <unknown>     80/TCP,8080/TCP   10y       dolla-bill-yall|
	// ||
}

func TestNormalizationFuncGlobalExistence(t *testing.T) {
	// This test can be safely deleted when we will not support multiple flag formats
	root := NewKubectlCommand(cmdutil.NewFactory(nil), os.Stdin, os.Stdout, os.Stderr)

	if root.Parent() != nil {
		t.Fatal("We expect the root command to be returned")
	}
	if root.GlobalNormalizationFunc() == nil {
		t.Fatal("We expect that root command has a global normalization function")
	}

	if reflect.ValueOf(root.GlobalNormalizationFunc()).Pointer() != reflect.ValueOf(root.Flags().GetNormalizeFunc()).Pointer() {
		t.Fatal("root command seems to have a wrong normalization function")
	}

	sub := root
	for sub.HasSubCommands() {
		sub = sub.Commands()[0]
	}

	// In case of failure of this test check this PR: spf13/cobra#110
	if reflect.ValueOf(sub.Flags().GetNormalizeFunc()).Pointer() != reflect.ValueOf(root.Flags().GetNormalizeFunc()).Pointer() {
		t.Fatal("child and root commands should have the same normalization functions")
	}
}
