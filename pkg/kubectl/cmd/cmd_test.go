/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	cmdutil "github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

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

func (*internalType) IsAnAPIObject()  {}
func (*externalType) IsAnAPIObject()  {}
func (*ExternalType2) IsAnAPIObject() {}

func newExternalScheme() (*runtime.Scheme, meta.RESTMapper, runtime.Codec) {
	scheme := runtime.NewScheme()
	scheme.AddKnownTypeWithName("", "Type", &internalType{})
	scheme.AddKnownTypeWithName("unlikelyversion", "Type", &externalType{})
	//This tests that kubectl will not confuse the external scheme with the internal scheme, even when they accidentally have versions of the same name.
	scheme.AddKnownTypeWithName(testapi.Version(), "Type", &ExternalType2{})

	codec := runtime.CodecFor(scheme, "unlikelyversion")
	validVersion := testapi.Version()
	mapper := meta.NewDefaultRESTMapper([]string{"unlikelyversion", validVersion}, func(version string) (*meta.VersionInterfaces, bool) {
		return &meta.VersionInterfaces{
			Codec:            runtime.CodecFor(scheme, version),
			ObjectConvertor:  scheme,
			MetadataAccessor: meta.NewAccessor(),
		}, (version == validVersion || version == "unlikelyversion")
	})
	for _, version := range []string{"unlikelyversion", validVersion} {
		for kind := range scheme.KnownTypes(version) {
			mixedCase := false
			scope := meta.RESTScopeNamespace
			mapper.Add(scope, kind, version, mixedCase)
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

type testDescriber struct {
	Name, Namespace string
	Output          string
	Err             error
}

func (t *testDescriber) Describe(namespace, name string) (output string, err error) {
	t.Namespace, t.Name = namespace, name
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
	ClientConfig *client.Config
	Err          error
}

func NewTestFactory() (*cmdutil.Factory, *testFactory, runtime.Codec) {
	scheme, mapper, codec := newExternalScheme()
	t := &testFactory{
		Validator: validation.NullSchema{},
		Mapper:    mapper,
		Typer:     scheme,
	}
	return &cmdutil.Factory{
		Object: func() (meta.RESTMapper, runtime.ObjectTyper) {
			return t.Mapper, t.Typer
		},
		RESTClient: func(*meta.RESTMapping) (resource.RESTClient, error) {
			return t.Client, t.Err
		},
		Describer: func(*meta.RESTMapping) (kubectl.Describer, error) {
			return t.Describer, t.Err
		},
		Printer: func(mapping *meta.RESTMapping, noHeaders, withNamespace bool, columnLabels []string) (kubectl.ResourcePrinter, error) {
			return t.Printer, t.Err
		},
		Validator: func() (validation.Schema, error) {
			return t.Validator, t.Err
		},
		DefaultNamespace: func() (string, error) {
			return t.Namespace, t.Err
		},
		ClientConfig: func() (*client.Config, error) {
			return t.ClientConfig, t.Err
		},
	}, t, codec
}

func NewMixedFactory(apiClient resource.RESTClient) (*cmdutil.Factory, *testFactory, runtime.Codec) {
	f, t, c := NewTestFactory()
	f.Object = func() (meta.RESTMapper, runtime.ObjectTyper) {
		return meta.MultiRESTMapper{t.Mapper, latest.RESTMapper}, runtime.MultiObjectTyper{t.Typer, api.Scheme}
	}
	f.RESTClient = func(m *meta.RESTMapping) (resource.RESTClient, error) {
		if m.ObjectConvertor == api.Scheme {
			return apiClient, t.Err
		}
		return t.Client, t.Err
	}
	return f, t, c
}

func NewAPIFactory() (*cmdutil.Factory, *testFactory, runtime.Codec) {
	t := &testFactory{
		Validator: validation.NullSchema{},
	}
	generators := map[string]kubectl.Generator{
		"run/v1":     kubectl.BasicReplicationController{},
		"service/v1": kubectl.ServiceGenerator{},
	}
	return &cmdutil.Factory{
		Object: func() (meta.RESTMapper, runtime.ObjectTyper) {
			return latest.RESTMapper, api.Scheme
		},
		Client: func() (*client.Client, error) {
			// Swap out the HTTP client out of the client with the fake's version.
			fakeClient := t.Client.(*client.FakeRESTClient)
			c := client.NewOrDie(t.ClientConfig)
			c.Client = fakeClient.Client
			return c, t.Err
		},
		RESTClient: func(*meta.RESTMapping) (resource.RESTClient, error) {
			return t.Client, t.Err
		},
		Describer: func(*meta.RESTMapping) (kubectl.Describer, error) {
			return t.Describer, t.Err
		},
		Printer: func(mapping *meta.RESTMapping, noHeaders, withNamespace bool, columnLabels []string) (kubectl.ResourcePrinter, error) {
			return t.Printer, t.Err
		},
		Validator: func() (validation.Schema, error) {
			return t.Validator, t.Err
		},
		DefaultNamespace: func() (string, error) {
			return t.Namespace, t.Err
		},
		ClientConfig: func() (*client.Config, error) {
			return t.ClientConfig, t.Err
		},
		Generator: func(name string) (kubectl.Generator, bool) {
			generator, ok := generators[name]
			return generator, ok
		},
	}, t, testapi.Codec()
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
//	version := testapi.Version()
//	mapping := &meta.RESTMapping{
//		APIVersion: version,
//	}
//	c, err := f.RESTClient(mapping)
//	if err != nil {
//		t.Errorf("unexpected error: %v", err)
//	}
//	client := c.(*client.RESTClient)
//	if client.APIVersion() != version {
//		t.Errorf("unexpected Client APIVersion: %s %v", client.APIVersion, client)
//	}
//}

func ExamplePrintReplicationController() {
	f, tf, codec := NewAPIFactory()
	tf.Printer = kubectl.NewHumanReadablePrinter(false, false, []string{})
	tf.Client = &client.FakeRESTClient{
		Codec:  codec,
		Client: nil,
	}
	cmd := NewCmdRun(f, os.Stdout)
	ctrl := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name:   "foo",
			Labels: map[string]string{"foo": "bar"},
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
	}
	err := f.PrintObject(cmd, ctrl, os.Stdout)
	if err != nil {
		fmt.Printf("Unexpected error: %v", err)
	}
	// Output:
	// CONTROLLER   CONTAINER(S)   IMAGE(S)    SELECTOR   REPLICAS
	// foo          foo            someimage   foo=bar    1
}

func TestNormalizationFuncGlobalExistance(t *testing.T) {
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
