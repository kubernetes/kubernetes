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
	"io"
	"io/ioutil"
	"net/http"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/printers"
)

func InitTestErrorHandler(t *testing.T) {
	cmdutil.BehaviorOnFatal(func(str string, code int) {
		t.Errorf("Error running command (exit code %d): %s", code, str)
	})
}

func DefaultHeader() http.Header {
	header := http.Header{}
	header.Set("Content-Type", runtime.ContentTypeJSON)
	return header
}

func DefaultClientConfig() *restclient.Config {
	return &restclient.Config{
		APIPath: "/api",
		ContentConfig: restclient.ContentConfig{
			NegotiatedSerializer: api.Codecs,
			ContentType:          runtime.ContentTypeJSON,
			GroupVersion:         &api.Registry.GroupOrDie(api.GroupName).GroupVersion,
		},
	}
}

func DefaultClientConfigForVersion(version *schema.GroupVersion) *restclient.Config {
	return &restclient.Config{
		APIPath: "/api",
		ContentConfig: restclient.ContentConfig{
			NegotiatedSerializer: api.Codecs,
			ContentType:          runtime.ContentTypeJSON,
			GroupVersion:         version,
		},
	}
}

type TestPrinter struct {
	Objects        []runtime.Object
	Err            error
	GenericPrinter bool
}

func (t *TestPrinter) PrintObj(obj runtime.Object, out io.Writer) error {
	t.Objects = append(t.Objects, obj)
	fmt.Fprintf(out, "%#v", obj)
	return t.Err
}

// TODO: implement HandledResources()
func (t *TestPrinter) HandledResources() []string {
	return []string{}
}

func (t *TestPrinter) AfterPrint(output io.Writer, res string) error {
	return nil
}

func (t *TestPrinter) IsGeneric() bool {
	return t.GenericPrinter
}

type TestDescriber struct {
	Name, Namespace string
	Settings        printers.DescriberSettings
	Output          string
	Err             error
}

func (t *TestDescriber) Describe(namespace, name string, describerSettings printers.DescriberSettings) (output string, err error) {
	t.Namespace, t.Name = namespace, name
	t.Settings = describerSettings
	return t.Output, t.Err
}

func ObjBody(codec runtime.Codec, obj runtime.Object) io.ReadCloser {
	return ioutil.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(codec, obj))))
}

func PolicyObjBody(obj runtime.Object) io.ReadCloser {
	return ioutil.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(testapi.Policy.Codec(), obj))))
}

func BytesBody(bodyBytes []byte) io.ReadCloser {
	return ioutil.NopCloser(bytes.NewReader(bodyBytes))
}

func StringBody(body string) io.ReadCloser {
	return ioutil.NopCloser(bytes.NewReader([]byte(body)))
}
