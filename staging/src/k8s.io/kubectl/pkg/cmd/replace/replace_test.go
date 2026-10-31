/*
Copyright 2015 The Kubernetes Authors.

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

package replace

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/openapi/openapitest"
	"k8s.io/client-go/rest/fake"
	cmddelete "k8s.io/kubectl/pkg/cmd/delete"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util/openapi"
	"k8s.io/kubectl/pkg/validation"
)

func TestReplaceObject(t *testing.T) {
	_, _, rc := cmdtesting.TestData()

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	deleted := false
	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/api/v1/namespaces/test" && m == http.MethodGet:
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &corev1.Namespace{})}, nil
			case p == "/namespaces/test/replicationcontrollers/redis-master" && m == http.MethodDelete:
				deleted = true
				fallthrough
			case p == "/namespaces/test/replicationcontrollers/redis-master" && m == http.MethodPut:
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &rc.Items[0])}, nil
			case p == "/namespaces/test/replicationcontrollers/redis-master" && m == http.MethodGet:
				statusCode := http.StatusOK
				if deleted {
					statusCode = http.StatusNotFound
				}
				return &http.Response{StatusCode: statusCode, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &rc.Items[0])}, nil
			case p == "/namespaces/test/replicationcontrollers" && m == http.MethodPost:
				return &http.Response{StatusCode: http.StatusCreated, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &rc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	streams, _, buf, _ := genericiooptions.NewTestIOStreams()

	cmd := NewCmdReplace(tf, streams)
	cmd.Flags().Set("filename", "../../../testdata/redis-master-controller.yaml")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	// uses the name from the file, not the response
	if buf.String() != "replicationcontroller/rc1\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}

	buf.Reset()
	cmd.Flags().Set("force", "true")
	cmd.Flags().Set("cascade", "false")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	if buf.String() != "replicationcontroller/redis-master\nreplicationcontroller/rc1\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestReplaceMultipleObject(t *testing.T) {
	_, svc, rc := cmdtesting.TestData()

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	redisMasterDeleted := false
	frontendDeleted := false
	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/api/v1/namespaces/test" && m == http.MethodGet:
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &corev1.Namespace{})}, nil
			case p == "/namespaces/test/replicationcontrollers/redis-master" && m == http.MethodDelete:
				redisMasterDeleted = true
				fallthrough
			case p == "/namespaces/test/replicationcontrollers/redis-master" && m == http.MethodPut:
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &rc.Items[0])}, nil
			case p == "/namespaces/test/replicationcontrollers/redis-master" && m == http.MethodGet:
				statusCode := http.StatusOK
				if redisMasterDeleted {
					statusCode = http.StatusNotFound
				}
				return &http.Response{StatusCode: statusCode, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &rc.Items[0])}, nil
			case p == "/namespaces/test/replicationcontrollers" && m == http.MethodPost:
				return &http.Response{StatusCode: http.StatusCreated, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &rc.Items[0])}, nil
			case p == "/namespaces/test/services/frontend" && m == http.MethodDelete:
				frontendDeleted = true
				fallthrough
			case p == "/namespaces/test/services/frontend" && m == http.MethodPut:
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &svc.Items[0])}, nil
			case p == "/namespaces/test/services/frontend" && m == http.MethodGet:
				statusCode := http.StatusOK
				if frontendDeleted {
					statusCode = http.StatusNotFound
				}
				return &http.Response{StatusCode: statusCode, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &svc.Items[0])}, nil
			case p == "/namespaces/test/services" && m == http.MethodPost:
				return &http.Response{StatusCode: http.StatusCreated, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &svc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	streams, _, buf, _ := genericiooptions.NewTestIOStreams()

	cmd := NewCmdReplace(tf, streams)
	cmd.Flags().Set("filename", "../../../testdata/redis-master-controller.yaml")
	cmd.Flags().Set("filename", "../../../testdata/frontend-service.yaml")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	if buf.String() != "replicationcontroller/rc1\nservice/baz\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}

	buf.Reset()
	cmd.Flags().Set("force", "true")
	cmd.Flags().Set("cascade", "false")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	if buf.String() != "replicationcontroller/redis-master\nservice/frontend\nreplicationcontroller/rc1\nservice/baz\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestReplaceDirectory(t *testing.T) {
	_, _, rc := cmdtesting.TestData()

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	created := map[string]bool{}
	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/api/v1/namespaces/test" && m == http.MethodGet:
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &corev1.Namespace{})}, nil
			case strings.HasPrefix(p, "/namespaces/test/replicationcontrollers/") && m == http.MethodPut:
				created[p] = true
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &rc.Items[0])}, nil
			case strings.HasPrefix(p, "/namespaces/test/replicationcontrollers/") && m == http.MethodGet:
				statusCode := http.StatusNotFound
				if created[p] {
					statusCode = http.StatusOK
				}
				return &http.Response{StatusCode: statusCode, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &rc.Items[0])}, nil
			case strings.HasPrefix(p, "/namespaces/test/replicationcontrollers/") && m == http.MethodDelete:
				delete(created, p)
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &rc.Items[0])}, nil
			case strings.HasPrefix(p, "/namespaces/test/replicationcontrollers") && m == http.MethodPost:
				return &http.Response{StatusCode: http.StatusCreated, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &rc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	streams, _, buf, _ := genericiooptions.NewTestIOStreams()

	cmd := NewCmdReplace(tf, streams)
	cmd.Flags().Set("filename", "../../../testdata/replace/legacy")
	cmd.Flags().Set("namespace", "test")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	if buf.String() != "replicationcontroller/rc1\nreplicationcontroller/rc1\nreplicationcontroller/rc1\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}

	buf.Reset()
	cmd.Flags().Set("force", "true")
	cmd.Flags().Set("cascade", "false")
	cmd.Run(cmd, []string{})

	if buf.String() != "replicationcontroller/frontend\nreplicationcontroller/redis-master\nreplicationcontroller/redis-slave\n"+
		"replicationcontroller/rc1\nreplicationcontroller/rc1\nreplicationcontroller/rc1\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestForceReplaceObjectNotFound(t *testing.T) {
	_, _, rc := cmdtesting.TestData()

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/api/v1/namespaces/test" && m == http.MethodGet:
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &corev1.Namespace{})}, nil
			case p == "/namespaces/test/replicationcontrollers/redis-master" && (m == http.MethodGet || m == http.MethodDelete):
				return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.StringBody("")}, nil
			case p == "/namespaces/test/replicationcontrollers" && m == http.MethodPost:
				return &http.Response{StatusCode: http.StatusCreated, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &rc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	streams, _, buf, _ := genericiooptions.NewTestIOStreams()

	cmd := NewCmdReplace(tf, streams)
	cmd.Flags().Set("filename", "../../../testdata/redis-master-controller.yaml")
	cmd.Flags().Set("force", "true")
	cmd.Flags().Set("cascade", "false")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	if buf.String() != "replicationcontroller/rc1\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestForceReplaceTypedList(t *testing.T) {
	filename := filepath.Join(t.TempDir(), "services.json")
	require.NoError(t, os.WriteFile(filename, []byte(`{
		"apiVersion":"v1",
		"kind":"ServiceList",
		"items":[{"metadata":{"name":"frontend"},"spec":{"ports":[{"port":80}]}}]
	}`), 0600))

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	tf.OpenAPISchemaFunc = func() (openapi.Resources, error) {
		t.Fatal("unexpected fallback to OpenAPI v2")
		return nil, nil
	}
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "frontend", Namespace: "test"},
		Spec:       corev1.ServiceSpec{Ports: []corev1.ServicePort{{Port: 80}}},
	}
	var requests []string
	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/api/v1/namespaces/test" && m == http.MethodGet:
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &corev1.Namespace{})}, nil
			case p == "/namespaces/test/services/frontend" && m == http.MethodDelete:
				requests = append(requests, m)
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, service)}, nil
			case p == "/namespaces/test/services/frontend" && m == http.MethodGet:
				requests = append(requests, m)
				notFound := apierrors.NewNotFound(schema.GroupResource{Resource: "services"}, "frontend")
				return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &notFound.ErrStatus)}, nil
			case p == "/namespaces/test/services" && m == http.MethodPost:
				requests = append(requests, m)
				var created corev1.Service
				require.NoError(t, json.NewDecoder(req.Body).Decode(&created))
				require.Equal(t, "Service", created.Kind)
				require.Equal(t, "v1", created.APIVersion)
				require.Equal(t, service.Name, created.Name)
				require.Equal(t, service.Namespace, created.Namespace)
				require.Equal(t, service.Spec.Ports, created.Spec.Ports)
				require.Equal(t, metav1.FieldValidationStrict, req.URL.Query().Get("fieldValidation"))
				return &http.Response{StatusCode: http.StatusCreated, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &created)}, nil
			default:
				t.Fatalf("unexpected request: %s %s", req.Method, req.URL)
				return nil, nil
			}
		}),
	}

	streams, _, out, _ := genericiooptions.NewTestIOStreams()
	o := NewReplaceOptions(streams)
	o.Builder = tf.NewBuilder
	o.Namespace = "test"
	o.Recorder = genericclioptions.NoopRecorder{}
	o.validationDirective = metav1.FieldValidationStrict
	o.Schema = validation.NewParamVerifyingSchema(
		validation.ConjunctiveSchema{validation.NewSchemaValidation(tf), validation.NoDoubleKeySchema{}},
		resource.NewQueryParamVerifierV3(tf.FakeDynamicClient, openapitest.NewEmbeddedFileClient(), resource.QueryParamFieldValidation),
		o.validationDirective,
	)
	o.DeleteOptions = &cmddelete.DeleteOptions{
		FilenameOptions:   resource.FilenameOptions{Filenames: []string{filename}},
		ForceDeletion:     true,
		GracePeriod:       -1,
		CascadingStrategy: metav1.DeletePropagationBackground,
		Output:            "name",
		IOStreams:         streams,
	}
	*o.PrintFlags.OutputFormat = "name"
	printer, err := o.PrintFlags.ToPrinter()
	require.NoError(t, err)
	o.PrintObj = func(obj runtime.Object) error {
		return printer.PrintObj(obj, out)
	}

	require.NoError(t, o.forceReplace())
	require.Equal(t, []string{http.MethodDelete, http.MethodGet, http.MethodPost}, requests)
	require.Equal(t, "service/frontend\nservice/frontend\n", out.String())
}
