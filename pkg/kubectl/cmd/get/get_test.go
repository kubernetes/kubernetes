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

package get

import (
	"bytes"
	encjson "encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	batchv1 "k8s.io/api/batch/v1"
	batchv1beta1 "k8s.io/api/batch/v1beta1"
	corev1 "k8s.io/api/core/v1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1beta1 "k8s.io/apimachinery/pkg/apis/meta/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	"k8s.io/apimachinery/pkg/runtime/serializer/streaming"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericclioptions/resource"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	restclientwatch "k8s.io/client-go/rest/watch"
	"k8s.io/kube-openapi/pkg/util/proto"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi"
	openapitesting "k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi/testing"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
)

var (
	openapiSchemaPath = filepath.Join("..", "..", "..", "..", "api", "openapi-spec", "swagger.json")

	grace              = int64(30)
	enableServiceLinks = corev1.DefaultEnableServiceLinks
)

func testComponentStatusData() *corev1.ComponentStatusList {
	good := corev1.ComponentStatus{
		Conditions: []corev1.ComponentCondition{
			{Type: corev1.ComponentHealthy, Status: corev1.ConditionTrue, Message: "ok"},
		},
		ObjectMeta: metav1.ObjectMeta{Name: "servergood"},
	}

	bad := corev1.ComponentStatus{
		Conditions: []corev1.ComponentCondition{
			{Type: corev1.ComponentHealthy, Status: corev1.ConditionFalse, Message: "", Error: "bad status: 500"},
		},
		ObjectMeta: metav1.ObjectMeta{Name: "serverbad"},
	}

	unknown := corev1.ComponentStatus{
		Conditions: []corev1.ComponentCondition{
			{Type: corev1.ComponentHealthy, Status: corev1.ConditionUnknown, Message: "", Error: "fizzbuzz error"},
		},
		ObjectMeta: metav1.ObjectMeta{Name: "serverunknown"},
	}

	return &corev1.ComponentStatusList{
		Items: []corev1.ComponentStatus{good, bad, unknown},
	}
}

// Verifies that schemas that are not in the master tree of Kubernetes can be retrieved via Get.
func TestGetUnknownSchemaObject(t *testing.T) {
	t.Skip("This test is completely broken.  The first thing it does is add the object to the scheme!")
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	_, _, codec := cmdtesting.NewExternalScheme()
	tf.OpenAPISchemaFunc = openapitesting.CreateOpenAPISchemaFunc(openapiSchemaPath)

	obj := &cmdtesting.ExternalType{
		Kind:       "Type",
		APIVersion: "apitest/unlikelyversion",
		Name:       "foo",
	}

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Resp: &http.Response{
			StatusCode: 200, Header: cmdtesting.DefaultHeader(),
			Body: cmdtesting.ObjBody(codec, obj),
		},
	}
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdGet("kubectl", tf, streams)
	cmd.SetOutput(buf)
	cmd.Run(cmd, []string{"type", "foo"})

	expected := []runtime.Object{cmdtesting.NewInternalType("", "", "foo")}
	actual := []runtime.Object{}
	if len(actual) != len(expected) {
		t.Fatalf("expected: %#v, but actual: %#v", expected, actual)
	}
	t.Logf("actual: %#v", actual[0])
	for i, obj := range actual {
		expectedJSON := runtime.EncodeOrDie(codec, expected[i])
		expectedMap := map[string]interface{}{}
		if err := encjson.Unmarshal([]byte(expectedJSON), &expectedMap); err != nil {
			t.Fatal(err)
		}

		actualJSON := runtime.EncodeOrDie(codec, obj)
		actualMap := map[string]interface{}{}
		if err := encjson.Unmarshal([]byte(actualJSON), &actualMap); err != nil {
			t.Fatal(err)
		}

		if !reflect.DeepEqual(expectedMap, actualMap) {
			t.Errorf("expectedMap: %#v, but actualMap: %#v", expectedMap, actualMap)
		}
	}
}

// Verifies that schemas that are not in the master tree of Kubernetes can be retrieved via Get.
func TestGetSchemaObject(t *testing.T) {
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(corev1.SchemeGroupVersion)
	t.Logf("%v", string(runtime.EncodeOrDie(codec, &corev1.ReplicationController{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})))

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Resp:                 &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &corev1.ReplicationController{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})},
	}
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdGet("kubectl", tf, streams)
	cmd.Run(cmd, []string{"replicationcontrollers", "foo"})

	if !strings.Contains(buf.String(), "foo") {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestGetObjectsWithOpenAPIOutputFormatPresent(t *testing.T) {
	pods, _, _ := cmdtesting.TestData()

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	// overide the openAPISchema function to return custom output
	// for Pod type.
	tf.OpenAPISchemaFunc = testOpenAPISchemaData
	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Resp:                 &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &pods.Items[0])},
	}

	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdGet("kubectl", tf, streams)
	cmd.SetOutput(buf)
	cmd.Flags().Set(useOpenAPIPrintColumnFlagLabel, "true")
	cmd.Run(cmd, []string{"pods", "foo"})

	expected := `NAME   RSRC
foo    10
`
	if e, a := expected, buf.String(); e != a {
		t.Errorf("expected\n%v\ngot\n%v", e, a)
	}
}

type FakeResources struct {
	resources map[schema.GroupVersionKind]proto.Schema
}

func (f FakeResources) LookupResource(s schema.GroupVersionKind) proto.Schema {
	return f.resources[s]
}

var _ openapi.Resources = &FakeResources{}

func testOpenAPISchemaData() (openapi.Resources, error) {
	return &FakeResources{
		resources: map[schema.GroupVersionKind]proto.Schema{
			{
				Version: "v1",
				Kind:    "Pod",
			}: &proto.Primitive{
				BaseSchema: proto.BaseSchema{
					Extensions: map[string]interface{}{
						"x-kubernetes-print-columns": "custom-columns=NAME:.metadata.name,RSRC:.metadata.resourceVersion",
					},
				},
			},
		},
	}, nil
}

func TestGetObjects(t *testing.T) {
	pods, _, _ := cmdtesting.TestData()

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Resp:                 &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &pods.Items[0])},
	}

	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdGet("kubectl", tf, streams)
	cmd.SetOutput(buf)
	cmd.Run(cmd, []string{"pods", "foo"})

	expected := `NAME   READY   STATUS   RESTARTS   AGE
foo    0/0              0          <unknown>
`
	if e, a := expected, buf.String(); e != a {
		t.Errorf("expected\n%v\ngot\n%v", e, a)
	}
}

func TestGetObjectsShowKind(t *testing.T) {
	pods, _, _ := cmdtesting.TestData()

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Resp:                 &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &pods.Items[0])},
	}

	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdGet("kubectl", tf, streams)
	cmd.SetOutput(buf)
	cmd.Flags().Set("show-kind", "true")
	cmd.Run(cmd, []string{"pods", "foo"})

	expected := `NAME      READY   STATUS   RESTARTS   AGE
pod/foo   0/0              0          <unknown>
`
	if e, a := expected, buf.String(); e != a {
		t.Errorf("expected\n%v\ngot\n%v", e, a)
	}
}

func TestGetMultipleResourceTypesShowKinds(t *testing.T) {
	pods, svcs, _ := cmdtesting.TestData()

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/pods" && m == "GET":
				return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)}, nil
			case p == "/namespaces/test/replicationcontrollers" && m == "GET":
				return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &corev1.ReplicationControllerList{})}, nil
			case p == "/namespaces/test/services" && m == "GET":
				return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, svcs)}, nil
			case p == "/namespaces/test/statefulsets" && m == "GET":
				return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &appsv1.StatefulSetList{})}, nil
			case p == "/namespaces/test/horizontalpodautoscalers" && m == "GET":
				return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &autoscalingv1.HorizontalPodAutoscalerList{})}, nil
			case p == "/namespaces/test/jobs" && m == "GET":
				return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &batchv1.JobList{})}, nil
			case p == "/namespaces/test/cronjobs" && m == "GET":
				return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &batchv1beta1.CronJobList{})}, nil
			case p == "/namespaces/test/daemonsets" && m == "GET":
				return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &appsv1.DaemonSetList{})}, nil
			case p == "/namespaces/test/deployments" && m == "GET":
				return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &extensionsv1beta1.DeploymentList{})}, nil
			case p == "/namespaces/test/replicasets" && m == "GET":
				return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &extensionsv1beta1.ReplicaSetList{})}, nil

			default:
				t.Fatalf("request url: %#v,and request: %#v", req.URL, req)
				return nil, nil
			}
		}),
	}

	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdGet("kubectl", tf, streams)
	cmd.SetOutput(buf)
	cmd.Run(cmd, []string{"all"})

	expected := `NAME      READY   STATUS   RESTARTS   AGE
pod/foo   0/0              0          <unknown>
pod/bar   0/0              0          <unknown>
NAME          TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)   AGE
service/baz   ClusterIP   <none>       <none>        <none>    <unknown>
`
	if e, a := expected, buf.String(); e != a {
		t.Errorf("expected\n%v\ngot\n%v", e, a)
	}
}

func TestGetObjectsShowLabels(t *testing.T) {
	pods, _, _ := cmdtesting.TestData()

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Resp:                 &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &pods.Items[0])},
	}

	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdGet("kubectl", tf, streams)
	cmd.SetOutput(buf)
	cmd.Flags().Set("show-labels", "true")
	cmd.Run(cmd, []string{"pods", "foo"})

	expected := `NAME   READY   STATUS   RESTARTS   AGE         LABELS
foo    0/0              0          <unknown>   <none>
`
	if e, a := expected, buf.String(); e != a {
		t.Errorf("expected\n%v\ngot\n%v", e, a)
	}
}

func TestGetObjectIgnoreNotFound(t *testing.T) {
	cmdtesting.InitTestErrorHandler(t)

	ns := &corev1.NamespaceList{
		ListMeta: metav1.ListMeta{
			ResourceVersion: "1",
		},
		Items: []corev1.Namespace{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "testns", Namespace: "test", ResourceVersion: "11"},
				Spec:       corev1.NamespaceSpec{},
			},
		},
	}

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/pods/nonexistentpod" && m == "GET":
				return &http.Response{StatusCode: 404, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.StringBody("")}, nil
			case p == "/api/v1/namespaces/test" && m == "GET":
				return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &ns.Items[0])}, nil
			default:
				t.Fatalf("request url: %#v,and request: %#v", req.URL, req)
				return nil, nil
			}
		}),
	}

	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdGet("kubectl", tf, streams)
	cmd.SetOutput(buf)
	cmd.Flags().Set("ignore-not-found", "true")
	cmd.Flags().Set("output", "yaml")
	cmd.Run(cmd, []string{"pods", "nonexistentpod"})

	if buf.String() != "" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestGetSortedObjects(t *testing.T) {
	pods := &corev1.PodList{
		ListMeta: metav1.ListMeta{
			ResourceVersion: "15",
		},
		Items: []corev1.Pod{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "c", Namespace: "test", ResourceVersion: "10"},
				Spec: corev1.PodSpec{
					RestartPolicy:                 corev1.RestartPolicyAlways,
					DNSPolicy:                     corev1.DNSClusterFirst,
					TerminationGracePeriodSeconds: &grace,
					SecurityContext:               &corev1.PodSecurityContext{},
					EnableServiceLinks:            &enableServiceLinks,
				},
			},
			{
				ObjectMeta: metav1.ObjectMeta{Name: "b", Namespace: "test", ResourceVersion: "11"},
				Spec: corev1.PodSpec{
					RestartPolicy:                 corev1.RestartPolicyAlways,
					DNSPolicy:                     corev1.DNSClusterFirst,
					TerminationGracePeriodSeconds: &grace,
					SecurityContext:               &corev1.PodSecurityContext{},
					EnableServiceLinks:            &enableServiceLinks,
				},
			},
			{
				ObjectMeta: metav1.ObjectMeta{Name: "a", Namespace: "test", ResourceVersion: "9"},
				Spec: corev1.PodSpec{
					RestartPolicy:                 corev1.RestartPolicyAlways,
					DNSPolicy:                     corev1.DNSClusterFirst,
					TerminationGracePeriodSeconds: &grace,
					SecurityContext:               &corev1.PodSecurityContext{},
					EnableServiceLinks:            &enableServiceLinks,
				},
			},
		},
	}

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Resp:                 &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)},
	}
	tf.ClientConfigVal = &restclient.Config{ContentConfig: restclient.ContentConfig{GroupVersion: &corev1.SchemeGroupVersion}}

	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdGet("kubectl", tf, streams)
	cmd.SetOutput(buf)

	// sorting with metedata.name
	cmd.Flags().Set("sort-by", ".metadata.name")
	cmd.Run(cmd, []string{"pods"})

	expected := `NAME   READY   STATUS   RESTARTS   AGE
a      0/0              0          <unknown>
b      0/0              0          <unknown>
c      0/0              0          <unknown>
`
	if e, a := expected, buf.String(); e != a {
		t.Errorf("expected\n%v\ngot\n%v", e, a)
	}
}

func sortTestData() []runtime.Object {
	return []runtime.Object{
		&corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "c", Namespace: "test", ResourceVersion: "10"},
			Spec: corev1.PodSpec{
				RestartPolicy:                 corev1.RestartPolicyAlways,
				DNSPolicy:                     corev1.DNSClusterFirst,
				TerminationGracePeriodSeconds: &grace,
				SecurityContext:               &corev1.PodSecurityContext{},
				EnableServiceLinks:            &enableServiceLinks,
			},
		},
		&corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "b", Namespace: "test", ResourceVersion: "11"},
			Spec: corev1.PodSpec{
				RestartPolicy:                 corev1.RestartPolicyAlways,
				DNSPolicy:                     corev1.DNSClusterFirst,
				TerminationGracePeriodSeconds: &grace,
				SecurityContext:               &corev1.PodSecurityContext{},
				EnableServiceLinks:            &enableServiceLinks,
			},
		},
		&corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "a", Namespace: "test", ResourceVersion: "9"},
			Spec: corev1.PodSpec{
				RestartPolicy:                 corev1.RestartPolicyAlways,
				DNSPolicy:                     corev1.DNSClusterFirst,
				TerminationGracePeriodSeconds: &grace,
				SecurityContext:               &corev1.PodSecurityContext{},
				EnableServiceLinks:            &enableServiceLinks,
			},
		},
	}
}

func sortTestTableData() []runtime.Object {
	return []runtime.Object{
		&metav1beta1.Table{
			TypeMeta: metav1.TypeMeta{Kind: "Table"},
			Rows: []metav1beta1.TableRow{
				{
					Object: runtime.RawExtension{
						Object: &corev1.Pod{
							ObjectMeta: metav1.ObjectMeta{Name: "c", Namespace: "test", ResourceVersion: "10"},
							Spec: corev1.PodSpec{
								RestartPolicy:                 corev1.RestartPolicyAlways,
								DNSPolicy:                     corev1.DNSClusterFirst,
								TerminationGracePeriodSeconds: &grace,
								SecurityContext:               &corev1.PodSecurityContext{},
								EnableServiceLinks:            &enableServiceLinks,
							},
						},
					},
				},
				{
					Object: runtime.RawExtension{
						Object: &corev1.Pod{
							ObjectMeta: metav1.ObjectMeta{Name: "b", Namespace: "test", ResourceVersion: "11"},
							Spec: corev1.PodSpec{
								RestartPolicy:                 corev1.RestartPolicyAlways,
								DNSPolicy:                     corev1.DNSClusterFirst,
								TerminationGracePeriodSeconds: &grace,
								SecurityContext:               &corev1.PodSecurityContext{},
								EnableServiceLinks:            &enableServiceLinks,
							},
						},
					},
				},
				{
					Object: runtime.RawExtension{
						Object: &corev1.Pod{
							ObjectMeta: metav1.ObjectMeta{Name: "a", Namespace: "test", ResourceVersion: "9"},
							Spec: corev1.PodSpec{
								RestartPolicy:                 corev1.RestartPolicyAlways,
								DNSPolicy:                     corev1.DNSClusterFirst,
								TerminationGracePeriodSeconds: &grace,
								SecurityContext:               &corev1.PodSecurityContext{},
								EnableServiceLinks:            &enableServiceLinks,
							},
						},
					},
				},
			},
		},
	}
}

func TestRuntimeSorter(t *testing.T) {
	tests := []struct {
		name        string
		field       string
		objs        []runtime.Object
		op          func(sorter *RuntimeSorter, objs []runtime.Object, out io.Writer) error
		expect      string
		expectError string
	}{
		{
			name:  "ensure sorter works with an empty object list",
			field: "metadata.name",
			objs:  []runtime.Object{},
			op: func(sorter *RuntimeSorter, objs []runtime.Object, out io.Writer) error {
				return nil
			},
			expect: "",
		},
		{
			name:  "ensure sorter returns original position",
			field: "metadata.name",
			objs:  sortTestData(),
			op: func(sorter *RuntimeSorter, objs []runtime.Object, out io.Writer) error {
				for idx := range objs {
					p := sorter.OriginalPosition(idx)
					fmt.Fprintf(out, "%v,", p)
				}
				return nil
			},
			expect: "2,1,0,",
		},
		{
			name:  "ensure sorter handles table object position",
			field: "metadata.name",
			objs:  sortTestTableData(),
			op: func(sorter *RuntimeSorter, objs []runtime.Object, out io.Writer) error {
				for idx := range objs {
					p := sorter.OriginalPosition(idx)
					fmt.Fprintf(out, "%v,", p)
				}
				return nil
			},
			expect: "0,",
		},
		{
			name:  "ensure sorter sorts table objects",
			field: "metadata.name",
			objs:  sortTestData(),
			op: func(sorter *RuntimeSorter, objs []runtime.Object, out io.Writer) error {
				for _, o := range objs {
					fmt.Fprintf(out, "%s,", o.(*corev1.Pod).Name)
				}
				return nil
			},
			expect: "a,b,c,",
		},
		{
			name:        "ensure sorter rejects mixed Table + non-Table object lists",
			field:       "metadata.name",
			objs:        append(sortTestData(), sortTestTableData()...),
			op:          func(sorter *RuntimeSorter, objs []runtime.Object, out io.Writer) error { return nil },
			expectError: "sorting is not supported on mixed Table",
		},
		{
			name:        "ensure sorter errors out on invalid jsonpath",
			field:       "metadata.unknown",
			objs:        sortTestData(),
			op:          func(sorter *RuntimeSorter, objs []runtime.Object, out io.Writer) error { return nil },
			expectError: "couldn't find any field with path",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sorter := NewRuntimeSorter(tc.objs, tc.field)
			if err := sorter.Sort(); err != nil {
				if len(tc.expectError) > 0 && strings.Contains(err.Error(), tc.expectError) {
					return
				}

				if len(tc.expectError) > 0 {
					t.Fatalf("unexpected error: expecting %s, but got %s", tc.expectError, err)
				}

				t.Fatalf("unexpected error: %v", err)
			}

			out := bytes.NewBuffer([]byte{})
			err := tc.op(sorter, tc.objs, out)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if tc.expect != out.String() {
				t.Fatalf("unexpected output: expecting %s, but got %s", tc.expect, out.String())
			}

		})
	}

}

func TestGetObjectsIdentifiedByFile(t *testing.T) {
	pods, _, _ := cmdtesting.TestData()

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Resp:                 &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &pods.Items[0])},
	}

	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdGet("kubectl", tf, streams)
	cmd.SetOutput(buf)
	cmd.Flags().Set("filename", "../../../../test/e2e/testing-manifests/statefulset/cassandra/controller.yaml")
	cmd.Run(cmd, []string{})

	expected := `NAME   READY   STATUS   RESTARTS   AGE
foo    0/0              0          <unknown>
`
	if e, a := expected, buf.String(); e != a {
		t.Errorf("expected\n%v\ngot\n%v", e, a)
	}
}

func TestGetListObjects(t *testing.T) {
	pods, _, _ := cmdtesting.TestData()

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Resp:                 &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)},
	}

	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdGet("kubectl", tf, streams)
	cmd.SetOutput(buf)
	cmd.Run(cmd, []string{"pods"})

	expected := `NAME   READY   STATUS   RESTARTS   AGE
foo    0/0              0          <unknown>
bar    0/0              0          <unknown>
`
	if e, a := expected, buf.String(); e != a {
		t.Errorf("expected\n%v\ngot\n%v", e, a)
	}
}

func TestGetListComponentStatus(t *testing.T) {
	statuses := testComponentStatusData()

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Resp:                 &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, statuses)},
	}

	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdGet("kubectl", tf, streams)
	cmd.SetOutput(buf)
	cmd.Run(cmd, []string{"componentstatuses"})

	expected := `NAME            STATUS      MESSAGE   ERROR
servergood      Healthy     ok        
serverbad       Unhealthy             bad status: 500
serverunknown   Unhealthy             fizzbuzz error
`
	if e, a := expected, buf.String(); e != a {
		t.Errorf("expected\n%v\ngot\n%v", e, a)
	}
}

func TestGetMixedGenericObjects(t *testing.T) {
	cmdtesting.InitTestErrorHandler(t)

	// ensure that a runtime.Object without
	// an ObjectMeta field is handled properly
	structuredObj := &metav1.Status{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Status",
			APIVersion: "v1",
		},
		Status:  "Success",
		Message: "",
		Reason:  "",
		Code:    0,
	}

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch req.URL.Path {
			case "/namespaces/test/pods":
				return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, structuredObj)}, nil
			default:
				t.Fatalf("request url: %#v,and request: %#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdGet("kubectl", tf, streams)
	cmd.SetOutput(buf)
	cmd.Flags().Set("output", "json")
	cmd.Run(cmd, []string{"pods"})

	expected := `{
    "apiVersion": "v1",
    "items": [
        {
            "apiVersion": "v1",
            "kind": "Status",
            "metadata": {},
            "status": "Success"
        }
    ],
    "kind": "List",
    "metadata": {
        "resourceVersion": "",
        "selfLink": ""
    }
}
`
	if e, a := expected, buf.String(); e != a {
		t.Errorf("expected\n%v\ngot\n%v", e, a)
	}
}

func TestGetMultipleTypeObjects(t *testing.T) {
	pods, svc, _ := cmdtesting.TestData()

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch req.URL.Path {
			case "/namespaces/test/pods":
				return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)}, nil
			case "/namespaces/test/services":
				return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, svc)}, nil
			default:
				t.Fatalf("request url: %#v,and request: %#v", req.URL, req)
				return nil, nil
			}
		}),
	}

	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdGet("kubectl", tf, streams)
	cmd.SetOutput(buf)
	cmd.Run(cmd, []string{"pods,services"})

	expected := `NAME      READY   STATUS   RESTARTS   AGE
pod/foo   0/0              0          <unknown>
pod/bar   0/0              0          <unknown>
NAME          TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)   AGE
service/baz   ClusterIP   <none>       <none>        <none>    <unknown>
`
	if e, a := expected, buf.String(); e != a {
		t.Errorf("expected\n%v\ngot\n%v", e, a)
	}
}

func TestGetMultipleTypeObjectsAsList(t *testing.T) {
	pods, svc, _ := cmdtesting.TestData()

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch req.URL.Path {
			case "/namespaces/test/pods":
				return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)}, nil
			case "/namespaces/test/services":
				return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, svc)}, nil
			default:
				t.Fatalf("request url: %#v,and request: %#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdGet("kubectl", tf, streams)
	cmd.SetOutput(buf)

	cmd.Flags().Set("output", "json")
	cmd.Run(cmd, []string{"pods,services"})

	expected := `{
    "apiVersion": "v1",
    "items": [
        {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "creationTimestamp": null,
                "name": "foo",
                "namespace": "test",
                "resourceVersion": "10"
            },
            "spec": {
                "containers": null,
                "dnsPolicy": "ClusterFirst",
                "enableServiceLinks": true,
                "restartPolicy": "Always",
                "securityContext": {},
                "terminationGracePeriodSeconds": 30
            },
            "status": {}
        },
        {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "creationTimestamp": null,
                "name": "bar",
                "namespace": "test",
                "resourceVersion": "11"
            },
            "spec": {
                "containers": null,
                "dnsPolicy": "ClusterFirst",
                "enableServiceLinks": true,
                "restartPolicy": "Always",
                "securityContext": {},
                "terminationGracePeriodSeconds": 30
            },
            "status": {}
        },
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "creationTimestamp": null,
                "name": "baz",
                "namespace": "test",
                "resourceVersion": "12"
            },
            "spec": {
                "sessionAffinity": "None",
                "type": "ClusterIP"
            },
            "status": {
                "loadBalancer": {}
            }
        }
    ],
    "kind": "List",
    "metadata": {
        "resourceVersion": "",
        "selfLink": ""
    }
}
`
	if e, a := expected, buf.String(); e != a {
		t.Errorf("did not match: %v", diff.StringDiff(e, a))
	}
}

func TestGetMultipleTypeObjectsWithLabelSelector(t *testing.T) {
	pods, svc, _ := cmdtesting.TestData()

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			if req.URL.Query().Get(metav1.LabelSelectorQueryParam("v1")) != "a=b" {
				t.Fatalf("request url: %#v,and request: %#v", req.URL, req)
			}
			switch req.URL.Path {
			case "/namespaces/test/pods":
				return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)}, nil
			case "/namespaces/test/services":
				return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, svc)}, nil
			default:
				t.Fatalf("request url: %#v,and request: %#v", req.URL, req)
				return nil, nil
			}
		}),
	}

	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdGet("kubectl", tf, streams)
	cmd.SetOutput(buf)

	cmd.Flags().Set("selector", "a=b")
	cmd.Run(cmd, []string{"pods,services"})

	expected := `NAME      READY   STATUS   RESTARTS   AGE
pod/foo   0/0              0          <unknown>
pod/bar   0/0              0          <unknown>
NAME          TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)   AGE
service/baz   ClusterIP   <none>       <none>        <none>    <unknown>
`
	if e, a := expected, buf.String(); e != a {
		t.Errorf("expected\n%v\ngot\n%v", e, a)
	}
}

func TestGetMultipleTypeObjectsWithFieldSelector(t *testing.T) {
	pods, svc, _ := cmdtesting.TestData()

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			if req.URL.Query().Get(metav1.FieldSelectorQueryParam("v1")) != "a=b" {
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
			}
			switch req.URL.Path {
			case "/namespaces/test/pods":
				return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)}, nil
			case "/namespaces/test/services":
				return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, svc)}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}

	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdGet("kubectl", tf, streams)
	cmd.SetOutput(buf)

	cmd.Flags().Set("field-selector", "a=b")
	cmd.Run(cmd, []string{"pods,services"})

	expected := `NAME      READY   STATUS   RESTARTS   AGE
pod/foo   0/0              0          <unknown>
pod/bar   0/0              0          <unknown>
NAME          TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)   AGE
service/baz   ClusterIP   <none>       <none>        <none>    <unknown>
`
	if e, a := expected, buf.String(); e != a {
		t.Errorf("expected\n%v\ngot\n%v", e, a)
	}
}

func TestGetMultipleTypeObjectsWithDirectReference(t *testing.T) {
	_, svc, _ := cmdtesting.TestData()
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
	}

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch req.URL.Path {
			case "/nodes/foo":
				return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, node)}, nil
			case "/namespaces/test/services/bar":
				return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &svc.Items[0])}, nil
			default:
				t.Fatalf("request url: %#v,and request: %#v", req.URL, req)
				return nil, nil
			}
		}),
	}

	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdGet("kubectl", tf, streams)
	cmd.SetOutput(buf)

	cmd.Run(cmd, []string{"services/bar", "node/foo"})

	expected := `NAME          TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)   AGE
service/baz   ClusterIP   <none>       <none>        <none>    <unknown>
NAME       STATUS    ROLES    AGE         VERSION
node/foo   Unknown   <none>   <unknown>   
`
	if e, a := expected, buf.String(); e != a {
		t.Errorf("expected\n%v\ngot\n%v", e, a)
	}
}

func watchTestData() ([]corev1.Pod, []watch.Event) {
	pods := []corev1.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:            "bar",
				Namespace:       "test",
				ResourceVersion: "9",
			},
			Spec: corev1.PodSpec{
				RestartPolicy:                 corev1.RestartPolicyAlways,
				DNSPolicy:                     corev1.DNSClusterFirst,
				TerminationGracePeriodSeconds: &grace,
				SecurityContext:               &corev1.PodSecurityContext{},
				EnableServiceLinks:            &enableServiceLinks,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:            "foo",
				Namespace:       "test",
				ResourceVersion: "10",
			},
			Spec: corev1.PodSpec{
				RestartPolicy:                 corev1.RestartPolicyAlways,
				DNSPolicy:                     corev1.DNSClusterFirst,
				TerminationGracePeriodSeconds: &grace,
				SecurityContext:               &corev1.PodSecurityContext{},
				EnableServiceLinks:            &enableServiceLinks,
			},
		},
	}
	events := []watch.Event{
		// current state events
		{
			Type: watch.Added,
			Object: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "bar",
					Namespace:       "test",
					ResourceVersion: "9",
				},
				Spec: corev1.PodSpec{
					RestartPolicy:                 corev1.RestartPolicyAlways,
					DNSPolicy:                     corev1.DNSClusterFirst,
					TerminationGracePeriodSeconds: &grace,
					SecurityContext:               &corev1.PodSecurityContext{},
					EnableServiceLinks:            &enableServiceLinks,
				},
			},
		},
		{
			Type: watch.Added,
			Object: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "foo",
					Namespace:       "test",
					ResourceVersion: "10",
				},
				Spec: corev1.PodSpec{
					RestartPolicy:                 corev1.RestartPolicyAlways,
					DNSPolicy:                     corev1.DNSClusterFirst,
					TerminationGracePeriodSeconds: &grace,
					SecurityContext:               &corev1.PodSecurityContext{},
					EnableServiceLinks:            &enableServiceLinks,
				},
			},
		},
		// resource events
		{
			Type: watch.Modified,
			Object: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "foo",
					Namespace:       "test",
					ResourceVersion: "11",
				},
				Spec: corev1.PodSpec{
					RestartPolicy:                 corev1.RestartPolicyAlways,
					DNSPolicy:                     corev1.DNSClusterFirst,
					TerminationGracePeriodSeconds: &grace,
					SecurityContext:               &corev1.PodSecurityContext{},
					EnableServiceLinks:            &enableServiceLinks,
				},
			},
		},
		{
			Type: watch.Deleted,
			Object: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "foo",
					Namespace:       "test",
					ResourceVersion: "12",
				},
				Spec: corev1.PodSpec{
					RestartPolicy:                 corev1.RestartPolicyAlways,
					DNSPolicy:                     corev1.DNSClusterFirst,
					TerminationGracePeriodSeconds: &grace,
					SecurityContext:               &corev1.PodSecurityContext{},
					EnableServiceLinks:            &enableServiceLinks,
				},
			},
		},
	}
	return pods, events
}

func TestWatchLabelSelector(t *testing.T) {
	pods, events := watchTestData()

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	podList := &corev1.PodList{
		Items: pods,
		ListMeta: metav1.ListMeta{
			ResourceVersion: "10",
		},
	}
	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			if req.URL.Query().Get(metav1.LabelSelectorQueryParam("v1")) != "a=b" {
				t.Fatalf("request url: %#v,and request: %#v", req.URL, req)
			}
			switch req.URL.Path {
			case "/namespaces/test/pods":
				if req.URL.Query().Get("watch") == "true" {
					return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: watchBody(codec, events[2:])}, nil
				}
				return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, podList)}, nil
			default:
				t.Fatalf("request url: %#v,and request: %#v", req.URL, req)
				return nil, nil
			}
		}),
	}

	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdGet("kubectl", tf, streams)
	cmd.SetOutput(buf)

	cmd.Flags().Set("watch", "true")
	cmd.Flags().Set("selector", "a=b")
	cmd.Run(cmd, []string{"pods"})

	expected := `NAME   READY   STATUS   RESTARTS   AGE
bar    0/0              0          <unknown>
foo    0/0              0          <unknown>
foo   0/0         0     <unknown>
foo   0/0         0     <unknown>
`
	if e, a := expected, buf.String(); e != a {
		t.Errorf("expected\n%v\ngot\n%v", e, a)
	}
}

func TestWatchFieldSelector(t *testing.T) {
	pods, events := watchTestData()

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	podList := &corev1.PodList{
		Items: pods,
		ListMeta: metav1.ListMeta{
			ResourceVersion: "10",
		},
	}
	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			if req.URL.Query().Get(metav1.FieldSelectorQueryParam("v1")) != "a=b" {
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
			}
			switch req.URL.Path {
			case "/namespaces/test/pods":
				if req.URL.Query().Get("watch") == "true" {
					return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: watchBody(codec, events[2:])}, nil
				}
				return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, podList)}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}

	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdGet("kubectl", tf, streams)
	cmd.SetOutput(buf)

	cmd.Flags().Set("watch", "true")
	cmd.Flags().Set("field-selector", "a=b")
	cmd.Run(cmd, []string{"pods"})

	expected := `NAME   READY   STATUS   RESTARTS   AGE
bar    0/0              0          <unknown>
foo    0/0              0          <unknown>
foo   0/0         0     <unknown>
foo   0/0         0     <unknown>
`
	if e, a := expected, buf.String(); e != a {
		t.Errorf("expected\n%v\ngot\n%v", e, a)
	}
}

func TestWatchResource(t *testing.T) {
	pods, events := watchTestData()

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch req.URL.Path {
			case "/namespaces/test/pods/foo":
				return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &pods[1])}, nil
			case "/namespaces/test/pods":
				if req.URL.Query().Get("watch") == "true" && req.URL.Query().Get("fieldSelector") == "metadata.name=foo" {
					return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: watchBody(codec, events[1:])}, nil
				}
				t.Fatalf("request url: %#v,and request: %#v", req.URL, req)
				return nil, nil
			default:
				t.Fatalf("request url: %#v,and request: %#v", req.URL, req)
				return nil, nil
			}
		}),
	}

	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdGet("kubectl", tf, streams)
	cmd.SetOutput(buf)

	cmd.Flags().Set("watch", "true")
	cmd.Run(cmd, []string{"pods", "foo"})

	expected := `NAME   READY   STATUS   RESTARTS   AGE
foo    0/0              0          <unknown>
foo   0/0         0     <unknown>
foo   0/0         0     <unknown>
`
	if e, a := expected, buf.String(); e != a {
		t.Errorf("expected\n%v\ngot\n%v", e, a)
	}
}

func TestWatchResourceIdentifiedByFile(t *testing.T) {
	pods, events := watchTestData()

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch req.URL.Path {
			case "/namespaces/test/replicationcontrollers/cassandra":
				return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &pods[1])}, nil
			case "/namespaces/test/replicationcontrollers":
				if req.URL.Query().Get("watch") == "true" && req.URL.Query().Get("fieldSelector") == "metadata.name=cassandra" {
					return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: watchBody(codec, events[1:])}, nil
				}
				t.Fatalf("request url: %#v,and request: %#v", req.URL, req)
				return nil, nil
			default:
				t.Fatalf("request url: %#v,and request: %#v", req.URL, req)
				return nil, nil
			}
		}),
	}

	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdGet("kubectl", tf, streams)
	cmd.SetOutput(buf)

	cmd.Flags().Set("watch", "true")
	cmd.Flags().Set("filename", "../../../../test/e2e/testing-manifests/statefulset/cassandra/controller.yaml")
	cmd.Run(cmd, []string{})

	expected := `NAME   READY   STATUS   RESTARTS   AGE
foo    0/0              0          <unknown>
foo   0/0         0     <unknown>
foo   0/0         0     <unknown>
`
	if e, a := expected, buf.String(); e != a {
		t.Errorf("expected\n%v\ngot\n%v", e, a)
	}
}

func TestWatchOnlyResource(t *testing.T) {
	pods, events := watchTestData()

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch req.URL.Path {
			case "/namespaces/test/pods/foo":
				return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &pods[1])}, nil
			case "/namespaces/test/pods":
				if req.URL.Query().Get("watch") == "true" && req.URL.Query().Get("fieldSelector") == "metadata.name=foo" {
					return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: watchBody(codec, events[1:])}, nil
				}
				t.Fatalf("request url: %#v,and request: %#v", req.URL, req)
				return nil, nil
			default:
				t.Fatalf("request url: %#v,and request: %#v", req.URL, req)
				return nil, nil
			}
		}),
	}

	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdGet("kubectl", tf, streams)
	cmd.SetOutput(buf)

	cmd.Flags().Set("watch-only", "true")
	cmd.Run(cmd, []string{"pods", "foo"})

	expected := `NAME   READY   STATUS   RESTARTS   AGE
foo    0/0              0          <unknown>
foo   0/0         0     <unknown>
`
	if e, a := expected, buf.String(); e != a {
		t.Errorf("expected\n%v\ngot\n%v", e, a)
	}
}

func TestWatchOnlyList(t *testing.T) {
	pods, events := watchTestData()

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	podList := &corev1.PodList{
		Items: pods,
		ListMeta: metav1.ListMeta{
			ResourceVersion: "10",
		},
	}
	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch req.URL.Path {
			case "/namespaces/test/pods":
				if req.URL.Query().Get("watch") == "true" {
					return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: watchBody(codec, events[2:])}, nil
				}
				return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, podList)}, nil
			default:
				t.Fatalf("request url: %#v,and request: %#v", req.URL, req)
				return nil, nil
			}
		}),
	}

	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdGet("kubectl", tf, streams)
	cmd.SetOutput(buf)

	cmd.Flags().Set("watch-only", "true")
	cmd.Run(cmd, []string{"pods"})

	expected := `NAME   READY   STATUS   RESTARTS   AGE
foo    0/0              0          <unknown>
foo   0/0         0     <unknown>
`
	if e, a := expected, buf.String(); e != a {
		t.Errorf("expected\n%v\ngot\n%v", e, a)
	}
}

func watchBody(codec runtime.Codec, events []watch.Event) io.ReadCloser {
	buf := bytes.NewBuffer([]byte{})
	enc := restclientwatch.NewEncoder(streaming.NewEncoder(buf, codec), codec)
	for i := range events {
		enc.Encode(&events[i])
	}
	return json.Framer.NewFrameReader(ioutil.NopCloser(buf))
}
