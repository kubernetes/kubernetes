/*
Copyright 2017 The Kubernetes Authors.

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

package set

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"path"
	"testing"

	"github.com/stretchr/testify/assert"
	appsv1 "k8s.io/api/apps/v1"
	appsv1beta1 "k8s.io/api/apps/v1beta1"
	appsv1beta2 "k8s.io/api/apps/v1beta2"
	batchv1 "k8s.io/api/batch/v1"
	"k8s.io/api/core/v1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/kubectl/categories"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
	"k8s.io/kubernetes/pkg/printers"
)

const serviceAccount = "serviceaccount1"
const serviceAccountMissingErrString = "serviceaccount is required"
const resourceMissingErrString = `You must provide one or more resources by argument or filename.
Example resource specifications include:
   '-f rsrc.yaml'
   '--filename=rsrc.json'
   '<resource> <name>'
   '<resource>'`

func TestSetServiceAccountLocal(t *testing.T) {
	inputs := []struct {
		yaml     string
		apiGroup string
	}{
		{yaml: "../../../../test/fixtures/doc-yaml/user-guide/replication.yaml", apiGroup: ""},
		{yaml: "../../../../test/fixtures/doc-yaml/admin/daemon.yaml", apiGroup: "extensions"},
		{yaml: "../../../../test/fixtures/doc-yaml/user-guide/replicaset/redis-slave.yaml", apiGroup: "extensions"},
		{yaml: "../../../../test/fixtures/doc-yaml/user-guide/job.yaml", apiGroup: "batch"},
		{yaml: "../../../../test/fixtures/doc-yaml/user-guide/deployment.yaml", apiGroup: "extensions"},
	}

	for i, input := range inputs {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			f, tf, _, _ := cmdtesting.NewAPIFactory()
			tf.Client = &fake.RESTClient{
				GroupVersion: schema.GroupVersion{Version: "v1"},
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					t.Fatalf("unexpected request: %s %#v\n%#v", req.Method, req.URL, req)
					return nil, nil
				}),
			}
			tf.Namespace = "test"
			out := new(bytes.Buffer)
			cmd := NewCmdServiceAccount(f, out, out)
			cmd.SetOutput(out)
			cmd.Flags().Set("output", "yaml")
			cmd.Flags().Set("local", "true")
			testapi.Default = testapi.Groups[input.apiGroup]
			tf.Printer = printers.NewVersionedPrinter(&printers.YAMLPrinter{}, testapi.Default.Converter(), *testapi.Default.GroupVersion())
			saConfig := serviceAccountConfig{fileNameOptions: resource.FilenameOptions{
				Filenames: []string{input.yaml}},
				out:   out,
				local: true}
			err := saConfig.Complete(f, cmd, []string{serviceAccount})
			assert.NoError(t, err)
			err = saConfig.Run()
			assert.NoError(t, err)
			assert.Contains(t, out.String(), "serviceAccountName: "+serviceAccount, fmt.Sprintf("serviceaccount not updated for %s", input.yaml))
		})
	}
}

func TestSetServiceAccountMultiLocal(t *testing.T) {
	testapi.Default = testapi.Groups[""]
	f, tf, codec, ns := cmdtesting.NewAPIFactory()
	tf.Client = &fake.RESTClient{
		GroupVersion:         schema.GroupVersion{Version: ""},
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			t.Fatalf("unexpected request: %s %#v\n%#v", req.Method, req.URL, req)
			return nil, nil
		}),
	}
	tf.Namespace = "test"
	tf.ClientConfig = &restclient.Config{ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Version: ""}}}

	buf := bytes.NewBuffer([]byte{})
	cmd := NewCmdServiceAccount(f, buf, buf)
	cmd.SetOutput(buf)
	cmd.Flags().Set("output", "name")
	cmd.Flags().Set("local", "true")
	mapper, typer := f.Object()
	tf.Printer = &printers.NamePrinter{Decoders: []runtime.Decoder{codec}, Typer: typer, Mapper: mapper}
	opts := serviceAccountConfig{fileNameOptions: resource.FilenameOptions{
		Filenames: []string{"../../../../test/fixtures/pkg/kubectl/cmd/set/multi-resource-yaml.yaml"}},
		out:   buf,
		local: true}

	err := opts.Complete(f, cmd, []string{serviceAccount})
	if err == nil {
		err = opts.Run()
	}
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expectedOut := "replicationcontrollers/first-rc\nreplicationcontrollers/second-rc\n"
	if buf.String() != expectedOut {
		t.Errorf("expected out:\n%s\nbut got:\n%s", expectedOut, buf.String())
	}
}

func TestSetServiceAccountRemote(t *testing.T) {
	inputs := []struct {
		object                          runtime.Object
		apiPrefix, apiGroup, apiVersion string
		testAPIGroup                    string
		args                            []string
	}{
		{
			object: &extensionsv1beta1.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
			},
			testAPIGroup: "extensions",
			apiPrefix:    "/apis", apiGroup: "extensions", apiVersion: "v1beta1",
			args: []string{"replicaset", "nginx", serviceAccount},
		},
		{
			object: &appsv1beta2.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
				Spec: appsv1beta2.ReplicaSetSpec{
					Template: v1.PodTemplateSpec{
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								{
									Name:  "nginx",
									Image: "nginx",
								},
							},
						},
					},
				},
			},
			testAPIGroup: "extensions",
			apiPrefix:    "/apis", apiGroup: "apps", apiVersion: "v1beta2",
			args: []string{"replicaset", "nginx", serviceAccount},
		},
		{
			object: &appsv1.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
				Spec: appsv1.ReplicaSetSpec{
					Template: v1.PodTemplateSpec{
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								{
									Name:  "nginx",
									Image: "nginx",
								},
							},
						},
					},
				},
			},
			testAPIGroup: "extensions",
			apiPrefix:    "/apis", apiGroup: "apps", apiVersion: "v1",
			args: []string{"replicaset", "nginx", serviceAccount},
		},
		{
			object: &extensionsv1beta1.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
			},
			testAPIGroup: "extensions",
			apiPrefix:    "/apis", apiGroup: "extensions", apiVersion: "v1beta1",
			args: []string{"daemonset", "nginx", serviceAccount},
		},
		{
			object: &appsv1beta2.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
			},
			testAPIGroup: "extensions",
			apiPrefix:    "/apis", apiGroup: "apps", apiVersion: "v1beta2",
			args: []string{"daemonset", "nginx", serviceAccount},
		},
		{
			object: &appsv1.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
			},
			testAPIGroup: "extensions",
			apiPrefix:    "/apis", apiGroup: "apps", apiVersion: "v1",
			args: []string{"daemonset", "nginx", serviceAccount},
		},
		{
			object: &extensionsv1beta1.Deployment{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
			},
			testAPIGroup: "extensions",
			apiPrefix:    "/apis", apiGroup: "extensions", apiVersion: "v1beta1",
			args: []string{"deployment", "nginx", serviceAccount},
		},
		{
			object: &appsv1beta1.Deployment{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
			},
			testAPIGroup: "extensions",
			apiPrefix:    "/apis", apiGroup: "apps", apiVersion: "v1beta1",
			args: []string{"deployment", "nginx", serviceAccount},
		},
		{
			object: &appsv1beta2.Deployment{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
			},
			testAPIGroup: "extensions",
			apiPrefix:    "/apis", apiGroup: "apps", apiVersion: "v1beta2",
			args: []string{"deployment", "nginx", serviceAccount},
		},
		{
			object: &appsv1.Deployment{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
				Spec: appsv1.DeploymentSpec{
					Template: v1.PodTemplateSpec{
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								{
									Name:  "nginx",
									Image: "nginx",
								},
							},
						},
					},
				},
			},
			testAPIGroup: "extensions",
			apiPrefix:    "/apis", apiGroup: "apps", apiVersion: "v1",
			args: []string{"deployment", "nginx", serviceAccount},
		},
		{
			object: &appsv1beta1.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
			},
			testAPIGroup: "apps",
			apiPrefix:    "/apis", apiGroup: "apps", apiVersion: "v1beta1",
			args: []string{"statefulset", "nginx", serviceAccount},
		},
		{
			object: &appsv1beta2.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
			},
			testAPIGroup: "apps",
			apiPrefix:    "/apis", apiGroup: "apps", apiVersion: "v1beta2",
			args: []string{"statefulset", "nginx", serviceAccount},
		},
		{
			object: &appsv1.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
				Spec: appsv1.StatefulSetSpec{
					Template: v1.PodTemplateSpec{
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								{
									Name:  "nginx",
									Image: "nginx",
								},
							},
						},
					},
				},
			},
			testAPIGroup: "apps",
			apiPrefix:    "/apis", apiGroup: "apps", apiVersion: "v1",
			args: []string{"statefulset", "nginx", serviceAccount},
		},
		{
			object: &batchv1.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
			},
			testAPIGroup: "batch",
			apiPrefix:    "/apis", apiGroup: "batch", apiVersion: "v1",
			args: []string{"job", "nginx", serviceAccount},
		},
		{
			object: &v1.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
			},
			testAPIGroup: "",
			apiPrefix:    "/api", apiGroup: "", apiVersion: "v1",
			args: []string{"replicationcontroller", "nginx", serviceAccount},
		},
	}
	for _, input := range inputs {
		groupVersion := schema.GroupVersion{Group: input.apiGroup, Version: input.apiVersion}
		testapi.Default = testapi.Groups[input.testAPIGroup]
		f, tf, _, ns := cmdtesting.NewAPIFactory()
		codec := scheme.Codecs.CodecForVersions(scheme.Codecs.LegacyCodec(groupVersion), scheme.Codecs.UniversalDecoder(groupVersion), groupVersion, groupVersion)
		tf.Printer = printers.NewVersionedPrinter(&printers.YAMLPrinter{}, testapi.Default.Converter(), *testapi.Default.GroupVersion())
		tf.Namespace = "test"
		tf.CategoryExpander = categories.LegacyCategoryExpander
		tf.Client = &fake.RESTClient{
			GroupVersion:         groupVersion,
			NegotiatedSerializer: ns,
			Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				resourcePath := testapi.Default.ResourcePath(input.args[0]+"s", tf.Namespace, input.args[1])
				switch p, m := req.URL.Path, req.Method; {
				case p == resourcePath && m == http.MethodGet:
					return &http.Response{StatusCode: http.StatusOK, Header: defaultHeader(), Body: objBody(codec, input.object)}, nil
				case p == resourcePath && m == http.MethodPatch:
					stream, err := req.GetBody()
					if err != nil {
						return nil, err
					}
					bytes, err := ioutil.ReadAll(stream)
					if err != nil {
						return nil, err
					}
					assert.Contains(t, string(bytes), `"serviceAccountName":`+`"`+serviceAccount+`"`, fmt.Sprintf("serviceaccount not updated for %#v", input.object))
					return &http.Response{StatusCode: http.StatusOK, Header: defaultHeader(), Body: objBody(codec, input.object)}, nil
				default:
					t.Errorf("%s: unexpected request: %s %#v\n%#v", "serviceaccount", req.Method, req.URL, req)
					return nil, fmt.Errorf("unexpected request")
				}
			}),
			VersionedAPIPath: path.Join(input.apiPrefix, testapi.Default.GroupVersion().String()),
		}
		out := new(bytes.Buffer)
		cmd := NewCmdServiceAccount(f, out, out)
		cmd.SetOutput(out)
		cmd.Flags().Set("output", "yaml")
		saConfig := serviceAccountConfig{
			out:   out,
			local: false}
		err := saConfig.Complete(f, cmd, input.args)
		assert.NoError(t, err)
		err = saConfig.Run()
		assert.NoError(t, err)
	}
}

func TestServiceAccountValidation(t *testing.T) {
	inputs := []struct {
		args        []string
		errorString string
	}{
		{args: []string{}, errorString: serviceAccountMissingErrString},
		{args: []string{serviceAccount}, errorString: resourceMissingErrString},
	}
	for _, input := range inputs {
		f, tf, _, _ := cmdtesting.NewAPIFactory()
		tf.Client = &fake.RESTClient{
			GroupVersion: schema.GroupVersion{Version: "v1"},
			Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				t.Fatalf("unexpected request: %s %#v\n%#v", req.Method, req.URL, req)
				return nil, nil
			}),
		}
		tf.Namespace = "test"
		out := bytes.NewBuffer([]byte{})
		cmd := NewCmdServiceAccount(f, out, out)
		cmd.SetOutput(out)

		saConfig := &serviceAccountConfig{}
		err := saConfig.Complete(f, cmd, input.args)
		assert.EqualError(t, err, input.errorString)
	}
}

func objBody(codec runtime.Codec, obj runtime.Object) io.ReadCloser {
	return bytesBody([]byte(runtime.EncodeOrDie(codec, obj)))
}

func defaultHeader() http.Header {
	header := http.Header{}
	header.Set("Content-Type", runtime.ContentTypeJSON)
	return header
}

func bytesBody(bodyBytes []byte) io.ReadCloser {
	return ioutil.NopCloser(bytes.NewReader(bodyBytes))
}
