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
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	appsv1 "k8s.io/api/apps/v1"
	appsv1beta1 "k8s.io/api/apps/v1beta1"
	appsv1beta2 "k8s.io/api/apps/v1beta2"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericclioptions/resource"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
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
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			tf.Client = &fake.RESTClient{
				GroupVersion: schema.GroupVersion{Version: "v1"},
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					t.Fatalf("unexpected request: %s %#v\n%#v", req.Method, req.URL, req)
					return nil, nil
				}),
			}

			outputFormat := "yaml"

			streams, _, buf, _ := genericclioptions.NewTestIOStreams()
			cmd := NewCmdServiceAccount(tf, streams)
			cmd.Flags().Set("output", outputFormat)
			cmd.Flags().Set("local", "true")
			saConfig := SetServiceAccountOptions{
				PrintFlags: genericclioptions.NewPrintFlags("").WithDefaultOutput(outputFormat).WithTypeSetter(scheme.Scheme),
				fileNameOptions: resource.FilenameOptions{
					Filenames: []string{input.yaml}},
				local:     true,
				IOStreams: streams,
			}
			err := saConfig.Complete(tf, cmd, []string{serviceAccount})
			assert.NoError(t, err)
			err = saConfig.Run()
			assert.NoError(t, err)
			assert.Contains(t, buf.String(), "serviceAccountName: "+serviceAccount, fmt.Sprintf("serviceaccount not updated for %s", input.yaml))
		})
	}
}

func TestSetServiceAccountMultiLocal(t *testing.T) {
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	tf.Client = &fake.RESTClient{
		GroupVersion:         schema.GroupVersion{Version: ""},
		NegotiatedSerializer: serializer.DirectCodecFactory{CodecFactory: scheme.Codecs},
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			t.Fatalf("unexpected request: %s %#v\n%#v", req.Method, req.URL, req)
			return nil, nil
		}),
	}
	tf.ClientConfigVal = &restclient.Config{ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Version: ""}}}

	outputFormat := "name"

	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdServiceAccount(tf, streams)
	cmd.Flags().Set("output", outputFormat)
	cmd.Flags().Set("local", "true")
	opts := SetServiceAccountOptions{
		PrintFlags: genericclioptions.NewPrintFlags("").WithDefaultOutput(outputFormat).WithTypeSetter(scheme.Scheme),
		fileNameOptions: resource.FilenameOptions{
			Filenames: []string{"../../../../test/fixtures/pkg/kubectl/cmd/set/multi-resource-yaml.yaml"}},
		local:     true,
		IOStreams: streams,
	}

	err := opts.Complete(tf, cmd, []string{serviceAccount})
	if err == nil {
		err = opts.Run()
	}
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expectedOut := "replicationcontroller/first-rc\nreplicationcontroller/second-rc\n"
	if buf.String() != expectedOut {
		t.Errorf("expected out:\n%s\nbut got:\n%s", expectedOut, buf.String())
	}
}

func TestSetServiceAccountRemote(t *testing.T) {
	inputs := []struct {
		object       runtime.Object
		groupVersion schema.GroupVersion
		path         string
		args         []string
	}{
		{
			object: &extensionsv1beta1.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
			},
			groupVersion: extensionsv1beta1.SchemeGroupVersion,
			path:         "/namespaces/test/replicasets/nginx",
			args:         []string{"replicaset", "nginx", serviceAccount},
		},
		{
			object: &appsv1beta2.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
				Spec: appsv1beta2.ReplicaSetSpec{
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Name:  "nginx",
									Image: "nginx",
								},
							},
						},
					},
				},
			},
			groupVersion: appsv1beta2.SchemeGroupVersion,
			path:         "/namespaces/test/replicasets/nginx",
			args:         []string{"replicaset", "nginx", serviceAccount},
		},
		{
			object: &appsv1.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
				Spec: appsv1.ReplicaSetSpec{
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Name:  "nginx",
									Image: "nginx",
								},
							},
						},
					},
				},
			},
			groupVersion: appsv1.SchemeGroupVersion,
			path:         "/namespaces/test/replicasets/nginx",
			args:         []string{"replicaset", "nginx", serviceAccount},
		},
		{
			object: &extensionsv1beta1.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
			},
			groupVersion: extensionsv1beta1.SchemeGroupVersion,
			path:         "/namespaces/test/daemonsets/nginx",
			args:         []string{"daemonset", "nginx", serviceAccount},
		},
		{
			object: &appsv1beta2.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
			},
			groupVersion: appsv1beta2.SchemeGroupVersion,
			path:         "/namespaces/test/daemonsets/nginx",
			args:         []string{"daemonset", "nginx", serviceAccount},
		},
		{
			object: &appsv1.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
			},
			groupVersion: appsv1.SchemeGroupVersion,
			path:         "/namespaces/test/daemonsets/nginx",
			args:         []string{"daemonset", "nginx", serviceAccount},
		},
		{
			object: &extensionsv1beta1.Deployment{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
			},
			groupVersion: extensionsv1beta1.SchemeGroupVersion,
			path:         "/namespaces/test/deployments/nginx",
			args:         []string{"deployment", "nginx", serviceAccount},
		},
		{
			object: &appsv1beta1.Deployment{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
			},
			groupVersion: appsv1beta1.SchemeGroupVersion,
			path:         "/namespaces/test/deployments/nginx",
			args:         []string{"deployment", "nginx", serviceAccount},
		},
		{
			object: &appsv1beta2.Deployment{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
			},
			groupVersion: appsv1beta2.SchemeGroupVersion,
			path:         "/namespaces/test/deployments/nginx",
			args:         []string{"deployment", "nginx", serviceAccount},
		},
		{
			object: &appsv1.Deployment{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
				Spec: appsv1.DeploymentSpec{
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Name:  "nginx",
									Image: "nginx",
								},
							},
						},
					},
				},
			},
			groupVersion: appsv1.SchemeGroupVersion,
			path:         "/namespaces/test/deployments/nginx",
			args:         []string{"deployment", "nginx", serviceAccount},
		},
		{
			object: &appsv1beta1.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
			},
			groupVersion: appsv1beta1.SchemeGroupVersion,
			path:         "/namespaces/test/statefulsets/nginx",
			args:         []string{"statefulset", "nginx", serviceAccount},
		},
		{
			object: &appsv1beta2.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
			},
			groupVersion: appsv1beta2.SchemeGroupVersion,
			path:         "/namespaces/test/statefulsets/nginx",
			args:         []string{"statefulset", "nginx", serviceAccount},
		},
		{
			object: &appsv1.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
				Spec: appsv1.StatefulSetSpec{
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Name:  "nginx",
									Image: "nginx",
								},
							},
						},
					},
				},
			},
			groupVersion: appsv1.SchemeGroupVersion,
			path:         "/namespaces/test/statefulsets/nginx",
			args:         []string{"statefulset", "nginx", serviceAccount},
		},
		{
			object: &batchv1.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
			},
			groupVersion: batchv1.SchemeGroupVersion,
			path:         "/namespaces/test/jobs/nginx",
			args:         []string{"job", "nginx", serviceAccount},
		},
		{
			object: &corev1.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
			},
			groupVersion: corev1.SchemeGroupVersion,
			path:         "/namespaces/test/replicationcontrollers/nginx",
			args:         []string{"replicationcontroller", "nginx", serviceAccount},
		},
	}
	for i, input := range inputs {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			tf.Client = &fake.RESTClient{
				GroupVersion:         input.groupVersion,
				NegotiatedSerializer: serializer.DirectCodecFactory{CodecFactory: scheme.Codecs},
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					switch p, m := req.URL.Path, req.Method; {
					case p == input.path && m == http.MethodGet:
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: objBody(input.object)}, nil
					case p == input.path && m == http.MethodPatch:
						stream, err := req.GetBody()
						if err != nil {
							return nil, err
						}
						bytes, err := ioutil.ReadAll(stream)
						if err != nil {
							return nil, err
						}
						assert.Contains(t, string(bytes), `"serviceAccountName":`+`"`+serviceAccount+`"`, fmt.Sprintf("serviceaccount not updated for %#v", input.object))
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: objBody(input.object)}, nil
					default:
						t.Errorf("%s: unexpected request: %s %#v\n%#v", "serviceaccount", req.Method, req.URL, req)
						return nil, fmt.Errorf("unexpected request")
					}
				}),
			}

			outputFormat := "yaml"

			streams := genericclioptions.NewTestIOStreamsDiscard()
			cmd := NewCmdServiceAccount(tf, streams)
			cmd.Flags().Set("output", outputFormat)
			saConfig := SetServiceAccountOptions{
				PrintFlags: genericclioptions.NewPrintFlags("").WithDefaultOutput(outputFormat).WithTypeSetter(scheme.Scheme),

				local:     false,
				IOStreams: streams,
			}
			err := saConfig.Complete(tf, cmd, input.args)
			assert.NoError(t, err)
			err = saConfig.Run()
			assert.NoError(t, err)
		})
	}
}

func TestServiceAccountValidation(t *testing.T) {
	inputs := []struct {
		name        string
		args        []string
		errorString string
	}{
		{name: "test service account missing", args: []string{}, errorString: serviceAccountMissingErrString},
		{name: "test service account resource missing", args: []string{serviceAccount}, errorString: resourceMissingErrString},
	}
	for _, input := range inputs {
		t.Run(input.name, func(t *testing.T) {
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			tf.Client = &fake.RESTClient{
				GroupVersion: schema.GroupVersion{Version: "v1"},
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					t.Fatalf("unexpected request: %s %#v\n%#v", req.Method, req.URL, req)
					return nil, nil
				}),
			}

			outputFormat := ""

			streams := genericclioptions.NewTestIOStreamsDiscard()
			cmd := NewCmdServiceAccount(tf, streams)

			saConfig := &SetServiceAccountOptions{
				PrintFlags: genericclioptions.NewPrintFlags("").WithDefaultOutput(outputFormat).WithTypeSetter(scheme.Scheme),
				IOStreams:  streams,
			}
			err := saConfig.Complete(tf, cmd, input.args)
			assert.EqualError(t, err, input.errorString)
		})
	}
}

func objBody(obj runtime.Object) io.ReadCloser {
	return cmdtesting.BytesBody([]byte(runtime.EncodeOrDie(scheme.DefaultJSONEncoder(), obj)))
}
