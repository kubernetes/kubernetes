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
	"io/ioutil"
	"net/http"
	"path"
	"strings"
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
	"k8s.io/apimachinery/pkg/runtime/serializer"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	"k8s.io/kubernetes/pkg/api/testapi"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	"k8s.io/kubernetes/pkg/kubectl/genericclioptions"
	"k8s.io/kubernetes/pkg/kubectl/genericclioptions/resource"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
)

func TestSetEnvLocal(t *testing.T) {
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

	streams, _, buf, bufErr := genericclioptions.NewTestIOStreams()
	opts := NewEnvOptions(streams)
	opts.PrintFlags = genericclioptions.NewPrintFlags("").WithDefaultOutput(outputFormat).WithTypeSetter(scheme.Scheme)
	opts.FilenameOptions = resource.FilenameOptions{
		Filenames: []string{"../../../../test/e2e/testing-manifests/statefulset/cassandra/controller.yaml"},
	}
	opts.Local = true

	err := opts.Complete(tf, NewCmdEnv(tf, streams), []string{"env=prod"})
	assert.NoError(t, err)
	err = opts.Validate()
	assert.NoError(t, err)
	err = opts.RunEnv()
	assert.NoError(t, err)
	if bufErr.Len() > 0 {
		t.Errorf("unexpected error: %s", string(bufErr.String()))
	}
	if !strings.Contains(buf.String(), "replicationcontroller/cassandra") {
		t.Errorf("did not set env: %s", buf.String())
	}
}

func TestSetMultiResourcesEnvLocal(t *testing.T) {
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
	streams, _, buf, bufErr := genericclioptions.NewTestIOStreams()
	opts := NewEnvOptions(streams)
	opts.PrintFlags = genericclioptions.NewPrintFlags("").WithDefaultOutput(outputFormat).WithTypeSetter(scheme.Scheme)
	opts.FilenameOptions = resource.FilenameOptions{
		Filenames: []string{"../../../../test/fixtures/pkg/kubectl/cmd/set/multi-resource-yaml.yaml"},
	}
	opts.Local = true

	err := opts.Complete(tf, NewCmdEnv(tf, streams), []string{"env=prod"})
	assert.NoError(t, err)
	err = opts.Validate()
	assert.NoError(t, err)
	err = opts.RunEnv()
	assert.NoError(t, err)
	if bufErr.Len() > 0 {
		t.Errorf("unexpected error: %s", string(bufErr.String()))
	}
	expectedOut := "replicationcontroller/first-rc\nreplicationcontroller/second-rc\n"
	if buf.String() != expectedOut {
		t.Errorf("expected out:\n%s\nbut got:\n%s", expectedOut, buf.String())
	}
}

func TestSetEnvRemote(t *testing.T) {
	inputs := []struct {
		name                            string
		object                          runtime.Object
		apiPrefix, apiGroup, apiVersion string
		testAPIGroup                    string
		args                            []string
	}{
		{
			name: "test extensions.v1beta1 replicaset",
			object: &extensionsv1beta1.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
				Spec: extensionsv1beta1.ReplicaSetSpec{
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
			apiPrefix:    "/apis", apiGroup: "extensions", apiVersion: "v1beta1",
			args: []string{"replicaset", "nginx", "env=prod"},
		},
		{
			name: "test apps.v1beta2 replicaset",
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
			args: []string{"replicaset", "nginx", "env=prod"},
		},
		{
			name: "test appsv1 replicaset",
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
			args: []string{"replicaset", "nginx", "env=prod"},
		},
		{
			name: "test extensions.v1beta1 daemonset",
			object: &extensionsv1beta1.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
				Spec: extensionsv1beta1.DaemonSetSpec{
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
			apiPrefix:    "/apis", apiGroup: "extensions", apiVersion: "v1beta1",
			args: []string{"daemonset", "nginx", "env=prod"},
		},
		{
			name: "test appsv1beta2 daemonset",
			object: &appsv1beta2.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
				Spec: appsv1beta2.DaemonSetSpec{
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
			args: []string{"daemonset", "nginx", "env=prod"},
		},
		{
			name: "test appsv1 daemonset",
			object: &appsv1.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
				Spec: appsv1.DaemonSetSpec{
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
			args: []string{"daemonset", "nginx", "env=prod"},
		},
		{
			name: "test extensions.v1beta1 deployment",
			object: &extensionsv1beta1.Deployment{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
				Spec: extensionsv1beta1.DeploymentSpec{
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
			apiPrefix:    "/apis", apiGroup: "extensions", apiVersion: "v1beta1",
			args: []string{"deployment", "nginx", "env=prod"},
		},
		{
			name: "test appsv1bta1 deployment",
			object: &appsv1beta1.Deployment{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
				Spec: appsv1beta1.DeploymentSpec{
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
			apiPrefix:    "/apis", apiGroup: "apps", apiVersion: "v1beta1",
			args: []string{"deployment", "nginx", "env=prod"},
		},
		{
			name: "test appsv1beta2n deployment",
			object: &appsv1beta2.Deployment{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
				Spec: appsv1beta2.DeploymentSpec{
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
			args: []string{"deployment", "nginx", "env=prod"},
		},
		{
			name: "test appsv1 deployment",
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
			args: []string{"deployment", "nginx", "env=prod"},
		},
		{
			name: "test appsv1beta1 statefulset",
			object: &appsv1beta1.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
				Spec: appsv1beta1.StatefulSetSpec{
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
			apiPrefix:    "/apis", apiGroup: "apps", apiVersion: "v1beta1",
			args: []string{"statefulset", "nginx", "env=prod"},
		},
		{
			name: "test appsv1beta2 statefulset",
			object: &appsv1beta2.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
				Spec: appsv1beta2.StatefulSetSpec{
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
			apiPrefix:    "/apis", apiGroup: "apps", apiVersion: "v1beta2",
			args: []string{"statefulset", "nginx", "env=prod"},
		},
		{
			name: "test appsv1 statefulset",
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
			args: []string{"statefulset", "nginx", "env=prod"},
		},
		{
			object: &batchv1.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
				Spec: batchv1.JobSpec{
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
			testAPIGroup: "batch",
			apiPrefix:    "/apis", apiGroup: "batch", apiVersion: "v1",
			args: []string{"job", "nginx", "env=prod"},
		},
		{
			object: &v1.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
				Spec: v1.ReplicationControllerSpec{
					Template: &v1.PodTemplateSpec{
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
			testAPIGroup: "",
			apiPrefix:    "/api", apiGroup: "", apiVersion: "v1",
			args: []string{"replicationcontroller", "nginx", "env=prod"},
		},
	}
	for _, input := range inputs {
		t.Run(input.name, func(t *testing.T) {
			groupVersion := schema.GroupVersion{Group: input.apiGroup, Version: input.apiVersion}
			testapi.Default = testapi.Groups[input.testAPIGroup]
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			tf.ClientConfigVal = &restclient.Config{ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Version: ""}}}
			defer tf.Cleanup()

			tf.Client = &fake.RESTClient{
				GroupVersion:         groupVersion,
				NegotiatedSerializer: serializer.DirectCodecFactory{CodecFactory: scheme.Codecs},
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					resourcePath := testapi.Default.ResourcePath(input.args[0]+"s", "test", input.args[1])
					switch p, m := req.URL.Path, req.Method; {
					case p == resourcePath && m == http.MethodGet:
						return &http.Response{StatusCode: http.StatusOK, Header: defaultHeader(), Body: objBody(input.object)}, nil
					case p == resourcePath && m == http.MethodPatch:
						stream, err := req.GetBody()
						if err != nil {
							return nil, err
						}
						bytes, err := ioutil.ReadAll(stream)
						if err != nil {
							return nil, err
						}
						assert.Contains(t, string(bytes), `"value":`+`"`+"prod"+`"`, fmt.Sprintf("env not updated for %#v", input.object))
						return &http.Response{StatusCode: http.StatusOK, Header: defaultHeader(), Body: objBody(input.object)}, nil
					default:
						t.Errorf("%s: unexpected request: %s %#v\n%#v", "image", req.Method, req.URL, req)
						return nil, fmt.Errorf("unexpected request")
					}
				}),
				VersionedAPIPath: path.Join(input.apiPrefix, testapi.Default.GroupVersion().String()),
			}

			outputFormat := "yaml"
			streams := genericclioptions.NewTestIOStreamsDiscard()
			opts := NewEnvOptions(streams)
			opts.PrintFlags = genericclioptions.NewPrintFlags("").WithDefaultOutput(outputFormat).WithTypeSetter(scheme.Scheme)
			opts.Local = false
			opts.IOStreams = streams
			err := opts.Complete(tf, NewCmdEnv(tf, streams), input.args)
			assert.NoError(t, err)
			err = opts.RunEnv()
			assert.NoError(t, err)
		})
	}
}

func TestSetEnvFromResource(t *testing.T) {
	mockConfigMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: "testconfigmap"},
		Data: map[string]string{
			"env":          "prod",
			"test-key":     "testValue",
			"test-key-two": "testValueTwo",
		},
	}

	mockSecret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: "testsecret"},
		Data: map[string][]byte{
			"env":          []byte("prod"),
			"test-key":     []byte("testValue"),
			"test-key-two": []byte("testValueTwo"),
		},
	}

	inputs := []struct {
		name           string
		args           []string
		from           string
		keys           []string
		assertIncludes []string
		assertExcludes []string
	}{
		{
			name: "test from configmap",
			args: []string{"deployment", "nginx"},
			from: "configmap/testconfigmap",
			keys: []string{},
			assertIncludes: []string{
				`{"name":"ENV","valueFrom":{"configMapKeyRef":{"key":"env","name":"testconfigmap"}}}`,
				`{"name":"TEST_KEY","valueFrom":{"configMapKeyRef":{"key":"test-key","name":"testconfigmap"}}}`,
				`{"name":"TEST_KEY_TWO","valueFrom":{"configMapKeyRef":{"key":"test-key-two","name":"testconfigmap"}}}`,
			},
			assertExcludes: []string{},
		},
		{
			name: "test from secret",
			args: []string{"deployment", "nginx"},
			from: "secret/testsecret",
			keys: []string{},
			assertIncludes: []string{
				`{"name":"ENV","valueFrom":{"secretKeyRef":{"key":"env","name":"testsecret"}}}`,
				`{"name":"TEST_KEY","valueFrom":{"secretKeyRef":{"key":"test-key","name":"testsecret"}}}`,
				`{"name":"TEST_KEY_TWO","valueFrom":{"secretKeyRef":{"key":"test-key-two","name":"testsecret"}}}`,
			},
			assertExcludes: []string{},
		},
		{
			name: "test from configmap with keys",
			args: []string{"deployment", "nginx"},
			from: "configmap/testconfigmap",
			keys: []string{"env", "test-key-two"},
			assertIncludes: []string{
				`{"name":"ENV","valueFrom":{"configMapKeyRef":{"key":"env","name":"testconfigmap"}}}`,
				`{"name":"TEST_KEY_TWO","valueFrom":{"configMapKeyRef":{"key":"test-key-two","name":"testconfigmap"}}}`,
			},
			assertExcludes: []string{`{"name":"TEST_KEY","valueFrom":{"configMapKeyRef":{"key":"test-key","name":"testconfigmap"}}}`},
		},
		{
			name: "test from secret with keys",
			args: []string{"deployment", "nginx"},
			from: "secret/testsecret",
			keys: []string{"env", "test-key-two"},
			assertIncludes: []string{
				`{"name":"ENV","valueFrom":{"secretKeyRef":{"key":"env","name":"testsecret"}}}`,
				`{"name":"TEST_KEY_TWO","valueFrom":{"secretKeyRef":{"key":"test-key-two","name":"testsecret"}}}`,
			},
			assertExcludes: []string{`{"name":"TEST_KEY","valueFrom":{"secretKeyRef":{"key":"test-key","name":"testsecret"}}}`},
		},
	}

	for _, input := range inputs {
		mockDeployment := &appsv1.Deployment{
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
		}
		t.Run(input.name, func(t *testing.T) {
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			tf.ClientConfigVal = &restclient.Config{ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Version: ""}}}
			tf.Client = &fake.RESTClient{
				GroupVersion:         schema.GroupVersion{Group: "", Version: "v1"},
				NegotiatedSerializer: serializer.DirectCodecFactory{CodecFactory: scheme.Codecs},
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					switch p, m := req.URL.Path, req.Method; {
					case p == "/namespaces/test/configmaps/testconfigmap" && m == http.MethodGet:
						return &http.Response{StatusCode: http.StatusOK, Header: defaultHeader(), Body: objBody(mockConfigMap)}, nil
					case p == "/namespaces/test/secrets/testsecret" && m == http.MethodGet:
						return &http.Response{StatusCode: http.StatusOK, Header: defaultHeader(), Body: objBody(mockSecret)}, nil
					case p == "/namespaces/test/deployments/nginx" && m == http.MethodGet:
						return &http.Response{StatusCode: http.StatusOK, Header: defaultHeader(), Body: objBody(mockDeployment)}, nil
					case p == "/namespaces/test/deployments/nginx" && m == http.MethodPatch:
						stream, err := req.GetBody()
						if err != nil {
							return nil, err
						}
						bytes, err := ioutil.ReadAll(stream)
						if err != nil {
							return nil, err
						}
						for _, include := range input.assertIncludes {
							assert.Contains(t, string(bytes), include)
						}
						for _, exclude := range input.assertExcludes {
							assert.NotContains(t, string(bytes), exclude)
						}
						return &http.Response{StatusCode: http.StatusOK, Header: defaultHeader(), Body: objBody(mockDeployment)}, nil
					default:
						t.Errorf("%s: unexpected request: %#v\n%#v", input.name, req.URL, req)
						return nil, nil
					}
				}),
			}

			outputFormat := "yaml"
			streams := genericclioptions.NewTestIOStreamsDiscard()
			opts := NewEnvOptions(streams)
			opts.From = input.from
			opts.Keys = input.keys
			opts.PrintFlags = genericclioptions.NewPrintFlags("").WithDefaultOutput(outputFormat).WithTypeSetter(scheme.Scheme)
			opts.Local = false
			opts.IOStreams = streams
			err := opts.Complete(tf, NewCmdEnv(tf, streams), input.args)
			assert.NoError(t, err)
			err = opts.RunEnv()
			assert.NoError(t, err)
		})
	}
}
