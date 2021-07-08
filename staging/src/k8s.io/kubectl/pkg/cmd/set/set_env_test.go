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
	"strings"
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
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/resource"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	"k8s.io/kubectl/pkg/scheme"
)

func TestSetEnvLocal(t *testing.T) {
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	tf.Client = &fake.RESTClient{
		GroupVersion:         schema.GroupVersion{Version: ""},
		NegotiatedSerializer: scheme.Codecs.WithoutConversion(),
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
		Filenames: []string{"../../../testdata/controller.yaml"},
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

func TestSetEnvLocalNamespace(t *testing.T) {
	tf := cmdtesting.NewTestFactory()
	defer tf.Cleanup()

	tf.Client = &fake.RESTClient{
		GroupVersion:         schema.GroupVersion{Version: ""},
		NegotiatedSerializer: scheme.Codecs.WithoutConversion(),
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			t.Fatalf("unexpected request: %s %#v\n%#v", req.Method, req.URL, req)
			return nil, nil
		}),
	}
	tf.ClientConfigVal = &restclient.Config{ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Version: ""}}}
	outputFormat := "yaml"

	streams, _, buf, bufErr := genericclioptions.NewTestIOStreams()
	opts := NewEnvOptions(streams)
	opts.PrintFlags = genericclioptions.NewPrintFlags("").WithDefaultOutput(outputFormat).WithTypeSetter(scheme.Scheme)
	opts.FilenameOptions = resource.FilenameOptions{
		Filenames: []string{"../../../testdata/set/namespaced-resource.yaml"},
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
	if !strings.Contains(buf.String(), "namespace: existing-ns") {
		t.Errorf("did not set env: %s", buf.String())
	}
}

func TestSetMultiResourcesEnvLocal(t *testing.T) {
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	tf.Client = &fake.RESTClient{
		GroupVersion:         schema.GroupVersion{Version: ""},
		NegotiatedSerializer: scheme.Codecs.WithoutConversion(),
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
		Filenames: []string{"../../../testdata/set/multi-resource-yaml.yaml"},
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
		name         string
		object       runtime.Object
		groupVersion schema.GroupVersion
		path         string
		args         []string
	}{
		{
			name: "test extensions.v1beta1 replicaset",
			object: &extensionsv1beta1.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
				Spec: extensionsv1beta1.ReplicaSetSpec{
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
			groupVersion: extensionsv1beta1.SchemeGroupVersion,
			path:         "/namespaces/test/replicasets/nginx",
			args:         []string{"replicaset", "nginx", "env=prod"},
		},
		{
			name: "test apps.v1beta2 replicaset",
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
			args:         []string{"replicaset", "nginx", "env=prod"},
		},
		{
			name: "test appsv1 replicaset",
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
			args:         []string{"replicaset", "nginx", "env=prod"},
		},
		{
			name: "test extensions.v1beta1 daemonset",
			object: &extensionsv1beta1.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
				Spec: extensionsv1beta1.DaemonSetSpec{
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
			groupVersion: extensionsv1beta1.SchemeGroupVersion,
			path:         "/namespaces/test/daemonsets/nginx",
			args:         []string{"daemonset", "nginx", "env=prod"},
		},
		{
			name: "test appsv1beta2 daemonset",
			object: &appsv1beta2.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
				Spec: appsv1beta2.DaemonSetSpec{
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
			path:         "/namespaces/test/daemonsets/nginx",
			args:         []string{"daemonset", "nginx", "env=prod"},
		},
		{
			name: "test appsv1 daemonset",
			object: &appsv1.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
				Spec: appsv1.DaemonSetSpec{
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
			path:         "/namespaces/test/daemonsets/nginx",
			args:         []string{"daemonset", "nginx", "env=prod"},
		},
		{
			name: "test extensions.v1beta1 deployment",
			object: &extensionsv1beta1.Deployment{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
				Spec: extensionsv1beta1.DeploymentSpec{
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
			groupVersion: extensionsv1beta1.SchemeGroupVersion,
			path:         "/namespaces/test/deployments/nginx",
			args:         []string{"deployment", "nginx", "env=prod"},
		},
		{
			name: "test appsv1beta1 deployment",
			object: &appsv1beta1.Deployment{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
				Spec: appsv1beta1.DeploymentSpec{
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
			groupVersion: appsv1beta1.SchemeGroupVersion,
			path:         "/namespaces/test/deployments/nginx",
			args:         []string{"deployment", "nginx", "env=prod"},
		},
		{
			name: "test appsv1beta2 deployment",
			object: &appsv1beta2.Deployment{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
				Spec: appsv1beta2.DeploymentSpec{
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
			path:         "/namespaces/test/deployments/nginx",
			args:         []string{"deployment", "nginx", "env=prod"},
		},
		{
			name: "test appsv1 deployment",
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
			args:         []string{"deployment", "nginx", "env=prod"},
		},
		{
			name: "test appsv1beta1 statefulset",
			object: &appsv1beta1.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
				Spec: appsv1beta1.StatefulSetSpec{
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
			groupVersion: appsv1beta1.SchemeGroupVersion,
			path:         "/namespaces/test/statefulsets/nginx",
			args:         []string{"statefulset", "nginx", "env=prod"},
		},
		{
			name: "test appsv1beta2 statefulset",
			object: &appsv1beta2.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
				Spec: appsv1beta2.StatefulSetSpec{
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
			path:         "/namespaces/test/statefulsets/nginx",
			args:         []string{"statefulset", "nginx", "env=prod"},
		},
		{
			name: "test appsv1 statefulset",
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
			args:         []string{"statefulset", "nginx", "env=prod"},
		},
		{
			name: "set image batchv1 CronJob",
			object: &batchv1.CronJob{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
				Spec: batchv1.CronJobSpec{
					JobTemplate: batchv1.JobTemplateSpec{
						Spec: batchv1.JobSpec{
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
				},
			},
			groupVersion: batchv1.SchemeGroupVersion,
			path:         "/namespaces/test/cronjobs/nginx",
			args:         []string{"cronjob", "nginx", "env=prod"},
		},
		{
			name: "test corev1 replication controller",
			object: &corev1.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
				Spec: corev1.ReplicationControllerSpec{
					Template: &corev1.PodTemplateSpec{
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
			groupVersion: corev1.SchemeGroupVersion,
			path:         "/namespaces/test/replicationcontrollers/nginx",
			args:         []string{"replicationcontroller", "nginx", "env=prod"},
		},
	}
	for _, input := range inputs {
		t.Run(input.name, func(t *testing.T) {
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			tf.Client = &fake.RESTClient{
				GroupVersion:         input.groupVersion,
				NegotiatedSerializer: scheme.Codecs.WithoutConversion(),
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
						assert.Contains(t, string(bytes), `"value":`+`"`+"prod"+`"`, fmt.Sprintf("env not updated for %#v", input.object))
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: objBody(input.object)}, nil
					default:
						t.Errorf("%s: unexpected request: %s %#v\n%#v", "image", req.Method, req.URL, req)
						return nil, fmt.Errorf("unexpected request")
					}
				}),
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
	mockConfigMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: "testconfigmap"},
		Data: map[string]string{
			"env":          "prod",
			"test-key":     "testValue",
			"test-key-two": "testValueTwo",
		},
	}

	mockSecret := &corev1.Secret{
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
		}
		t.Run(input.name, func(t *testing.T) {
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			tf.ClientConfigVal = &restclient.Config{ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Version: ""}}}
			tf.Client = &fake.RESTClient{
				GroupVersion:         schema.GroupVersion{Group: "", Version: "v1"},
				NegotiatedSerializer: scheme.Codecs.WithoutConversion(),
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					switch p, m := req.URL.Path, req.Method; {
					case p == "/namespaces/test/configmaps/testconfigmap" && m == http.MethodGet:
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: objBody(mockConfigMap)}, nil
					case p == "/namespaces/test/secrets/testsecret" && m == http.MethodGet:
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: objBody(mockSecret)}, nil
					case p == "/namespaces/test/deployments/nginx" && m == http.MethodGet:
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: objBody(mockDeployment)}, nil
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
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: objBody(mockDeployment)}, nil
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

func TestSetEnvRemoteWithSpecificContainers(t *testing.T) {
	inputs := []struct {
		name     string
		args     []string
		selector string

		expectedContainers int
	}{
		{
			name:               "all containers",
			args:               []string{"deployments", "redis", "env=prod"},
			selector:           "*",
			expectedContainers: 2,
		},
		{
			name:               "use wildcards to select some containers",
			args:               []string{"deployments", "redis", "env=prod"},
			selector:           "red*",
			expectedContainers: 1,
		},
		{
			name:               "single container",
			args:               []string{"deployments", "redis", "env=prod"},
			selector:           "redis",
			expectedContainers: 1,
		},
	}

	for _, input := range inputs {
		mockDeployment := &appsv1.Deployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "redis",
				Namespace: "test",
			},
			Spec: appsv1.DeploymentSpec{
				Template: corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						InitContainers: []corev1.Container{
							{
								Name:  "init",
								Image: "redis",
							},
						},
						Containers: []corev1.Container{
							{
								Name:  "redis",
								Image: "redis",
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
				NegotiatedSerializer: scheme.Codecs.WithoutConversion(),
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					switch p, m := req.URL.Path, req.Method; {
					case p == "/namespaces/test/deployments/redis" && m == http.MethodGet:
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: objBody(mockDeployment)}, nil
					case p == "/namespaces/test/deployments/redis" && m == http.MethodPatch:
						stream, err := req.GetBody()
						if err != nil {
							return nil, err
						}
						bytes, err := ioutil.ReadAll(stream)
						if err != nil {
							return nil, err
						}
						updated := strings.Count(string(bytes), `"value":`+`"`+"prod"+`"`)
						if updated != input.expectedContainers {
							t.Errorf("expected %d containers to be selected but got %d \n", input.expectedContainers, updated)
						}
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: objBody(mockDeployment)}, nil
					default:
						t.Errorf("%s: unexpected request: %#v\n%#v", input.name, req.URL, req)
						return nil, nil
					}
				}),
			}
			streams := genericclioptions.NewTestIOStreamsDiscard()
			opts := &EnvOptions{
				PrintFlags:        genericclioptions.NewPrintFlags("").WithDefaultOutput("yaml").WithTypeSetter(scheme.Scheme),
				ContainerSelector: input.selector,
				Overwrite:         true,
				IOStreams:         streams,
			}
			err := opts.Complete(tf, NewCmdEnv(tf, streams), input.args)
			assert.NoError(t, err)
			err = opts.RunEnv()
			assert.NoError(t, err)
		})
	}
}

func TestSetEnvDoubleStdinUsage(t *testing.T) {
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	tf.Client = &fake.RESTClient{
		GroupVersion:         schema.GroupVersion{Version: ""},
		NegotiatedSerializer: scheme.Codecs.WithoutConversion(),
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			t.Fatalf("unexpected request: %s %#v\n%#v", req.Method, req.URL, req)
			return nil, nil
		}),
	}
	tf.ClientConfigVal = &restclient.Config{ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Version: ""}}}

	streams, bufIn, _, _ := genericclioptions.NewTestIOStreams()
	bufIn.WriteString("SOME_ENV_VAR_KEY=SOME_ENV_VAR_VAL")
	opts := NewEnvOptions(streams)
	opts.FilenameOptions = resource.FilenameOptions{
		Filenames: []string{"-"},
	}

	err := opts.Complete(tf, NewCmdEnv(tf, streams), []string{"-"})
	assert.NoError(t, err)
	err = opts.Validate()
	assert.NoError(t, err)
	err = opts.RunEnv()
	assert.ErrorIs(t, err, resource.StdinMultiUseError)
}
