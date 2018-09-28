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
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericclioptions/resource"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	"k8s.io/kubernetes/pkg/api/testapi"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
)

func TestResourcesLocal(t *testing.T) {
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
	cmd := NewCmdResources(tf, streams)
	cmd.SetOutput(buf)
	cmd.Flags().Set("output", outputFormat)
	cmd.Flags().Set("local", "true")

	opts := SetResourcesOptions{
		PrintFlags: genericclioptions.NewPrintFlags("").WithDefaultOutput(outputFormat).WithTypeSetter(scheme.Scheme),
		FilenameOptions: resource.FilenameOptions{
			Filenames: []string{"../../../../test/e2e/testing-manifests/statefulset/cassandra/controller.yaml"}},
		Local:             true,
		Limits:            "cpu=200m,memory=512Mi",
		Requests:          "cpu=200m,memory=512Mi",
		ContainerSelector: "*",
		IOStreams:         streams,
	}

	err := opts.Complete(tf, cmd, []string{})
	if err == nil {
		err = opts.Validate()
	}
	if err == nil {
		err = opts.Run()
	}
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(buf.String(), "replicationcontroller/cassandra") {
		t.Errorf("did not set resources: %s", buf.String())
	}
}

func TestSetMultiResourcesLimitsLocal(t *testing.T) {
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
	cmd := NewCmdResources(tf, streams)
	cmd.SetOutput(buf)
	cmd.Flags().Set("output", outputFormat)
	cmd.Flags().Set("local", "true")

	opts := SetResourcesOptions{
		PrintFlags: genericclioptions.NewPrintFlags("").WithDefaultOutput(outputFormat).WithTypeSetter(scheme.Scheme),
		FilenameOptions: resource.FilenameOptions{
			Filenames: []string{"../../../../test/fixtures/pkg/kubectl/cmd/set/multi-resource-yaml.yaml"}},
		Local:             true,
		Limits:            "cpu=200m,memory=512Mi",
		Requests:          "cpu=200m,memory=512Mi",
		ContainerSelector: "*",
		IOStreams:         streams,
	}

	err := opts.Complete(tf, cmd, []string{})
	if err == nil {
		err = opts.Validate()
	}
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

func TestSetResourcesRemote(t *testing.T) {
	inputs := []struct {
		object                          runtime.Object
		apiPrefix, apiGroup, apiVersion string
		testAPIGroup                    string
		args                            []string
	}{
		{
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
			args: []string{"replicaset", "nginx"},
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
			args: []string{"replicaset", "nginx"},
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
			args: []string{"replicaset", "nginx"},
		},
		{
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
			args: []string{"daemonset", "nginx"},
		},
		{
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
			args: []string{"daemonset", "nginx"},
		},
		{
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
			args: []string{"daemonset", "nginx"},
		},
		{
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
			args: []string{"deployment", "nginx"},
		},
		{
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
			args: []string{"deployment", "nginx"},
		},
		{
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
			args: []string{"deployment", "nginx"},
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
			args: []string{"deployment", "nginx"},
		},
		{
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
			args: []string{"statefulset", "nginx"},
		},
		{
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
			args: []string{"statefulset", "nginx"},
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
			args: []string{"statefulset", "nginx"},
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
			args: []string{"job", "nginx"},
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
			args: []string{"replicationcontroller", "nginx"},
		},
	}
	for i, input := range inputs {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			groupVersion := schema.GroupVersion{Group: input.apiGroup, Version: input.apiVersion}
			testapi.Default = testapi.Groups[input.testAPIGroup]
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
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
						assert.Contains(t, string(bytes), "200m", fmt.Sprintf("resources not updated for %#v", input.object))
						return &http.Response{StatusCode: http.StatusOK, Header: defaultHeader(), Body: objBody(input.object)}, nil
					default:
						t.Errorf("%s: unexpected request: %s %#v\n%#v", "resources", req.Method, req.URL, req)
						return nil, fmt.Errorf("unexpected request")
					}
				}),
				VersionedAPIPath: path.Join(input.apiPrefix, testapi.Default.GroupVersion().String()),
			}

			outputFormat := "yaml"

			streams := genericclioptions.NewTestIOStreamsDiscard()
			cmd := NewCmdResources(tf, streams)
			cmd.Flags().Set("output", outputFormat)
			opts := SetResourcesOptions{
				PrintFlags: genericclioptions.NewPrintFlags("").WithDefaultOutput(outputFormat).WithTypeSetter(scheme.Scheme),

				Limits:            "cpu=200m,memory=512Mi",
				ContainerSelector: "*",
				IOStreams:         streams,
			}
			err := opts.Complete(tf, cmd, input.args)
			if err == nil {
				err = opts.Validate()
			}
			if err == nil {
				err = opts.Run()
			}
			assert.NoError(t, err)
		})
	}
}
