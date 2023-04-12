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

package set

import (
	"fmt"
	"io"
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
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/resource"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	"k8s.io/kubectl/pkg/scheme"
)

func TestImageLocal(t *testing.T) {
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

	streams, _, buf, _ := genericiooptions.NewTestIOStreams()
	cmd := NewCmdImage(tf, streams)
	cmd.SetOut(buf)
	cmd.SetErr(buf)
	cmd.Flags().Set("output", outputFormat)
	cmd.Flags().Set("local", "true")

	opts := SetImageOptions{
		PrintFlags: genericclioptions.NewPrintFlags("").WithDefaultOutput(outputFormat).WithTypeSetter(scheme.Scheme),
		FilenameOptions: resource.FilenameOptions{
			Filenames: []string{"../../../testdata/controller.yaml"}},
		Local:     true,
		IOStreams: streams,
	}
	err := opts.Complete(tf, cmd, []string{"cassandra=thingy"})
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
		t.Errorf("did not set image: %s", buf.String())
	}
}

func TestSetImageValidation(t *testing.T) {
	printFlags := genericclioptions.NewPrintFlags("").WithTypeSetter(scheme.Scheme)

	testCases := []struct {
		name         string
		imageOptions *SetImageOptions
		expectErr    string
	}{
		{
			name:         "test resource < 1 and filenames empty",
			imageOptions: &SetImageOptions{PrintFlags: printFlags},
			expectErr:    "[one or more resources must be specified as <resource> <name> or <resource>/<name>, at least one image update is required]",
		},
		{
			name: "test containerImages < 1",
			imageOptions: &SetImageOptions{
				PrintFlags: printFlags,
				Resources:  []string{"a", "b", "c"},

				FilenameOptions: resource.FilenameOptions{
					Filenames: []string{"testFile"},
				},
			},
			expectErr: "at least one image update is required",
		},
		{
			name: "test containerImages > 1 and all containers are already specified by *",
			imageOptions: &SetImageOptions{
				PrintFlags: printFlags,
				Resources:  []string{"a", "b", "c"},
				FilenameOptions: resource.FilenameOptions{
					Filenames: []string{"testFile"},
				},
				ContainerImages: map[string]string{
					"test": "test",
					"*":    "test",
				},
			},
			expectErr: "all containers are already specified by *, but saw more than one container_name=container_image pairs",
		},
		{
			name: "success case",
			imageOptions: &SetImageOptions{
				PrintFlags: printFlags,
				Resources:  []string{"a", "b", "c"},
				FilenameOptions: resource.FilenameOptions{
					Filenames: []string{"testFile"},
				},
				ContainerImages: map[string]string{
					"test": "test",
				},
			},
			expectErr: "",
		},
	}
	for _, testCase := range testCases {
		err := testCase.imageOptions.Validate()
		if err != nil {
			if err.Error() != testCase.expectErr {
				t.Errorf("[%s]:expect err:%s got err:%s", testCase.name, testCase.expectErr, err.Error())
			}
		}
		if err == nil && (testCase.expectErr != "") {
			t.Errorf("[%s]:expect err:%s got err:%v", testCase.name, testCase.expectErr, err)
		}
	}
}

func TestSetMultiResourcesImageLocal(t *testing.T) {
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

	streams, _, buf, _ := genericiooptions.NewTestIOStreams()
	cmd := NewCmdImage(tf, streams)
	cmd.SetOut(buf)
	cmd.SetErr(buf)
	cmd.Flags().Set("output", outputFormat)
	cmd.Flags().Set("local", "true")

	opts := SetImageOptions{
		PrintFlags: genericclioptions.NewPrintFlags("").WithDefaultOutput(outputFormat).WithTypeSetter(scheme.Scheme),
		FilenameOptions: resource.FilenameOptions{
			Filenames: []string{"../../../testdata/set/multi-resource-yaml.yaml"}},
		Local:     true,
		IOStreams: streams,
	}
	err := opts.Complete(tf, cmd, []string{"*=thingy"})
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

func TestSetImageRemote(t *testing.T) {
	inputs := []struct {
		name         string
		object       runtime.Object
		groupVersion schema.GroupVersion
		path         string
		args         []string
	}{
		{
			name: "set image extensionsv1beta1 ReplicaSet",
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
							InitContainers: []corev1.Container{
								{
									Name:  "busybox",
									Image: "busybox",
								},
							},
						},
					},
				},
			},
			groupVersion: extensionsv1beta1.SchemeGroupVersion,
			path:         "/namespaces/test/replicasets/nginx",
			args:         []string{"replicaset", "nginx", "*=thingy"},
		},
		{
			name: "set image appsv1beta2 ReplicaSet",
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
							InitContainers: []corev1.Container{
								{
									Name:  "busybox",
									Image: "busybox",
								},
							},
						},
					},
				},
			},
			groupVersion: appsv1beta2.SchemeGroupVersion,
			path:         "/namespaces/test/replicasets/nginx",
			args:         []string{"replicaset", "nginx", "*=thingy"},
		},
		{
			name: "set image appsv1 ReplicaSet",
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
							InitContainers: []corev1.Container{
								{
									Name:  "busybox",
									Image: "busybox",
								},
							},
						},
					},
				},
			},
			groupVersion: appsv1.SchemeGroupVersion,
			path:         "/namespaces/test/replicasets/nginx",
			args:         []string{"replicaset", "nginx", "*=thingy"},
		},
		{
			name: "set image extensionsv1beta1 DaemonSet",
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
							InitContainers: []corev1.Container{
								{
									Name:  "busybox",
									Image: "busybox",
								},
							},
						},
					},
				},
			},
			groupVersion: extensionsv1beta1.SchemeGroupVersion,
			path:         "/namespaces/test/daemonsets/nginx",
			args:         []string{"daemonset", "nginx", "*=thingy"},
		},
		{
			name: "set image appsv1beta2 DaemonSet",
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
							InitContainers: []corev1.Container{
								{
									Name:  "busybox",
									Image: "busybox",
								},
							},
						},
					},
				},
			},
			groupVersion: appsv1beta2.SchemeGroupVersion,
			path:         "/namespaces/test/daemonsets/nginx",
			args:         []string{"daemonset", "nginx", "*=thingy"},
		},
		{
			name: "set image appsv1 DaemonSet",
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
							InitContainers: []corev1.Container{
								{
									Name:  "busybox",
									Image: "busybox",
								},
							},
						},
					},
				},
			},
			groupVersion: appsv1.SchemeGroupVersion,
			path:         "/namespaces/test/daemonsets/nginx",
			args:         []string{"daemonset", "nginx", "*=thingy"},
		},
		{
			name: "set image extensionsv1beta1 Deployment",
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
							InitContainers: []corev1.Container{
								{
									Name:  "busybox",
									Image: "busybox",
								},
							},
						},
					},
				},
			},
			groupVersion: extensionsv1beta1.SchemeGroupVersion,
			path:         "/namespaces/test/deployments/nginx",
			args:         []string{"deployment", "nginx", "*=thingy"},
		},
		{
			name: "set image appsv1beta1 Deployment",
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
							InitContainers: []corev1.Container{
								{
									Name:  "busybox",
									Image: "busybox",
								},
							},
						},
					},
				},
			},
			groupVersion: appsv1beta1.SchemeGroupVersion,
			path:         "/namespaces/test/deployments/nginx",
			args:         []string{"deployment", "nginx", "*=thingy"},
		},
		{
			name: "set image appsv1beta2 Deployment",
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
							InitContainers: []corev1.Container{
								{
									Name:  "busybox",
									Image: "busybox",
								},
							},
						},
					},
				},
			},
			groupVersion: appsv1beta2.SchemeGroupVersion,
			path:         "/namespaces/test/deployments/nginx",
			args:         []string{"deployment", "nginx", "*=thingy"},
		},
		{
			name: "set image appsv1 Deployment",
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
							InitContainers: []corev1.Container{
								{
									Name:  "busybox",
									Image: "busybox",
								},
							},
						},
					},
				},
			},
			groupVersion: appsv1.SchemeGroupVersion,
			path:         "/namespaces/test/deployments/nginx",
			args:         []string{"deployment", "nginx", "*=thingy"},
		},
		{
			name: "set image appsv1beta1 StatefulSet",
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
							InitContainers: []corev1.Container{
								{
									Name:  "busybox",
									Image: "busybox",
								},
							},
						},
					},
				},
			},
			groupVersion: appsv1beta1.SchemeGroupVersion,
			path:         "/namespaces/test/statefulsets/nginx",
			args:         []string{"statefulset", "nginx", "*=thingy"},
		},
		{
			name: "set image appsv1beta2 StatefulSet",
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
							InitContainers: []corev1.Container{
								{
									Name:  "busybox",
									Image: "busybox",
								},
							},
						},
					},
				},
			},
			groupVersion: appsv1beta2.SchemeGroupVersion,
			path:         "/namespaces/test/statefulsets/nginx",
			args:         []string{"statefulset", "nginx", "*=thingy"},
		},
		{
			name: "set image appsv1 StatefulSet",
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
							InitContainers: []corev1.Container{
								{
									Name:  "busybox",
									Image: "busybox",
								},
							},
						},
					},
				},
			},
			groupVersion: appsv1.SchemeGroupVersion,
			path:         "/namespaces/test/statefulsets/nginx",
			args:         []string{"statefulset", "nginx", "*=thingy"},
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
									InitContainers: []corev1.Container{
										{
											Name:  "busybox",
											Image: "busybox",
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
			args:         []string{"cronjob", "nginx", "*=thingy"},
		},
		{
			name: "set image corev1.ReplicationController",
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
							InitContainers: []corev1.Container{
								{
									Name:  "busybox",
									Image: "busybox",
								},
							},
						},
					},
				},
			},
			groupVersion: corev1.SchemeGroupVersion,
			path:         "/namespaces/test/replicationcontrollers/nginx",
			args:         []string{"replicationcontroller", "nginx", "*=thingy"},
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
						bytes, err := io.ReadAll(stream)
						if err != nil {
							return nil, err
						}
						assert.Contains(t, string(bytes), `"image":`+`"`+"thingy"+`"`, fmt.Sprintf("image not updated for %#v", input.object))
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: objBody(input.object)}, nil
					default:
						t.Errorf("%s: unexpected request: %s %#v\n%#v", "image", req.Method, req.URL, req)
						return nil, fmt.Errorf("unexpected request")
					}
				}),
			}

			outputFormat := "yaml"

			streams := genericiooptions.NewTestIOStreamsDiscard()
			cmd := NewCmdImage(tf, streams)
			cmd.Flags().Set("output", outputFormat)
			opts := SetImageOptions{
				PrintFlags: genericclioptions.NewPrintFlags("").WithDefaultOutput(outputFormat).WithTypeSetter(scheme.Scheme),

				Local:     false,
				IOStreams: streams,
			}
			err := opts.Complete(tf, cmd, input.args)
			assert.NoError(t, err)
			err = opts.Run()
			assert.NoError(t, err)
		})
	}
}

func TestSetImageRemoteWithSpecificContainers(t *testing.T) {
	inputs := []struct {
		name         string
		object       runtime.Object
		groupVersion schema.GroupVersion
		path         string
		args         []string
	}{
		{
			name: "set container image only",
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
							InitContainers: []corev1.Container{
								{
									Name:  "busybox",
									Image: "busybox",
								},
							},
						},
					},
				},
			},
			groupVersion: extensionsv1beta1.SchemeGroupVersion,
			path:         "/namespaces/test/replicasets/nginx",
			args:         []string{"replicaset", "nginx", "nginx=thingy"},
		},
		{
			name: "set initContainer image only",
			object: &appsv1beta2.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{Name: "nginx"},
				Spec: appsv1beta2.ReplicaSetSpec{
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Name:  "busybox",
									Image: "busybox",
								},
							},
							InitContainers: []corev1.Container{
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
			args:         []string{"replicaset", "nginx", "nginx=thingy"},
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
						bytes, err := io.ReadAll(stream)
						if err != nil {
							return nil, err
						}
						assert.Contains(t, string(bytes), `"image":"`+"thingy"+`","name":`+`"nginx"`, fmt.Sprintf("image not updated for %#v", input.object))
						assert.NotContains(t, string(bytes), `"image":"`+"thingy"+`","name":`+`"busybox"`, fmt.Sprintf("image updated for %#v", input.object))
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: objBody(input.object)}, nil
					default:
						t.Errorf("%s: unexpected request: %s %#v\n%#v", "image", req.Method, req.URL, req)
						return nil, fmt.Errorf("unexpected request")
					}
				}),
			}

			outputFormat := "yaml"

			streams := genericiooptions.NewTestIOStreamsDiscard()
			cmd := NewCmdImage(tf, streams)
			cmd.Flags().Set("output", outputFormat)
			opts := SetImageOptions{
				PrintFlags: genericclioptions.NewPrintFlags("").WithDefaultOutput(outputFormat).WithTypeSetter(scheme.Scheme),

				Local:     false,
				IOStreams: streams,
			}
			err := opts.Complete(tf, cmd, input.args)
			assert.NoError(t, err)
			err = opts.Run()
			assert.NoError(t, err)
		})
	}
}

func TestSetImageResolver(t *testing.T) {
	f := func(in string) (string, error) {
		return "custom", nil
	}

	ImageResolver = f

	out, err := ImageResolver("my-image")
	if err != nil {
		t.Errorf("unexpected error from ImageResolver: %v", err)
	} else if out != "custom" {
		t.Errorf("expected: %s, found: %s", "custom", out)
	}
}
