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

package kubectl

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/extensions"
)

func TestGenerate(t *testing.T) {
	tests := []struct {
		description string
		params      map[string]interface{}
		expected    *api.ReplicationController
		expectErr   bool
	}{
		{
			description: "pullpolicy always 1 replica",
			params: map[string]interface{}{
				"name":              "foo",
				"image":             "someimage",
				"image-pull-policy": "Always",
				"replicas":          "1",
				"port":              "",
			},
			expected: &api.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"run": "foo"},
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 1,
					Selector: map[string]string{"run": "foo"},
					Template: &api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{"run": "foo"},
						},
						Spec: api.PodSpec{
							Containers: []api.Container{
								{
									Name:            "foo",
									Image:           "someimage",
									ImagePullPolicy: api.PullAlways,
								},
							},
						},
					},
				},
			},
		},

		{
			description: "environment variables are passed to podSpec",
			params: map[string]interface{}{
				"name":     "foo",
				"image":    "someimage",
				"replicas": "1",
				"port":     "",
				"env":      []string{"a=b", "c=d"},
			},
			expected: &api.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"run": "foo"},
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 1,
					Selector: map[string]string{"run": "foo"},
					Template: &api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{"run": "foo"},
						},
						Spec: api.PodSpec{
							Containers: []api.Container{
								{
									Name:  "foo",
									Image: "someimage",
									Env: []api.EnvVar{
										{
											Name:  "a",
											Value: "b",
										},
										{
											Name:  "c",
											Value: "d",
										},
									},
								},
							},
						},
					},
				},
			},
		},

		{
			description: "pullPolicy never, args",
			params: map[string]interface{}{
				"name":              "foo",
				"image":             "someimage",
				"image-pull-policy": "Never",
				"replicas":          "1",
				"port":              "",
				"args":              []string{"bar", "baz", "blah"},
			},
			expected: &api.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"run": "foo"},
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 1,
					Selector: map[string]string{"run": "foo"},
					Template: &api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{"run": "foo"},
						},
						Spec: api.PodSpec{
							Containers: []api.Container{
								{
									Name:            "foo",
									Image:           "someimage",
									ImagePullPolicy: api.PullNever,
									Args:            []string{"bar", "baz", "blah"},
								},
							},
						},
					},
				},
			},
		},
		{
			description: "command",
			params: map[string]interface{}{
				"name":     "foo",
				"image":    "someimage",
				"replicas": "1",
				"port":     "",
				"args":     []string{"bar", "baz", "blah"},
				"command":  "true",
			},
			expected: &api.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"run": "foo"},
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 1,
					Selector: map[string]string{"run": "foo"},
					Template: &api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{"run": "foo"},
						},
						Spec: api.PodSpec{
							Containers: []api.Container{
								{
									Name:    "foo",
									Image:   "someimage",
									Command: []string{"bar", "baz", "blah"},
								},
							},
						},
					},
				},
			},
		},
		{
			description: "containerPort",
			params: map[string]interface{}{
				"name":     "foo",
				"image":    "someimage",
				"replicas": "1",
				"port":     "80",
			},
			expected: &api.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"run": "foo"},
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 1,
					Selector: map[string]string{"run": "foo"},
					Template: &api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{"run": "foo"},
						},
						Spec: api.PodSpec{
							Containers: []api.Container{
								{
									Name:  "foo",
									Image: "someimage",
									Ports: []api.ContainerPort{
										{
											ContainerPort: 80,
										},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			description: "hostPort containerPort",
			params: map[string]interface{}{
				"name":              "foo",
				"image":             "someimage",
				"image-pull-policy": "IfNotPresent",
				"replicas":          "1",
				"port":              "80",
				"hostport":          "80",
			},
			expected: &api.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"run": "foo"},
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 1,
					Selector: map[string]string{"run": "foo"},
					Template: &api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{"run": "foo"},
						},
						Spec: api.PodSpec{
							Containers: []api.Container{
								{
									Name:            "foo",
									Image:           "someimage",
									ImagePullPolicy: api.PullIfNotPresent,
									Ports: []api.ContainerPort{
										{
											ContainerPort: 80,
											HostPort:      80,
										},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			description: "labels",
			params: map[string]interface{}{
				"name":     "foo",
				"image":    "someimage",
				"replicas": "1",
				"labels":   "foo=bar,baz=blah",
			},
			expected: &api.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"foo": "bar", "baz": "blah"},
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 1,
					Selector: map[string]string{"foo": "bar", "baz": "blah"},
					Template: &api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{"foo": "bar", "baz": "blah"},
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
			},
		},
		{
			description: "error no name",
			params: map[string]interface{}{
				"name":     "foo",
				"image":    "someimage",
				"replicas": "1",
				"hostport": "80",
			},
			expected:  nil,
			expectErr: true,
		},
		{
			description: "error incorrect requests format no space",
			params: map[string]interface{}{
				"name":     "foo",
				"image":    "someimage",
				"replicas": "1",
				"labels":   "foo=bar,baz=blah",
				"requests": "cpu100m,memory=100Mi",
			},
			expected:  nil,
			expectErr: true,
		},
		{
			description: "error incorrect request format &",
			params: map[string]interface{}{
				"name":     "foo",
				"image":    "someimage",
				"replicas": "1",
				"labels":   "foo=bar,baz=blah",
				"requests": "cpu=100m&memory=100Mi",
			},
			expected:  nil,
			expectErr: true,
		},
		{
			description: "error incorrect request format key no value",
			params: map[string]interface{}{
				"name":     "foo",
				"image":    "someimage",
				"replicas": "1",
				"labels":   "foo=bar,baz=blah",
				"requests": "cpu=",
			},
			expected:  nil,
			expectErr: true,
		},
		{
			description: "requests and limits",
			params: map[string]interface{}{
				"name":     "foo",
				"image":    "someimage",
				"replicas": "1",
				"labels":   "foo=bar,baz=blah",
				"requests": "cpu=100m,memory=100Mi",
				"limits":   "cpu=400m,memory=200Mi",
			},
			expected: &api.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"foo": "bar", "baz": "blah"},
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 1,
					Selector: map[string]string{"foo": "bar", "baz": "blah"},
					Template: &api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{"foo": "bar", "baz": "blah"},
						},
						Spec: api.PodSpec{
							Containers: []api.Container{
								{
									Name:  "foo",
									Image: "someimage",
									Resources: api.ResourceRequirements{
										Requests: api.ResourceList{
											api.ResourceCPU:    resource.MustParse("100m"),
											api.ResourceMemory: resource.MustParse("100Mi"),
										},
										Limits: api.ResourceList{
											api.ResourceCPU:    resource.MustParse("400m"),
											api.ResourceMemory: resource.MustParse("200Mi"),
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}
	generator := BasicReplicationController{}
	for _, test := range tests {
		t.Run(test.description, func(t *testing.T) {
			t.Parallel()
			obj, err := generator.Generate(test.params)
			if !test.expectErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if test.expectErr && err != nil {
				return
			}
			if !reflect.DeepEqual(obj.(*api.ReplicationController).Spec.Template, test.expected.Spec.Template) {
				t.Errorf("\nexpected:\n%#v\nsaw:\n%#v", test.expected.Spec.Template, obj.(*api.ReplicationController).Spec.Template)
			}
		})
	}
}

func TestGeneratePod(t *testing.T) {
	tests := []struct {
		description string
		params      map[string]interface{}
		expected    *api.Pod
		expectErr   bool
	}{
		{
			description: "default generate pod",
			params: map[string]interface{}{
				"name":  "foo",
				"image": "someimage",
				"port":  "",
			},
			expected: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:            "foo",
							Image:           "someimage",
							ImagePullPolicy: api.PullIfNotPresent,
						},
					},
					DNSPolicy:     api.DNSClusterFirst,
					RestartPolicy: api.RestartPolicyAlways,
				},
			},
		},
		{
			description: "error env vars key no value",
			params: map[string]interface{}{
				"name":  "foo",
				"image": "someimage",
				"env":   []string{"a", "c"},
			},

			expected:  nil,
			expectErr: true,
		},
		{
			description: "pull policy always, env vars",
			params: map[string]interface{}{
				"name":              "foo",
				"image":             "someimage",
				"image-pull-policy": "Always",
				"env":               []string{"a=b", "c=d"},
			},
			expected: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:            "foo",
							Image:           "someimage",
							ImagePullPolicy: api.PullAlways,
							Env: []api.EnvVar{
								{
									Name:  "a",
									Value: "b",
								},
								{
									Name:  "c",
									Value: "d",
								},
							},
						},
					},
					DNSPolicy:     api.DNSClusterFirst,
					RestartPolicy: api.RestartPolicyAlways,
				},
			},
		},
		{
			description: "port",
			params: map[string]interface{}{
				"name":  "foo",
				"image": "someimage",
				"port":  "80",
			},
			expected: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:            "foo",
							Image:           "someimage",
							ImagePullPolicy: api.PullIfNotPresent,
							Ports: []api.ContainerPort{
								{
									ContainerPort: 80,
								},
							},
						},
					},
					DNSPolicy:     api.DNSClusterFirst,
					RestartPolicy: api.RestartPolicyAlways,
				},
			},
		},
		{
			description: "port and hostport",
			params: map[string]interface{}{
				"name":     "foo",
				"image":    "someimage",
				"port":     "80",
				"hostport": "80",
			},
			expected: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:            "foo",
							Image:           "someimage",
							ImagePullPolicy: api.PullIfNotPresent,
							Ports: []api.ContainerPort{
								{
									ContainerPort: 80,
									HostPort:      80,
								},
							},
						},
					},
					DNSPolicy:     api.DNSClusterFirst,
					RestartPolicy: api.RestartPolicyAlways,
				},
			},
		},
		{
			description: "hostport",
			params: map[string]interface{}{
				"name":     "foo",
				"image":    "someimage",
				"hostport": "80",
			},
			expected:  nil,
			expectErr: true,
		},
		{
			description: "labels",
			params: map[string]interface{}{
				"name":     "foo",
				"image":    "someimage",
				"replicas": "1",
				"labels":   "foo=bar,baz=blah",
			},
			expected: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"foo": "bar", "baz": "blah"},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:            "foo",
							Image:           "someimage",
							ImagePullPolicy: api.PullIfNotPresent,
						},
					},
					DNSPolicy:     api.DNSClusterFirst,
					RestartPolicy: api.RestartPolicyAlways,
				},
			},
		},
		{
			description: "from stdin",
			params: map[string]interface{}{
				"name":     "foo",
				"image":    "someimage",
				"replicas": "1",
				"labels":   "foo=bar,baz=blah",
				"stdin":    "true",
			},
			expected: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"foo": "bar", "baz": "blah"},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:            "foo",
							Image:           "someimage",
							ImagePullPolicy: api.PullIfNotPresent,
							Stdin:           true,
							StdinOnce:       true,
						},
					},
					DNSPolicy:     api.DNSClusterFirst,
					RestartPolicy: api.RestartPolicyAlways,
				},
			},
		},
		{
			description: "leave stdin open",
			params: map[string]interface{}{
				"name":             "foo",
				"image":            "someimage",
				"replicas":         "1",
				"labels":           "foo=bar,baz=blah",
				"stdin":            "true",
				"leave-stdin-open": "true",
			},
			expected: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"foo": "bar", "baz": "blah"},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:            "foo",
							Image:           "someimage",
							ImagePullPolicy: api.PullIfNotPresent,
							Stdin:           true,
							StdinOnce:       false,
						},
					},
					DNSPolicy:     api.DNSClusterFirst,
					RestartPolicy: api.RestartPolicyAlways,
				},
			},
		},
	}
	generator := BasicPod{}
	for _, test := range tests {
		t.Run(test.description, func(t *testing.T) {
			t.Parallel()
			obj, err := generator.Generate(test.params)
			if !test.expectErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if test.expectErr && err != nil {
				return
			}
			if !reflect.DeepEqual(obj.(*api.Pod), test.expected) {
				t.Errorf("\nexpected:\n%#v\nsaw:\n%#v", test.expected, obj.(*api.Pod))
			}
		})
	}
}

func TestGenerateDeployment(t *testing.T) {
	tests := []struct {
		description string
		params      map[string]interface{}
		expected    *extensions.Deployment
		expectErr   bool
	}{
		{
			description: "multiple options",
			params: map[string]interface{}{
				"labels":            "foo=bar,baz=blah",
				"name":              "foo",
				"replicas":          "3",
				"image":             "someimage",
				"image-pull-policy": "Always",
				"port":              "80",
				"hostport":          "80",
				"stdin":             "true",
				"command":           "true",
				"args":              []string{"bar", "baz", "blah"},
				"env":               []string{"a=b", "c=d"},
				"requests":          "cpu=100m,memory=100Mi",
				"limits":            "cpu=400m,memory=200Mi",
			},
			expected: &extensions.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"foo": "bar", "baz": "blah"},
				},
				Spec: extensions.DeploymentSpec{
					Replicas: 3,
					Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar", "baz": "blah"}},
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{"foo": "bar", "baz": "blah"},
						},
						Spec: api.PodSpec{
							Containers: []api.Container{
								{
									Name:            "foo",
									Image:           "someimage",
									ImagePullPolicy: api.PullAlways,
									Stdin:           true,
									Ports: []api.ContainerPort{
										{
											ContainerPort: 80,
											HostPort:      80,
										},
									},
									Command: []string{"bar", "baz", "blah"},
									Env: []api.EnvVar{
										{
											Name:  "a",
											Value: "b",
										},
										{
											Name:  "c",
											Value: "d",
										},
									},
									Resources: api.ResourceRequirements{
										Requests: api.ResourceList{
											api.ResourceCPU:    resource.MustParse("100m"),
											api.ResourceMemory: resource.MustParse("100Mi"),
										},
										Limits: api.ResourceList{
											api.ResourceCPU:    resource.MustParse("400m"),
											api.ResourceMemory: resource.MustParse("200Mi"),
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	generator := DeploymentV1Beta1{}
	for _, test := range tests {
		t.Run(test.description, func(t *testing.T) {
			t.Parallel()
			obj, err := generator.Generate(test.params)
			if !test.expectErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if test.expectErr && err != nil {
				return
			}
			if !reflect.DeepEqual(obj.(*extensions.Deployment), test.expected) {
				t.Errorf("\nexpected:\n%#v\nsaw:\n%#v", test.expected, obj.(*extensions.Deployment))
			}
		})
	}
}

func TestGenerateJob(t *testing.T) {
	tests := []struct {
		description string
		params      map[string]interface{}
		expected    *batch.Job
		expectErr   bool
	}{
		{
			description: "multiple options",
			params: map[string]interface{}{
				"labels":           "foo=bar,baz=blah",
				"name":             "foo",
				"image":            "someimage",
				"port":             "80",
				"hostport":         "80",
				"stdin":            "true",
				"leave-stdin-open": "true",
				"command":          "true",
				"args":             []string{"bar", "baz", "blah"},
				"env":              []string{"a=b", "c=d"},
				"requests":         "cpu=100m,memory=100Mi",
				"limits":           "cpu=400m,memory=200Mi",
				"restart":          "OnFailure",
			},
			expected: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"foo": "bar", "baz": "blah"},
				},
				Spec: batch.JobSpec{
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{"foo": "bar", "baz": "blah"},
						},
						Spec: api.PodSpec{
							RestartPolicy: api.RestartPolicyOnFailure,
							Containers: []api.Container{
								{
									Name:      "foo",
									Image:     "someimage",
									Stdin:     true,
									StdinOnce: false,
									Ports: []api.ContainerPort{
										{
											ContainerPort: 80,
											HostPort:      80,
										},
									},
									Command: []string{"bar", "baz", "blah"},
									Env: []api.EnvVar{
										{
											Name:  "a",
											Value: "b",
										},
										{
											Name:  "c",
											Value: "d",
										},
									},
									Resources: api.ResourceRequirements{
										Requests: api.ResourceList{
											api.ResourceCPU:    resource.MustParse("100m"),
											api.ResourceMemory: resource.MustParse("100Mi"),
										},
										Limits: api.ResourceList{
											api.ResourceCPU:    resource.MustParse("400m"),
											api.ResourceMemory: resource.MustParse("200Mi"),
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	generator := JobV1{}
	for _, test := range tests {
		t.Run(test.description, func(t *testing.T) {
			t.Parallel()
			obj, err := generator.Generate(test.params)
			if !test.expectErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if test.expectErr && err != nil {
				return
			}
			if !reflect.DeepEqual(obj.(*batch.Job), test.expected) {
				t.Errorf("\nexpected:\n%#v\nsaw:\n%#v", test.expected, obj.(*batch.Job))
			}
		})
	}
}

func TestGenerateCronJob(t *testing.T) {
	tests := []struct {
		description string
		params      map[string]interface{}
		expected    *batch.CronJob
		expectErr   bool
	}{
		{
			description: "multiple options",
			params: map[string]interface{}{
				"labels":           "foo=bar,baz=blah",
				"name":             "foo",
				"image":            "someimage",
				"port":             "80",
				"hostport":         "80",
				"stdin":            "true",
				"leave-stdin-open": "true",
				"command":          "true",
				"args":             []string{"bar", "baz", "blah"},
				"env":              []string{"a=b", "c=d"},
				"requests":         "cpu=100m,memory=100Mi",
				"limits":           "cpu=400m,memory=200Mi",
				"restart":          "OnFailure",
				"schedule":         "0/5 * * * ?",
			},
			expected: &batch.CronJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"foo": "bar", "baz": "blah"},
				},
				Spec: batch.CronJobSpec{
					Schedule:          "0/5 * * * ?",
					ConcurrencyPolicy: batch.AllowConcurrent,
					JobTemplate: batch.JobTemplateSpec{
						Spec: batch.JobSpec{
							Template: api.PodTemplateSpec{
								ObjectMeta: metav1.ObjectMeta{
									Labels: map[string]string{"foo": "bar", "baz": "blah"},
								},
								Spec: api.PodSpec{
									RestartPolicy: api.RestartPolicyOnFailure,
									Containers: []api.Container{
										{
											Name:      "foo",
											Image:     "someimage",
											Stdin:     true,
											StdinOnce: false,
											Ports: []api.ContainerPort{
												{
													ContainerPort: 80,
													HostPort:      80,
												},
											},
											Command: []string{"bar", "baz", "blah"},
											Env: []api.EnvVar{
												{
													Name:  "a",
													Value: "b",
												},
												{
													Name:  "c",
													Value: "d",
												},
											},
											Resources: api.ResourceRequirements{
												Requests: api.ResourceList{
													api.ResourceCPU:    resource.MustParse("100m"),
													api.ResourceMemory: resource.MustParse("100Mi"),
												},
												Limits: api.ResourceList{
													api.ResourceCPU:    resource.MustParse("400m"),
													api.ResourceMemory: resource.MustParse("200Mi"),
												},
											},
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	generator := CronJobV2Alpha1{}
	for _, test := range tests {
		t.Run(test.description, func(t *testing.T) {
			obj, err := generator.Generate(test.params)
			if !test.expectErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if test.expectErr && err != nil {
				return
			}
			if !reflect.DeepEqual(obj.(*batch.CronJob), test.expected) {
				t.Errorf("\nexpected:\n%#v\nsaw:\n%#v", test.expected, obj.(*batch.CronJob))
			}
		})
	}
}

func TestParseEnv(t *testing.T) {
	tests := []struct {
		description string
		envArray    []string
		expected    []api.EnvVar
		expectErr   bool
	}{
		{
			description: "no special characters",
			envArray: []string{
				"THIS_ENV=isOK",
			},
			expected: []api.EnvVar{
				{
					Name:  "THIS_ENV",
					Value: "isOK",
				},
			},
		},
		{
			description: "with commas",
			envArray: []string{
				"HAS_COMMAS=foo,bar",
			},
			expected: []api.EnvVar{
				{
					Name:  "HAS_COMMAS",
					Value: "foo,bar",
				},
			},
		},
		{
			description: "has equals",
			envArray: []string{
				"HAS_EQUALS=jJnro54iUu75xNy==",
			},
			expected: []api.EnvVar{
				{
					Name:  "HAS_EQUALS",
					Value: "jJnro54iUu75xNy==",
				},
			},
		},
		{
			description: "multiple envs",
			envArray: []string{
				"ENV_1=one",
				"ENV_2=two",
			},
			expected: []api.EnvVar{
				{
					Name:  "ENV_1",
					Value: "one",
				},
				{
					Name:  "ENV_2",
					Value: "two",
				},
			},
		},
		{
			description: "error key no value",
			envArray: []string{
				"WITH_OUT_EQUALS",
			},
			expected:  []api.EnvVar{},
			expectErr: true,
		},
		{
			description: "error key equals no value",
			envArray: []string{
				"WITH_OUT_VALUES=",
			},
			expected: []api.EnvVar{
				{
					Name:  "WITH_OUT_VALUES",
					Value: "",
				},
			},
			expectErr: false,
		},
		{
			description: "error value no key",
			envArray: []string{
				"=WITH_OUT_NAME",
			},
			expected:  []api.EnvVar{},
			expectErr: true,
		},
	}

	for _, test := range tests {
		t.Run(test.description, func(t *testing.T) {
			t.Parallel()
			envs, err := parseEnvs(test.envArray)
			if !test.expectErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if test.expectErr && err != nil {
				return
			}
			if !reflect.DeepEqual(envs, test.expected) {
				t.Errorf("\nexpected:\n%#v\nsaw:\n%#v", test.expected, envs)
			}
		})
	}
}
