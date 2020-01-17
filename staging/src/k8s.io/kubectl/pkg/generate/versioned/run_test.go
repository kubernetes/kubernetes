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

package versioned

import (
	"reflect"
	"testing"

	appsv1beta1 "k8s.io/api/apps/v1beta1"
	batchv1 "k8s.io/api/batch/v1"
	batchv1beta1 "k8s.io/api/batch/v1beta1"
	batchv2alpha1 "k8s.io/api/batch/v2alpha1"
	"k8s.io/api/core/v1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestGenerate(t *testing.T) {
	one := int32(1)
	tests := []struct {
		name      string
		params    map[string]interface{}
		expected  *v1.ReplicationController
		expectErr bool
	}{
		{
			name: "test1",
			params: map[string]interface{}{
				"name":              "foo",
				"image":             "someimage",
				"image-pull-policy": "Always",
				"replicas":          "1",
				"port":              "",
			},
			expected: &v1.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"run": "foo"},
				},
				Spec: v1.ReplicationControllerSpec{
					Replicas: &one,
					Selector: map[string]string{"run": "foo"},
					Template: &v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{"run": "foo"},
						},
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								{
									Name:            "foo",
									Image:           "someimage",
									ImagePullPolicy: v1.PullAlways,
								},
							},
						},
					},
				},
			},
		},

		{
			name: "test2",
			params: map[string]interface{}{
				"name":     "foo",
				"image":    "someimage",
				"replicas": "1",
				"port":     "",
				"env":      []string{"a=b", "c=d"},
			},
			expected: &v1.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"run": "foo"},
				},
				Spec: v1.ReplicationControllerSpec{
					Replicas: &one,
					Selector: map[string]string{"run": "foo"},
					Template: &v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{"run": "foo"},
						},
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								{
									Name:  "foo",
									Image: "someimage",
									Env: []v1.EnvVar{
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
			name: "test3",
			params: map[string]interface{}{
				"name":              "foo",
				"image":             "someimage",
				"image-pull-policy": "Never",
				"replicas":          "1",
				"port":              "",
				"args":              []string{"bar", "baz", "blah"},
			},
			expected: &v1.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"run": "foo"},
				},
				Spec: v1.ReplicationControllerSpec{
					Replicas: &one,
					Selector: map[string]string{"run": "foo"},
					Template: &v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{"run": "foo"},
						},
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								{
									Name:            "foo",
									Image:           "someimage",
									ImagePullPolicy: v1.PullNever,
									Args:            []string{"bar", "baz", "blah"},
								},
							},
						},
					},
				},
			},
		},
		{
			name: "test3",
			params: map[string]interface{}{
				"name":     "foo",
				"image":    "someimage",
				"replicas": "1",
				"port":     "",
				"args":     []string{"bar", "baz", "blah"},
				"command":  "true",
			},
			expected: &v1.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"run": "foo"},
				},
				Spec: v1.ReplicationControllerSpec{
					Replicas: &one,
					Selector: map[string]string{"run": "foo"},
					Template: &v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{"run": "foo"},
						},
						Spec: v1.PodSpec{
							Containers: []v1.Container{
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
			name: "test4",
			params: map[string]interface{}{
				"name":     "foo",
				"image":    "someimage",
				"replicas": "1",
				"port":     "80",
			},
			expected: &v1.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"run": "foo"},
				},
				Spec: v1.ReplicationControllerSpec{
					Replicas: &one,
					Selector: map[string]string{"run": "foo"},
					Template: &v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{"run": "foo"},
						},
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								{
									Name:  "foo",
									Image: "someimage",
									Ports: []v1.ContainerPort{
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
			name: "test5",
			params: map[string]interface{}{
				"name":              "foo",
				"image":             "someimage",
				"image-pull-policy": "IfNotPresent",
				"replicas":          "1",
				"port":              "80",
				"hostport":          "80",
			},
			expected: &v1.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"run": "foo"},
				},
				Spec: v1.ReplicationControllerSpec{
					Replicas: &one,
					Selector: map[string]string{"run": "foo"},
					Template: &v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{"run": "foo"},
						},
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								{
									Name:            "foo",
									Image:           "someimage",
									ImagePullPolicy: v1.PullIfNotPresent,
									Ports: []v1.ContainerPort{
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
			name: "test6",
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
			name: "test7",
			params: map[string]interface{}{
				"name":     "foo",
				"image":    "someimage",
				"replicas": "1",
				"labels":   "foo=bar,baz=blah",
			},
			expected: &v1.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"foo": "bar", "baz": "blah"},
				},
				Spec: v1.ReplicationControllerSpec{
					Replicas: &one,
					Selector: map[string]string{"foo": "bar", "baz": "blah"},
					Template: &v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{"foo": "bar", "baz": "blah"},
						},
						Spec: v1.PodSpec{
							Containers: []v1.Container{
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
			name: "test8",
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
			name: "test9",
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
			name: "test10",
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
			name: "test11",
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
			name: "test12",
			params: map[string]interface{}{
				"name":     "foo",
				"image":    "someimage",
				"replicas": "1",
				"labels":   "foo=bar,baz=blah",
				"requests": "cpu=100m,memory=100Mi",
				"limits":   "cpu=400m,memory=200Mi",
			},
			expected: &v1.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"foo": "bar", "baz": "blah"},
				},
				Spec: v1.ReplicationControllerSpec{
					Replicas: &one,
					Selector: map[string]string{"foo": "bar", "baz": "blah"},
					Template: &v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{"foo": "bar", "baz": "blah"},
						},
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								{
									Name:  "foo",
									Image: "someimage",
									Resources: v1.ResourceRequirements{
										Requests: v1.ResourceList{
											v1.ResourceCPU:    resource.MustParse("100m"),
											v1.ResourceMemory: resource.MustParse("100Mi"),
										},
										Limits: v1.ResourceList{
											v1.ResourceCPU:    resource.MustParse("400m"),
											v1.ResourceMemory: resource.MustParse("200Mi"),
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
	for i, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			obj, err := generator.Generate(tt.params)
			t.Logf("%d: %#v", i, obj)
			if !tt.expectErr && err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}
			if tt.expectErr && err != nil {
				return
			}
			if !reflect.DeepEqual(obj.(*v1.ReplicationController).Spec.Template, tt.expected.Spec.Template) {
				t.Errorf("\nexpected:\n%#v\nsaw:\n%#v", tt.expected.Spec.Template, obj.(*v1.ReplicationController).Spec.Template)
			}
		})
	}
}

func TestGeneratePod(t *testing.T) {
	tests := []struct {
		name      string
		params    map[string]interface{}
		expected  *v1.Pod
		expectErr bool
	}{
		{
			name: "test1",
			params: map[string]interface{}{
				"name":  "foo",
				"image": "someimage",
				"port":  "",
			},
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"run": "foo"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "foo",
							Image: "someimage",
						},
					},
					DNSPolicy:     v1.DNSClusterFirst,
					RestartPolicy: v1.RestartPolicyAlways,
				},
			},
		},
		{
			name: "test2",
			params: map[string]interface{}{
				"name":  "foo",
				"image": "someimage",
				"env":   []string{"a", "c"},
			},

			expected:  nil,
			expectErr: true,
		},
		{
			name: "test3",
			params: map[string]interface{}{
				"name":              "foo",
				"image":             "someimage",
				"image-pull-policy": "Always",
				"env":               []string{"a=b", "c=d"},
			},
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"run": "foo"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:            "foo",
							Image:           "someimage",
							ImagePullPolicy: v1.PullAlways,
							Env: []v1.EnvVar{
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
					DNSPolicy:     v1.DNSClusterFirst,
					RestartPolicy: v1.RestartPolicyAlways,
				},
			},
		},
		{
			name: "test4",
			params: map[string]interface{}{
				"name":  "foo",
				"image": "someimage",
				"port":  "80",
			},
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"run": "foo"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "foo",
							Image: "someimage",
							Ports: []v1.ContainerPort{
								{
									ContainerPort: 80,
								},
							},
						},
					},
					DNSPolicy:     v1.DNSClusterFirst,
					RestartPolicy: v1.RestartPolicyAlways,
				},
			},
		},
		{
			name: "test5",
			params: map[string]interface{}{
				"name":     "foo",
				"image":    "someimage",
				"port":     "80",
				"hostport": "80",
			},
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"run": "foo"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "foo",
							Image: "someimage",
							Ports: []v1.ContainerPort{
								{
									ContainerPort: 80,
									HostPort:      80,
								},
							},
						},
					},
					DNSPolicy:     v1.DNSClusterFirst,
					RestartPolicy: v1.RestartPolicyAlways,
				},
			},
		},
		{
			name: "test6",
			params: map[string]interface{}{
				"name":     "foo",
				"image":    "someimage",
				"hostport": "80",
			},
			expected:  nil,
			expectErr: true,
		},
		{
			name: "test7",
			params: map[string]interface{}{
				"name":     "foo",
				"image":    "someimage",
				"replicas": "1",
				"labels":   "foo=bar,baz=blah",
			},
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"foo": "bar", "baz": "blah"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "foo",
							Image: "someimage",
						},
					},
					DNSPolicy:     v1.DNSClusterFirst,
					RestartPolicy: v1.RestartPolicyAlways,
				},
			},
		},
		{
			name: "test8",
			params: map[string]interface{}{
				"name":     "foo",
				"image":    "someimage",
				"replicas": "1",
				"labels":   "foo=bar,baz=blah",
				"stdin":    "true",
			},
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"foo": "bar", "baz": "blah"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:      "foo",
							Image:     "someimage",
							Stdin:     true,
							StdinOnce: true,
						},
					},
					DNSPolicy:     v1.DNSClusterFirst,
					RestartPolicy: v1.RestartPolicyAlways,
				},
			},
		},
		{
			name: "test9",
			params: map[string]interface{}{
				"name":             "foo",
				"image":            "someimage",
				"replicas":         "1",
				"labels":           "foo=bar,baz=blah",
				"stdin":            "true",
				"leave-stdin-open": "true",
			},
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"foo": "bar", "baz": "blah"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:      "foo",
							Image:     "someimage",
							Stdin:     true,
							StdinOnce: false,
						},
					},
					DNSPolicy:     v1.DNSClusterFirst,
					RestartPolicy: v1.RestartPolicyAlways,
				},
			},
		},
	}
	generator := BasicPod{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			obj, err := generator.Generate(tt.params)
			if !tt.expectErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if tt.expectErr && err != nil {
				return
			}
			if !reflect.DeepEqual(obj.(*v1.Pod), tt.expected) {
				t.Errorf("\nexpected:\n%#v\nsaw:\n%#v", tt.expected, obj.(*v1.Pod))
			}
		})
	}
}

func TestGenerateDeployment(t *testing.T) {
	three := int32(3)
	tests := []struct {
		name      string
		params    map[string]interface{}
		expected  *extensionsv1beta1.Deployment
		expectErr bool
	}{
		{
			name: "test1",
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
			expected: &extensionsv1beta1.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"foo": "bar", "baz": "blah"},
				},
				Spec: extensionsv1beta1.DeploymentSpec{
					Replicas: &three,
					Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar", "baz": "blah"}},
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{"foo": "bar", "baz": "blah"},
						},
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								{
									Name:            "foo",
									Image:           "someimage",
									ImagePullPolicy: v1.PullAlways,
									Stdin:           true,
									Ports: []v1.ContainerPort{
										{
											ContainerPort: 80,
											HostPort:      80,
										},
									},
									Command: []string{"bar", "baz", "blah"},
									Env: []v1.EnvVar{
										{
											Name:  "a",
											Value: "b",
										},
										{
											Name:  "c",
											Value: "d",
										},
									},
									Resources: v1.ResourceRequirements{
										Requests: v1.ResourceList{
											v1.ResourceCPU:    resource.MustParse("100m"),
											v1.ResourceMemory: resource.MustParse("100Mi"),
										},
										Limits: v1.ResourceList{
											v1.ResourceCPU:    resource.MustParse("400m"),
											v1.ResourceMemory: resource.MustParse("200Mi"),
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
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			obj, err := generator.Generate(tt.params)
			if !tt.expectErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if tt.expectErr && err != nil {
				return
			}
			if !reflect.DeepEqual(obj.(*extensionsv1beta1.Deployment), tt.expected) {
				t.Errorf("\nexpected:\n%#v\nsaw:\n%#v", tt.expected, obj.(*extensionsv1beta1.Deployment))
			}
		})
	}
}

func TestGenerateAppsDeployment(t *testing.T) {
	three := int32(3)
	tests := []struct {
		name      string
		params    map[string]interface{}
		expected  *appsv1beta1.Deployment
		expectErr bool
	}{
		{
			name: "test1",
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
			expected: &appsv1beta1.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"foo": "bar", "baz": "blah"},
				},
				Spec: appsv1beta1.DeploymentSpec{
					Replicas: &three,
					Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar", "baz": "blah"}},
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{"foo": "bar", "baz": "blah"},
						},
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								{
									Name:            "foo",
									Image:           "someimage",
									ImagePullPolicy: v1.PullAlways,
									Stdin:           true,
									Ports: []v1.ContainerPort{
										{
											ContainerPort: 80,
											HostPort:      80,
										},
									},
									Command: []string{"bar", "baz", "blah"},
									Env: []v1.EnvVar{
										{
											Name:  "a",
											Value: "b",
										},
										{
											Name:  "c",
											Value: "d",
										},
									},
									Resources: v1.ResourceRequirements{
										Requests: v1.ResourceList{
											v1.ResourceCPU:    resource.MustParse("100m"),
											v1.ResourceMemory: resource.MustParse("100Mi"),
										},
										Limits: v1.ResourceList{
											v1.ResourceCPU:    resource.MustParse("400m"),
											v1.ResourceMemory: resource.MustParse("200Mi"),
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

	generator := DeploymentAppsV1Beta1{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			obj, err := generator.Generate(tt.params)
			if !tt.expectErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if tt.expectErr && err != nil {
				return
			}
			if !reflect.DeepEqual(obj.(*appsv1beta1.Deployment), tt.expected) {
				t.Errorf("\nexpected:\n%#v\nsaw:\n%#v", tt.expected, obj.(*appsv1beta1.Deployment))
			}
		})
	}
}

func TestGenerateJob(t *testing.T) {
	tests := []struct {
		name      string
		params    map[string]interface{}
		expected  *batchv1.Job
		expectErr bool
	}{
		{
			name: "test1",
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
			expected: &batchv1.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"foo": "bar", "baz": "blah"},
				},
				Spec: batchv1.JobSpec{
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{"foo": "bar", "baz": "blah"},
						},
						Spec: v1.PodSpec{
							RestartPolicy: v1.RestartPolicyOnFailure,
							Containers: []v1.Container{
								{
									Name:      "foo",
									Image:     "someimage",
									Stdin:     true,
									StdinOnce: false,
									Ports: []v1.ContainerPort{
										{
											ContainerPort: 80,
											HostPort:      80,
										},
									},
									Command: []string{"bar", "baz", "blah"},
									Env: []v1.EnvVar{
										{
											Name:  "a",
											Value: "b",
										},
										{
											Name:  "c",
											Value: "d",
										},
									},
									Resources: v1.ResourceRequirements{
										Requests: v1.ResourceList{
											v1.ResourceCPU:    resource.MustParse("100m"),
											v1.ResourceMemory: resource.MustParse("100Mi"),
										},
										Limits: v1.ResourceList{
											v1.ResourceCPU:    resource.MustParse("400m"),
											v1.ResourceMemory: resource.MustParse("200Mi"),
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
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			obj, err := generator.Generate(tt.params)
			if !tt.expectErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if tt.expectErr && err != nil {
				return
			}
			if !reflect.DeepEqual(obj.(*batchv1.Job), tt.expected) {
				t.Errorf("\nexpected:\n%#v\nsaw:\n%#v", tt.expected, obj.(*batchv1.Job))
			}
		})
	}
}

func TestGenerateCronJobAlpha(t *testing.T) {
	tests := []struct {
		name      string
		params    map[string]interface{}
		expected  *batchv2alpha1.CronJob
		expectErr bool
	}{
		{
			name: "test1",
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
			expected: &batchv2alpha1.CronJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"foo": "bar", "baz": "blah"},
				},
				Spec: batchv2alpha1.CronJobSpec{
					Schedule:          "0/5 * * * ?",
					ConcurrencyPolicy: batchv2alpha1.AllowConcurrent,
					JobTemplate: batchv2alpha1.JobTemplateSpec{
						Spec: batchv1.JobSpec{
							Template: v1.PodTemplateSpec{
								ObjectMeta: metav1.ObjectMeta{
									Labels: map[string]string{"foo": "bar", "baz": "blah"},
								},
								Spec: v1.PodSpec{
									RestartPolicy: v1.RestartPolicyOnFailure,
									Containers: []v1.Container{
										{
											Name:      "foo",
											Image:     "someimage",
											Stdin:     true,
											StdinOnce: false,
											Ports: []v1.ContainerPort{
												{
													ContainerPort: 80,
													HostPort:      80,
												},
											},
											Command: []string{"bar", "baz", "blah"},
											Env: []v1.EnvVar{
												{
													Name:  "a",
													Value: "b",
												},
												{
													Name:  "c",
													Value: "d",
												},
											},
											Resources: v1.ResourceRequirements{
												Requests: v1.ResourceList{
													v1.ResourceCPU:    resource.MustParse("100m"),
													v1.ResourceMemory: resource.MustParse("100Mi"),
												},
												Limits: v1.ResourceList{
													v1.ResourceCPU:    resource.MustParse("400m"),
													v1.ResourceMemory: resource.MustParse("200Mi"),
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
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			obj, err := generator.Generate(tt.params)
			if !tt.expectErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if tt.expectErr && err != nil {
				return
			}
			if !reflect.DeepEqual(obj.(*batchv2alpha1.CronJob), tt.expected) {
				t.Errorf("\nexpected:\n%#v\nsaw:\n%#v", tt.expected, obj.(*batchv2alpha1.CronJob))
			}
		})
	}
}

func TestGenerateCronJobBeta(t *testing.T) {
	tests := []struct {
		name      string
		params    map[string]interface{}
		expected  *batchv1beta1.CronJob
		expectErr bool
	}{
		{
			name: "test1",
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
			expected: &batchv1beta1.CronJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"foo": "bar", "baz": "blah"},
				},
				Spec: batchv1beta1.CronJobSpec{
					Schedule:          "0/5 * * * ?",
					ConcurrencyPolicy: batchv1beta1.AllowConcurrent,
					JobTemplate: batchv1beta1.JobTemplateSpec{
						Spec: batchv1.JobSpec{
							Template: v1.PodTemplateSpec{
								ObjectMeta: metav1.ObjectMeta{
									Labels: map[string]string{"foo": "bar", "baz": "blah"},
								},
								Spec: v1.PodSpec{
									RestartPolicy: v1.RestartPolicyOnFailure,
									Containers: []v1.Container{
										{
											Name:      "foo",
											Image:     "someimage",
											Stdin:     true,
											StdinOnce: false,
											Ports: []v1.ContainerPort{
												{
													ContainerPort: 80,
													HostPort:      80,
												},
											},
											Command: []string{"bar", "baz", "blah"},
											Env: []v1.EnvVar{
												{
													Name:  "a",
													Value: "b",
												},
												{
													Name:  "c",
													Value: "d",
												},
											},
											Resources: v1.ResourceRequirements{
												Requests: v1.ResourceList{
													v1.ResourceCPU:    resource.MustParse("100m"),
													v1.ResourceMemory: resource.MustParse("100Mi"),
												},
												Limits: v1.ResourceList{
													v1.ResourceCPU:    resource.MustParse("400m"),
													v1.ResourceMemory: resource.MustParse("200Mi"),
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

	generator := CronJobV1Beta1{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			obj, err := generator.Generate(tt.params)
			if !tt.expectErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if tt.expectErr && err != nil {
				return
			}
			if !reflect.DeepEqual(obj.(*batchv1beta1.CronJob), tt.expected) {
				t.Errorf("\nexpected:\n%#v\nsaw:\n%#v", tt.expected, obj.(*batchv1beta1.CronJob))
			}
		})
	}
}

func TestParseEnv(t *testing.T) {
	tests := []struct {
		name      string
		envArray  []string
		expected  []v1.EnvVar
		expectErr bool
		test      string
	}{
		{
			name: "test1",
			envArray: []string{
				"THIS_ENV=isOK",
				"this.dotted.env=isOKToo",
				"HAS_COMMAS=foo,bar",
				"HAS_EQUALS=jJnro54iUu75xNy==",
			},
			expected: []v1.EnvVar{
				{
					Name:  "THIS_ENV",
					Value: "isOK",
				},
				{
					Name:  "this.dotted.env",
					Value: "isOKToo",
				},
				{
					Name:  "HAS_COMMAS",
					Value: "foo,bar",
				},
				{
					Name:  "HAS_EQUALS",
					Value: "jJnro54iUu75xNy==",
				},
			},
			expectErr: false,
			test:      "test case 1",
		},
		{
			name: "test2",
			envArray: []string{
				"WITH_OUT_EQUALS",
			},
			expected:  []v1.EnvVar{},
			expectErr: true,
			test:      "test case 2",
		},
		{
			name: "test3",
			envArray: []string{
				"WITH_OUT_VALUES=",
			},
			expected: []v1.EnvVar{
				{
					Name:  "WITH_OUT_VALUES",
					Value: "",
				},
			},
			expectErr: false,
			test:      "test case 3",
		},
		{
			name: "test4",
			envArray: []string{
				"=WITH_OUT_NAME",
			},
			expected:  []v1.EnvVar{},
			expectErr: true,
			test:      "test case 4",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			envs, err := parseEnvs(tt.envArray)
			if !tt.expectErr && err != nil {
				t.Errorf("unexpected error: %v (%s)", err, tt.test)
			}
			if tt.expectErr && err != nil {
				return
			}
			if !reflect.DeepEqual(envs, tt.expected) {
				t.Errorf("\nexpected:\n%#v\nsaw:\n%#v (%s)", tt.expected, envs, tt.test)
			}
		})
	}
}
