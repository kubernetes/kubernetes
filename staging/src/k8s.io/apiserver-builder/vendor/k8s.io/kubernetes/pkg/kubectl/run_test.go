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
	"k8s.io/kubernetes/pkg/api/v1"
	appsv1beta1 "k8s.io/kubernetes/pkg/apis/apps/v1beta1"
	batchv1 "k8s.io/kubernetes/pkg/apis/batch/v1"
	batchv2alpha1 "k8s.io/kubernetes/pkg/apis/batch/v2alpha1"
	extensionsv1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
)

func TestGenerate(t *testing.T) {
	one := int32(1)
	tests := []struct {
		params    map[string]interface{}
		expected  *v1.ReplicationController
		expectErr bool
	}{
		{
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
	for i, test := range tests {
		obj, err := generator.Generate(test.params)
		t.Logf("%d: %#v", i, obj)
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}
		if test.expectErr && err != nil {
			continue
		}
		if !reflect.DeepEqual(obj.(*v1.ReplicationController).Spec.Template, test.expected.Spec.Template) {
			t.Errorf("\nexpected:\n%#v\nsaw:\n%#v", test.expected.Spec.Template, obj.(*v1.ReplicationController).Spec.Template)
		}
	}
}

func TestGeneratePod(t *testing.T) {
	tests := []struct {
		params    map[string]interface{}
		expected  *v1.Pod
		expectErr bool
	}{
		{
			params: map[string]interface{}{
				"name":  "foo",
				"image": "someimage",
				"port":  "",
			},
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:            "foo",
							Image:           "someimage",
							ImagePullPolicy: v1.PullIfNotPresent,
						},
					},
					DNSPolicy:     v1.DNSClusterFirst,
					RestartPolicy: v1.RestartPolicyAlways,
				},
			},
		},
		{
			params: map[string]interface{}{
				"name":  "foo",
				"image": "someimage",
				"env":   []string{"a", "c"},
			},

			expected:  nil,
			expectErr: true,
		},
		{
			params: map[string]interface{}{
				"name":              "foo",
				"image":             "someimage",
				"image-pull-policy": "Always",
				"env":               []string{"a=b", "c=d"},
			},
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
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
			params: map[string]interface{}{
				"name":  "foo",
				"image": "someimage",
				"port":  "80",
			},
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
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
			params: map[string]interface{}{
				"name":     "foo",
				"image":    "someimage",
				"port":     "80",
				"hostport": "80",
			},
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
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
					DNSPolicy:     v1.DNSClusterFirst,
					RestartPolicy: v1.RestartPolicyAlways,
				},
			},
		},
		{
			params: map[string]interface{}{
				"name":     "foo",
				"image":    "someimage",
				"hostport": "80",
			},
			expected:  nil,
			expectErr: true,
		},
		{
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
							Name:            "foo",
							Image:           "someimage",
							ImagePullPolicy: v1.PullIfNotPresent,
						},
					},
					DNSPolicy:     v1.DNSClusterFirst,
					RestartPolicy: v1.RestartPolicyAlways,
				},
			},
		},
		{
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
							Name:            "foo",
							Image:           "someimage",
							ImagePullPolicy: v1.PullIfNotPresent,
							Stdin:           true,
							StdinOnce:       true,
						},
					},
					DNSPolicy:     v1.DNSClusterFirst,
					RestartPolicy: v1.RestartPolicyAlways,
				},
			},
		},
		{
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
							Name:            "foo",
							Image:           "someimage",
							ImagePullPolicy: v1.PullIfNotPresent,
							Stdin:           true,
							StdinOnce:       false,
						},
					},
					DNSPolicy:     v1.DNSClusterFirst,
					RestartPolicy: v1.RestartPolicyAlways,
				},
			},
		},
	}
	generator := BasicPod{}
	for _, test := range tests {
		obj, err := generator.Generate(test.params)
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if test.expectErr && err != nil {
			continue
		}
		if !reflect.DeepEqual(obj.(*v1.Pod), test.expected) {
			t.Errorf("\nexpected:\n%#v\nsaw:\n%#v", test.expected, obj.(*v1.Pod))
		}
	}
}

func TestGenerateDeployment(t *testing.T) {
	three := int32(3)
	tests := []struct {
		params    map[string]interface{}
		expected  *extensionsv1beta1.Deployment
		expectErr bool
	}{
		{
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
	for _, test := range tests {
		obj, err := generator.Generate(test.params)
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if test.expectErr && err != nil {
			continue
		}
		if !reflect.DeepEqual(obj.(*extensionsv1beta1.Deployment), test.expected) {
			t.Errorf("\nexpected:\n%#v\nsaw:\n%#v", test.expected, obj.(*extensionsv1beta1.Deployment))
		}
	}
}

func TestGenerateAppsDeployment(t *testing.T) {
	three := int32(3)
	tests := []struct {
		params    map[string]interface{}
		expected  *appsv1beta1.Deployment
		expectErr bool
	}{
		{
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
	for _, test := range tests {
		obj, err := generator.Generate(test.params)
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if test.expectErr && err != nil {
			continue
		}
		if !reflect.DeepEqual(obj.(*appsv1beta1.Deployment), test.expected) {
			t.Errorf("\nexpected:\n%#v\nsaw:\n%#v", test.expected, obj.(*appsv1beta1.Deployment))
		}
	}
}

func TestGenerateJob(t *testing.T) {
	tests := []struct {
		params    map[string]interface{}
		expected  *batchv1.Job
		expectErr bool
	}{
		{
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
	for _, test := range tests {
		obj, err := generator.Generate(test.params)
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if test.expectErr && err != nil {
			continue
		}
		if !reflect.DeepEqual(obj.(*batchv1.Job), test.expected) {
			t.Errorf("\nexpected:\n%#v\nsaw:\n%#v", test.expected, obj.(*batchv1.Job))
		}
	}
}

func TestGenerateCronJob(t *testing.T) {
	tests := []struct {
		params    map[string]interface{}
		expected  *batchv2alpha1.CronJob
		expectErr bool
	}{
		{
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
	for _, test := range tests {
		obj, err := generator.Generate(test.params)
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if test.expectErr && err != nil {
			continue
		}
		if !reflect.DeepEqual(obj.(*batchv2alpha1.CronJob), test.expected) {
			t.Errorf("\nexpected:\n%#v\nsaw:\n%#v", test.expected, obj.(*batchv2alpha1.CronJob))
		}
	}
}

func TestParseEnv(t *testing.T) {
	tests := []struct {
		envArray  []string
		expected  []v1.EnvVar
		expectErr bool
		test      string
	}{
		{
			envArray: []string{
				"THIS_ENV=isOK",
				"HAS_COMMAS=foo,bar",
				"HAS_EQUALS=jJnro54iUu75xNy==",
			},
			expected: []v1.EnvVar{
				{
					Name:  "THIS_ENV",
					Value: "isOK",
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
			envArray: []string{
				"WITH_OUT_EQUALS",
			},
			expected:  []v1.EnvVar{},
			expectErr: true,
			test:      "test case 2",
		},
		{
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
			envArray: []string{
				"=WITH_OUT_NAME",
			},
			expected:  []v1.EnvVar{},
			expectErr: true,
			test:      "test case 4",
		},
	}

	for _, test := range tests {
		envs, err := parseEnvs(test.envArray)
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v (%s)", err, test.test)
		}
		if test.expectErr && err != nil {
			continue
		}
		if !reflect.DeepEqual(envs, test.expected) {
			t.Errorf("\nexpected:\n%#v\nsaw:\n%#v (%s)", test.expected, envs, test.test)
		}
	}
}
