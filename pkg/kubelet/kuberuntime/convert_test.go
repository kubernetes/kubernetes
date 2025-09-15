/*
Copyright 2020 The Kubernetes Authors.

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

package kuberuntime

import (
	"testing"

	"github.com/stretchr/testify/assert"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

func TestConvertToKubeContainerImageSpec(t *testing.T) {
	testCases := []struct {
		input    *runtimeapi.Image
		expected kubecontainer.ImageSpec
	}{
		{
			input: &runtimeapi.Image{
				Id:   "test",
				Spec: nil,
			},
			expected: kubecontainer.ImageSpec{
				Image:       "test",
				Annotations: []kubecontainer.Annotation(nil),
			},
		},
		{
			input: &runtimeapi.Image{
				Id: "test",
				Spec: &runtimeapi.ImageSpec{
					Annotations: nil,
				},
			},
			expected: kubecontainer.ImageSpec{
				Image:       "test",
				Annotations: []kubecontainer.Annotation(nil),
			},
		},
		{
			input: &runtimeapi.Image{
				Id: "test",
				Spec: &runtimeapi.ImageSpec{
					Annotations: map[string]string{},
				},
			},
			expected: kubecontainer.ImageSpec{
				Image:       "test",
				Annotations: []kubecontainer.Annotation(nil),
			},
		},
		{
			input: &runtimeapi.Image{
				Id: "test",
				Spec: &runtimeapi.ImageSpec{
					Annotations: map[string]string{
						"kubernetes.io/os":             "linux",
						"kubernetes.io/runtimehandler": "handler",
					},
				},
			},
			expected: kubecontainer.ImageSpec{
				Image: "test",
				Annotations: []kubecontainer.Annotation{
					{
						Name:  "kubernetes.io/os",
						Value: "linux",
					},
					{
						Name:  "kubernetes.io/runtimehandler",
						Value: "handler",
					},
				},
			},
		},
	}

	for _, test := range testCases {
		actual := toKubeContainerImageSpec(test.input)
		assert.Equal(t, test.expected, actual)
	}
}

func TestConvertToRuntimeAPIImageSpec(t *testing.T) {
	testCases := []struct {
		input    kubecontainer.ImageSpec
		expected *runtimeapi.ImageSpec
	}{
		{
			input: kubecontainer.ImageSpec{
				Image:          "test",
				RuntimeHandler: "",
				Annotations:    nil,
			},
			expected: &runtimeapi.ImageSpec{
				Image:          "test",
				RuntimeHandler: "",
				Annotations:    map[string]string{},
			},
		},
		{
			input: kubecontainer.ImageSpec{
				Image:          "test",
				RuntimeHandler: "",
				Annotations:    []kubecontainer.Annotation{},
			},
			expected: &runtimeapi.ImageSpec{
				Image:          "test",
				RuntimeHandler: "",
				Annotations:    map[string]string{},
			},
		},
		{
			input: kubecontainer.ImageSpec{
				Image:          "test",
				RuntimeHandler: "",
				Annotations: []kubecontainer.Annotation{
					{
						Name:  "kubernetes.io/os",
						Value: "linux",
					},
					{
						Name:  "kubernetes.io/runtimehandler",
						Value: "handler",
					},
				},
			},
			expected: &runtimeapi.ImageSpec{
				Image:          "test",
				RuntimeHandler: "",
				Annotations: map[string]string{
					"kubernetes.io/os":             "linux",
					"kubernetes.io/runtimehandler": "handler",
				},
			},
		},
	}

	for _, test := range testCases {
		actual := toRuntimeAPIImageSpec(test.input)
		assert.Equal(t, test.expected, actual)
	}
}

func TestConvertToKubeContainerImageSpecWithRuntimeHandlerInImageSpecCri(t *testing.T) {
	testCases := []struct {
		input    *runtimeapi.Image
		expected kubecontainer.ImageSpec
	}{
		{
			input: &runtimeapi.Image{
				Id:   "test",
				Spec: nil,
			},
			expected: kubecontainer.ImageSpec{
				Image:          "test",
				RuntimeHandler: "",
				Annotations:    []kubecontainer.Annotation(nil),
			},
		},
		{
			input: &runtimeapi.Image{
				Id: "test",
				Spec: &runtimeapi.ImageSpec{
					Annotations: nil,
				},
			},
			expected: kubecontainer.ImageSpec{
				Image:          "test",
				RuntimeHandler: "",
				Annotations:    []kubecontainer.Annotation(nil),
			},
		},
		{
			input: &runtimeapi.Image{
				Id: "test",
				Spec: &runtimeapi.ImageSpec{
					Annotations: map[string]string{},
				},
			},
			expected: kubecontainer.ImageSpec{
				Image:          "test",
				RuntimeHandler: "",
				Annotations:    []kubecontainer.Annotation(nil),
			},
		},
		{
			input: &runtimeapi.Image{
				Id: "test",
				Spec: &runtimeapi.ImageSpec{
					RuntimeHandler: "test-runtimeHandler",
					Annotations: map[string]string{
						"kubernetes.io/os":             "linux",
						"kubernetes.io/runtimehandler": "handler",
					},
				},
			},
			expected: kubecontainer.ImageSpec{
				Image:          "test",
				RuntimeHandler: "test-runtimeHandler",
				Annotations: []kubecontainer.Annotation{
					{
						Name:  "kubernetes.io/os",
						Value: "linux",
					},
					{
						Name:  "kubernetes.io/runtimehandler",
						Value: "handler",
					},
				},
			},
		},
	}

	for _, test := range testCases {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RuntimeClassInImageCriAPI, true)
		actual := toKubeContainerImageSpec(test.input)
		assert.Equal(t, test.expected, actual)
	}
}

func TestConvertToRuntimeAPIImageSpecWithRuntimeHandlerInImageSpecCri(t *testing.T) {
	testCases := []struct {
		input    kubecontainer.ImageSpec
		expected *runtimeapi.ImageSpec
	}{
		{
			input: kubecontainer.ImageSpec{
				Image:          "test",
				RuntimeHandler: "",
				Annotations:    nil,
			},
			expected: &runtimeapi.ImageSpec{
				Image:          "test",
				RuntimeHandler: "",
				Annotations:    map[string]string{},
			},
		},
		{
			input: kubecontainer.ImageSpec{
				Image:          "test",
				RuntimeHandler: "",
				Annotations:    []kubecontainer.Annotation{},
			},
			expected: &runtimeapi.ImageSpec{
				Image:          "test",
				RuntimeHandler: "",
				Annotations:    map[string]string{},
			},
		},
		{
			input: kubecontainer.ImageSpec{
				Image:          "test",
				RuntimeHandler: "test-runtimeHandler",
				Annotations: []kubecontainer.Annotation{
					{
						Name:  "kubernetes.io/os",
						Value: "linux",
					},
					{
						Name:  "kubernetes.io/runtimehandler",
						Value: "handler",
					},
				},
			},
			expected: &runtimeapi.ImageSpec{
				Image:          "test",
				RuntimeHandler: "test-runtimeHandler",
				Annotations: map[string]string{
					"kubernetes.io/os":             "linux",
					"kubernetes.io/runtimehandler": "handler",
				},
			},
		},
	}

	for _, test := range testCases {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RuntimeClassInImageCriAPI, true)
		actual := toRuntimeAPIImageSpec(test.input)
		assert.Equal(t, test.expected, actual)
	}
}
