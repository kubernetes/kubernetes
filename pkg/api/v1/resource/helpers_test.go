/*
Copyright 2015 The Kubernetes Authors.

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

package resource

import (
	"math/big"

	"gopkg.in/inf.v0"
	"math"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
)

const (
	containerName     = "test-container"
	initContainerName = "init-test-container"
)

func TestResourceHelpers(t *testing.T) {
	cpuLimit := resource.MustParse("10")
	memoryLimit := resource.MustParse("10G")
	resourceSpec := v1.ResourceRequirements{
		Limits: v1.ResourceList{
			v1.ResourceCPU:    cpuLimit,
			v1.ResourceMemory: memoryLimit,
		},
	}
	if res := resourceSpec.Limits.Cpu(); res.Cmp(cpuLimit) != 0 {
		t.Errorf("expected cpulimit %v, got %v", cpuLimit, res)
	}
	if res := resourceSpec.Limits.Memory(); res.Cmp(memoryLimit) != 0 {
		t.Errorf("expected memorylimit %v, got %v", memoryLimit, res)
	}
	resourceSpec = v1.ResourceRequirements{
		Limits: v1.ResourceList{
			v1.ResourceMemory: memoryLimit,
		},
	}
	if res := resourceSpec.Limits.Cpu(); res.Value() != 0 {
		t.Errorf("expected cpulimit %v, got %v", 0, res)
	}
	if res := resourceSpec.Limits.Memory(); res.Cmp(memoryLimit) != 0 {
		t.Errorf("expected memorylimit %v, got %v", memoryLimit, res)
	}
}

func TestDefaultResourceHelpers(t *testing.T) {
	resourceList := v1.ResourceList{}
	if resourceList.Cpu().Format != resource.DecimalSI {
		t.Errorf("expected %v, actual %v", resource.DecimalSI, resourceList.Cpu().Format)
	}
	if resourceList.Memory().Format != resource.BinarySI {
		t.Errorf("expected %v, actual %v", resource.BinarySI, resourceList.Memory().Format)
	}
}

func TestGetResourceRequest(t *testing.T) {
	cases := []struct {
		pod                      *v1.Pod
		cName                    string
		resourceName             v1.ResourceName
		expectedValue            int64
		podLevelResourcesEnabled bool
	}{
		{
			pod:           getPod(containerName, resources{cpuRequest: "9"}),
			resourceName:  v1.ResourceCPU,
			expectedValue: 9000,
		},
		{
			pod:           getPod(containerName, resources{memoryRequest: "90Mi"}),
			resourceName:  v1.ResourceMemory,
			expectedValue: 94371840,
		},
		{
			cName:         "just-overhead for cpu",
			pod:           getPod(containerName, resources{cpuOverhead: "5", memoryOverhead: "5"}),
			resourceName:  v1.ResourceCPU,
			expectedValue: 0,
		},
		{
			cName:         "just-overhead for memory",
			pod:           getPod(containerName, resources{memoryOverhead: "5"}),
			resourceName:  v1.ResourceMemory,
			expectedValue: 0,
		},
		{
			cName:         "cpu overhead and req",
			pod:           getPod(containerName, resources{cpuRequest: "2", cpuOverhead: "5", memoryOverhead: "5"}),
			resourceName:  v1.ResourceCPU,
			expectedValue: 7000,
		},
		{
			cName:         "mem overhead and req",
			pod:           getPod(containerName, resources{cpuRequest: "2", memoryRequest: "1024", cpuOverhead: "5", memoryOverhead: "5"}),
			resourceName:  v1.ResourceMemory,
			expectedValue: 1029,
		},
		{
			cName:                    "pod level resources cpu req, container cpu req",
			pod:                      getPodWithPodLevelResources(containerName, resources{cpuRequest: "10"}, resources{cpuRequest: "8"}),
			resourceName:             v1.ResourceCPU,
			expectedValue:            10000,
			podLevelResourcesEnabled: true,
		},
		{
			cName:                    "pod level resources mem req, container mem req",
			pod:                      getPodWithPodLevelResources(containerName, resources{memoryRequest: "100Mi"}, resources{memoryRequest: "80Mi"}),
			resourceName:             v1.ResourceMemory,
			expectedValue:            104857600,
			podLevelResourcesEnabled: true,
		},
		{
			cName:                    "pod level resources cpu req, container mem req",
			pod:                      getPodWithPodLevelResources(containerName, resources{cpuRequest: "5"}, resources{memoryRequest: "50Mi"}),
			resourceName:             v1.ResourceCPU,
			expectedValue:            5000,
			podLevelResourcesEnabled: true,
		},
		{
			cName:                    "pod level resources mem req, container cpu req",
			pod:                      getPodWithPodLevelResources(containerName, resources{memoryRequest: "100Mi"}, resources{cpuRequest: "8"}),
			resourceName:             v1.ResourceMemory,
			expectedValue:            104857600,
			podLevelResourcesEnabled: true,
		},
		{
			cName:                    "pod level resources cpu req, container no req",
			pod:                      getPodWithPodLevelResources(containerName, resources{cpuRequest: "10"}, resources{}),
			resourceName:             v1.ResourceCPU,
			expectedValue:            10000,
			podLevelResourcesEnabled: true,
		},
		{
			cName:                    "pod level resources mem req, container no req",
			pod:                      getPodWithPodLevelResources(containerName, resources{memoryRequest: "100Mi"}, resources{}),
			resourceName:             v1.ResourceMemory,
			expectedValue:            104857600,
			podLevelResourcesEnabled: true,
		},
		{
			cName:                    "cpu pod without pod level resources but pod level resources feature enabled",
			pod:                      getPod(containerName, resources{cpuRequest: "9"}),
			resourceName:             v1.ResourceCPU,
			expectedValue:            9000,
			podLevelResourcesEnabled: true,
		},
		{
			cName:                    "mem pod without pod level resources but pod level resources feature enabled",
			pod:                      getPod(containerName, resources{memoryRequest: "90Mi"}),
			resourceName:             v1.ResourceMemory,
			expectedValue:            94371840,
			podLevelResourcesEnabled: true,
		},
	}
	as := assert.New(t)
	for idx, tc := range cases {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, tc.podLevelResourcesEnabled)

		actual := GetResourceRequest(tc.pod, tc.resourceName)
		as.Equal(tc.expectedValue, actual, "expected test case [%d] %v: to return %q; got %q instead", idx, tc.cName, tc.expectedValue, actual)
	}
}

func TestExtractResourceValue(t *testing.T) {
	cases := []struct {
		fs            *v1.ResourceFieldSelector
		pod           *v1.Pod
		cName         string
		expectedValue string
		expectedError error
	}{
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "limits.cpu",
			},
			cName:         containerName,
			pod:           getPod(containerName, resources{cpuLimit: "9"}),
			expectedValue: "9",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.cpu",
			},
			cName:         containerName,
			pod:           getPod(containerName, resources{}),
			expectedValue: "0",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.cpu",
			},
			cName:         containerName,
			pod:           getPod(containerName, resources{cpuRequest: "8"}),
			expectedValue: "8",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.cpu",
			},
			cName:         containerName,
			pod:           getPod(containerName, resources{cpuRequest: "100m"}),
			expectedValue: "1",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.cpu",
				Divisor:  resource.MustParse("100m"),
			},
			cName:         containerName,
			pod:           getPod(containerName, resources{cpuRequest: "1200m"}),
			expectedValue: "12",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.memory",
			},
			cName:         containerName,
			pod:           getPod(containerName, resources{memoryRequest: "100Mi"}),
			expectedValue: "104857600",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.memory",
				Divisor:  resource.MustParse("1Mi"),
			},
			cName:         containerName,
			pod:           getPod(containerName, resources{memoryRequest: "100Mi", memoryLimit: "1Gi"}),
			expectedValue: "100",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "limits.memory",
			},
			cName:         containerName,
			pod:           getPod(containerName, resources{memoryRequest: "10Mi", memoryLimit: "100Mi"}),
			expectedValue: "104857600",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "limits.cpu",
			},
			cName:         initContainerName,
			pod:           getPod(containerName, resources{cpuLimit: "9"}),
			expectedValue: "9",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.cpu",
			},
			cName:         initContainerName,
			pod:           getPod(containerName, resources{}),
			expectedValue: "0",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.cpu",
			},
			cName:         initContainerName,
			pod:           getPod(containerName, resources{cpuRequest: "8"}),
			expectedValue: "8",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.cpu",
			},
			cName:         initContainerName,
			pod:           getPod(containerName, resources{cpuRequest: "100m"}),
			expectedValue: "1",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.cpu",
				Divisor:  resource.MustParse("100m"),
			},
			cName:         initContainerName,
			pod:           getPod(containerName, resources{cpuRequest: "1200m"}),
			expectedValue: "12",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.memory",
			},
			cName:         initContainerName,
			pod:           getPod(containerName, resources{memoryRequest: "100Mi"}),
			expectedValue: "104857600",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.memory",
				Divisor:  resource.MustParse("1Mi"),
			},
			cName:         initContainerName,
			pod:           getPod(containerName, resources{memoryRequest: "100Mi", memoryLimit: "1Gi"}),
			expectedValue: "100",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "limits.memory",
			},
			cName: initContainerName,
			pod:   getPod(containerName, resources{memoryRequest: "10Mi", memoryLimit: "100Mi"}),

			expectedValue: "104857600",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.cpu",
				Divisor:  resource.MustParse("1n"),
			},
			cName:         containerName,
			pod:           getPod(containerName, resources{cpuRequest: "1m"}),
			expectedValue: "1000000",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.cpu",
				Divisor:  resource.MustParse("1u"),
			},
			cName:         containerName,
			pod:           getPod(containerName, resources{cpuRequest: "1m"}),
			expectedValue: "1000",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.memory",
				Divisor:  resource.MustParse("0"),
			},
			cName:         containerName,
			pod:           getPod(containerName, resources{memoryRequest: "100Mi"}),
			expectedValue: "104857600",
		},
	}
	as := assert.New(t)
	for idx, tc := range cases {
		actual, err := ExtractResourceValueByContainerName(tc.fs, tc.pod, tc.cName)
		if tc.expectedError != nil {
			require.EqualError(t, err, tc.expectedError.Error(), "expected test case [%d] to fail with error %v; got %v", idx, tc.expectedError, err)
		} else {
			require.NoError(t, err, "expected test case [%d] to not return an error; got %v", idx, err)
			as.Equal(tc.expectedValue, actual, "expected test case [%d] to return %q; got %q instead", idx, tc.expectedValue, actual)
		}
	}
}

type resources struct {
	cpuRequest, cpuLimit, memoryRequest, memoryLimit, cpuOverhead, memoryOverhead string
}

func defineResources(resources resources) v1.ResourceRequirements {
	r := v1.ResourceRequirements{
		Limits:   make(v1.ResourceList),
		Requests: make(v1.ResourceList),
	}

	if resources.cpuLimit != "" {
		r.Limits[v1.ResourceCPU] = resource.MustParse(resources.cpuLimit)
	}
	if resources.memoryLimit != "" {
		r.Limits[v1.ResourceMemory] = resource.MustParse(resources.memoryLimit)
	}
	if resources.cpuRequest != "" {
		r.Requests[v1.ResourceCPU] = resource.MustParse(resources.cpuRequest)
	}
	if resources.memoryRequest != "" {
		r.Requests[v1.ResourceMemory] = resource.MustParse(resources.memoryRequest)
	}

	return r
}

func getPod(cname string, resources resources) *v1.Pod {
	r := defineResources(resources)

	overhead := make(v1.ResourceList)

	if resources.cpuOverhead != "" {
		overhead[v1.ResourceCPU] = resource.MustParse(resources.cpuOverhead)
	}
	if resources.memoryOverhead != "" {
		overhead[v1.ResourceMemory] = resource.MustParse(resources.memoryOverhead)
	}

	return &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:      cname,
					Resources: r,
				},
			},
			InitContainers: []v1.Container{
				{
					Name:      "init-" + cname,
					Resources: r,
				},
			},
			Overhead: overhead,
		},
	}
}

func getPodWithPodLevelResources(cname string, podResources resources, resources resources) *v1.Pod {
	pod := getPod(cname, resources)

	r := defineResources(podResources)
	pod.Spec.Resources = &r

	return pod
}

// TestConvertQuantityToStringBounds covers quantities whose scale is far from
// the divisor's. Aligning the two scales multiplies one side by 10^(scale
// difference), so without a bound a value written as 1e1000000 makes a million
// digit intermediate, and a scale near math.MaxInt32 would try to allocate far
// more than that. These cases must return promptly.
func TestConvertQuantityToStringBounds(t *testing.T) {
	// paddedOne is 1 written as 10^10000 scaled by 10000. A parsed quantity
	// never carries a coefficient like this, but NewDecimalQuantity allows it,
	// and it is the case where a bit length estimate is furthest from the
	// value it describes.
	paddedOne := resource.NewDecimalQuantity(*inf.NewDecBig(new(big.Int).Exp(big.NewInt(10), big.NewInt(10000), nil), 10000), resource.DecimalSI)

	testCases := []struct {
		name      string
		quantity  string
		divisor   string
		quantityQ *resource.Quantity
		divisorQ  *resource.Quantity
		expected  string
	}{
		{
			name:     "huge exponent saturates instead of expanding",
			quantity: "1e1000000",
			divisor:  "1",
			expected: strconv.FormatInt(math.MaxInt64, 10),
		},
		{
			name:     "huge exponent against a huge divisor of the same magnitude",
			quantity: "1e1000000",
			divisor:  "1e1000000",
			expected: "1",
		},
		{
			name:     "divisor dwarfs the value",
			quantity: "1",
			divisor:  "1e1000000",
			expected: "1",
		},
		{
			name:     "value just past int64 saturates",
			quantity: "100E",
			divisor:  "1",
			expected: strconv.FormatInt(math.MaxInt64, 10),
		},
		{
			name:     "ordinary ratio is unaffected",
			quantity: "100Mi",
			divisor:  "1Mi",
			expected: "100",
		},
		{
			name:     "ordinary ceiling is unaffected",
			quantity: "1500m",
			divisor:  "1",
			expected: "2",
		},
		{
			name:     "one past int64 saturates",
			quantity: "9223372036854775808",
			divisor:  "1",
			expected: strconv.FormatInt(math.MaxInt64, 10),
		},
		{
			name:     "10E saturates",
			quantity: "10E",
			divisor:  "1",
			expected: strconv.FormatInt(math.MaxInt64, 10),
		},
		{
			// 7e19 and 8e19 straddle the shortcut, so both have to saturate
			// for the result to stay monotonic.
			name:     "below the shortcut but past int64",
			quantity: "7e19",
			divisor:  "1",
			expected: strconv.FormatInt(math.MaxInt64, 10),
		},
		{
			name:     "above the shortcut and past int64",
			quantity: "8e19",
			divisor:  "1",
			expected: strconv.FormatInt(math.MaxInt64, 10),
		},
		{
			// The scale here is math.MaxInt32 wide. Where int is 32 bits the
			// scale difference wraps, so neither shortcut fires and the
			// alignment asks for 10^2147483647.
			name:     "scale too wide for a 32 bit int",
			quantity: "8e2147483647",
			divisor:  "1",
			expected: strconv.FormatInt(math.MaxInt64, 10),
		},
		{
			name:     "divisor with a padded coefficient is still one",
			quantity: "2",
			divisorQ: paddedOne,
			expected: "2",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			q := tc.quantityQ
			if q == nil {
				parsed := resource.MustParse(tc.quantity)
				q = &parsed
			}
			divisor := tc.divisorQ
			if divisor == nil {
				parsed := resource.MustParse(tc.divisor)
				divisor = &parsed
			}
			got, err := convertQuantityToString(q, *divisor)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.expected {
				t.Errorf("got %q, want %q", got, tc.expected)
			}
		})
	}
}
