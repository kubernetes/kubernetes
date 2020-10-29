package clusterresourceoverride

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"reflect"
	"testing"

	"github.com/openshift/library-go/pkg/config/helpers"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/autoscaling/apis/clusterresourceoverride"
	clusterresourceoverridev1 "k8s.io/kubernetes/openshift-kube-apiserver/admission/autoscaling/apis/clusterresourceoverride/v1"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/autoscaling/apis/clusterresourceoverride/validation"
	kapi "k8s.io/kubernetes/pkg/apis/core"
)

const (
	yamlConfig = `
apiVersion: autoscaling.openshift.io/v1
kind: ClusterResourceOverrideConfig
limitCPUToMemoryPercent: 100
cpuRequestToLimitPercent: 10
memoryRequestToLimitPercent: 25
`
	invalidConfig = `
apiVersion: autoscaling.openshift.io/v1
kind: ClusterResourceOverrideConfig
cpuRequestToLimitPercent: 200
`
	invalidConfig2 = `
apiVersion: autoscaling.openshift.io/v1
kind: ClusterResourceOverrideConfig
`
)

var (
	deserializedYamlConfig = &clusterresourceoverride.ClusterResourceOverrideConfig{
		LimitCPUToMemoryPercent:     100,
		CPURequestToLimitPercent:    10,
		MemoryRequestToLimitPercent: 25,
	}
)

func TestConfigReader(t *testing.T) {
	initialConfig := testConfig(10, 20, 30)
	serializedConfig, serializationErr := helpers.WriteYAML(initialConfig, clusterresourceoverridev1.Install)
	if serializationErr != nil {
		t.Fatalf("WriteYAML: config serialize failed: %v", serializationErr)
	}

	tests := []struct {
		name           string
		config         io.Reader
		expectErr      bool
		expectNil      bool
		expectInvalid  bool
		expectedConfig *clusterresourceoverride.ClusterResourceOverrideConfig
	}{
		{
			name:      "process nil config",
			config:    nil,
			expectNil: true,
		}, {
			name:           "deserialize initialConfig yaml",
			config:         bytes.NewReader(serializedConfig),
			expectedConfig: initialConfig,
		}, {
			name:      "completely broken config",
			config:    bytes.NewReader([]byte("asdfasdfasdF")),
			expectErr: true,
		}, {
			name:           "deserialize yamlConfig",
			config:         bytes.NewReader([]byte(yamlConfig)),
			expectedConfig: deserializedYamlConfig,
		}, {
			name:          "choke on out-of-bounds ratio",
			config:        bytes.NewReader([]byte(invalidConfig)),
			expectInvalid: true,
			expectErr:     true,
		}, {
			name:          "complain about no settings",
			config:        bytes.NewReader([]byte(invalidConfig2)),
			expectInvalid: true,
			expectErr:     true,
		},
	}
	for _, test := range tests {
		config, err := ReadConfig(test.config)
		if test.expectErr && err == nil {
			t.Errorf("%s: expected error", test.name)
		} else if !test.expectErr && err != nil {
			t.Errorf("%s: expected no error, saw %v", test.name, err)
		}
		if err == nil {
			if test.expectNil && config != nil {
				t.Errorf("%s: expected nil config, but saw: %v", test.name, config)
			} else if !test.expectNil && config == nil {
				t.Errorf("%s: expected config, but got nil", test.name)
			}
		}
		if config != nil {
			if test.expectedConfig != nil && *test.expectedConfig != *config {
				t.Errorf("%s: expected %v from reader, but got %v", test.name, test.expectErr, config)
			}
			if err := validation.Validate(config); test.expectInvalid && len(err) == 0 {
				t.Errorf("%s: expected validation to fail, but it passed", test.name)
			} else if !test.expectInvalid && len(err) > 0 {
				t.Errorf("%s: expected validation to pass, but it failed with %v", test.name, err)
			}
		}
	}
}

func TestLimitRequestAdmission(t *testing.T) {
	tests := []struct {
		name               string
		config             *clusterresourceoverride.ClusterResourceOverrideConfig
		pod                *kapi.Pod
		expectedMemRequest resource.Quantity
		expectedCpuLimit   resource.Quantity
		expectedCpuRequest resource.Quantity
		namespace          *corev1.Namespace
		namespaceLimits    []*corev1.LimitRange
	}{
		{
			name:               "ignore pods that have no memory limit specified",
			config:             testConfig(100, 50, 50),
			pod:                testBestEffortPod(),
			expectedMemRequest: resource.MustParse("0"),
			expectedCpuLimit:   resource.MustParse("0"),
			expectedCpuRequest: resource.MustParse("0"),
			namespace:          fakeNamespace(true),
		},
		{
			name:               "with namespace limits, ignore pods that have no memory limit specified",
			config:             testConfig(100, 50, 50),
			pod:                testBestEffortPod(),
			expectedMemRequest: resource.MustParse("0"),
			expectedCpuLimit:   resource.MustParse("0"),
			expectedCpuRequest: resource.MustParse("0"),
			namespace:          fakeNamespace(true),
			namespaceLimits: []*corev1.LimitRange{
				fakeMinCPULimitRange("567m"),
				fakeMinCPULimitRange("678m"),
				fakeMinMemoryLimitRange("700Gi"),
				fakeMinMemoryLimitRange("456Gi"),
			},
		},
		{
			name:               "test floor for memory and cpu",
			config:             testConfig(100, 50, 50),
			pod:                testPod("1Mi", "0", "0", "0"),
			expectedMemRequest: resource.MustParse("1Mi"),
			expectedCpuLimit:   resource.MustParse("1m"),
			expectedCpuRequest: resource.MustParse("1m"),
			namespace:          fakeNamespace(true),
		},
		{
			name:               "with namespace limits, test floor for memory and cpu",
			config:             testConfig(100, 50, 50),
			pod:                testPod("1Mi", "0", "0", "0"),
			expectedMemRequest: resource.MustParse("456Gi"),
			expectedCpuLimit:   resource.MustParse("567m"),
			expectedCpuRequest: resource.MustParse("567m"),
			namespace:          fakeNamespace(true),
			namespaceLimits: []*corev1.LimitRange{
				fakeMinCPULimitRange("567m"),
				fakeMinCPULimitRange("678m"),
				fakeMinMemoryLimitRange("700Gi"),
				fakeMinMemoryLimitRange("456Gi"),
			},
		},
		{
			name:               "nil config",
			config:             nil,
			pod:                testPod("1", "1", "1", "1"),
			expectedMemRequest: resource.MustParse("1"),
			expectedCpuLimit:   resource.MustParse("1"),
			expectedCpuRequest: resource.MustParse("1"),
			namespace:          fakeNamespace(true),
		},
		{
			name:               "with namespace limits, nil config",
			config:             nil,
			pod:                testPod("1", "1", "1", "1"),
			expectedMemRequest: resource.MustParse("1"),
			expectedCpuLimit:   resource.MustParse("1"),
			expectedCpuRequest: resource.MustParse("1"),
			namespace:          fakeNamespace(true),
			namespaceLimits: []*corev1.LimitRange{
				fakeMinCPULimitRange("567m"),
				fakeMinCPULimitRange("678m"),
				fakeMinMemoryLimitRange("700Gi"),
				fakeMinMemoryLimitRange("456Gi"),
			},
		},
		{
			name:               "all values are adjusted",
			config:             testConfig(100, 50, 50),
			pod:                testPod("1Gi", "0", "2000m", "0"),
			expectedMemRequest: resource.MustParse("512Mi"),
			expectedCpuLimit:   resource.MustParse("1"),
			expectedCpuRequest: resource.MustParse("500m"),
			namespace:          fakeNamespace(true),
		},
		{
			name:               "with namespace limits, all values are adjusted to floor of namespace limits",
			config:             testConfig(100, 50, 50),
			pod:                testPod("1Gi", "0", "2000m", "0"),
			expectedMemRequest: resource.MustParse("456Gi"),
			expectedCpuLimit:   resource.MustParse("10567m"),
			expectedCpuRequest: resource.MustParse("10567m"),
			namespace:          fakeNamespace(true),
			namespaceLimits: []*corev1.LimitRange{
				fakeMinCPULimitRange("10567m"),
				fakeMinCPULimitRange("20678m"),
				fakeMinMemoryLimitRange("700Gi"),
				fakeMinMemoryLimitRange("456Gi"),
			},
		},
		{
			name:               "just requests are adjusted",
			config:             testConfig(0, 50, 50),
			pod:                testPod("10Mi", "0", "50m", "0"),
			expectedMemRequest: resource.MustParse("5Mi"),
			expectedCpuLimit:   resource.MustParse("50m"),
			expectedCpuRequest: resource.MustParse("25m"),
			namespace:          fakeNamespace(true),
		},
		{
			name:               "with namespace limits, all requests are adjusted to floor of namespace limits",
			config:             testConfig(0, 50, 50),
			pod:                testPod("10Mi", "0", "50m", "0"),
			expectedMemRequest: resource.MustParse("456Gi"),
			expectedCpuLimit:   resource.MustParse("50m"),
			expectedCpuRequest: resource.MustParse("10567m"),
			namespace:          fakeNamespace(true),
			namespaceLimits: []*corev1.LimitRange{
				fakeMinCPULimitRange("10567m"),
				fakeMinCPULimitRange("20678m"),
				fakeMinMemoryLimitRange("700Gi"),
				fakeMinMemoryLimitRange("456Gi"),
			},
		},
		{
			name:               "project annotation disables overrides",
			config:             testConfig(0, 50, 50),
			pod:                testPod("10Mi", "0", "50m", "0"),
			expectedMemRequest: resource.MustParse("0"),
			expectedCpuLimit:   resource.MustParse("50m"),
			expectedCpuRequest: resource.MustParse("0"),
			namespace:          fakeNamespace(false),
		},
		{
			name:               "with namespace limits, project annotation disables overrides",
			config:             testConfig(0, 50, 50),
			pod:                testPod("10Mi", "0", "50m", "0"),
			expectedMemRequest: resource.MustParse("0"),
			expectedCpuLimit:   resource.MustParse("50m"),
			expectedCpuRequest: resource.MustParse("0"),
			namespace:          fakeNamespace(false),
			namespaceLimits: []*corev1.LimitRange{
				fakeMinCPULimitRange("10567m"),
				fakeMinCPULimitRange("20678m"),
				fakeMinMemoryLimitRange("700Gi"),
				fakeMinMemoryLimitRange("456Gi"),
			},
		},
		{
			name:               "large values don't overflow",
			config:             testConfig(100, 50, 50),
			pod:                testPod("1Ti", "0", "0", "0"),
			expectedMemRequest: resource.MustParse("512Gi"),
			expectedCpuLimit:   resource.MustParse("1024"),
			expectedCpuRequest: resource.MustParse("512"),
			namespace:          fakeNamespace(true),
		},
		{
			name:               "little values mess things up",
			config:             testConfig(500, 10, 10),
			pod:                testPod("1.024Mi", "0", "0", "0"),
			expectedMemRequest: resource.MustParse("1Mi"),
			expectedCpuLimit:   resource.MustParse("5m"),
			expectedCpuRequest: resource.MustParse("1m"),
			namespace:          fakeNamespace(true),
		},
		{
			name:               "test fractional memory requests round up",
			config:             testConfig(500, 10, 60),
			pod:                testPod("512Mi", "0", "0", "0"),
			expectedMemRequest: resource.MustParse("307Mi"),
			expectedCpuLimit:   resource.MustParse("2.5"),
			expectedCpuRequest: resource.MustParse("250m"),
			namespace:          fakeNamespace(true),
		},
		{
			name:               "test only containers types are considered with namespace limits",
			config:             testConfig(100, 50, 50),
			pod:                testPod("1Gi", "0", "2000m", "0"),
			expectedMemRequest: resource.MustParse("512Mi"),
			expectedCpuLimit:   resource.MustParse("1"),
			expectedCpuRequest: resource.MustParse("500m"),
			namespace:          fakeNamespace(true),
			namespaceLimits: []*corev1.LimitRange{
				fakeMinStorageLimitRange("1567Mi"),
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			c, err := newClusterResourceOverride(test.config)
			if err != nil {
				t.Fatalf("%s: config de/serialize failed: %v", test.name, err)
			}
			// Override LimitRanger with limits from test case
			c.(*clusterResourceOverridePlugin).limitRangesLister = fakeLimitRangeLister{
				namespaceLister: fakeLimitRangeNamespaceLister{
					limits: test.namespaceLimits,
				},
			}
			c.(*clusterResourceOverridePlugin).nsLister = fakeNamespaceLister(test.namespace)
			attrs := admission.NewAttributesRecord(test.pod, nil, schema.GroupVersionKind{}, test.namespace.Name, "name", kapi.Resource("pods").WithVersion("version"), "", admission.Create, nil, false, fakeUser())
			clone := test.pod.DeepCopy()
			if err = c.(admission.MutationInterface).Admit(context.TODO(), attrs, nil); err != nil {
				t.Fatalf("%s: admission controller returned error: %v", test.name, err)
			}
			if err = c.(admission.ValidationInterface).Validate(context.TODO(), attrs, nil); err != nil {
				t.Fatalf("%s: admission controller returned error: %v", test.name, err)
			}

			if !reflect.DeepEqual(test.pod, clone) {
				attrs := admission.NewAttributesRecord(clone, nil, schema.GroupVersionKind{}, test.namespace.Name, "name", kapi.Resource("pods").WithVersion("version"), "", admission.Create, nil, false, fakeUser())
				if err = c.(admission.ValidationInterface).Validate(context.TODO(), attrs, nil); err == nil {
					t.Fatalf("%s: admission controller returned no error, but should", test.name)
				}
			}

			resources := test.pod.Spec.InitContainers[0].Resources // only test one container
			if actual := resources.Requests[kapi.ResourceMemory]; test.expectedMemRequest.Cmp(actual) != 0 {
				t.Errorf("%s: memory requests do not match; %v should be %v", test.name, actual, test.expectedMemRequest)
			}
			if actual := resources.Requests[kapi.ResourceCPU]; test.expectedCpuRequest.Cmp(actual) != 0 {
				t.Errorf("%s: cpu requests do not match; %v should be %v", test.name, actual, test.expectedCpuRequest)
			}
			if actual := resources.Limits[kapi.ResourceCPU]; test.expectedCpuLimit.Cmp(actual) != 0 {
				t.Errorf("%s: cpu limits do not match; %v should be %v", test.name, actual, test.expectedCpuLimit)
			}

			resources = test.pod.Spec.Containers[0].Resources // only test one container
			if actual := resources.Requests[kapi.ResourceMemory]; test.expectedMemRequest.Cmp(actual) != 0 {
				t.Errorf("%s: memory requests do not match; %v should be %v", test.name, actual, test.expectedMemRequest)
			}
			if actual := resources.Requests[kapi.ResourceCPU]; test.expectedCpuRequest.Cmp(actual) != 0 {
				t.Errorf("%s: cpu requests do not match; %v should be %v", test.name, actual, test.expectedCpuRequest)
			}
			if actual := resources.Limits[kapi.ResourceCPU]; test.expectedCpuLimit.Cmp(actual) != 0 {
				t.Errorf("%s: cpu limits do not match; %v should be %v", test.name, actual, test.expectedCpuLimit)
			}
		})
	}
}

func testBestEffortPod() *kapi.Pod {
	return &kapi.Pod{
		Spec: kapi.PodSpec{
			InitContainers: []kapi.Container{
				{
					Resources: kapi.ResourceRequirements{},
				},
			},
			Containers: []kapi.Container{
				{
					Resources: kapi.ResourceRequirements{},
				},
			},
		},
	}
}

func testPod(memLimit string, memRequest string, cpuLimit string, cpuRequest string) *kapi.Pod {
	return &kapi.Pod{
		Spec: kapi.PodSpec{
			InitContainers: []kapi.Container{
				{
					Resources: kapi.ResourceRequirements{
						Limits: kapi.ResourceList{
							kapi.ResourceCPU:    resource.MustParse(cpuLimit),
							kapi.ResourceMemory: resource.MustParse(memLimit),
						},
						Requests: kapi.ResourceList{
							kapi.ResourceCPU:    resource.MustParse(cpuRequest),
							kapi.ResourceMemory: resource.MustParse(memRequest),
						},
					},
				},
			},
			Containers: []kapi.Container{
				{
					Resources: kapi.ResourceRequirements{
						Limits: kapi.ResourceList{
							kapi.ResourceCPU:    resource.MustParse(cpuLimit),
							kapi.ResourceMemory: resource.MustParse(memLimit),
						},
						Requests: kapi.ResourceList{
							kapi.ResourceCPU:    resource.MustParse(cpuRequest),
							kapi.ResourceMemory: resource.MustParse(memRequest),
						},
					},
				},
			},
		},
	}
}

func fakeUser() user.Info {
	return &user.DefaultInfo{
		Name: "testuser",
	}
}

var nsIndex = 0

func fakeNamespace(pluginEnabled bool) *corev1.Namespace {
	nsIndex++
	ns := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name:        fmt.Sprintf("fakeNS%d", nsIndex),
			Annotations: map[string]string{},
		},
	}
	if !pluginEnabled {
		ns.Annotations[clusterResourceOverrideAnnotation] = "false"
	}
	return ns
}

func fakeNamespaceLister(ns *corev1.Namespace) corev1listers.NamespaceLister {
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	indexer.Add(ns)
	return corev1listers.NewNamespaceLister(indexer)
}

func testConfig(lc2mr int64, cr2lr int64, mr2lr int64) *clusterresourceoverride.ClusterResourceOverrideConfig {
	return &clusterresourceoverride.ClusterResourceOverrideConfig{
		LimitCPUToMemoryPercent:     lc2mr,
		CPURequestToLimitPercent:    cr2lr,
		MemoryRequestToLimitPercent: mr2lr,
	}
}

func fakeMinLimitRange(limitType corev1.LimitType, resourceType corev1.ResourceName, limits ...string) *corev1.LimitRange {
	r := &corev1.LimitRange{}

	for i := range limits {
		rl := corev1.ResourceList{}
		rl[resourceType] = resource.MustParse(limits[i])
		r.Spec.Limits = append(r.Spec.Limits,
			corev1.LimitRangeItem{
				Type: limitType,
				Min:  rl,
			},
		)
	}

	return r
}

func fakeMinMemoryLimitRange(limits ...string) *corev1.LimitRange {
	return fakeMinLimitRange(corev1.LimitTypeContainer, corev1.ResourceMemory, limits...)
}

func fakeMinCPULimitRange(limits ...string) *corev1.LimitRange {
	return fakeMinLimitRange(corev1.LimitTypeContainer, corev1.ResourceCPU, limits...)
}

func fakeMinStorageLimitRange(limits ...string) *corev1.LimitRange {
	return fakeMinLimitRange(corev1.LimitTypePersistentVolumeClaim, corev1.ResourceStorage, limits...)
}

type fakeLimitRangeLister struct {
	corev1listers.LimitRangeLister
	namespaceLister fakeLimitRangeNamespaceLister
}

type fakeLimitRangeNamespaceLister struct {
	corev1listers.LimitRangeNamespaceLister
	limits []*corev1.LimitRange
}

func (f fakeLimitRangeLister) LimitRanges(namespace string) corev1listers.LimitRangeNamespaceLister {
	return f.namespaceLister
}

func (f fakeLimitRangeNamespaceLister) List(selector labels.Selector) ([]*corev1.LimitRange, error) {
	return f.limits, nil
}
