package managementcpusoverride

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"testing"

	configv1 "github.com/openshift/api/config/v1"
	configv1listers "github.com/openshift/client-go/config/listers/config/v1"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/client-go/kubernetes/fake"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	coreapi "k8s.io/kubernetes/pkg/apis/core"
	kapi "k8s.io/kubernetes/pkg/apis/core"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

const (
	// workloadTypeManagement contains the type for the management workload
	workloadTypeManagement = "management"
	// managedCapacityLabel contains the name of the new management resource that will available under the node
	managedCapacityLabel = "management.workload.openshift.io/cores"
)

func getMockManagementCPUsOverride(namespace *corev1.Namespace, nodes []*corev1.Node, infra *configv1.Infrastructure) (*managementCPUsOverride, error) {
	m := &managementCPUsOverride{
		Handler:               admission.NewHandler(admission.Create),
		client:                &fake.Clientset{},
		nsLister:              fakeNamespaceLister(namespace),
		nsListerSynced:        func() bool { return true },
		nodeLister:            fakeNodeLister(nodes),
		nodeListSynced:        func() bool { return true },
		infraConfigLister:     fakeInfraConfigLister(infra),
		infraConfigListSynced: func() bool { return true },
	}
	if err := m.ValidateInitialization(); err != nil {
		return nil, err
	}

	return m, nil
}

func fakeNamespaceLister(ns *corev1.Namespace) corev1listers.NamespaceLister {
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	_ = indexer.Add(ns)
	return corev1listers.NewNamespaceLister(indexer)
}

func fakeNodeLister(nodes []*corev1.Node) corev1listers.NodeLister {
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	for _, node := range nodes {
		_ = indexer.Add(node)
	}
	return corev1listers.NewNodeLister(indexer)
}

func fakeInfraConfigLister(infra *configv1.Infrastructure) configv1listers.InfrastructureLister {
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	if infra != nil {
		_ = indexer.Add(infra)
	}
	return configv1listers.NewInfrastructureLister(indexer)
}

func TestAdmit(t *testing.T) {
	tests := []struct {
		name                string
		pod                 *kapi.Pod
		namespace           *corev1.Namespace
		nodes               []*corev1.Node
		infra               *configv1.Infrastructure
		expectedCpuRequest  resource.Quantity
		expectedAnnotations map[string]string
		expectedError       error
	}{
		{
			name:               "should return admission error when the pod namespace does not allow the workload type",
			pod:                testManagedPod("500m", "250m", "500Mi", "250Mi"),
			expectedCpuRequest: resource.MustParse("250m"),
			namespace:          testNamespace(),
			nodes:              []*corev1.Node{testNodeWithManagementResource()},
			infra:              testClusterSNOInfra(),
			expectedError:      fmt.Errorf("the pod namespace %q does not allow the workload type management", "namespace"),
		},
		{
			name:               "should ignore pods that do not have managed annotation",
			pod:                testPod("500m", "250m", "500Mi", "250Mi"),
			expectedCpuRequest: resource.MustParse("250m"),
			namespace:          testManagedNamespace(),
			nodes:              []*corev1.Node{testNodeWithManagementResource()},
		},
		{
			name: "should return admission error when the pod has more than one workload annotation",
			pod: testManagedPodWithAnnotations(
				"500m",
				"250m",
				"500Mi",
				"250Mi",
				map[string]string{
					fmt.Sprintf("%s%s", podWorkloadTargetAnnotationPrefix, workloadTypeManagement): "",
					fmt.Sprintf("%stest", podWorkloadTargetAnnotationPrefix):                       "",
				},
			),
			expectedCpuRequest: resource.MustParse("250m"),
			namespace:          testManagedNamespace(),
			nodes:              []*corev1.Node{testNodeWithManagementResource()},
			infra:              testClusterSNOInfra(),
			expectedError:      fmt.Errorf("the pod can not have more than one workload annotations"),
		},
		{
			name: "should return admission error when the pod has incorrect workload annotation",
			pod: testManagedPodWithAnnotations(
				"500m",
				"250m",
				"500Mi",
				"250Mi",
				map[string]string{
					podWorkloadTargetAnnotationPrefix: "",
				},
			),
			expectedCpuRequest: resource.MustParse("250m"),
			namespace:          testManagedNamespace(),
			nodes:              []*corev1.Node{testNodeWithManagementResource()},
			infra:              testClusterSNOInfra(),
			expectedError:      fmt.Errorf("the workload annotation key should have format %s<workload_type>", podWorkloadTargetAnnotationPrefix),
		},
		{
			name: "should return admission error when the pod has incorrect workload annotation effect",
			pod: testManagedPodWithAnnotations(
				"500m",
				"250m",
				"500Mi",
				"250Mi",
				map[string]string{
					fmt.Sprintf("%s%s", podWorkloadTargetAnnotationPrefix, workloadTypeManagement): "{",
				},
			),
			expectedCpuRequest: resource.MustParse("250m"),
			namespace:          testManagedNamespace(),
			nodes:              []*corev1.Node{testNodeWithManagementResource()},
			infra:              testClusterSNOInfra(),
			expectedError:      fmt.Errorf(`failed to get workload annotation effect: failed to parse "{" annotation value: unexpected end of JSON input`),
		},
		{
			name: "should return admission error when the pod has workload annotation without effect value",
			pod: testManagedPodWithAnnotations(
				"500m",
				"250m",
				"500Mi",
				"250Mi",
				map[string]string{
					fmt.Sprintf("%s%s", podWorkloadTargetAnnotationPrefix, workloadTypeManagement): `{"test": "test"}`,
				},
			),
			expectedCpuRequest: resource.MustParse("250m"),
			namespace:          testManagedNamespace(),
			nodes:              []*corev1.Node{testNodeWithManagementResource()},
			expectedError:      fmt.Errorf(`failed to get workload annotation effect: the workload annotation value map["test":"test"] does not have "effect" key`),
			infra:              testClusterSNOInfra(),
		},
		{
			name:               "should delete CPU requests and update workload CPU annotations for the burstable pod with managed annotation",
			pod:                testManagedPod("", "250m", "500Mi", "250Mi"),
			expectedCpuRequest: resource.Quantity{},
			namespace:          testManagedNamespace(),
			expectedAnnotations: map[string]string{
				fmt.Sprintf("%s%s", containerResourcesAnnotationPrefix, "test"):                fmt.Sprintf(`{"%s": 256}`, containerResourcesAnnotationValueKeyCPUShares),
				fmt.Sprintf("%s%s", containerResourcesAnnotationPrefix, "initTest"):            fmt.Sprintf(`{"%s": 256}`, containerResourcesAnnotationValueKeyCPUShares),
				fmt.Sprintf("%s%s", podWorkloadTargetAnnotationPrefix, workloadTypeManagement): fmt.Sprintf(`{"%s":"%s"}`, podWorkloadAnnotationEffect, workloadEffectPreferredDuringScheduling),
			},
			nodes: []*corev1.Node{testNodeWithManagementResource()},
			infra: testClusterSNOInfra(),
		},
		{
			name:               "should update workload CPU annotations for the best-effort pod with managed annotation",
			pod:                testManagedPod("", "", "", ""),
			expectedCpuRequest: resource.Quantity{},
			namespace:          testManagedNamespace(),
			expectedAnnotations: map[string]string{
				fmt.Sprintf("%s%s", containerResourcesAnnotationPrefix, "test"):                fmt.Sprintf(`{"%s": 2}`, containerResourcesAnnotationValueKeyCPUShares),
				fmt.Sprintf("%s%s", containerResourcesAnnotationPrefix, "initTest"):            fmt.Sprintf(`{"%s": 2}`, containerResourcesAnnotationValueKeyCPUShares),
				fmt.Sprintf("%s%s", podWorkloadTargetAnnotationPrefix, workloadTypeManagement): fmt.Sprintf(`{"%s":"%s"}`, podWorkloadAnnotationEffect, workloadEffectPreferredDuringScheduling),
			},
			nodes: []*corev1.Node{testNodeWithManagementResource()},
			infra: testClusterSNOInfra(),
		},
		{
			name:               "should skip static pod mutation",
			pod:                testManagedStaticPod("500m", "250m", "500Mi", "250Mi"),
			expectedCpuRequest: resource.MustParse("250m"),
			namespace:          testManagedNamespace(),
			expectedAnnotations: map[string]string{
				fmt.Sprintf("%s%s", podWorkloadTargetAnnotationPrefix, workloadTypeManagement): fmt.Sprintf(`{"%s":"%s"}`, podWorkloadAnnotationEffect, workloadEffectPreferredDuringScheduling),
				kubetypes.ConfigSourceAnnotationKey:                                            kubetypes.FileSource,
			},
			nodes: []*corev1.Node{testNodeWithManagementResource()},
			infra: testClusterSNOInfra(),
		},
		{
			name:               "should ignore guaranteed pod",
			pod:                testManagedPod("500m", "500m", "500Mi", "500Mi"),
			expectedCpuRequest: resource.MustParse("500m"),
			namespace:          testManagedNamespace(),
			expectedAnnotations: map[string]string{
				workloadAdmissionWarning: "skip pod CPUs requests modifications because it has guaranteed QoS class",
			},
			nodes: []*corev1.Node{testNodeWithManagementResource()},
			infra: testClusterSNOInfra(),
		},
		{
			name:               "should ignore pod when one of pod containers have both CPU limit and request",
			pod:                testManagedPod("500m", "250m", "500Mi", ""),
			expectedCpuRequest: resource.MustParse("250m"),
			namespace:          testManagedNamespace(),
			expectedAnnotations: map[string]string{
				workloadAdmissionWarning: fmt.Sprintf("skip pod CPUs requests modifications because pod container has both CPU limit and request"),
			},
			nodes: []*corev1.Node{testNodeWithManagementResource()},
			infra: testClusterSNOInfra(),
		},
		{
			name:               "should ignore pod when removing the CPU request will change the pod QoS class to best-effort",
			pod:                testManagedPod("", "250m", "", ""),
			expectedCpuRequest: resource.MustParse("250m"),
			namespace:          testManagedNamespace(),
			expectedAnnotations: map[string]string{
				workloadAdmissionWarning: fmt.Sprintf("skip pod CPUs requests modifications because it will change the pod QoS class from %s to %s", corev1.PodQOSBurstable, corev1.PodQOSBestEffort),
			},
			nodes: []*corev1.Node{testNodeWithManagementResource()},
			infra: testClusterSNOInfra(),
		},
		{
			name:               "should not mutate the pod when at least one node does not have management resources",
			pod:                testManagedPod("500m", "250m", "500Mi", "250Mi"),
			expectedCpuRequest: resource.MustParse("250m"),
			namespace:          testManagedNamespace(),
			nodes:              []*corev1.Node{testNode()},
			infra:              testClusterSNOInfra(),
		},
		{
			name:               "should return admission error when the cluster does not have any nodes",
			pod:                testManagedPod("500m", "250m", "500Mi", "250Mi"),
			expectedCpuRequest: resource.MustParse("250m"),
			namespace:          testManagedNamespace(),
			nodes:              []*corev1.Node{},
			infra:              testClusterSNOInfra(),
			expectedError:      fmt.Errorf("the cluster does not have any nodes"),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			m, err := getMockManagementCPUsOverride(test.namespace, test.nodes, test.infra)
			if err != nil {
				t.Fatalf("%s: failed to get mock managementCPUsOverride: %v", test.name, err)
			}

			test.pod.Namespace = test.namespace.Name

			attrs := admission.NewAttributesRecord(test.pod, nil, schema.GroupVersionKind{}, test.pod.Namespace, test.pod.Name, kapi.Resource("pods").WithVersion("version"), "", admission.Create, nil, false, fakeUser())
			err = m.Admit(context.TODO(), attrs, nil)
			if err != nil {
				if test.expectedError == nil {
					t.Fatalf("%s: admission controller returned error: %v", test.name, err)
				}

				if !strings.Contains(err.Error(), test.expectedError.Error()) {
					t.Fatalf("%s: the expected error %v, got %v", test.name, test.expectedError, err)
				}
			}

			if err == nil && test.expectedError != nil {
				t.Fatalf("%s: the expected error %v, got nil", test.name, test.expectedError)
			}

			if test.expectedAnnotations != nil && !reflect.DeepEqual(test.expectedAnnotations, test.pod.Annotations) {
				t.Fatalf("%s: the pod annotations do not match; %v should be %v", test.name, test.pod.Annotations, test.expectedAnnotations)
			}

			resources := test.pod.Spec.InitContainers[0].Resources // only test one container
			if actual := resources.Requests[kapi.ResourceCPU]; test.expectedCpuRequest.Cmp(actual) != 0 {
				t.Fatalf("%s: cpu requests do not match; %v should be %v", test.name, actual, test.expectedCpuRequest)
			}

			resources = test.pod.Spec.Containers[0].Resources // only test one container
			if actual := resources.Requests[kapi.ResourceCPU]; test.expectedCpuRequest.Cmp(actual) != 0 {
				t.Fatalf("%s: cpu requests do not match; %v should be %v", test.name, actual, test.expectedCpuRequest)
			}
		})
	}
}

func TestGetPodQoSClass(t *testing.T) {
	tests := []struct {
		name             string
		pod              *kapi.Pod
		expectedQoSClass coreapi.PodQOSClass
	}{
		{
			name:             "should recognize best-effort pod",
			pod:              testManagedPod("", "", "", ""),
			expectedQoSClass: coreapi.PodQOSBestEffort,
		},
		{
			name:             "should recognize guaranteed pod",
			pod:              testManagedPod("100m", "100m", "100Mi", "100Mi"),
			expectedQoSClass: coreapi.PodQOSGuaranteed,
		},
		{
			name:             "should recognize guaranteed pod when CPU request equals to 0",
			pod:              testManagedPod("100m", "0", "100Mi", "100Mi"),
			expectedQoSClass: coreapi.PodQOSGuaranteed,
		},
		{
			name:             "should recognize burstable pod with only CPU limit",
			pod:              testManagedPod("100m", "", "", ""),
			expectedQoSClass: coreapi.PodQOSBurstable,
		},
		{
			name:             "should recognize burstable pod with only CPU request",
			pod:              testManagedPod("", "100m", "", ""),
			expectedQoSClass: coreapi.PodQOSBurstable,
		},
		{
			name:             "should recognize burstable pod with only memory limit",
			pod:              testManagedPod("", "", "100Mi", ""),
			expectedQoSClass: coreapi.PodQOSBurstable,
		},
		{
			name:             "should recognize burstable pod with only memory request",
			pod:              testManagedPod("", "", "", "100Mi"),
			expectedQoSClass: coreapi.PodQOSBurstable,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			allContainers := append([]coreapi.Container{}, test.pod.Spec.InitContainers...)
			allContainers = append(allContainers, test.pod.Spec.Containers...)
			qosClass := getPodQoSClass(allContainers)
			if qosClass != test.expectedQoSClass {
				t.Fatalf("%s: pod has QoS class %s; should be %s", test.name, qosClass, test.expectedQoSClass)
			}
		})
	}
}

func TestValidate(t *testing.T) {
	tests := []struct {
		name          string
		pod           *kapi.Pod
		namespace     *corev1.Namespace
		nodes         []*corev1.Node
		expectedError error
	}{
		{
			name: "should return invalid error when the pod has more than one workload annotation",
			pod: testManagedPodWithAnnotations(
				"500m",
				"250m",
				"500Mi",
				"250Mi",
				map[string]string{
					fmt.Sprintf("%s%s", podWorkloadTargetAnnotationPrefix, workloadTypeManagement): "",
					fmt.Sprintf("%stest", podWorkloadTargetAnnotationPrefix):                       "",
				},
			),
			namespace:     testManagedNamespace(),
			nodes:         []*corev1.Node{testNodeWithManagementResource()},
			expectedError: fmt.Errorf("the pod can not have more than one workload annotations"),
		},
		{
			name: "should return invalid error when the pod has incorrect workload annotation",
			pod: testManagedPodWithAnnotations(
				"500m",
				"250m",
				"500Mi",
				"250Mi",
				map[string]string{
					podWorkloadTargetAnnotationPrefix: "",
				},
			),
			namespace:     testManagedNamespace(),
			nodes:         []*corev1.Node{testNodeWithManagementResource()},
			expectedError: fmt.Errorf("the workload annotation key should have format %s<workload_type>", podWorkloadTargetAnnotationPrefix),
		},
		{
			name: "should return invalid error when the pod has cpuset resource annotation",
			pod: testManagedPodWithAnnotations(
				"500m",
				"250m",
				"500Mi",
				"250Mi",
				map[string]string{
					fmt.Sprintf("%s%s", podWorkloadTargetAnnotationPrefix, workloadTypeManagement): fmt.Sprintf(`{"%s":"%s"}`, podWorkloadAnnotationEffect, workloadEffectPreferredDuringScheduling),
					fmt.Sprintf("%s%s", containerResourcesAnnotationPrefix, "test"):                `{"cpuset": 1}`,
				},
			),
			namespace:     testManagedNamespace(),
			nodes:         []*corev1.Node{testNodeWithManagementResource()},
			expectedError: fmt.Errorf("he pod resource annotation value should have only cpushares key"),
		},
		{
			name: "should return invalid error when the pod does not have workload annotation, but has resource annotation",
			pod: testManagedPodWithAnnotations(
				"500m",
				"250m",
				"500Mi",
				"250Mi",
				map[string]string{
					fmt.Sprintf("%s%s", containerResourcesAnnotationPrefix, "test"): fmt.Sprintf(`{"%s": 2}`, containerResourcesAnnotationValueKeyCPUShares),
				},
			),
			namespace:     testManagedNamespace(),
			nodes:         []*corev1.Node{testNodeWithManagementResource()},
			expectedError: fmt.Errorf("the pod without workload annotation can not have resource annotation"),
		},
		{
			name: "should return invalid error when the pod does not have workload annotation, but the container has management resource",
			pod: testPodWithManagedResource(
				"500m",
				"250m",
				"500Mi",
				"250Mi",
			),
			namespace:     testManagedNamespace(),
			nodes:         []*corev1.Node{testNodeWithManagementResource()},
			expectedError: fmt.Errorf("the pod without workload annotations can not have containers with workload resources %q", "management.workload.openshift.io/cores"),
		},
		{
			name: "should return invalid error when the pod has workload annotation, but the pod namespace does not have allowed annotation",
			pod: testManagedPod(
				"500m",
				"250m",
				"500Mi",
				"250Mi",
			),
			namespace:     testNamespace(),
			nodes:         []*corev1.Node{testNodeWithManagementResource()},
			expectedError: fmt.Errorf("the pod can not have workload annotation, when the namespace %q does not allow it", "namespace"),
		},
		{
			name: "should not return any errors when the pod and namespace valid",
			pod: testManagedPodWithAnnotations(
				"500m",
				"250m",
				"500Mi",
				"250Mi",
				map[string]string{
					fmt.Sprintf("%s%s", containerResourcesAnnotationPrefix, "test"):                fmt.Sprintf(`{"%s": 256}`, containerResourcesAnnotationValueKeyCPUShares),
					fmt.Sprintf("%s%s", containerResourcesAnnotationPrefix, "initTest"):            fmt.Sprintf(`{"%s": 256}`, containerResourcesAnnotationValueKeyCPUShares),
					fmt.Sprintf("%s%s", podWorkloadTargetAnnotationPrefix, workloadTypeManagement): fmt.Sprintf(`{"%s":"%s"}`, podWorkloadAnnotationEffect, workloadEffectPreferredDuringScheduling),
				},
			),
			namespace: testManagedNamespace(),
			nodes:     []*corev1.Node{testNodeWithManagementResource()},
		},
		{
			name: "should skip static pod validation",
			pod: testManagedPodWithAnnotations(
				"500m",
				"250m",
				"500Mi",
				"250Mi",
				map[string]string{
					fmt.Sprintf("%s%s", podWorkloadTargetAnnotationPrefix, workloadTypeManagement): "",
					fmt.Sprintf("%stest", podWorkloadTargetAnnotationPrefix):                       "",
					kubetypes.ConfigSourceAnnotationKey:                                            kubetypes.FileSource,
				},
			),
			namespace: testManagedNamespace(),
			nodes:     []*corev1.Node{testNodeWithManagementResource()},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			m, err := getMockManagementCPUsOverride(test.namespace, test.nodes, nil)
			if err != nil {
				t.Fatalf("%s: failed to get mock managementCPUsOverride: %v", test.name, err)
			}
			test.pod.Namespace = test.namespace.Name

			attrs := admission.NewAttributesRecord(test.pod, nil, schema.GroupVersionKind{}, test.pod.Namespace, test.pod.Name, kapi.Resource("pods").WithVersion("version"), "", admission.Create, nil, false, fakeUser())
			err = m.Validate(context.TODO(), attrs, nil)
			if err != nil {
				if test.expectedError == nil {
					t.Fatalf("%s: admission controller returned error: %v", test.name, err)
				}

				if !strings.Contains(err.Error(), test.expectedError.Error()) {
					t.Fatalf("%s: the expected error %v, got %v", test.name, test.expectedError, err)
				}
			}

			if err == nil && test.expectedError != nil {
				t.Fatalf("%s: the expected error %v, got nil", test.name, test.expectedError)
			}
		})
	}
}

func testPodWithManagedResource(cpuLimit, cpuRequest, memoryLimit, memoryRequest string) *kapi.Pod {
	pod := testPod(cpuLimit, cpuRequest, memoryLimit, memoryRequest)

	managedResourceName := fmt.Sprintf("%s.%s", workloadTypeManagement, containerWorkloadResourceSuffix)

	managedResourceQuantity := resource.MustParse("26")
	pod.Spec.Containers[0].Resources.Requests[kapi.ResourceName(managedResourceName)] = managedResourceQuantity
	return pod
}

func testManagedPodWithAnnotations(cpuLimit, cpuRequest, memoryLimit, memoryRequest string, annotations map[string]string) *kapi.Pod {
	pod := testManagedPod(cpuLimit, cpuRequest, memoryLimit, memoryRequest)
	pod.Annotations = annotations
	return pod
}

func testManagedStaticPod(cpuLimit, cpuRequest, memoryLimit, memoryRequest string) *kapi.Pod {
	pod := testManagedPod(cpuLimit, cpuRequest, memoryLimit, memoryRequest)
	pod.Annotations[kubetypes.ConfigSourceAnnotationKey] = kubetypes.FileSource
	return pod
}

func testManagedPod(cpuLimit, cpuRequest, memoryLimit, memoryRequest string) *kapi.Pod {
	pod := testPod(cpuLimit, cpuRequest, memoryLimit, memoryRequest)

	pod.Annotations = map[string]string{}
	for _, c := range pod.Spec.InitContainers {
		cpusetAnnotation := fmt.Sprintf("%s%s", containerResourcesAnnotationPrefix, c.Name)
		pod.Annotations[cpusetAnnotation] = `{"cpuset": "0-1"}`
	}
	for _, c := range pod.Spec.Containers {
		cpusetAnnotation := fmt.Sprintf("%s%s", containerResourcesAnnotationPrefix, c.Name)
		pod.Annotations[cpusetAnnotation] = `{"cpuset": "0-1"}`
	}

	managementWorkloadAnnotation := fmt.Sprintf("%s%s", podWorkloadTargetAnnotationPrefix, workloadTypeManagement)
	pod.Annotations = map[string]string{
		managementWorkloadAnnotation: fmt.Sprintf(`{"%s":"%s"}`, podWorkloadAnnotationEffect, workloadEffectPreferredDuringScheduling),
	}

	return pod
}

func testPod(cpuLimit, cpuRequest, memoryLimit, memoryRequest string) *kapi.Pod {
	pod := &kapi.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			Namespace: "test",
		},
		Spec: kapi.PodSpec{
			InitContainers: []kapi.Container{
				{
					Name: "initTest",
				},
			},
			Containers: []kapi.Container{
				{
					Name: "test",
				},
			},
		},
	}

	var limits kapi.ResourceList
	// we need this kind of statement to verify assignment to entry in nil map
	if cpuLimit != "" || memoryLimit != "" {
		limits = kapi.ResourceList{}
		if cpuLimit != "" {
			limits[kapi.ResourceCPU] = resource.MustParse(cpuLimit)
		}

		if memoryLimit != "" {
			limits[kapi.ResourceMemory] = resource.MustParse(memoryLimit)
		}

		pod.Spec.InitContainers[0].Resources.Limits = limits.DeepCopy()
		pod.Spec.Containers[0].Resources.Limits = limits.DeepCopy()
	}

	var requests kapi.ResourceList
	// we need this kind of statement to verify assignment to entry in nil map
	if cpuRequest != "" || memoryRequest != "" {
		requests = kapi.ResourceList{}
		if cpuRequest != "" {
			requests[kapi.ResourceCPU] = resource.MustParse(cpuRequest)
		}
		if memoryRequest != "" {
			requests[kapi.ResourceMemory] = resource.MustParse(memoryRequest)
		}

		pod.Spec.InitContainers[0].Resources.Requests = requests.DeepCopy()
		pod.Spec.Containers[0].Resources.Requests = requests.DeepCopy()
	}

	return pod
}

func fakeUser() user.Info {
	return &user.DefaultInfo{
		Name: "testuser",
	}
}

func testNamespace() *corev1.Namespace {
	return &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: "namespace",
		},
	}
}

func testManagedNamespace() *corev1.Namespace {
	return &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: "managed-namespace",
			Annotations: map[string]string{
				namespaceAllowedAnnotation: fmt.Sprintf("%s,test", workloadTypeManagement),
			},
		},
	}
}

func testNode() *corev1.Node {
	return &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node",
		},
	}
}

func testNodeWithManagementResource() *corev1.Node {
	q := resource.NewQuantity(16000, resource.DecimalSI)
	return &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "managed-node",
		},
		Status: corev1.NodeStatus{
			Allocatable: corev1.ResourceList{
				managedCapacityLabel: *q,
			},
		},
	}
}

func testClusterInfraWithoutAnyStatusFields() *configv1.Infrastructure {
	return &configv1.Infrastructure{
		ObjectMeta: metav1.ObjectMeta{
			Name: infraClusterName,
		},
	}
}

func testClusterSNOInfra() *configv1.Infrastructure {
	return &configv1.Infrastructure{
		ObjectMeta: metav1.ObjectMeta{
			Name: infraClusterName,
		},
		Status: configv1.InfrastructureStatus{
			APIServerURL:           "test",
			ControlPlaneTopology:   configv1.SingleReplicaTopologyMode,
			InfrastructureTopology: configv1.SingleReplicaTopologyMode,
			CPUPartitioning:        configv1.CPUPartitioningAllNodes,
		},
	}
}

func testClusterInfraWithoutTopologyFields() *configv1.Infrastructure {
	infra := testClusterSNOInfra()
	infra.Status.ControlPlaneTopology = ""
	infra.Status.InfrastructureTopology = ""
	return infra
}
