package mixedcpus

import (
	"context"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/client-go/kubernetes/fake"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	coreapi "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/test/e2e/framework/pod"
)

func TestAdmit(t *testing.T) {
	testCases := []struct {
		name              string
		pod               *coreapi.Pod
		ns                *corev1.Namespace
		expectedPodStatus *errors.StatusError
		// container names that should have the runtime annotation
		expectedContainersWithAnnotations []string
	}{
		{
			name: "one container, requests single resources",
			pod: makePod("test1", withNs("foo"),
				withGuaranteedContainer("cnt1",
					map[coreapi.ResourceName]resource.Quantity{
						coreapi.ResourceCPU:          resource.MustParse("1"),
						coreapi.ResourceMemory:       resource.MustParse("100Mi"),
						containerResourceRequestName: resource.MustParse("1"),
					},
				)),
			ns:                                makeNs("foo", map[string]string{namespaceAllowedAnnotation: ""}),
			expectedContainersWithAnnotations: []string{"cnt1"},
			expectedPodStatus:                 nil,
		},
		{
			name: "two containers, only one of them requests single resource",
			pod: makePod("test1", withNs("foo"),
				withGuaranteedContainer("cnt1",
					map[coreapi.ResourceName]resource.Quantity{
						coreapi.ResourceCPU:    resource.MustParse("1"),
						coreapi.ResourceMemory: resource.MustParse("100Mi"),
					},
				),
				withGuaranteedContainer("cnt2",
					map[coreapi.ResourceName]resource.Quantity{
						coreapi.ResourceCPU:          resource.MustParse("1"),
						coreapi.ResourceMemory:       resource.MustParse("100Mi"),
						containerResourceRequestName: resource.MustParse("1"),
					},
				)),
			ns:                                makeNs("foo", map[string]string{namespaceAllowedAnnotation: ""}),
			expectedContainersWithAnnotations: []string{"cnt2"},
			expectedPodStatus:                 nil,
		},
		{
			name: "two containers, one of them requests more than single resource",
			pod: makePod("test1", withNs("bar"),
				withGuaranteedContainer("cnt1",
					map[coreapi.ResourceName]resource.Quantity{
						coreapi.ResourceCPU:          resource.MustParse("1"),
						coreapi.ResourceMemory:       resource.MustParse("100Mi"),
						containerResourceRequestName: resource.MustParse("1"),
					},
				),
				withGuaranteedContainer("cnt2",
					map[coreapi.ResourceName]resource.Quantity{
						coreapi.ResourceCPU:          resource.MustParse("1"),
						coreapi.ResourceMemory:       resource.MustParse("100Mi"),
						containerResourceRequestName: resource.MustParse("2"),
					},
				)),
			ns:                                makeNs("bar", map[string]string{namespaceAllowedAnnotation: ""}),
			expectedContainersWithAnnotations: []string{},
			expectedPodStatus:                 errors.NewForbidden(schema.GroupResource{}, "", nil),
		},
		{
			name: "one container, pod is not Guaranteed QoS class",
			pod: makePod("test1", withNs("bar"),
				withContainer("cnt1",
					map[coreapi.ResourceName]resource.Quantity{
						coreapi.ResourceCPU:          resource.MustParse("1"),
						coreapi.ResourceMemory:       resource.MustParse("100Mi"),
						containerResourceRequestName: resource.MustParse("1"),
					},
				),
			),
			ns:                                makeNs("bar", map[string]string{namespaceAllowedAnnotation: ""}),
			expectedContainersWithAnnotations: []string{},
			expectedPodStatus:                 errors.NewForbidden(schema.GroupResource{}, "", nil),
		},
		{
			name: "one container, pod is not in allowed namespace",
			pod: makePod("test1",
				withGuaranteedContainer("cnt1",
					map[coreapi.ResourceName]resource.Quantity{
						coreapi.ResourceCPU:          resource.MustParse("1"),
						coreapi.ResourceMemory:       resource.MustParse("100Mi"),
						containerResourceRequestName: resource.MustParse("1"),
					},
				),
			),
			ns:                                makeNs("bar", map[string]string{namespaceAllowedAnnotation: ""}),
			expectedContainersWithAnnotations: []string{},
			expectedPodStatus:                 errors.NewForbidden(schema.GroupResource{}, "", nil),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			testPod := tc.pod
			mutation, err := getMockMixedCPUsMutation(testPod, tc.ns)
			if err != nil {
				t.Fatalf("%v", err)
			}
			attrs := admission.NewAttributesRecord(testPod,
				nil,
				schema.GroupVersionKind{},
				testPod.Namespace,
				testPod.Name,
				coreapi.Resource("pods").WithVersion("version"),
				"",
				admission.Create,
				nil,
				false,
				fakeUser())

			err = mutation.Admit(context.TODO(), attrs, nil)
			if err != nil && tc.expectedPodStatus == nil {
				t.Errorf("%s: unexpected error %v", tc.name, err)
			}

			if err != nil {
				if !errors.IsForbidden(tc.expectedPodStatus) {
					t.Errorf("%s: forbidden error was expected. got %v instead", tc.name, err)
				}
			}

			testPod, _ = attrs.GetObject().(*coreapi.Pod)
			for _, cntName := range tc.expectedContainersWithAnnotations {
				if v, ok := testPod.Annotations[getRuntimeAnnotationName(cntName)]; !ok || v != annotationEnable {
					t.Errorf("%s: container %s is missing runtime annotation", tc.name, cntName)
				}
			}
		})
	}
}

func fakeUser() user.Info {
	return &user.DefaultInfo{
		Name: "testuser",
	}
}

func makeNs(name string, annotations map[string]string) *corev1.Namespace {
	return &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
			Annotations: annotations,
		},
	}
}

func makePod(name string, opts ...func(pod *coreapi.Pod)) *coreapi.Pod {
	p := &coreapi.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
	}
	for _, opt := range opts {
		opt(p)
	}
	return p
}

func withContainer(name string, requests coreapi.ResourceList) func(p *coreapi.Pod) {
	return func(p *coreapi.Pod) {
		cnt := coreapi.Container{
			Name:  name,
			Image: pod.GetDefaultTestImage(),
			Resources: coreapi.ResourceRequirements{
				Requests: requests,
			},
		}
		p.Spec.Containers = append(p.Spec.Containers, cnt)
	}
}

func withGuaranteedContainer(name string, requests coreapi.ResourceList) func(p *coreapi.Pod) {
	return func(p *coreapi.Pod) {
		withContainer(name, requests)(p)
		for i := 0; i < len(p.Spec.Containers); i++ {
			cnt := &p.Spec.Containers[i]
			if cnt.Name == name {
				cnt.Resources.Limits = cnt.Resources.Requests
			}
		}
	}
}

func withNs(name string) func(p *coreapi.Pod) {
	return func(p *coreapi.Pod) {
		p.Namespace = name
	}
}

func fakePodLister(pod *coreapi.Pod) corev1listers.PodLister {
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	if pod != nil {
		_ = indexer.Add(pod)
	}
	return corev1listers.NewPodLister(indexer)
}

func fakeNsLister(ns *corev1.Namespace) corev1listers.NamespaceLister {
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	_ = indexer.Add(ns)
	return corev1listers.NewNamespaceLister(indexer)
}

func getMockMixedCPUsMutation(pod *coreapi.Pod, ns *corev1.Namespace) (*mixedCPUsMutation, error) {
	m := &mixedCPUsMutation{
		Handler:         admission.NewHandler(admission.Create),
		client:          &fake.Clientset{},
		podListerSynced: func() bool { return true },
		podLister:       fakePodLister(pod),
		nsListerSynced:  func() bool { return true },
		nsLister:        fakeNsLister(ns),
	}
	if err := m.ValidateInitialization(); err != nil {
		return nil, err
	}

	return m, nil
}
