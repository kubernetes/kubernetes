/*
Copyright 2019 The Kubernetes Authors.

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

package endpoint

import (
	"fmt"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/intstr"
	corelisters "k8s.io/client-go/listers/core/v1"
)

const (
	namespace = "NS1"
)

var (
	frontend1Pod = createPod(namespace, "frontend1", "app:frontend")
	frontend2Pod = createPod(namespace, "frontend2", "app:frontend")
	backend1Pod  = createPod(namespace, "backend1", "app:backend")
	backend2Pod  = createPod(namespace, "backend2", "app:backend")

	frontendService = createService(namespace, "frontend", "app:frontend")
	backendService  = createService(namespace, "backend", "app:backend")
)

func TestGetServiceAndPods(t *testing.T) {
	cache := newCacheUnderTest(t)

	cache.initialize(
		[]*v1.Service{frontendService, backendService},
		[]*v1.Pod{frontend1Pod, frontend2Pod, backend1Pod, backend2Pod})

	cache.whenGetServiceAndPods(namespace, "frontend").
		expectService(frontendService).
		expectPods(frontend1Pod, frontend2Pod)
}

func TestGetServiceAndPods_ServiceDeleted(t *testing.T) {
	cache := newCacheUnderTest(t)

	cache.initialize(
		[]*v1.Service{frontendService, backendService},
		[]*v1.Pod{frontend1Pod, frontend2Pod, backend1Pod, backend2Pod})

	cache.DeleteService(frontendService)

	// Sync can happen after service was deleted, this is a valid call and should return nil service
	// and no pods.
	cache.whenGetServiceAndPods(namespace, "frontend").
		expectService(nil).
		expectPods()
}

func TestGetServiceAndPods_ServiceAddedNoPods(t *testing.T) {
	cache := newCacheUnderTest(t)

	cache.initialize(
		[]*v1.Service{frontendService, backendService},
		[]*v1.Pod{frontend1Pod, frontend2Pod, backend1Pod, backend2Pod})

	service := createService(namespace, "my-service", "app:new-app")
	cache.AddService(service)

	cache.whenGetServiceAndPods(namespace, "my-service").
		expectService(service).
		expectPods() // No Pods
}

func TestGetServiceAndPods_ServiceAddedMatchingPods(t *testing.T) {
	cache := newCacheUnderTest(t)

	pod1 := createPod(namespace, "pod1", "app:new-app")
	pod2 := createPod(namespace, "pod2", "app:new-app")

	cache.initialize(
		[]*v1.Service{},
		[]*v1.Pod{pod1, pod2})

	service := createService(namespace, "my-service", "app:new-app")
	cache.AddService(service)

	cache.whenGetServiceAndPods(namespace, "my-service").
		expectService(service).
		expectPods(pod1, pod2) // No Pods
}

func TestGetServiceAndPods_ServiceWithEmptySelector(t *testing.T) {
	cache := newCacheUnderTest(t)

	cache.initialize(
		[]*v1.Service{frontendService, backendService},
		[]*v1.Pod{frontend1Pod, frontend2Pod, backend1Pod, backend2Pod})

	service := createService(namespace, "my-service") // No selector.
	cache.AddService(service)

	cache.whenGetServiceAndPods(namespace, "my-service").
		expectService(service).
		expectPods() // No Pods, no selector = nothing.
}

func TestGetServiceAndPods_ServiceUpdated(t *testing.T) {
	cache := newCacheUnderTest(t)

	pod1 := createPod(namespace, "pod1", "app:frontend", "env:dev")
	pod2 := createPod(namespace, "pod2", "app:frontend", "env:prod")

	service := createService(namespace, "my-service", "app:frontend")

	cache.initialize(
		[]*v1.Service{frontendService, service},
		[]*v1.Pod{pod1, pod2})

	cache.whenGetServiceAndPods(namespace, "my-service").
		expectService(service).
		expectPods(pod1, pod2)

	// Updated service matches only "prod" pods.
	updatedService := createService(namespace, "my-service", "app:frontend", "env:prod")
	cache.UpdateService(service, updatedService)

	cache.whenGetServiceAndPods(namespace, "my-service").
		expectService(updatedService).
		expectPods(pod2)
}

func TestGetServiceAndPods_PodAdded(t *testing.T) {
	cache := newCacheUnderTest(t)

	cache.initialize(
		[]*v1.Service{frontendService, backendService},
		[]*v1.Pod{frontend1Pod, frontend2Pod, backend1Pod, backend2Pod})

	cache.whenGetServiceAndPods(namespace, "frontend").
		expectService(frontendService).
		expectPods(frontend1Pod, frontend2Pod)

	pod := createPod(namespace, "pod1", "app:frontend")
	cache.AddPod(pod)

	cache.whenGetServiceAndPods(namespace, "frontend").
		expectService(frontendService).
		expectPods(frontend1Pod, frontend2Pod, pod)
}

func TestGetServiceAndPods_PodUpdated(t *testing.T) {
	cache := newCacheUnderTest(t)

	pod := createPod(namespace, "pod1", "app:frontend")

	cache.initialize(
		[]*v1.Service{frontendService, backendService},
		[]*v1.Pod{frontend1Pod, frontend2Pod, backend1Pod, backend2Pod, pod})

	cache.whenGetServiceAndPods(namespace, "frontend").
		expectService(frontendService).
		expectPods(frontend1Pod, frontend2Pod, pod)
	cache.whenGetServiceAndPods(namespace, "backend").
		expectService(backendService).
		expectPods(backend1Pod, backend2Pod)

	updatedPod := createPod(namespace, "pod1", "app:backend") // Now pod belongs to backend.
	cache.UpdatePod(pod, updatedPod)

	cache.whenGetServiceAndPods(namespace, "frontend").
		expectService(frontendService).
		expectPods(frontend1Pod, frontend2Pod)
	cache.whenGetServiceAndPods(namespace, "backend").
		expectService(backendService).
		expectPods(backend1Pod, backend2Pod, updatedPod)
}

func TestGetServiceAndPods_PodDeleted(t *testing.T) {
	cache := newCacheUnderTest(t)

	cache.initialize(
		[]*v1.Service{frontendService, backendService},
		[]*v1.Pod{frontend1Pod, frontend2Pod, backend1Pod, backend2Pod})

	cache.whenGetServiceAndPods(namespace, "frontend").
		expectService(frontendService).
		expectPods(frontend1Pod, frontend2Pod)

	cache.DeletePod(frontend1Pod)

	cache.whenGetServiceAndPods(namespace, "frontend").
		expectService(frontendService).
		expectPods(frontend2Pod)
}

func TestGetServiceAndPods_PodInDIfferentNamespace(t *testing.T) {
	cache := newCacheUnderTest(t)

	ns2Pod := createPod("NS2", "frontend1", "app:frontend")
	ns2Service := createService("NS2", "frontend", "app:frontend")

	cache.initialize(
		[]*v1.Service{frontendService, backendService, ns2Service},
		[]*v1.Pod{frontend1Pod, frontend2Pod, backend1Pod, backend2Pod, ns2Pod})

	cache.whenGetServiceAndPods(namespace, "frontend").
		expectService(frontendService).
		expectPods(frontend1Pod, frontend2Pod) // No ns2Pod.

	cache.whenGetServiceAndPods("NS2", "frontend").
		expectService(ns2Service).
		expectPods(ns2Pod)
}

func TestAddPod(t *testing.T) {
	cache := newCacheUnderTest(t)

	cache.initialize(
		[]*v1.Service{frontendService, backendService},
		[]*v1.Pod{frontend1Pod, frontend2Pod, backend1Pod})

	cache.whenAddPod(backend2Pod).expect("NS1/backend")
}

func TestAddPod_NoMatchingService(t *testing.T) {
	cache := newCacheUnderTest(t)

	cache.initialize(
		[]*v1.Service{frontendService, backendService},
		[]*v1.Pod{frontend1Pod, frontend2Pod, backend1Pod})

	pod := createPod(namespace, "pod", "app:new-app")
	cache.whenAddPod(pod).expect()
}

func TestAddPod_MultiServicePod(t *testing.T) {
	cache := newCacheUnderTest(t)

	pod := createPod(namespace, "my-pod", "app:frontend", "env:dev")
	service := createService(namespace, "dev", "env:dev")

	cache.initialize(
		[]*v1.Service{frontendService, backendService, service},
		[]*v1.Pod{frontend1Pod, frontend2Pod, backend1Pod, pod})

	cache.whenAddPod(pod).expect("NS1/dev", "NS1/frontend")
}

func TestUpdatePod(t *testing.T) {
	cache := newCacheUnderTest(t)

	cache.initialize(
		[]*v1.Service{frontendService, backendService},
		[]*v1.Pod{frontend1Pod, frontend2Pod, backend1Pod, backend2Pod})

	updatedPod := createPod(namespace, "frontend2", "app:backend")
	cache.whenUpdatePod(frontend2Pod, updatedPod).
		// Pod moved from backend to frontend, both services should be synced.
		expect("NS1/frontend", "NS1/backend")
}

func TestDeletePod(t *testing.T) {
	cache := newCacheUnderTest(t)

	cache.initialize(
		[]*v1.Service{frontendService, backendService},
		[]*v1.Pod{frontend1Pod, frontend2Pod, backend1Pod, backend2Pod})

	cache.whenDeletePod(backend1Pod).expect("NS1/backend")
}

func TestAddPod_DifferentNamespace(t *testing.T) {
	cache := newCacheUnderTest(t)

	ns2Service := createService("NS2", "frontend", "app:frontend")

	cache.initialize(
		[]*v1.Service{frontendService, backendService, ns2Service},
		[]*v1.Pod{frontend1Pod, frontend2Pod, backend1Pod})

	pod := createPod("NS2", "frontend1", "app:frontend")
	// NS1/frontend not returned, even though selector matches.
	cache.whenAddPod(pod).expect("NS2/frontend")
}

// ------- Test Utils -------

type tester struct {
	*endpointsControllerCache
	t *testing.T
}

func newCacheUnderTest(t *testing.T) *tester {
	return &tester{newEndpointsControllerCache(), t}
}

func (t *tester) initialize(services []*v1.Service, pods []*v1.Pod) {
	t.Initialize(serviceLister(services), podLister(pods))
}

func (t *tester) whenGetServiceAndPods(namespace, name string) *serviceAndPodsSubject {
	service, pods := t.GetServiceAndPods(namespace, name)
	sortPods(pods)
	return &serviceAndPodsSubject{service, pods, t.t}
}

func (t *tester) whenAddPod(pod *v1.Pod) *serviceNameSliceSubject {
	return newServiceNameSliceSubject(t.AddPod(pod), t.t)
}

func (t *tester) whenUpdatePod(before, after *v1.Pod) *serviceNameSliceSubject {
	return newServiceNameSliceSubject(t.UpdatePod(before, after), t.t)
}

func (t *tester) whenDeletePod(pod *v1.Pod) *serviceNameSliceSubject {
	return newServiceNameSliceSubject(t.DeletePod(pod), t.t)
}

type serviceAndPodsSubject struct {
	gotService *v1.Service
	gotPods    []*v1.Pod
	t          *testing.T
}

func (s *serviceAndPodsSubject) expectService(expected *v1.Service) *serviceAndPodsSubject {
	assertEqual(s.gotService, expected, s.t)
	return s
}

func (s *serviceAndPodsSubject) expectPods(expected ...*v1.Pod) *serviceAndPodsSubject {
	sortPods(expected)
	assertEqual(s.gotPods, expected, s.t)
	return s
}

type serviceNameSliceSubject struct {
	got []string
	t   *testing.T
}

func newServiceNameSliceSubject(got []string, t *testing.T) *serviceNameSliceSubject {
	sort.Strings(got)
	return &serviceNameSliceSubject{got, t}
}

func (s *serviceNameSliceSubject) expect(expected ...string) {
	if expected == nil {
		expected = []string{}
	}
	sort.Strings(expected)
	assertEqual(s.got, expected, s.t)
}

func assertEqual(got interface{}, expected interface{}, t *testing.T) {
	if !reflect.DeepEqual(got, expected) {
		_, fn, line, _ := runtime.Caller(2)
		t.Errorf("Failed assertion in %s:%d expected %s, got %s",
			fn, line, toString(expected), toString(got))
	}
}

func toString(o interface{}) string {
	if o == nil {
		return "<nil>"
	} else if s, ok := o.(string); ok {
		return s
	} else if pod, ok := o.(*v1.Pod); ok {
		return fmt.Sprintf("Pod %s/%s", pod.Namespace, pod.Name)
	} else if service, ok := o.(*v1.Service); ok {
		return fmt.Sprintf("Service %s/%s", service.Namespace, service.Name)
	} else if reflect.TypeOf(o).Kind() == reflect.Slice {
		slice := reflect.ValueOf(o)
		stringSlice := make([]string, slice.Len())
		for i := 0; i < slice.Len(); i++ {
			stringSlice[i] = toString(slice.Index(i).Interface())
		}
		return "[ " + strings.Join(stringSlice, ", ") + " ]"
	}

	panic(fmt.Sprintf("Unexpected object: %s", o))
}

func sortPods(pods []*v1.Pod) {
	sort.Slice(pods, func(i, j int) bool {
		key := func(i int) string { return pods[i].Namespace + "/" + pods[i].Name }
		return key(i) < key(j)
	})
}

func parseLabels(labelsAsStrings []string) (labels map[string]string) {
	if len(labelsAsStrings) == 0 {
		return nil
	}
	labels = make(map[string]string)
	for _, label := range labelsAsStrings {
		keyAndValue := strings.Split(label, ":")
		labels[keyAndValue[0]] = keyAndValue[1]
	}
	return labels
}

func createPod(namespace, name string, labelsAsStrings ...string) *v1.Pod {
	return &v1.Pod{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1"},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
			Labels:    parseLabels(labelsAsStrings),
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{Ports: []v1.ContainerPort{}}},
		},
		Status: v1.PodStatus{
			PodIP: "1.2.3.4",
			Conditions: []v1.PodCondition{
				{
					Type:   v1.PodReady,
					Status: v1.ConditionTrue,
				},
			},
		},
	}
}

func createService(namespace, name string, selectorAsStrings ...string) *v1.Service {
	return &v1.Service{
		TypeMeta:   metav1.TypeMeta{APIVersion: "v1"},
		ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: name},
		Spec: v1.ServiceSpec{
			Selector: parseLabels(selectorAsStrings),
			Ports:    []v1.ServicePort{{Port: 80, Protocol: "TCP", TargetPort: intstr.FromInt(8080)}},
		},
	}
}

type serviceLister []*v1.Service

func (s serviceLister) List(selector labels.Selector) (ret []*v1.Service, err error) { return s, nil }
func (serviceLister) Services(namespace string) corelisters.ServiceNamespaceLister {
	panic("Not Implemented")
}
func (serviceLister) GetPodServices(pod *v1.Pod) ([]*v1.Service, error) { panic("Not Implemented") }

type podLister []*v1.Pod

func (p podLister) List(selector labels.Selector) (ret []*v1.Pod, err error) { return p, nil }
func (podLister) Pods(namespace string) corelisters.PodNamespaceLister       { panic("Not Implemented!") }
