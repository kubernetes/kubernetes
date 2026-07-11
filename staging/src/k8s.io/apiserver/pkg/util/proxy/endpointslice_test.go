/*
Copyright 2025 The Kubernetes Authors.

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

package proxy

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	discoveryv1 "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/cache"
)

func makeEndpointSlice(namespace, service, slice int, ip string) *discoveryv1.EndpointSlice {
	namespaceName := fmt.Sprintf("namespace%d", namespace)
	serviceName := fmt.Sprintf("service%d", service)
	sliceName := fmt.Sprintf("service%d-%d%d%d", service, slice, slice, slice)
	return &discoveryv1.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      sliceName,
			Namespace: namespaceName,
			Labels: map[string]string{
				discoveryv1.LabelServiceName: serviceName,
			},
		},
		Endpoints: []discoveryv1.Endpoint{
			{
				Addresses: []string{ip},
			},
		},
	}
}

// Note that we resuse the same service names in the two namespaces to test proper
// indexing/namespacing.
var (
	// namespace1 service1:
	// - initial: 1 slice
	// - phase1:  add 1 slice
	// - phase2:  no change
	n1s1FirstSlice    = makeEndpointSlice(1, 1, 1, "10.1.1.1")
	n1s1SecondSlice   = makeEndpointSlice(1, 1, 2, "10.1.1.2")
	n1s1InitialSlices = []*discoveryv1.EndpointSlice{n1s1FirstSlice}
	n1s1Phase1Slices  = []*discoveryv1.EndpointSlice{n1s1FirstSlice, n1s1SecondSlice}
	n1s1Phase2Slices  = []*discoveryv1.EndpointSlice{n1s1FirstSlice, n1s1SecondSlice}

	// namespace1 service2:
	// - initial: 1 slice
	// - phase1:  update slice
	// - phase2:  delete slice
	n1s2FirstSlice    = makeEndpointSlice(1, 2, 1, "10.1.2.1")
	n1s2UpdatedSlice  = makeEndpointSlice(1, 2, 1, "10.1.2.99")
	n1s2InitialSlices = []*discoveryv1.EndpointSlice{n1s2FirstSlice}
	n1s2Phase1Slices  = []*discoveryv1.EndpointSlice{n1s2UpdatedSlice}
	n1s2Phase2Slices  = []*discoveryv1.EndpointSlice{}

	// namespace2 service 1:
	// - initial: 2 slices
	// - phase1:  delete first slice
	// - phase2:  delete second slice
	n2s1FirstSlice    = makeEndpointSlice(2, 1, 1, "10.2.1.1")
	n2s1SecondSlice   = makeEndpointSlice(2, 1, 2, "10.2.1.2")
	n2s1InitialSlices = []*discoveryv1.EndpointSlice{n2s1FirstSlice, n2s1SecondSlice}
	n2s1Phase1Slices  = []*discoveryv1.EndpointSlice{n2s1SecondSlice}
	n2s1Phase2Slices  = []*discoveryv1.EndpointSlice{}

	// namespace2 service 2:
	// - initial: no slices
	// - phase1:  no change
	// - phase2:  create slice
	n2s2FirstSlice    = makeEndpointSlice(2, 2, 1, "10.2.2.1")
	n2s2InitialSlices = []*discoveryv1.EndpointSlice{}
	n2s2Phase1Slices  = []*discoveryv1.EndpointSlice{}
	n2s2Phase2Slices  = []*discoveryv1.EndpointSlice{n2s2FirstSlice}

	initialSlices = []runtime.Object{
		n1s1FirstSlice, n1s2FirstSlice, n2s1FirstSlice, n2s1SecondSlice,
	}
)

func assertSlices(t *testing.T, getter EndpointSliceGetter, namespace, service string, expected []*discoveryv1.EndpointSlice) {
	t.Helper()

	// Poll because the informers may not sync immediately
	var lastErr error
	err := wait.PollUntilContextTimeout(context.Background(), 10*time.Millisecond, time.Second, true, func(ctx context.Context) (bool, error) {
		slices, err := getter.GetEndpointSlices(namespace, service)
		if err != nil {
			lastErr = fmt.Errorf("unexpected error getting %s/%s slices: %w", namespace, service, err)
			return false, nil
		}
		// cmp.Diff doesn't deal with nil vs []
		if len(expected) == 0 && len(slices) == 0 {
			return true, nil
		}
		if diff := cmp.Diff(expected, slices); diff != "" {
			lastErr = fmt.Errorf("slices for %s/%s did not match expectation:\n%s", namespace, service, diff)
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		t.Errorf("%s", lastErr)
	}
}

func testGetter(t *testing.T, client clientset.Interface, getter EndpointSliceGetter) {
	ctx := context.Background()

	// Check initial state
	assertSlices(t, getter, "namespace1", "service1", n1s1InitialSlices)
	assertSlices(t, getter, "namespace1", "service2", n1s2InitialSlices)
	assertSlices(t, getter, "namespace2", "service1", n2s1InitialSlices)
	assertSlices(t, getter, "namespace2", "service2", n2s2InitialSlices)

	_, err := client.DiscoveryV1().EndpointSlices("namespace1").Create(ctx, n1s1SecondSlice, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_, err = client.DiscoveryV1().EndpointSlices("namespace1").Update(ctx, n1s2UpdatedSlice, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	err = client.DiscoveryV1().EndpointSlices("namespace2").Delete(ctx, n2s1FirstSlice.Name, metav1.DeleteOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	assertSlices(t, getter, "namespace1", "service1", n1s1Phase1Slices)
	assertSlices(t, getter, "namespace1", "service2", n1s2Phase1Slices)
	assertSlices(t, getter, "namespace2", "service1", n2s1Phase1Slices)
	assertSlices(t, getter, "namespace2", "service2", n2s2Phase1Slices)

	err = client.DiscoveryV1().EndpointSlices("namespace1").Delete(ctx, n1s2FirstSlice.Name, metav1.DeleteOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	err = client.DiscoveryV1().EndpointSlices("namespace2").Delete(ctx, n2s1SecondSlice.Name, metav1.DeleteOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_, err = client.DiscoveryV1().EndpointSlices("namespace2").Create(ctx, n2s2FirstSlice, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	assertSlices(t, getter, "namespace1", "service1", n1s1Phase2Slices)
	assertSlices(t, getter, "namespace1", "service2", n1s2Phase2Slices)
	assertSlices(t, getter, "namespace2", "service1", n2s1Phase2Slices)
	assertSlices(t, getter, "namespace2", "service2", n2s2Phase2Slices)
}

func TestNewEndpointSliceIndexerGetter(t *testing.T) {
	client := clientsetfake.NewSimpleClientset(initialSlices...)
	informerFactory := informers.NewSharedInformerFactory(client, 30*time.Second)
	getter, err := NewEndpointSliceIndexerGetter(informerFactory.Discovery().V1().EndpointSlices())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	informerFactory.Start(wait.NeverStop)
	cache.WaitForCacheSync(nil, informerFactory.Discovery().V1().EndpointSlices().Informer().HasSynced)

	testGetter(t, client, getter)
}

func TestNewEndpointSliceListerGetter(t *testing.T) {
	client := clientsetfake.NewSimpleClientset(initialSlices...)
	informerFactory := informers.NewSharedInformerFactory(client, 30*time.Second)
	getter, err := NewEndpointSliceListerGetter(informerFactory.Discovery().V1().EndpointSlices().Lister())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	informerFactory.Start(wait.NeverStop)
	cache.WaitForCacheSync(nil, informerFactory.Discovery().V1().EndpointSlices().Informer().HasSynced)

	testGetter(t, client, getter)
}
