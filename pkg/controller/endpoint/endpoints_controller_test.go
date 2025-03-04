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

package endpoint

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	clientscheme "k8s.io/client-go/kubernetes/scheme"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	utiltesting "k8s.io/client-go/util/testing"
	endptspkg "k8s.io/kubernetes/pkg/api/v1/endpoints"
	api "k8s.io/kubernetes/pkg/apis/core"
	controllerpkg "k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/test/utils/ktesting"
	utilnet "k8s.io/utils/net"
	"k8s.io/utils/pointer"
)

var alwaysReady = func() bool { return true }
var neverReady = func() bool { return false }
var emptyNodeName string
var triggerTime = time.Date(2018, 01, 01, 0, 0, 0, 0, time.UTC)
var triggerTimeString = triggerTime.Format(time.RFC3339Nano)
var oldTriggerTimeString = triggerTime.Add(-time.Hour).Format(time.RFC3339Nano)

var ipv4only = []v1.IPFamily{v1.IPv4Protocol}
var ipv6only = []v1.IPFamily{v1.IPv6Protocol}
var ipv4ipv6 = []v1.IPFamily{v1.IPv4Protocol, v1.IPv6Protocol}
var ipv6ipv4 = []v1.IPFamily{v1.IPv6Protocol, v1.IPv4Protocol}

func testPod(namespace string, id int, nPorts int, isReady bool, ipFamilies []v1.IPFamily) *v1.Pod {
	p := &v1.Pod{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1"},
		ObjectMeta: metav1.ObjectMeta{
			Namespace:       namespace,
			Name:            fmt.Sprintf("pod%d", id),
			Labels:          map[string]string{"foo": "bar"},
			ResourceVersion: fmt.Sprint(id),
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{Ports: []v1.ContainerPort{}}},
		},
		Status: v1.PodStatus{
			Conditions: []v1.PodCondition{
				{
					Type:   v1.PodReady,
					Status: v1.ConditionTrue,
				},
			},
		},
	}
	if !isReady {
		p.Status.Conditions[0].Status = v1.ConditionFalse
	}
	for j := 0; j < nPorts; j++ {
		p.Spec.Containers[0].Ports = append(p.Spec.Containers[0].Ports,
			v1.ContainerPort{Name: fmt.Sprintf("port%d", j), ContainerPort: int32(8080 + j)})
	}
	for _, family := range ipFamilies {
		var ip string
		if family == v1.IPv4Protocol {
			// if id=1, ip=1.2.3.4
			// if id=2, ip=1.2.3.5
			// if id=999, ip=1.2.6.235
			ip = fmt.Sprintf("1.2.%d.%d", 3+((4+id)/256), (4+id)%256)
		} else {
			ip = fmt.Sprintf("2000::%x", 4+id)
		}
		p.Status.PodIPs = append(p.Status.PodIPs, v1.PodIP{IP: ip})
	}
	p.Status.PodIP = p.Status.PodIPs[0].IP

	return p
}

func addPods(store cache.Store, namespace string, nPods int, nPorts int, nNotReady int, ipFamilies []v1.IPFamily) {
	for i := 0; i < nPods+nNotReady; i++ {
		isReady := i < nPods
		pod := testPod(namespace, i, nPorts, isReady, ipFamilies)
		store.Add(pod)
	}
}

func addBadIPPod(store cache.Store, namespace string, ipFamilies []v1.IPFamily) {
	pod := testPod(namespace, 0, 1, true, ipFamilies)
	pod.Status.PodIPs[0].IP = "0" + pod.Status.PodIPs[0].IP
	pod.Status.PodIP = pod.Status.PodIPs[0].IP
	store.Add(pod)
}

func addNotReadyPodsWithSpecifiedRestartPolicyAndPhase(store cache.Store, namespace string, nPods int, nPorts int, restartPolicy v1.RestartPolicy, podPhase v1.PodPhase) {
	for i := 0; i < nPods; i++ {
		p := &v1.Pod{
			TypeMeta: metav1.TypeMeta{APIVersion: "v1"},
			ObjectMeta: metav1.ObjectMeta{
				Namespace: namespace,
				Name:      fmt.Sprintf("pod%d", i),
				Labels:    map[string]string{"foo": "bar"},
			},
			Spec: v1.PodSpec{
				RestartPolicy: restartPolicy,
				Containers:    []v1.Container{{Ports: []v1.ContainerPort{}}},
			},
			Status: v1.PodStatus{
				PodIP: fmt.Sprintf("1.2.3.%d", 4+i),
				Phase: podPhase,
				Conditions: []v1.PodCondition{
					{
						Type:   v1.PodReady,
						Status: v1.ConditionFalse,
					},
				},
			},
		}
		for j := 0; j < nPorts; j++ {
			p.Spec.Containers[0].Ports = append(p.Spec.Containers[0].Ports,
				v1.ContainerPort{Name: fmt.Sprintf("port%d", j), ContainerPort: int32(8080 + j)})
		}
		store.Add(p)
	}
}

func makeTestServer(t *testing.T, namespace string) (*httptest.Server, *utiltesting.FakeHandler) {
	fakeEndpointsHandler := utiltesting.FakeHandler{
		StatusCode:   http.StatusOK,
		ResponseBody: runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{}),
	}
	mux := http.NewServeMux()
	if namespace == "" {
		t.Fatal("namespace cannot be empty")
	}
	mux.Handle("/api/v1/namespaces/"+namespace+"/endpoints", &fakeEndpointsHandler)
	mux.Handle("/api/v1/namespaces/"+namespace+"/endpoints/", &fakeEndpointsHandler)
	mux.HandleFunc("/", func(res http.ResponseWriter, req *http.Request) {
		t.Errorf("unexpected request: %v", req.RequestURI)
		http.Error(res, "", http.StatusNotFound)
	})
	return httptest.NewServer(mux), &fakeEndpointsHandler
}

// makeBlockingEndpointTestServer will signal the blockNextAction channel on endpoint "POST", "PUT", and "DELETE"
// requests. "POST" and "PUT" requests will wait on a blockUpdate signal if provided, while "DELETE" requests will wait
// on a blockDelete signal if provided. If controller is nil, an error will be sent in the response.
func makeBlockingEndpointTestServer(t *testing.T, controller *endpointController, endpoint *v1.Endpoints, blockUpdate, blockDelete, blockNextAction chan struct{}, namespace string) *httptest.Server {

	handlerFunc := func(res http.ResponseWriter, req *http.Request) {
		if controller == nil {
			res.WriteHeader(http.StatusInternalServerError)
			res.Write([]byte("controller has not been set yet"))
			return
		}

		if req.Method == "POST" || req.Method == "PUT" {
			if blockUpdate != nil {
				go func() {
					// Delay the update of endpoints to make endpoints cache out of sync
					<-blockUpdate
					_ = controller.endpointsStore.Add(endpoint)
				}()
			} else {
				_ = controller.endpointsStore.Add(endpoint)
			}
			blockNextAction <- struct{}{}
		}

		if req.Method == "DELETE" {
			if blockDelete != nil {
				go func() {
					// Delay the deletion of endpoints to make endpoints cache out of sync
					<-blockDelete
					_ = controller.endpointsStore.Delete(endpoint)
					controller.onEndpointsDelete(endpoint)
				}()
			} else {
				_ = controller.endpointsStore.Delete(endpoint)
				controller.onEndpointsDelete(endpoint)
			}
			blockNextAction <- struct{}{}
		}

		res.Header().Set("Content-Type", "application/json")
		res.WriteHeader(http.StatusOK)
		_, _ = res.Write([]byte(runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), endpoint)))
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/api/v1/namespaces/"+namespace+"/endpoints", handlerFunc)
	mux.HandleFunc("/api/v1/namespaces/"+namespace+"/endpoints/", handlerFunc)
	mux.HandleFunc("/api/v1/namespaces/"+namespace+"/events", func(res http.ResponseWriter, req *http.Request) {})
	mux.HandleFunc("/", func(res http.ResponseWriter, req *http.Request) {
		t.Errorf("unexpected request: %v", req.RequestURI)
		http.Error(res, "", http.StatusNotFound)
	})
	return httptest.NewServer(mux)

}

type endpointController struct {
	*Controller
	podStore       cache.Store
	serviceStore   cache.Store
	endpointsStore cache.Store
}

func newController(ctx context.Context, url string, batchPeriod time.Duration) *endpointController {
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: url, ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}, ContentType: runtime.ContentTypeJSON}})
	informerFactory := informers.NewSharedInformerFactory(client, controllerpkg.NoResyncPeriodFunc())
	endpoints := NewEndpointController(ctx, informerFactory.Core().V1().Pods(), informerFactory.Core().V1().Services(),
		informerFactory.Core().V1().Endpoints(), client, batchPeriod)
	endpoints.podsSynced = alwaysReady
	endpoints.servicesSynced = alwaysReady
	endpoints.endpointsSynced = alwaysReady
	return &endpointController{
		endpoints,
		informerFactory.Core().V1().Pods().Informer().GetStore(),
		informerFactory.Core().V1().Services().Informer().GetStore(),
		informerFactory.Core().V1().Endpoints().Informer().GetStore(),
	}
}

func newFakeController(ctx context.Context, batchPeriod time.Duration) (*fake.Clientset, *endpointController) {
	client := fake.NewSimpleClientset()
	informerFactory := informers.NewSharedInformerFactory(client, controllerpkg.NoResyncPeriodFunc())

	eController := NewEndpointController(
		ctx,
		informerFactory.Core().V1().Pods(),
		informerFactory.Core().V1().Services(),
		informerFactory.Core().V1().Endpoints(),
		client,
		batchPeriod)

	eController.podsSynced = alwaysReady
	eController.servicesSynced = alwaysReady
	eController.endpointsSynced = alwaysReady

	return client, &endpointController{
		eController,
		informerFactory.Core().V1().Pods().Informer().GetStore(),
		informerFactory.Core().V1().Services().Informer().GetStore(),
		informerFactory.Core().V1().Endpoints().Informer().GetStore(),
	}
}

func TestSyncEndpointsItemsPreserveNoSelector(t *testing.T) {
	ns := metav1.NamespaceDefault
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()

	tCtx := ktesting.Init(t)
	endpoints := newController(tCtx, testServer.URL, 0*time.Second)
	endpoints.endpointsStore.Add(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
		},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "6.7.8.9", NodeName: &emptyNodeName}},
			Ports:     []v1.EndpointPort{{Port: 1000}},
		}},
	})
	endpoints.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
		Spec:       v1.ServiceSpec{Ports: []v1.ServicePort{{Port: 80}}},
	})
	err := endpoints.syncService(tCtx, ns+"/foo")
	if err != nil {
		t.Errorf("Unexpected error syncing service %v", err)
	}
	endpointsHandler.ValidateRequestCount(t, 0)
}

func TestSyncEndpointsExistingNilSubsets(t *testing.T) {
	ns := metav1.NamespaceDefault
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()

	tCtx := ktesting.Init(t)
	endpoints := newController(tCtx, testServer.URL, 0*time.Second)
	endpoints.endpointsStore.Add(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
				labelManagedBy: controllerName,
			},
		},
		Subsets: nil,
	})
	endpoints.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: v1.ServiceSpec{
			Selector: map[string]string{"foo": "bar"},
			Ports:    []v1.ServicePort{{Port: 80}},
		},
	})
	err := endpoints.syncService(tCtx, ns+"/foo")
	if err != nil {
		t.Errorf("Unexpected error syncing service %v", err)
	}
	endpointsHandler.ValidateRequestCount(t, 0)
}

func TestSyncEndpointsExistingEmptySubsets(t *testing.T) {
	ns := metav1.NamespaceDefault
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()

	tCtx := ktesting.Init(t)
	endpoints := newController(tCtx, testServer.URL, 0*time.Second)
	endpoints.endpointsStore.Add(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
				labelManagedBy: controllerName,
			},
		},
		Subsets: []v1.EndpointSubset{},
	})
	endpoints.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: v1.ServiceSpec{
			Selector: map[string]string{"foo": "bar"},
			Ports:    []v1.ServicePort{{Port: 80}},
		},
	})
	err := endpoints.syncService(tCtx, ns+"/foo")
	if err != nil {
		t.Errorf("Unexpected error syncing service %v", err)
	}
	endpointsHandler.ValidateRequestCount(t, 0)
}

func TestSyncEndpointsWithPodResourceVersionUpdateOnly(t *testing.T) {
	ns := metav1.NamespaceDefault
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()
	pod0 := testPod(ns, 0, 1, true, ipv4only)
	pod1 := testPod(ns, 1, 1, false, ipv4only)
	tCtx := ktesting.Init(t)
	endpoints := newController(tCtx, testServer.URL, 0*time.Second)
	endpoints.endpointsStore.Add(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
				labelManagedBy: controllerName,
			},
		},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{
				{
					IP:        pod0.Status.PodIPs[0].IP,
					NodeName:  &emptyNodeName,
					TargetRef: &v1.ObjectReference{Kind: "Pod", Name: pod0.Name, Namespace: ns, ResourceVersion: "1"},
				},
			},
			NotReadyAddresses: []v1.EndpointAddress{
				{
					IP:        pod1.Status.PodIPs[0].IP,
					NodeName:  &emptyNodeName,
					TargetRef: &v1.ObjectReference{Kind: "Pod", Name: pod1.Name, Namespace: ns, ResourceVersion: "2"},
				},
			},
			Ports: []v1.EndpointPort{{Port: 8080, Protocol: "TCP"}},
		}},
	})
	endpoints.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: v1.ServiceSpec{
			Selector:   map[string]string{"foo": "bar"},
			Ports:      []v1.ServicePort{{Port: 80, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)}},
			IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
		},
	})
	pod0.ResourceVersion = "3"
	pod1.ResourceVersion = "4"
	endpoints.podStore.Add(pod0)
	endpoints.podStore.Add(pod1)
	err := endpoints.syncService(tCtx, ns+"/foo")
	if err != nil {
		t.Errorf("Unexpected error syncing service %v", err)
	}

	endpointsHandler.ValidateRequestCount(t, 0)
}

func TestSyncEndpointsNewNoSubsets(t *testing.T) {
	ns := metav1.NamespaceDefault
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()
	tCtx := ktesting.Init(t)
	endpoints := newController(tCtx, testServer.URL, 0*time.Second)
	endpoints.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: v1.ServiceSpec{
			Selector: map[string]string{"foo": "bar"},
			Ports:    []v1.ServicePort{{Port: 80}},
		},
	})
	err := endpoints.syncService(tCtx, ns+"/foo")
	if err != nil {
		t.Errorf("Unexpected error syncing service %v", err)
	}
	endpointsHandler.ValidateRequestCount(t, 1)
}

func TestCheckLeftoverEndpoints(t *testing.T) {
	ns := metav1.NamespaceDefault
	testServer, _ := makeTestServer(t, ns)
	defer testServer.Close()

	tCtx := ktesting.Init(t)
	endpoints := newController(tCtx, testServer.URL, 0*time.Second)
	endpoints.endpointsStore.Add(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
		},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "6.7.8.9", NodeName: &emptyNodeName}},
			Ports:     []v1.EndpointPort{{Port: 1000}},
		}},
	})
	endpoints.checkLeftoverEndpoints()
	if e, a := 1, endpoints.queue.Len(); e != a {
		t.Fatalf("Expected %v, got %v", e, a)
	}
	got, _ := endpoints.queue.Get()
	if e, a := ns+"/foo", got; e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}
}

func TestSyncEndpointsProtocolTCP(t *testing.T) {
	ns := "other"
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()
	tCtx := ktesting.Init(t)
	endpoints := newController(tCtx, testServer.URL, 0*time.Second)
	endpoints.endpointsStore.Add(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
		},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "6.7.8.9", NodeName: &emptyNodeName}},
			Ports:     []v1.EndpointPort{{Port: 1000, Protocol: "TCP"}},
		}},
	})
	addPods(endpoints.podStore, ns, 1, 1, 0, ipv4only)
	endpoints.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: v1.ServiceSpec{
			Selector:   map[string]string{},
			Ports:      []v1.ServicePort{{Port: 80, TargetPort: intstr.FromInt32(8080), Protocol: "TCP"}},
			IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
		},
	})
	err := endpoints.syncService(tCtx, ns+"/foo")
	if err != nil {
		t.Errorf("Unexpected error syncing service %v", err)
	}

	endpointsHandler.ValidateRequestCount(t, 1)
	data := runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
				labelManagedBy:       controllerName,
				v1.IsHeadlessService: "",
			},
		},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "1.2.3.4", NodeName: &emptyNodeName, TargetRef: &v1.ObjectReference{Kind: "Pod", Name: "pod0", Namespace: ns}}},
			Ports:     []v1.EndpointPort{{Port: 8080, Protocol: "TCP"}},
		}},
	})
	endpointsHandler.ValidateRequest(t, "/api/v1/namespaces/"+ns+"/endpoints/foo", "PUT", &data)
}

func TestSyncEndpointsHeadlessServiceLabel(t *testing.T) {
	ns := metav1.NamespaceDefault
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()
	tCtx := ktesting.Init(t)
	endpoints := newController(tCtx, testServer.URL, 0*time.Second)
	endpoints.endpointsStore.Add(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
				labelManagedBy:       controllerName,
				v1.IsHeadlessService: "",
			},
		},
		Subsets: []v1.EndpointSubset{},
	})
	endpoints.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: v1.ServiceSpec{
			Selector: map[string]string{"foo": "bar"},
			Ports:    []v1.ServicePort{{Port: 80}},
		},
	})
	err := endpoints.syncService(tCtx, ns+"/foo")
	if err != nil {
		t.Errorf("Unexpected error syncing service %v", err)
	}

	endpointsHandler.ValidateRequestCount(t, 0)
}

func TestSyncServiceExternalNameType(t *testing.T) {
	serviceName := "testing-1"
	namespace := metav1.NamespaceDefault

	testCases := []struct {
		desc    string
		service *v1.Service
	}{
		{
			desc: "External name with selector and ports should not receive endpoints",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: serviceName, Namespace: namespace},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{"foo": "bar"},
					Ports:    []v1.ServicePort{{Port: 80}},
					Type:     v1.ServiceTypeExternalName,
				},
			},
		},
		{
			desc: "External name with ports should not receive endpoints",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: serviceName, Namespace: namespace},
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{{Port: 80}},
					Type:  v1.ServiceTypeExternalName,
				},
			},
		},
		{
			desc: "External name with selector should not receive endpoints",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: serviceName, Namespace: namespace},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{"foo": "bar"},
					Type:     v1.ServiceTypeExternalName,
				},
			},
		},
		{
			desc: "External name without selector and ports should not receive endpoints",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: serviceName, Namespace: namespace},
				Spec: v1.ServiceSpec{
					Type: v1.ServiceTypeExternalName,
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			testServer, endpointsHandler := makeTestServer(t, namespace)

			defer testServer.Close()
			tCtx := ktesting.Init(t)
			endpoints := newController(tCtx, testServer.URL, 0*time.Second)
			err := endpoints.serviceStore.Add(tc.service)
			if err != nil {
				t.Fatalf("Error adding service to service store: %v", err)
			}
			err = endpoints.syncService(tCtx, namespace+"/"+serviceName)
			if err != nil {
				t.Fatalf("Error syncing service: %v", err)
			}
			endpointsHandler.ValidateRequestCount(t, 0)
		})
	}
}

func TestSyncEndpointsProtocolUDP(t *testing.T) {
	ns := "other"
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()
	tCtx := ktesting.Init(t)
	endpoints := newController(tCtx, testServer.URL, 0*time.Second)
	endpoints.endpointsStore.Add(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
		},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "6.7.8.9", NodeName: &emptyNodeName}},
			Ports:     []v1.EndpointPort{{Port: 1000, Protocol: "UDP"}},
		}},
	})
	addPods(endpoints.podStore, ns, 1, 1, 0, ipv4only)
	endpoints.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: v1.ServiceSpec{
			Selector:   map[string]string{},
			Ports:      []v1.ServicePort{{Port: 80, TargetPort: intstr.FromInt32(8080), Protocol: "UDP"}},
			IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
		},
	})
	err := endpoints.syncService(tCtx, ns+"/foo")
	if err != nil {
		t.Errorf("Unexpected error syncing service %v", err)
	}

	endpointsHandler.ValidateRequestCount(t, 1)
	data := runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
				labelManagedBy:       controllerName,
				v1.IsHeadlessService: "",
			},
		},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "1.2.3.4", NodeName: &emptyNodeName, TargetRef: &v1.ObjectReference{Kind: "Pod", Name: "pod0", Namespace: ns}}},
			Ports:     []v1.EndpointPort{{Port: 8080, Protocol: "UDP"}},
		}},
	})
	endpointsHandler.ValidateRequest(t, "/api/v1/namespaces/"+ns+"/endpoints/foo", "PUT", &data)
}

func TestSyncEndpointsProtocolSCTP(t *testing.T) {
	ns := "other"
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()
	tCtx := ktesting.Init(t)
	endpoints := newController(tCtx, testServer.URL, 0*time.Second)
	endpoints.endpointsStore.Add(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
		},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "6.7.8.9", NodeName: &emptyNodeName}},
			Ports:     []v1.EndpointPort{{Port: 1000, Protocol: "SCTP"}},
		}},
	})
	addPods(endpoints.podStore, ns, 1, 1, 0, ipv4only)
	endpoints.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: v1.ServiceSpec{
			Selector:   map[string]string{},
			Ports:      []v1.ServicePort{{Port: 80, TargetPort: intstr.FromInt32(8080), Protocol: "SCTP"}},
			IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
		},
	})
	err := endpoints.syncService(tCtx, ns+"/foo")
	if err != nil {
		t.Errorf("Unexpected error syncing service %v", err)
	}

	endpointsHandler.ValidateRequestCount(t, 1)
	data := runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
				labelManagedBy:       controllerName,
				v1.IsHeadlessService: "",
			},
		},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "1.2.3.4", NodeName: &emptyNodeName, TargetRef: &v1.ObjectReference{Kind: "Pod", Name: "pod0", Namespace: ns}}},
			Ports:     []v1.EndpointPort{{Port: 8080, Protocol: "SCTP"}},
		}},
	})
	endpointsHandler.ValidateRequest(t, "/api/v1/namespaces/"+ns+"/endpoints/foo", "PUT", &data)
}

func TestSyncEndpointsItemsEmptySelectorSelectsAll(t *testing.T) {
	ns := "other"
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()
	tCtx := ktesting.Init(t)
	endpoints := newController(tCtx, testServer.URL, 0*time.Second)
	endpoints.endpointsStore.Add(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
		},
		Subsets: []v1.EndpointSubset{},
	})
	addPods(endpoints.podStore, ns, 1, 1, 0, ipv4only)
	endpoints.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: v1.ServiceSpec{
			Selector:   map[string]string{},
			Ports:      []v1.ServicePort{{Port: 80, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)}},
			IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
		},
	})
	err := endpoints.syncService(tCtx, ns+"/foo")
	if err != nil {
		t.Errorf("Unexpected error syncing service %v", err)
	}

	data := runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
				labelManagedBy:       controllerName,
				v1.IsHeadlessService: "",
			},
		},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "1.2.3.4", NodeName: &emptyNodeName, TargetRef: &v1.ObjectReference{Kind: "Pod", Name: "pod0", Namespace: ns}}},
			Ports:     []v1.EndpointPort{{Port: 8080, Protocol: "TCP"}},
		}},
	})
	endpointsHandler.ValidateRequest(t, "/api/v1/namespaces/"+ns+"/endpoints/foo", "PUT", &data)
}

func TestSyncEndpointsItemsEmptySelectorSelectsAllNotReady(t *testing.T) {
	ns := "other"
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()

	tCtx := ktesting.Init(t)
	endpoints := newController(tCtx, testServer.URL, 0*time.Second)
	endpoints.endpointsStore.Add(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
		},
		Subsets: []v1.EndpointSubset{},
	})
	addPods(endpoints.podStore, ns, 0, 1, 1, ipv4only)
	endpoints.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: v1.ServiceSpec{
			Selector:   map[string]string{},
			Ports:      []v1.ServicePort{{Port: 80, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)}},
			IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
		},
	})
	err := endpoints.syncService(tCtx, ns+"/foo")
	if err != nil {
		t.Errorf("Unexpected error syncing service %v", err)
	}

	data := runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
				labelManagedBy:       controllerName,
				v1.IsHeadlessService: "",
			},
		},
		Subsets: []v1.EndpointSubset{{
			NotReadyAddresses: []v1.EndpointAddress{{IP: "1.2.3.4", NodeName: &emptyNodeName, TargetRef: &v1.ObjectReference{Kind: "Pod", Name: "pod0", Namespace: ns}}},
			Ports:             []v1.EndpointPort{{Port: 8080, Protocol: "TCP"}},
		}},
	})
	endpointsHandler.ValidateRequest(t, "/api/v1/namespaces/"+ns+"/endpoints/foo", "PUT", &data)
}

func TestSyncEndpointsItemsEmptySelectorSelectsAllMixed(t *testing.T) {
	ns := "other"
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()

	tCtx := ktesting.Init(t)
	endpoints := newController(tCtx, testServer.URL, 0*time.Second)
	endpoints.endpointsStore.Add(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
		},
		Subsets: []v1.EndpointSubset{},
	})
	addPods(endpoints.podStore, ns, 1, 1, 1, ipv4only)
	endpoints.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: v1.ServiceSpec{
			Selector:   map[string]string{},
			Ports:      []v1.ServicePort{{Port: 80, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)}},
			IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
		},
	})
	err := endpoints.syncService(tCtx, ns+"/foo")
	if err != nil {
		t.Errorf("Unexpected error syncing service %v", err)
	}

	data := runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
				labelManagedBy:       controllerName,
				v1.IsHeadlessService: "",
			},
		},
		Subsets: []v1.EndpointSubset{{
			Addresses:         []v1.EndpointAddress{{IP: "1.2.3.4", NodeName: &emptyNodeName, TargetRef: &v1.ObjectReference{Kind: "Pod", Name: "pod0", Namespace: ns}}},
			NotReadyAddresses: []v1.EndpointAddress{{IP: "1.2.3.5", NodeName: &emptyNodeName, TargetRef: &v1.ObjectReference{Kind: "Pod", Name: "pod1", Namespace: ns}}},
			Ports:             []v1.EndpointPort{{Port: 8080, Protocol: "TCP"}},
		}},
	})
	endpointsHandler.ValidateRequest(t, "/api/v1/namespaces/"+ns+"/endpoints/foo", "PUT", &data)
}

func TestSyncEndpointsItemsPreexisting(t *testing.T) {
	ns := "bar"
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()
	tCtx := ktesting.Init(t)
	endpoints := newController(tCtx, testServer.URL, 0*time.Second)
	endpoints.endpointsStore.Add(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
				labelManagedBy: controllerName,
			},
		},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "6.7.8.9", NodeName: &emptyNodeName}},
			Ports:     []v1.EndpointPort{{Port: 1000}},
		}},
	})
	addPods(endpoints.podStore, ns, 1, 1, 0, ipv4only)
	endpoints.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: v1.ServiceSpec{
			Selector:   map[string]string{"foo": "bar"},
			Ports:      []v1.ServicePort{{Port: 80, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)}},
			IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
		},
	})
	err := endpoints.syncService(tCtx, ns+"/foo")
	if err != nil {
		t.Errorf("Unexpected error syncing service %v", err)
	}

	data := runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
				labelManagedBy:       controllerName,
				v1.IsHeadlessService: "",
			},
		},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "1.2.3.4", NodeName: &emptyNodeName, TargetRef: &v1.ObjectReference{Kind: "Pod", Name: "pod0", Namespace: ns}}},
			Ports:     []v1.EndpointPort{{Port: 8080, Protocol: "TCP"}},
		}},
	})
	endpointsHandler.ValidateRequest(t, "/api/v1/namespaces/"+ns+"/endpoints/foo", "PUT", &data)
}

func TestSyncEndpointsItemsPreexistingIdentical(t *testing.T) {
	ns := metav1.NamespaceDefault
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()
	tCtx := ktesting.Init(t)
	endpoints := newController(tCtx, testServer.URL, 0*time.Second)
	endpoints.endpointsStore.Add(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			ResourceVersion: "1",
			Name:            "foo",
			Namespace:       ns,
			Labels: map[string]string{
				labelManagedBy: controllerName,
			},
		},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "1.2.3.4", NodeName: &emptyNodeName, TargetRef: &v1.ObjectReference{Kind: "Pod", Name: "pod0", Namespace: ns}}},
			Ports:     []v1.EndpointPort{{Port: 8080, Protocol: "TCP"}},
		}},
	})
	addPods(endpoints.podStore, metav1.NamespaceDefault, 1, 1, 0, ipv4only)
	endpoints.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: metav1.NamespaceDefault},
		Spec: v1.ServiceSpec{
			Selector:   map[string]string{"foo": "bar"},
			Ports:      []v1.ServicePort{{Port: 80, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)}},
			IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
		},
	})
	err := endpoints.syncService(tCtx, ns+"/foo")
	if err != nil {
		t.Errorf("Unexpected error syncing service %v", err)
	}
	endpointsHandler.ValidateRequestCount(t, 0)
}

func TestSyncEndpointsItems(t *testing.T) {
	ns := "other"
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()
	tCtx := ktesting.Init(t)
	endpoints := newController(tCtx, testServer.URL, 0*time.Second)
	addPods(endpoints.podStore, ns, 3, 2, 0, ipv4only)
	addPods(endpoints.podStore, "blah", 5, 2, 0, ipv4only) // make sure these aren't found!

	endpoints.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: v1.ServiceSpec{
			Selector: map[string]string{"foo": "bar"},
			Ports: []v1.ServicePort{
				{Name: "port0", Port: 80, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)},
				{Name: "port1", Port: 88, Protocol: "TCP", TargetPort: intstr.FromInt32(8088)},
			},
			IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
		},
	})
	err := endpoints.syncService(tCtx, "other/foo")
	if err != nil {
		t.Errorf("Unexpected error syncing service %v", err)
	}

	expectedSubsets := []v1.EndpointSubset{{
		Addresses: []v1.EndpointAddress{
			{IP: "1.2.3.4", NodeName: &emptyNodeName, TargetRef: &v1.ObjectReference{Kind: "Pod", Name: "pod0", Namespace: ns}},
			{IP: "1.2.3.5", NodeName: &emptyNodeName, TargetRef: &v1.ObjectReference{Kind: "Pod", Name: "pod1", Namespace: ns}},
			{IP: "1.2.3.6", NodeName: &emptyNodeName, TargetRef: &v1.ObjectReference{Kind: "Pod", Name: "pod2", Namespace: ns}},
		},
		Ports: []v1.EndpointPort{
			{Name: "port0", Port: 8080, Protocol: "TCP"},
			{Name: "port1", Port: 8088, Protocol: "TCP"},
		},
	}}
	data := runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			ResourceVersion: "",
			Name:            "foo",
			Labels: map[string]string{
				labelManagedBy:       controllerName,
				v1.IsHeadlessService: "",
			},
		},
		Subsets: endptspkg.SortSubsets(expectedSubsets),
	})
	endpointsHandler.ValidateRequestCount(t, 1)
	endpointsHandler.ValidateRequest(t, "/api/v1/namespaces/"+ns+"/endpoints", "POST", &data)
}

func TestSyncEndpointsItemsWithLabels(t *testing.T) {
	ns := "other"
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()
	tCtx := ktesting.Init(t)
	endpoints := newController(tCtx, testServer.URL, 0*time.Second)
	addPods(endpoints.podStore, ns, 3, 2, 0, ipv4only)
	serviceLabels := map[string]string{"foo": "bar"}
	endpoints.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: ns,
			Labels:    serviceLabels,
		},
		Spec: v1.ServiceSpec{
			Selector: map[string]string{"foo": "bar"},
			Ports: []v1.ServicePort{
				{Name: "port0", Port: 80, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)},
				{Name: "port1", Port: 88, Protocol: "TCP", TargetPort: intstr.FromInt32(8088)},
			},
			IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
		},
	})
	err := endpoints.syncService(tCtx, ns+"/foo")
	if err != nil {
		t.Errorf("Unexpected error syncing service %v", err)
	}

	expectedSubsets := []v1.EndpointSubset{{
		Addresses: []v1.EndpointAddress{
			{IP: "1.2.3.4", NodeName: &emptyNodeName, TargetRef: &v1.ObjectReference{Kind: "Pod", Name: "pod0", Namespace: ns}},
			{IP: "1.2.3.5", NodeName: &emptyNodeName, TargetRef: &v1.ObjectReference{Kind: "Pod", Name: "pod1", Namespace: ns}},
			{IP: "1.2.3.6", NodeName: &emptyNodeName, TargetRef: &v1.ObjectReference{Kind: "Pod", Name: "pod2", Namespace: ns}},
		},
		Ports: []v1.EndpointPort{
			{Name: "port0", Port: 8080, Protocol: "TCP"},
			{Name: "port1", Port: 8088, Protocol: "TCP"},
		},
	}}

	serviceLabels[v1.IsHeadlessService] = ""
	serviceLabels[labelManagedBy] = controllerName
	data := runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			ResourceVersion: "",
			Name:            "foo",
			Labels:          serviceLabels,
		},
		Subsets: endptspkg.SortSubsets(expectedSubsets),
	})
	endpointsHandler.ValidateRequestCount(t, 1)
	endpointsHandler.ValidateRequest(t, "/api/v1/namespaces/"+ns+"/endpoints", "POST", &data)
}

func TestSyncEndpointsItemsPreexistingLabelsChange(t *testing.T) {
	ns := "bar"
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()
	tCtx := ktesting.Init(t)
	endpoints := newController(tCtx, testServer.URL, 0*time.Second)
	endpoints.endpointsStore.Add(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
				"foo": "bar",
			},
		},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "6.7.8.9", NodeName: &emptyNodeName}},
			Ports:     []v1.EndpointPort{{Port: 1000}},
		}},
	})
	addPods(endpoints.podStore, ns, 1, 1, 0, ipv4only)
	serviceLabels := map[string]string{"baz": "blah"}
	endpoints.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: ns,
			Labels:    serviceLabels,
		},
		Spec: v1.ServiceSpec{
			Selector:   map[string]string{"foo": "bar"},
			Ports:      []v1.ServicePort{{Port: 80, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)}},
			IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
		},
	})
	err := endpoints.syncService(tCtx, ns+"/foo")
	if err != nil {
		t.Errorf("Unexpected error syncing service %v", err)
	}

	serviceLabels[v1.IsHeadlessService] = ""
	serviceLabels[labelManagedBy] = controllerName
	data := runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels:          serviceLabels,
		},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "1.2.3.4", NodeName: &emptyNodeName, TargetRef: &v1.ObjectReference{Kind: "Pod", Name: "pod0", Namespace: ns}}},
			Ports:     []v1.EndpointPort{{Port: 8080, Protocol: "TCP"}},
		}},
	})
	endpointsHandler.ValidateRequest(t, "/api/v1/namespaces/"+ns+"/endpoints/foo", "PUT", &data)
}

func TestWaitsForAllInformersToBeSynced2(t *testing.T) {
	var tests = []struct {
		podsSynced            func() bool
		servicesSynced        func() bool
		endpointsSynced       func() bool
		shouldUpdateEndpoints bool
	}{
		{neverReady, alwaysReady, alwaysReady, false},
		{alwaysReady, neverReady, alwaysReady, false},
		{alwaysReady, alwaysReady, neverReady, false},
		{alwaysReady, alwaysReady, alwaysReady, true},
	}

	for _, test := range tests {
		func() {
			ns := "other"
			testServer, endpointsHandler := makeTestServer(t, ns)
			defer testServer.Close()
			tCtx := ktesting.Init(t)
			endpoints := newController(tCtx, testServer.URL, 0*time.Second)
			addPods(endpoints.podStore, ns, 1, 1, 0, ipv4only)

			service := &v1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
				Spec: v1.ServiceSpec{
					Selector:   map[string]string{},
					Ports:      []v1.ServicePort{{Port: 80, TargetPort: intstr.FromInt32(8080), Protocol: "TCP"}},
					IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
				},
			}
			endpoints.serviceStore.Add(service)
			endpoints.onServiceUpdate(service)
			endpoints.podsSynced = test.podsSynced
			endpoints.servicesSynced = test.servicesSynced
			endpoints.endpointsSynced = test.endpointsSynced
			endpoints.workerLoopPeriod = 10 * time.Millisecond
			go endpoints.Run(tCtx, 1)

			// cache.WaitForNamedCacheSync has a 100ms poll period, and the endpoints worker has a 10ms period.
			// To ensure we get all updates, including unexpected ones, we need to wait at least as long as
			// a single cache sync period and worker period, with some fudge room.
			time.Sleep(150 * time.Millisecond)
			if test.shouldUpdateEndpoints {
				// Ensure the work queue has been processed by looping for up to a second to prevent flakes.
				wait.PollImmediate(50*time.Millisecond, 1*time.Second, func() (bool, error) {
					return endpoints.queue.Len() == 0, nil
				})
				endpointsHandler.ValidateRequestCount(t, 1)
			} else {
				endpointsHandler.ValidateRequestCount(t, 0)
			}
		}()
	}
}

func TestSyncEndpointsHeadlessService(t *testing.T) {
	ns := "headless"
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()
	tCtx := ktesting.Init(t)
	endpoints := newController(tCtx, testServer.URL, 0*time.Second)
	endpoints.endpointsStore.Add(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
		},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "6.7.8.9", NodeName: &emptyNodeName}},
			Ports:     []v1.EndpointPort{{Port: 1000, Protocol: "TCP"}},
		}},
	})
	addPods(endpoints.podStore, ns, 1, 1, 0, ipv4only)
	service := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns, Labels: map[string]string{"a": "b"}},
		Spec: v1.ServiceSpec{
			Selector:   map[string]string{},
			ClusterIP:  api.ClusterIPNone,
			Ports:      []v1.ServicePort{},
			IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
		},
	}
	originalService := service.DeepCopy()
	endpoints.serviceStore.Add(service)
	err := endpoints.syncService(tCtx, ns+"/foo")
	if err != nil {
		t.Errorf("Unexpected error syncing service %v", err)
	}
	data := runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
				labelManagedBy:       controllerName,
				"a":                  "b",
				v1.IsHeadlessService: "",
			},
		},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "1.2.3.4", NodeName: &emptyNodeName, TargetRef: &v1.ObjectReference{Kind: "Pod", Name: "pod0", Namespace: ns}}},
			Ports:     []v1.EndpointPort{},
		}},
	})
	if !reflect.DeepEqual(originalService, service) {
		t.Fatalf("syncing endpoints changed service: %s", cmp.Diff(service, originalService))
	}
	endpointsHandler.ValidateRequestCount(t, 1)
	endpointsHandler.ValidateRequest(t, "/api/v1/namespaces/"+ns+"/endpoints/foo", "PUT", &data)
}

func TestSyncEndpointsItemsExcludeNotReadyPodsWithRestartPolicyNeverAndPhaseFailed(t *testing.T) {
	ns := "other"
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()

	tCtx := ktesting.Init(t)
	endpoints := newController(tCtx, testServer.URL, 0*time.Second)
	endpoints.endpointsStore.Add(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
				labelManagedBy: controllerName,
				"foo":          "bar",
			},
		},
		Subsets: []v1.EndpointSubset{},
	})
	addNotReadyPodsWithSpecifiedRestartPolicyAndPhase(endpoints.podStore, ns, 1, 1, v1.RestartPolicyNever, v1.PodFailed)
	endpoints.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: v1.ServiceSpec{
			Selector:   map[string]string{"foo": "bar"},
			Ports:      []v1.ServicePort{{Port: 80, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)}},
			IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
		},
	})
	err := endpoints.syncService(tCtx, ns+"/foo")
	if err != nil {
		t.Errorf("Unexpected error syncing service %v", err)
	}
	data := runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
				labelManagedBy:       controllerName,
				v1.IsHeadlessService: "",
			},
		},
		Subsets: []v1.EndpointSubset{},
	})
	endpointsHandler.ValidateRequest(t, "/api/v1/namespaces/"+ns+"/endpoints/foo", "PUT", &data)
}

func TestSyncEndpointsItemsExcludeNotReadyPodsWithRestartPolicyNeverAndPhaseSucceeded(t *testing.T) {
	ns := "other"
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()

	tCtx := ktesting.Init(t)
	endpoints := newController(tCtx, testServer.URL, 0*time.Second)
	endpoints.endpointsStore.Add(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
				"foo": "bar",
			},
		},
		Subsets: []v1.EndpointSubset{},
	})
	addNotReadyPodsWithSpecifiedRestartPolicyAndPhase(endpoints.podStore, ns, 1, 1, v1.RestartPolicyNever, v1.PodSucceeded)
	endpoints.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: v1.ServiceSpec{
			Selector:   map[string]string{"foo": "bar"},
			Ports:      []v1.ServicePort{{Port: 80, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)}},
			IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
		},
	})
	err := endpoints.syncService(tCtx, ns+"/foo")
	if err != nil {
		t.Errorf("Unexpected error syncing service %v", err)
	}
	data := runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
				labelManagedBy:       controllerName,
				v1.IsHeadlessService: "",
			},
		},
		Subsets: []v1.EndpointSubset{},
	})
	endpointsHandler.ValidateRequest(t, "/api/v1/namespaces/"+ns+"/endpoints/foo", "PUT", &data)
}

func TestSyncEndpointsItemsExcludeNotReadyPodsWithRestartPolicyOnFailureAndPhaseSucceeded(t *testing.T) {
	ns := "other"
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()

	tCtx := ktesting.Init(t)
	endpoints := newController(tCtx, testServer.URL, 0*time.Second)
	endpoints.endpointsStore.Add(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
				"foo": "bar",
			},
		},
		Subsets: []v1.EndpointSubset{},
	})
	addNotReadyPodsWithSpecifiedRestartPolicyAndPhase(endpoints.podStore, ns, 1, 1, v1.RestartPolicyOnFailure, v1.PodSucceeded)
	endpoints.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: v1.ServiceSpec{
			Selector:   map[string]string{"foo": "bar"},
			Ports:      []v1.ServicePort{{Port: 80, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)}},
			IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
		},
	})
	err := endpoints.syncService(tCtx, ns+"/foo")
	if err != nil {
		t.Errorf("Unexpected error syncing service %v", err)
	}

	data := runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
				labelManagedBy:       controllerName,
				v1.IsHeadlessService: "",
			},
		},
		Subsets: []v1.EndpointSubset{},
	})
	endpointsHandler.ValidateRequest(t, "/api/v1/namespaces/"+ns+"/endpoints/foo", "PUT", &data)
}

func TestSyncEndpointsHeadlessWithoutPort(t *testing.T) {
	ns := metav1.NamespaceDefault
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()
	tCtx := ktesting.Init(t)
	endpoints := newController(tCtx, testServer.URL, 0*time.Second)
	endpoints.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: v1.ServiceSpec{
			Selector:   map[string]string{"foo": "bar"},
			ClusterIP:  "None",
			Ports:      nil,
			IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
		},
	})
	addPods(endpoints.podStore, ns, 1, 1, 0, ipv4only)

	err := endpoints.syncService(tCtx, ns+"/foo")
	if err != nil {
		t.Errorf("Unexpected error syncing service %v", err)
	}
	endpointsHandler.ValidateRequestCount(t, 1)
	data := runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
			Labels: map[string]string{
				labelManagedBy:       controllerName,
				v1.IsHeadlessService: "",
			},
		},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "1.2.3.4", NodeName: &emptyNodeName, TargetRef: &v1.ObjectReference{Kind: "Pod", Name: "pod0", Namespace: ns}}},
			Ports:     nil,
		}},
	})
	endpointsHandler.ValidateRequest(t, "/api/v1/namespaces/"+ns+"/endpoints", "POST", &data)
}

func TestPodToEndpointAddressForService(t *testing.T) {
	ipv4 := v1.IPv4Protocol
	ipv6 := v1.IPv6Protocol

	testCases := []struct {
		name                   string
		ipFamilies             []v1.IPFamily
		service                v1.Service
		expectedEndpointFamily v1.IPFamily
		expectError            bool
	}{
		{
			name:       "v4 service, in a single stack cluster",
			ipFamilies: ipv4only,
			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIP:  "10.0.0.1",
					IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
				},
			},
			expectedEndpointFamily: ipv4,
		},
		{
			name:       "v4 service, in a dual stack cluster",
			ipFamilies: ipv4ipv6,
			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIP:  "10.0.0.1",
					IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
				},
			},
			expectedEndpointFamily: ipv4,
		},
		{
			name:       "v4 service, in a dual stack ipv6-primary cluster",
			ipFamilies: ipv6ipv4,
			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIP:  "10.0.0.1",
					IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
				},
			},
			expectedEndpointFamily: ipv4,
		},
		{
			name:       "v4 headless service, in a single stack cluster",
			ipFamilies: ipv4only,
			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIP:  v1.ClusterIPNone,
					IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
				},
			},
			expectedEndpointFamily: ipv4,
		},
		{
			name:       "v4 headless service, in a dual stack cluster",
			ipFamilies: ipv4ipv6,
			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIP:  v1.ClusterIPNone,
					IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
				},
			},
			expectedEndpointFamily: ipv4,
		},
		{
			name:       "v6 service, in a dual stack cluster",
			ipFamilies: ipv4ipv6,
			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIP:  "3000::1",
					IPFamilies: []v1.IPFamily{v1.IPv6Protocol},
				},
			},
			expectedEndpointFamily: ipv6,
		},
		{
			name:       "v6 headless service, in a single stack cluster",
			ipFamilies: ipv6only,
			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIP:  v1.ClusterIPNone,
					IPFamilies: []v1.IPFamily{v1.IPv6Protocol},
				},
			},
			expectedEndpointFamily: ipv6,
		},
		{
			name:       "v6 headless service, in a dual stack cluster",
			ipFamilies: ipv4ipv6,
			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIP:  v1.ClusterIPNone,
					IPFamilies: []v1.IPFamily{v1.IPv6Protocol}, // <- set by a api-server defaulting logic
				},
			},
			expectedEndpointFamily: ipv6,
		},
		{
			name:       "v6 service, in a v4 only cluster",
			ipFamilies: ipv4only,
			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIP:  "3000::1",
					IPFamilies: []v1.IPFamily{v1.IPv6Protocol},
				},
			},
			expectError: true,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			podStore := cache.NewStore(cache.DeletionHandlingMetaNamespaceKeyFunc)
			ns := "test"
			// We use addBadIPPod to test that podToEndpointAddressForService
			// fixes up the bad IP.
			addBadIPPod(podStore, ns, tc.ipFamilies)
			pods := podStore.List()
			if len(pods) != 1 {
				t.Fatalf("podStore size: expected: %d, got: %d", 1, len(pods))
			}
			pod := pods[0].(*v1.Pod)
			epa, err := podToEndpointAddressForService(&tc.service, pod)

			if err != nil && !tc.expectError {
				t.Fatalf("podToEndpointAddressForService returned unexpected error %v", err)
			}

			if err == nil && tc.expectError {
				t.Fatalf("podToEndpointAddressForService should have returned error but it did not")
			}

			if err != nil && tc.expectError {
				return
			}

			if utilnet.IsIPv6String(epa.IP) != (tc.expectedEndpointFamily == ipv6) {
				t.Fatalf("IP: expected %s, got: %s", tc.expectedEndpointFamily, epa.IP)
			}
			if strings.HasPrefix(epa.IP, "0") {
				t.Fatalf("IP: expected valid, got: %s", epa.IP)
			}
			if *(epa.NodeName) != pod.Spec.NodeName {
				t.Fatalf("NodeName: expected: %s, got: %s", pod.Spec.NodeName, *(epa.NodeName))
			}
			if epa.TargetRef.Kind != "Pod" {
				t.Fatalf("TargetRef.Kind: expected: %s, got: %s", "Pod", epa.TargetRef.Kind)
			}
			if epa.TargetRef.Namespace != pod.ObjectMeta.Namespace {
				t.Fatalf("TargetRef.Namespace: expected: %s, got: %s", pod.ObjectMeta.Namespace, epa.TargetRef.Namespace)
			}
			if epa.TargetRef.Name != pod.ObjectMeta.Name {
				t.Fatalf("TargetRef.Name: expected: %s, got: %s", pod.ObjectMeta.Name, epa.TargetRef.Name)
			}
			if epa.TargetRef.UID != pod.ObjectMeta.UID {
				t.Fatalf("TargetRef.UID: expected: %s, got: %s", pod.ObjectMeta.UID, epa.TargetRef.UID)
			}
			if epa.TargetRef.ResourceVersion != "" {
				t.Fatalf("TargetRef.ResourceVersion: expected empty, got: %s", epa.TargetRef.ResourceVersion)
			}
		})
	}

}

func TestLastTriggerChangeTimeAnnotation(t *testing.T) {
	ns := "other"
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()

	tCtx := ktesting.Init(t)
	endpoints := newController(tCtx, testServer.URL, 0*time.Second)
	endpoints.endpointsStore.Add(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
		},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "6.7.8.9", NodeName: &emptyNodeName}},
			Ports:     []v1.EndpointPort{{Port: 1000, Protocol: "TCP"}},
		}},
	})
	addPods(endpoints.podStore, ns, 1, 1, 0, ipv4only)
	endpoints.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns, CreationTimestamp: metav1.NewTime(triggerTime)},
		Spec: v1.ServiceSpec{
			Selector:   map[string]string{},
			Ports:      []v1.ServicePort{{Port: 80, TargetPort: intstr.FromInt32(8080), Protocol: "TCP"}},
			IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
		},
	})
	err := endpoints.syncService(tCtx, ns+"/foo")
	if err != nil {
		t.Errorf("Unexpected error syncing service %v", err)
	}

	endpointsHandler.ValidateRequestCount(t, 1)
	data := runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Annotations: map[string]string{
				v1.EndpointsLastChangeTriggerTime: triggerTimeString,
			},
			Labels: map[string]string{
				labelManagedBy:       controllerName,
				v1.IsHeadlessService: "",
			},
		},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "1.2.3.4", NodeName: &emptyNodeName, TargetRef: &v1.ObjectReference{Kind: "Pod", Name: "pod0", Namespace: ns}}},
			Ports:     []v1.EndpointPort{{Port: 8080, Protocol: "TCP"}},
		}},
	})
	endpointsHandler.ValidateRequest(t, "/api/v1/namespaces/"+ns+"/endpoints/foo", "PUT", &data)
}

func TestLastTriggerChangeTimeAnnotation_AnnotationOverridden(t *testing.T) {
	ns := "other"
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()

	tCtx := ktesting.Init(t)
	endpoints := newController(tCtx, testServer.URL, 0*time.Second)
	endpoints.endpointsStore.Add(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Annotations: map[string]string{
				v1.EndpointsLastChangeTriggerTime: oldTriggerTimeString,
			},
		},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "6.7.8.9", NodeName: &emptyNodeName}},
			Ports:     []v1.EndpointPort{{Port: 1000, Protocol: "TCP"}},
		}},
	})
	addPods(endpoints.podStore, ns, 1, 1, 0, ipv4only)
	endpoints.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns, CreationTimestamp: metav1.NewTime(triggerTime)},
		Spec: v1.ServiceSpec{
			Selector:   map[string]string{},
			Ports:      []v1.ServicePort{{Port: 80, TargetPort: intstr.FromInt32(8080), Protocol: "TCP"}},
			IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
		},
	})
	err := endpoints.syncService(tCtx, ns+"/foo")
	if err != nil {
		t.Errorf("Unexpected error syncing service %v", err)
	}

	endpointsHandler.ValidateRequestCount(t, 1)
	data := runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Annotations: map[string]string{
				v1.EndpointsLastChangeTriggerTime: triggerTimeString,
			},
			Labels: map[string]string{
				labelManagedBy:       controllerName,
				v1.IsHeadlessService: "",
			},
		},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "1.2.3.4", NodeName: &emptyNodeName, TargetRef: &v1.ObjectReference{Kind: "Pod", Name: "pod0", Namespace: ns}}},
			Ports:     []v1.EndpointPort{{Port: 8080, Protocol: "TCP"}},
		}},
	})
	endpointsHandler.ValidateRequest(t, "/api/v1/namespaces/"+ns+"/endpoints/foo", "PUT", &data)
}

func TestLastTriggerChangeTimeAnnotation_AnnotationCleared(t *testing.T) {
	ns := "other"
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()

	tCtx := ktesting.Init(t)
	endpoints := newController(tCtx, testServer.URL, 0*time.Second)
	endpoints.endpointsStore.Add(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Annotations: map[string]string{
				v1.EndpointsLastChangeTriggerTime: triggerTimeString,
			},
		},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "6.7.8.9", NodeName: &emptyNodeName}},
			Ports:     []v1.EndpointPort{{Port: 1000, Protocol: "TCP"}},
		}},
	})
	// Neither pod nor service has trigger time, this should cause annotation to be cleared.
	addPods(endpoints.podStore, ns, 1, 1, 0, ipv4only)
	endpoints.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: v1.ServiceSpec{
			Selector:   map[string]string{},
			Ports:      []v1.ServicePort{{Port: 80, TargetPort: intstr.FromInt32(8080), Protocol: "TCP"}},
			IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
		},
	})
	err := endpoints.syncService(tCtx, ns+"/foo")
	if err != nil {
		t.Errorf("Unexpected error syncing service %v", err)
	}

	endpointsHandler.ValidateRequestCount(t, 1)
	data := runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
				labelManagedBy:       controllerName,
				v1.IsHeadlessService: "",
			}, // Annotation not set anymore.
		},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "1.2.3.4", NodeName: &emptyNodeName, TargetRef: &v1.ObjectReference{Kind: "Pod", Name: "pod0", Namespace: ns}}},
			Ports:     []v1.EndpointPort{{Port: 8080, Protocol: "TCP"}},
		}},
	})
	endpointsHandler.ValidateRequest(t, "/api/v1/namespaces/"+ns+"/endpoints/foo", "PUT", &data)
}

// TestPodUpdatesBatching verifies that endpoint updates caused by pod updates are batched together.
// This test uses real time.Sleep, as there is no easy way to mock time in endpoints controller now.
// TODO(mborsz): Migrate this test to mock clock when possible.
func TestPodUpdatesBatching(t *testing.T) {
	type podUpdate struct {
		delay   time.Duration
		podName string
		podIP   string
	}

	tests := []struct {
		name             string
		batchPeriod      time.Duration
		podsCount        int
		updates          []podUpdate
		finalDelay       time.Duration
		wantRequestCount int
	}{
		{
			name:        "three updates with no batching",
			batchPeriod: 0 * time.Second,
			podsCount:   10,
			updates: []podUpdate{
				{
					// endpoints.Run needs ~100 ms to start processing updates.
					delay:   200 * time.Millisecond,
					podName: "pod0",
					podIP:   "10.0.0.0",
				},
				{
					delay:   100 * time.Millisecond,
					podName: "pod1",
					podIP:   "10.0.0.1",
				},
				{
					delay:   100 * time.Millisecond,
					podName: "pod2",
					podIP:   "10.0.0.2",
				},
			},
			finalDelay:       3 * time.Second,
			wantRequestCount: 3,
		},
		{
			name:        "three updates in one batch",
			batchPeriod: 1 * time.Second,
			podsCount:   10,
			updates: []podUpdate{
				{
					// endpoints.Run needs ~100 ms to start processing updates.
					delay:   200 * time.Millisecond,
					podName: "pod0",
					podIP:   "10.0.0.0",
				},
				{
					delay:   100 * time.Millisecond,
					podName: "pod1",
					podIP:   "10.0.0.1",
				},
				{
					delay:   100 * time.Millisecond,
					podName: "pod2",
					podIP:   "10.0.0.2",
				},
			},
			finalDelay:       3 * time.Second,
			wantRequestCount: 1,
		},
		{
			name:        "three updates in two batches",
			batchPeriod: 1 * time.Second,
			podsCount:   10,
			updates: []podUpdate{
				{
					// endpoints.Run needs ~100 ms to start processing updates.
					delay:   200 * time.Millisecond,
					podName: "pod0",
					podIP:   "10.0.0.0",
				},
				{
					delay:   100 * time.Millisecond,
					podName: "pod1",
					podIP:   "10.0.0.1",
				},
				{
					delay:   1 * time.Second,
					podName: "pod2",
					podIP:   "10.0.0.2",
				},
			},
			finalDelay:       3 * time.Second,
			wantRequestCount: 2,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ns := "other"
			resourceVersion := 1
			testServer, endpointsHandler := makeTestServer(t, ns)
			defer testServer.Close()

			tCtx := ktesting.Init(t)
			endpoints := newController(tCtx, testServer.URL, tc.batchPeriod)
			endpoints.podsSynced = alwaysReady
			endpoints.servicesSynced = alwaysReady
			endpoints.endpointsSynced = alwaysReady
			endpoints.workerLoopPeriod = 10 * time.Millisecond

			go endpoints.Run(tCtx, 1)

			addPods(endpoints.podStore, ns, tc.podsCount, 1, 0, ipv4only)

			endpoints.serviceStore.Add(&v1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
				Spec: v1.ServiceSpec{
					Selector:   map[string]string{"foo": "bar"},
					Ports:      []v1.ServicePort{{Port: 80}},
					IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
				},
			})

			for _, update := range tc.updates {
				time.Sleep(update.delay)

				old, exists, err := endpoints.podStore.GetByKey(fmt.Sprintf("%s/%s", ns, update.podName))
				if err != nil {
					t.Fatalf("Error while retrieving old value of %q: %v", update.podName, err)
				}
				if !exists {
					t.Fatalf("Pod %q doesn't exist", update.podName)
				}
				oldPod := old.(*v1.Pod)
				newPod := oldPod.DeepCopy()
				newPod.Status.PodIP = update.podIP
				newPod.Status.PodIPs[0].IP = update.podIP
				newPod.ResourceVersion = strconv.Itoa(resourceVersion)
				resourceVersion++

				endpoints.podStore.Update(newPod)
				endpoints.updatePod(oldPod, newPod)
			}

			time.Sleep(tc.finalDelay)
			endpointsHandler.ValidateRequestCount(t, tc.wantRequestCount)
		})
	}
}

// TestPodAddsBatching verifies that endpoint updates caused by pod addition are batched together.
// This test uses real time.Sleep, as there is no easy way to mock time in endpoints controller now.
// TODO(mborsz): Migrate this test to mock clock when possible.
func TestPodAddsBatching(t *testing.T) {
	type podAdd struct {
		delay time.Duration
	}

	tests := []struct {
		name             string
		batchPeriod      time.Duration
		adds             []podAdd
		finalDelay       time.Duration
		wantRequestCount int
	}{
		{
			name:        "three adds with no batching",
			batchPeriod: 0 * time.Second,
			adds: []podAdd{
				{
					// endpoints.Run needs ~100 ms to start processing updates.
					delay: 200 * time.Millisecond,
				},
				{
					delay: 100 * time.Millisecond,
				},
				{
					delay: 100 * time.Millisecond,
				},
			},
			finalDelay:       3 * time.Second,
			wantRequestCount: 3,
		},
		{
			name:        "three adds in one batch",
			batchPeriod: 1 * time.Second,
			adds: []podAdd{
				{
					// endpoints.Run needs ~100 ms to start processing updates.
					delay: 200 * time.Millisecond,
				},
				{
					delay: 100 * time.Millisecond,
				},
				{
					delay: 100 * time.Millisecond,
				},
			},
			finalDelay:       3 * time.Second,
			wantRequestCount: 1,
		},
		{
			name:        "three adds in two batches",
			batchPeriod: 1 * time.Second,
			adds: []podAdd{
				{
					// endpoints.Run needs ~100 ms to start processing updates.
					delay: 200 * time.Millisecond,
				},
				{
					delay: 100 * time.Millisecond,
				},
				{
					delay: 1 * time.Second,
				},
			},
			finalDelay:       3 * time.Second,
			wantRequestCount: 2,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ns := "other"
			testServer, endpointsHandler := makeTestServer(t, ns)
			defer testServer.Close()

			tCtx := ktesting.Init(t)
			endpoints := newController(tCtx, testServer.URL, tc.batchPeriod)
			endpoints.podsSynced = alwaysReady
			endpoints.servicesSynced = alwaysReady
			endpoints.endpointsSynced = alwaysReady
			endpoints.workerLoopPeriod = 10 * time.Millisecond

			go endpoints.Run(tCtx, 1)

			endpoints.serviceStore.Add(&v1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
				Spec: v1.ServiceSpec{
					Selector:   map[string]string{"foo": "bar"},
					Ports:      []v1.ServicePort{{Port: 80}},
					IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
				},
			})

			for i, add := range tc.adds {
				time.Sleep(add.delay)

				p := testPod(ns, i, 1, true, ipv4only)
				endpoints.podStore.Add(p)
				endpoints.addPod(p)
			}

			time.Sleep(tc.finalDelay)
			endpointsHandler.ValidateRequestCount(t, tc.wantRequestCount)
		})
	}
}

// TestPodDeleteBatching verifies that endpoint updates caused by pod deletion are batched together.
// This test uses real time.Sleep, as there is no easy way to mock time in endpoints controller now.
// TODO(mborsz): Migrate this test to mock clock when possible.
func TestPodDeleteBatching(t *testing.T) {
	type podDelete struct {
		delay   time.Duration
		podName string
	}

	tests := []struct {
		name             string
		batchPeriod      time.Duration
		podsCount        int
		deletes          []podDelete
		finalDelay       time.Duration
		wantRequestCount int
	}{
		{
			name:        "three deletes with no batching",
			batchPeriod: 0 * time.Second,
			podsCount:   10,
			deletes: []podDelete{
				{
					// endpoints.Run needs ~100 ms to start processing updates.
					delay:   200 * time.Millisecond,
					podName: "pod0",
				},
				{
					delay:   100 * time.Millisecond,
					podName: "pod1",
				},
				{
					delay:   100 * time.Millisecond,
					podName: "pod2",
				},
			},
			finalDelay:       3 * time.Second,
			wantRequestCount: 3,
		},
		{
			name:        "three deletes in one batch",
			batchPeriod: 1 * time.Second,
			podsCount:   10,
			deletes: []podDelete{
				{
					// endpoints.Run needs ~100 ms to start processing updates.
					delay:   200 * time.Millisecond,
					podName: "pod0",
				},
				{
					delay:   100 * time.Millisecond,
					podName: "pod1",
				},
				{
					delay:   100 * time.Millisecond,
					podName: "pod2",
				},
			},
			finalDelay:       3 * time.Second,
			wantRequestCount: 1,
		},
		{
			name:        "three deletes in two batches",
			batchPeriod: 1 * time.Second,
			podsCount:   10,
			deletes: []podDelete{
				{
					// endpoints.Run needs ~100 ms to start processing updates.
					delay:   200 * time.Millisecond,
					podName: "pod0",
				},
				{
					delay:   100 * time.Millisecond,
					podName: "pod1",
				},
				{
					delay:   1 * time.Second,
					podName: "pod2",
				},
			},
			finalDelay:       3 * time.Second,
			wantRequestCount: 2,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ns := "other"
			testServer, endpointsHandler := makeTestServer(t, ns)
			defer testServer.Close()

			tCtx := ktesting.Init(t)
			endpoints := newController(tCtx, testServer.URL, tc.batchPeriod)
			endpoints.podsSynced = alwaysReady
			endpoints.servicesSynced = alwaysReady
			endpoints.endpointsSynced = alwaysReady
			endpoints.workerLoopPeriod = 10 * time.Millisecond

			go endpoints.Run(tCtx, 1)

			addPods(endpoints.podStore, ns, tc.podsCount, 1, 0, ipv4only)

			endpoints.serviceStore.Add(&v1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
				Spec: v1.ServiceSpec{
					Selector:   map[string]string{"foo": "bar"},
					Ports:      []v1.ServicePort{{Port: 80}},
					IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
				},
			})

			for _, update := range tc.deletes {
				time.Sleep(update.delay)

				old, exists, err := endpoints.podStore.GetByKey(fmt.Sprintf("%s/%s", ns, update.podName))
				if err != nil {
					t.Fatalf("Error while retrieving old value of %q: %v", update.podName, err)
				}
				if !exists {
					t.Fatalf("Pod %q doesn't exist", update.podName)
				}
				endpoints.podStore.Delete(old)
				endpoints.deletePod(old)
			}

			time.Sleep(tc.finalDelay)
			endpointsHandler.ValidateRequestCount(t, tc.wantRequestCount)
		})
	}
}

func TestSyncEndpointsServiceNotFound(t *testing.T) {
	ns := metav1.NamespaceDefault
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()

	tCtx := ktesting.Init(t)
	endpoints := newController(tCtx, testServer.URL, 0)
	endpoints.endpointsStore.Add(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
		},
	})
	err := endpoints.syncService(tCtx, ns+"/foo")
	if err != nil {
		t.Errorf("Unexpected error syncing service %v", err)
	}
	endpointsHandler.ValidateRequestCount(t, 1)
	endpointsHandler.ValidateRequest(t, "/api/v1/namespaces/"+ns+"/endpoints/foo", "DELETE", nil)
}

func TestSyncServiceOverCapacity(t *testing.T) {
	testCases := []struct {
		name                string
		startingAnnotation  *string
		numExisting         int
		numDesired          int
		numDesiredNotReady  int
		numExpectedReady    int
		numExpectedNotReady int
		expectedAnnotation  bool
	}{{
		name:                "empty",
		startingAnnotation:  nil,
		numExisting:         0,
		numDesired:          0,
		numExpectedReady:    0,
		numExpectedNotReady: 0,
		expectedAnnotation:  false,
	}, {
		name:                "annotation added past capacity, < than maxCapacity of Ready Addresses",
		startingAnnotation:  nil,
		numExisting:         maxCapacity - 1,
		numDesired:          maxCapacity - 3,
		numDesiredNotReady:  4,
		numExpectedReady:    maxCapacity - 3,
		numExpectedNotReady: 3,
		expectedAnnotation:  true,
	}, {
		name:                "annotation added past capacity, maxCapacity of Ready Addresses ",
		startingAnnotation:  nil,
		numExisting:         maxCapacity - 1,
		numDesired:          maxCapacity,
		numDesiredNotReady:  10,
		numExpectedReady:    maxCapacity,
		numExpectedNotReady: 0,
		expectedAnnotation:  true,
	}, {
		name:                "annotation removed below capacity",
		startingAnnotation:  pointer.String("truncated"),
		numExisting:         maxCapacity - 1,
		numDesired:          maxCapacity - 1,
		numDesiredNotReady:  0,
		numExpectedReady:    maxCapacity - 1,
		numExpectedNotReady: 0,
		expectedAnnotation:  false,
	}, {
		name:                "annotation was set to warning previously, annotation removed at capacity",
		startingAnnotation:  pointer.String("warning"),
		numExisting:         maxCapacity,
		numDesired:          maxCapacity,
		numDesiredNotReady:  0,
		numExpectedReady:    maxCapacity,
		numExpectedNotReady: 0,
		expectedAnnotation:  false,
	}, {
		name:                "annotation was set to warning previously but still over capacity",
		startingAnnotation:  pointer.String("warning"),
		numExisting:         maxCapacity + 1,
		numDesired:          maxCapacity + 1,
		numDesiredNotReady:  0,
		numExpectedReady:    maxCapacity,
		numExpectedNotReady: 0,
		expectedAnnotation:  true,
	}, {
		name:                "annotation removed at capacity",
		startingAnnotation:  pointer.String("truncated"),
		numExisting:         maxCapacity,
		numDesired:          maxCapacity,
		numDesiredNotReady:  0,
		numExpectedReady:    maxCapacity,
		numExpectedNotReady: 0,
		expectedAnnotation:  false,
	}, {
		name:                "no endpoints change, annotation value corrected",
		startingAnnotation:  pointer.String("invalid"),
		numExisting:         maxCapacity + 1,
		numDesired:          maxCapacity + 1,
		numDesiredNotReady:  0,
		numExpectedReady:    maxCapacity,
		numExpectedNotReady: 0,
		expectedAnnotation:  true,
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			ns := "test"
			client, c := newFakeController(tCtx, 0*time.Second)

			addPods(c.podStore, ns, tc.numDesired, 1, tc.numDesiredNotReady, ipv4only)
			pods := c.podStore.List()

			svc := &v1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
				Spec: v1.ServiceSpec{
					Selector:   map[string]string{"foo": "bar"},
					Ports:      []v1.ServicePort{{Port: 80}},
					IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
				},
			}
			c.serviceStore.Add(svc)

			subset := v1.EndpointSubset{}
			for i := 0; i < tc.numExisting; i++ {
				pod := pods[i].(*v1.Pod)
				epa, _ := podToEndpointAddressForService(svc, pod)
				subset.Addresses = append(subset.Addresses, *epa)
			}
			endpoints := &v1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:            svc.Name,
					Namespace:       ns,
					ResourceVersion: "1",
					Annotations:     map[string]string{},
				},
				Subsets: []v1.EndpointSubset{subset},
			}
			if tc.startingAnnotation != nil {
				endpoints.Annotations[v1.EndpointsOverCapacity] = *tc.startingAnnotation
			}
			c.endpointsStore.Add(endpoints)
			_, err := client.CoreV1().Endpoints(ns).Create(tCtx, endpoints, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("unexpected error creating endpoints: %v", err)
			}

			err = c.syncService(tCtx, fmt.Sprintf("%s/%s", ns, svc.Name))
			if err != nil {
				t.Errorf("Unexpected error syncing service %v", err)
			}

			actualEndpoints, err := client.CoreV1().Endpoints(ns).Get(tCtx, endpoints.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("unexpected error getting endpoints: %v", err)
			}

			actualAnnotation, ok := actualEndpoints.Annotations[v1.EndpointsOverCapacity]
			if tc.expectedAnnotation {
				if !ok {
					t.Errorf("Expected EndpointsOverCapacity annotation to be set")
				} else if actualAnnotation != "truncated" {
					t.Errorf("Expected EndpointsOverCapacity annotation to be 'truncated', got %s", actualAnnotation)
				}
			} else {
				if ok {
					t.Errorf("Expected EndpointsOverCapacity annotation not to be set, got %s", actualAnnotation)
				}
			}
			numActualReady := 0
			numActualNotReady := 0
			for _, subset := range actualEndpoints.Subsets {
				numActualReady += len(subset.Addresses)
				numActualNotReady += len(subset.NotReadyAddresses)
			}
			if numActualReady != tc.numExpectedReady {
				t.Errorf("Unexpected number of actual ready Endpoints: got %d endpoints, want %d endpoints", numActualReady, tc.numExpectedReady)
			}
			if numActualNotReady != tc.numExpectedNotReady {
				t.Errorf("Unexpected number of actual not ready Endpoints: got %d endpoints, want %d endpoints", numActualNotReady, tc.numExpectedNotReady)
			}
		})
	}
}

func TestTruncateEndpoints(t *testing.T) {
	testCases := []struct {
		desc string
		// subsetsReady, subsetsNotReady, expectedReady, expectedNotReady
		// must all be the same length
		subsetsReady     []int
		subsetsNotReady  []int
		expectedReady    []int
		expectedNotReady []int
	}{{
		desc:             "empty",
		subsetsReady:     []int{},
		subsetsNotReady:  []int{},
		expectedReady:    []int{},
		expectedNotReady: []int{},
	}, {
		desc:             "total endpoints < max capacity",
		subsetsReady:     []int{50, 100, 100, 100, 100},
		subsetsNotReady:  []int{50, 100, 100, 100, 100},
		expectedReady:    []int{50, 100, 100, 100, 100},
		expectedNotReady: []int{50, 100, 100, 100, 100},
	}, {
		desc:             "total endpoints = max capacity",
		subsetsReady:     []int{100, 100, 100, 100, 100},
		subsetsNotReady:  []int{100, 100, 100, 100, 100},
		expectedReady:    []int{100, 100, 100, 100, 100},
		expectedNotReady: []int{100, 100, 100, 100, 100},
	}, {
		desc:             "total ready endpoints < max capacity, but total endpoints > max capacity",
		subsetsReady:     []int{90, 110, 50, 10, 20},
		subsetsNotReady:  []int{101, 200, 200, 201, 298},
		expectedReady:    []int{90, 110, 50, 10, 20},
		expectedNotReady: []int{73, 144, 144, 145, 214},
	}, {
		desc:             "total ready endpoints > max capacity",
		subsetsReady:     []int{205, 400, 402, 400, 693},
		subsetsNotReady:  []int{100, 200, 200, 200, 300},
		expectedReady:    []int{98, 191, 192, 191, 328},
		expectedNotReady: []int{0, 0, 0, 0, 0},
	}}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			var subsets []v1.EndpointSubset
			for subsetIndex, numReady := range tc.subsetsReady {
				subset := v1.EndpointSubset{}
				for i := 0; i < numReady; i++ {
					subset.Addresses = append(subset.Addresses, v1.EndpointAddress{})
				}

				numNotReady := tc.subsetsNotReady[subsetIndex]
				for i := 0; i < numNotReady; i++ {
					subset.NotReadyAddresses = append(subset.NotReadyAddresses, v1.EndpointAddress{})
				}
				subsets = append(subsets, subset)
			}

			endpoints := &v1.Endpoints{Subsets: subsets}
			truncateEndpoints(endpoints)

			for i, subset := range endpoints.Subsets {
				if len(subset.Addresses) != tc.expectedReady[i] {
					t.Errorf("Unexpected number of actual ready Endpoints for subset %d: got %d endpoints, want %d endpoints", i, len(subset.Addresses), tc.expectedReady[i])
				}
				if len(subset.NotReadyAddresses) != tc.expectedNotReady[i] {
					t.Errorf("Unexpected number of actual not ready Endpoints for subset %d: got %d endpoints, want %d endpoints", i, len(subset.NotReadyAddresses), tc.expectedNotReady[i])
				}
			}
		})
	}
}

func TestEndpointPortFromServicePort(t *testing.T) {
	http := pointer.String("http")
	testCases := map[string]struct {
		serviceAppProtocol           *string
		expectedEndpointsAppProtocol *string
	}{
		"empty app protocol": {
			serviceAppProtocol:           nil,
			expectedEndpointsAppProtocol: nil,
		},
		"http app protocol": {
			serviceAppProtocol:           http,
			expectedEndpointsAppProtocol: http,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			epp := endpointPortFromServicePort(&v1.ServicePort{Name: "test", AppProtocol: tc.serviceAppProtocol}, 80)

			if epp.AppProtocol != tc.expectedEndpointsAppProtocol {
				t.Errorf("Expected Endpoints AppProtocol to be %s, got %s", stringVal(tc.expectedEndpointsAppProtocol), stringVal(epp.AppProtocol))
			}
		})
	}
}

// TestMultipleServiceChanges tests that endpoints that are not created because of an out of sync endpoints cache are eventually recreated
// A service will be created. After the endpoints exist, the service will be deleted and the endpoints will not be deleted from the cache immediately.
// After the service is recreated, the endpoints will be deleted replicating an out of sync cache. Expect that eventually the endpoints will be recreated.
func TestMultipleServiceChanges(t *testing.T) {
	ns := metav1.NamespaceDefault
	expectedSubsets := []v1.EndpointSubset{{
		Addresses: []v1.EndpointAddress{
			{IP: "1.2.3.4", NodeName: &emptyNodeName, TargetRef: &v1.ObjectReference{Kind: "Pod", Name: "pod0", Namespace: ns}},
		},
	}}
	endpoint := &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns, ResourceVersion: "1"},
		Subsets:    expectedSubsets,
	}

	controller := &endpointController{}
	blockDelete := make(chan struct{})
	blockNextAction := make(chan struct{})
	stopChan := make(chan struct{})
	testServer := makeBlockingEndpointTestServer(t, controller, endpoint, nil, blockDelete, blockNextAction, ns)
	defer testServer.Close()

	tCtx := ktesting.Init(t)
	*controller = *newController(tCtx, testServer.URL, 0*time.Second)
	addPods(controller.podStore, ns, 1, 1, 0, ipv4only)

	go func() { controller.Run(tCtx, 1) }()

	svc := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: v1.ServiceSpec{
			Selector:   map[string]string{"foo": "bar"},
			ClusterIP:  "None",
			Ports:      nil,
			IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
		},
	}

	controller.serviceStore.Add(svc)
	controller.onServiceUpdate(svc)
	// blockNextAction should eventually unblock once server gets endpoint request.
	waitForChanReceive(t, 1*time.Second, blockNextAction, "Service Add should have caused a request to be sent to the test server")

	controller.serviceStore.Delete(svc)
	controller.onServiceDelete(svc)
	waitForChanReceive(t, 1*time.Second, blockNextAction, "Service Delete should have caused a request to be sent to the test server")

	// If endpoints cache has not updated before service update is registered
	// Services add will not trigger a Create endpoint request.
	controller.serviceStore.Add(svc)
	controller.onServiceUpdate(svc)

	// Ensure the work queue has been processed by looping for up to a second to prevent flakes.
	wait.PollImmediate(50*time.Millisecond, 1*time.Second, func() (bool, error) {
		return controller.queue.Len() == 0, nil
	})

	// Cause test server to delete endpoints
	close(blockDelete)
	waitForChanReceive(t, 1*time.Second, blockNextAction, "Endpoint should have been recreated")

	close(blockNextAction)
	close(stopChan)
}

// TestMultiplePodChanges tests that endpoints that are not updated because of an out of sync endpoints cache are
// eventually resynced after multiple Pod changes.
func TestMultiplePodChanges(t *testing.T) {
	ns := metav1.NamespaceDefault

	readyEndpoints := &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns, ResourceVersion: "1"},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{
				{IP: "1.2.3.4", NodeName: &emptyNodeName, TargetRef: &v1.ObjectReference{Kind: "Pod", Name: "pod0", Namespace: ns}},
			},
			Ports: []v1.EndpointPort{{Port: 8080, Protocol: v1.ProtocolTCP}},
		}},
	}
	notReadyEndpoints := &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns, ResourceVersion: "2"},
		Subsets: []v1.EndpointSubset{{
			NotReadyAddresses: []v1.EndpointAddress{
				{IP: "1.2.3.4", NodeName: &emptyNodeName, TargetRef: &v1.ObjectReference{Kind: "Pod", Name: "pod0", Namespace: ns}},
			},
			Ports: []v1.EndpointPort{{Port: 8080, Protocol: v1.ProtocolTCP}},
		}},
	}

	controller := &endpointController{}
	blockUpdate := make(chan struct{})
	blockNextAction := make(chan struct{})
	stopChan := make(chan struct{})
	testServer := makeBlockingEndpointTestServer(t, controller, notReadyEndpoints, blockUpdate, nil, blockNextAction, ns)
	defer testServer.Close()

	tCtx := ktesting.Init(t)
	*controller = *newController(tCtx, testServer.URL, 0*time.Second)
	pod := testPod(ns, 0, 1, true, ipv4only)
	_ = controller.podStore.Add(pod)
	_ = controller.endpointsStore.Add(readyEndpoints)
	_ = controller.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: v1.ServiceSpec{
			Selector:   map[string]string{"foo": "bar"},
			ClusterIP:  "10.0.0.1",
			Ports:      []v1.ServicePort{{Port: 80, Protocol: "TCP", TargetPort: intstr.FromInt32(8080)}},
			IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
		},
	})

	go func() { controller.Run(tCtx, 1) }()

	// Rapidly update the Pod: Ready -> NotReady -> Ready.
	pod2 := pod.DeepCopy()
	pod2.ResourceVersion = "2"
	pod2.Status.Conditions[0].Status = v1.ConditionFalse
	_ = controller.podStore.Update(pod2)
	controller.updatePod(pod, pod2)
	// blockNextAction should eventually unblock once server gets endpoints request.
	waitForChanReceive(t, 1*time.Second, blockNextAction, "Pod Update should have caused a request to be sent to the test server")
	// The endpoints update hasn't been applied to the cache yet.
	pod3 := pod.DeepCopy()
	pod3.ResourceVersion = "3"
	pod3.Status.Conditions[0].Status = v1.ConditionTrue
	_ = controller.podStore.Update(pod3)
	controller.updatePod(pod2, pod3)
	// It shouldn't get endpoints request as the endpoints in the cache is out-of-date.
	timer := time.NewTimer(100 * time.Millisecond)
	select {
	case <-timer.C:
	case <-blockNextAction:
		t.Errorf("Pod Update shouldn't have caused a request to be sent to the test server")
	}

	// Applying the endpoints update to the cache should cause test server to update endpoints.
	close(blockUpdate)
	waitForChanReceive(t, 1*time.Second, blockNextAction, "Endpoints should have been updated")

	close(blockNextAction)
	close(stopChan)
}

func TestSyncServiceAddresses(t *testing.T) {
	makeService := func(tolerateUnready bool) *v1.Service {
		return &v1.Service{
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "ns"},
			Spec: v1.ServiceSpec{
				Selector:                 map[string]string{"foo": "bar"},
				PublishNotReadyAddresses: tolerateUnready,
				Type:                     v1.ServiceTypeClusterIP,
				ClusterIP:                "1.1.1.1",
				Ports:                    []v1.ServicePort{{Port: 80}},
				IPFamilies:               []v1.IPFamily{v1.IPv4Protocol},
			},
		}
	}

	makePod := func(phase v1.PodPhase, isReady bool, terminating bool) *v1.Pod {
		statusCondition := v1.ConditionFalse
		if isReady {
			statusCondition = v1.ConditionTrue
		}

		now := metav1.Now()
		deletionTimestamp := &now
		if !terminating {
			deletionTimestamp = nil
		}
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Namespace:         "ns",
				Name:              "fakepod",
				DeletionTimestamp: deletionTimestamp,
				Labels:            map[string]string{"foo": "bar"},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{{Ports: []v1.ContainerPort{
					{Name: "port1", ContainerPort: int32(8080)},
				}}},
			},
			Status: v1.PodStatus{
				Phase: phase,
				Conditions: []v1.PodCondition{
					{
						Type:   v1.PodReady,
						Status: statusCondition,
					},
				},
				PodIP: "10.1.1.1",
				PodIPs: []v1.PodIP{
					{IP: "10.1.1.1"},
				},
			},
		}
	}

	testCases := []struct {
		name            string
		pod             *v1.Pod
		service         *v1.Service
		expectedReady   int
		expectedUnready int
	}{
		{
			name:            "pod running phase",
			pod:             makePod(v1.PodRunning, true, false),
			service:         makeService(false),
			expectedReady:   1,
			expectedUnready: 0,
		},
		{
			name:            "pod running phase being deleted",
			pod:             makePod(v1.PodRunning, true, true),
			service:         makeService(false),
			expectedReady:   0,
			expectedUnready: 0,
		},
		{
			name:            "pod unknown phase container ready",
			pod:             makePod(v1.PodUnknown, true, false),
			service:         makeService(false),
			expectedReady:   1,
			expectedUnready: 0,
		},
		{
			name:            "pod unknown phase container ready being deleted",
			pod:             makePod(v1.PodUnknown, true, true),
			service:         makeService(false),
			expectedReady:   0,
			expectedUnready: 0,
		},
		{
			name:            "pod pending phase container ready",
			pod:             makePod(v1.PodPending, true, false),
			service:         makeService(false),
			expectedReady:   1,
			expectedUnready: 0,
		},
		{
			name:            "pod pending phase container ready being deleted",
			pod:             makePod(v1.PodPending, true, true),
			service:         makeService(false),
			expectedReady:   0,
			expectedUnready: 0,
		},
		{
			name:            "pod unknown phase container not ready",
			pod:             makePod(v1.PodUnknown, false, false),
			service:         makeService(false),
			expectedReady:   0,
			expectedUnready: 1,
		},
		{
			name:            "pod pending phase container not ready",
			pod:             makePod(v1.PodPending, false, false),
			service:         makeService(false),
			expectedReady:   0,
			expectedUnready: 1,
		},
		{
			name:            "pod failed phase",
			pod:             makePod(v1.PodFailed, false, false),
			service:         makeService(false),
			expectedReady:   0,
			expectedUnready: 0,
		},
		{
			name:            "pod succeeded phase",
			pod:             makePod(v1.PodSucceeded, false, false),
			service:         makeService(false),
			expectedReady:   0,
			expectedUnready: 0,
		},
		{
			name:            "pod running phase and tolerate unready",
			pod:             makePod(v1.PodRunning, false, false),
			service:         makeService(true),
			expectedReady:   1,
			expectedUnready: 0,
		},
		{
			name:            "pod running phase and tolerate unready being deleted",
			pod:             makePod(v1.PodRunning, false, true),
			service:         makeService(true),
			expectedReady:   1,
			expectedUnready: 0,
		},
		{
			name:            "pod unknown phase and tolerate unready",
			pod:             makePod(v1.PodUnknown, false, false),
			service:         makeService(true),
			expectedReady:   1,
			expectedUnready: 0,
		},
		{
			name:            "pod unknown phase and tolerate unready being deleted",
			pod:             makePod(v1.PodUnknown, false, true),
			service:         makeService(true),
			expectedReady:   1,
			expectedUnready: 0,
		},
		{
			name:            "pod pending phase and tolerate unready",
			pod:             makePod(v1.PodPending, false, false),
			service:         makeService(true),
			expectedReady:   1,
			expectedUnready: 0,
		},
		{
			name:            "pod pending phase and tolerate unready being deleted",
			pod:             makePod(v1.PodPending, false, true),
			service:         makeService(true),
			expectedReady:   1,
			expectedUnready: 0,
		},
		{
			name:            "pod failed phase and tolerate unready",
			pod:             makePod(v1.PodFailed, false, false),
			service:         makeService(true),
			expectedReady:   0,
			expectedUnready: 0,
		},
		{
			name:            "pod succeeded phase and tolerate unready endpoints",
			pod:             makePod(v1.PodSucceeded, false, false),
			service:         makeService(true),
			expectedReady:   0,
			expectedUnready: 0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)

			ns := tc.service.Namespace
			client, c := newFakeController(tCtx, 0*time.Second)

			err := c.podStore.Add(tc.pod)
			if err != nil {
				t.Errorf("Unexpected error adding pod %v", err)
			}
			err = c.serviceStore.Add(tc.service)
			if err != nil {
				t.Errorf("Unexpected error adding service %v", err)
			}
			err = c.syncService(tCtx, fmt.Sprintf("%s/%s", ns, tc.service.Name))
			if err != nil {
				t.Errorf("Unexpected error syncing service %v", err)
			}

			endpoints, err := client.CoreV1().Endpoints(ns).Get(tCtx, tc.service.Name, metav1.GetOptions{})
			if err != nil {
				t.Errorf("Unexpected error %v", err)
			}

			readyEndpoints := 0
			unreadyEndpoints := 0
			for _, subset := range endpoints.Subsets {
				readyEndpoints += len(subset.Addresses)
				unreadyEndpoints += len(subset.NotReadyAddresses)
			}

			if tc.expectedReady != readyEndpoints {
				t.Errorf("Expected %d ready endpoints, got %d", tc.expectedReady, readyEndpoints)
			}

			if tc.expectedUnready != unreadyEndpoints {
				t.Errorf("Expected %d ready endpoints, got %d", tc.expectedUnready, unreadyEndpoints)
			}
		})
	}
}

func TestEndpointsDeletionEvents(t *testing.T) {
	ns := metav1.NamespaceDefault
	testServer, _ := makeTestServer(t, ns)
	defer testServer.Close()

	tCtx := ktesting.Init(t)
	controller := newController(tCtx, testServer.URL, 0)
	store := controller.endpointsStore
	ep1 := &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "rv1",
		},
	}

	// Test Unexpected and Expected Deletes
	store.Delete(ep1)
	controller.onEndpointsDelete(ep1)

	if controller.queue.Len() != 1 {
		t.Errorf("Expected one service to be in the queue, found %d", controller.queue.Len())
	}
}

func stringVal(str *string) string {
	if str == nil {
		return "nil"
	}
	return *str
}

// waitForChanReceive blocks up to the timeout waiting for the receivingChan to receive
func waitForChanReceive(t *testing.T, timeout time.Duration, receivingChan chan struct{}, errorMsg string) {
	timer := time.NewTimer(timeout)
	select {
	case <-timer.C:
		t.Error(errorMsg)
	case <-receivingChan:
	}
}

func TestEndpointSubsetsEqualIgnoreResourceVersion(t *testing.T) {
	copyAndMutateEndpointSubset := func(orig *v1.EndpointSubset, mutator func(*v1.EndpointSubset)) *v1.EndpointSubset {
		newSubSet := orig.DeepCopy()
		mutator(newSubSet)
		return newSubSet
	}
	es1 := &v1.EndpointSubset{
		Addresses: []v1.EndpointAddress{
			{
				IP:        "1.1.1.1",
				TargetRef: &v1.ObjectReference{Kind: "Pod", Name: "pod1-1", Namespace: "ns", ResourceVersion: "1"},
			},
		},
		NotReadyAddresses: []v1.EndpointAddress{
			{
				IP:        "1.1.1.2",
				TargetRef: &v1.ObjectReference{Kind: "Pod", Name: "pod1-2", Namespace: "ns2", ResourceVersion: "2"},
			},
		},
		Ports: []v1.EndpointPort{{Port: 8081, Protocol: "TCP"}},
	}
	es2 := &v1.EndpointSubset{
		Addresses: []v1.EndpointAddress{
			{
				IP:        "2.2.2.1",
				TargetRef: &v1.ObjectReference{Kind: "Pod", Name: "pod2-1", Namespace: "ns", ResourceVersion: "3"},
			},
		},
		NotReadyAddresses: []v1.EndpointAddress{
			{
				IP:        "2.2.2.2",
				TargetRef: &v1.ObjectReference{Kind: "Pod", Name: "pod2-2", Namespace: "ns2", ResourceVersion: "4"},
			},
		},
		Ports: []v1.EndpointPort{{Port: 8082, Protocol: "TCP"}},
	}
	tests := []struct {
		name     string
		subsets1 []v1.EndpointSubset
		subsets2 []v1.EndpointSubset
		expected bool
	}{
		{
			name:     "Subsets removed",
			subsets1: []v1.EndpointSubset{*es1, *es2},
			subsets2: []v1.EndpointSubset{*es1},
			expected: false,
		},
		{
			name:     "Ready Pod IP changed",
			subsets1: []v1.EndpointSubset{*es1, *es2},
			subsets2: []v1.EndpointSubset{*copyAndMutateEndpointSubset(es1, func(es *v1.EndpointSubset) {
				es.Addresses[0].IP = "1.1.1.10"
			}), *es2},
			expected: false,
		},
		{
			name:     "NotReady Pod IP changed",
			subsets1: []v1.EndpointSubset{*es1, *es2},
			subsets2: []v1.EndpointSubset{*es1, *copyAndMutateEndpointSubset(es2, func(es *v1.EndpointSubset) {
				es.NotReadyAddresses[0].IP = "2.2.2.10"
			})},
			expected: false,
		},
		{
			name:     "Pod ResourceVersion changed",
			subsets1: []v1.EndpointSubset{*es1, *es2},
			subsets2: []v1.EndpointSubset{*es1, *copyAndMutateEndpointSubset(es2, func(es *v1.EndpointSubset) {
				es.Addresses[0].TargetRef.ResourceVersion = "100"
			})},
			expected: true,
		},
		{
			name:     "Pod ResourceVersion removed",
			subsets1: []v1.EndpointSubset{*es1, *es2},
			subsets2: []v1.EndpointSubset{*es1, *copyAndMutateEndpointSubset(es2, func(es *v1.EndpointSubset) {
				es.Addresses[0].TargetRef.ResourceVersion = ""
			})},
			expected: true,
		},
		{
			name:     "Ports changed",
			subsets1: []v1.EndpointSubset{*es1, *es2},
			subsets2: []v1.EndpointSubset{*es1, *copyAndMutateEndpointSubset(es1, func(es *v1.EndpointSubset) {
				es.Ports[0].Port = 8082
			})},
			expected: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := endpointSubsetsEqualIgnoreResourceVersion(tt.subsets1, tt.subsets2); got != tt.expected {
				t.Errorf("semanticIgnoreResourceVersion.DeepEqual() = %v, expected %v", got, tt.expected)
			}
		})
	}
}
