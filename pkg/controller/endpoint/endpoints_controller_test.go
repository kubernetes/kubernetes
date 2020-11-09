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
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strconv"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	clientscheme "k8s.io/client-go/kubernetes/scheme"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	utiltesting "k8s.io/client-go/util/testing"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	endptspkg "k8s.io/kubernetes/pkg/api/v1/endpoints"
	api "k8s.io/kubernetes/pkg/apis/core"
	controllerpkg "k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/features"
	utilnet "k8s.io/utils/net"
	utilpointer "k8s.io/utils/pointer"
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
			Namespace: namespace,
			Name:      fmt.Sprintf("pod%d", id),
			Labels:    map[string]string{"foo": "bar"},
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
			ip = fmt.Sprintf("1.2.3.%d", 4+id)
		} else {
			ip = fmt.Sprintf("2000::%d", 4+id)
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

// makeBlockingEndpointDeleteTestServer will signal the blockNextAction channel on endpoint "POST" & "DELETE" requests. All
// block endpoint "DELETE" requestsi will wait on a blockDelete signal to delete endpoint. If controller is nil, a error will
// be sent in the response.
func makeBlockingEndpointDeleteTestServer(t *testing.T, controller *endpointController, endpoint *v1.Endpoints, blockDelete, blockNextAction chan struct{}, namespace string) *httptest.Server {

	handlerFunc := func(res http.ResponseWriter, req *http.Request) {
		if controller == nil {
			res.WriteHeader(http.StatusInternalServerError)
			res.Write([]byte("controller has not been set yet"))
			return
		}

		if req.Method == "POST" {
			controller.endpointsStore.Add(endpoint)
			blockNextAction <- struct{}{}
		}

		if req.Method == "DELETE" {
			go func() {
				// Delay the deletion of endoints to make endpoint cache out of sync
				<-blockDelete
				controller.endpointsStore.Delete(endpoint)
				controller.onEndpointsDelete(endpoint)
			}()
			blockNextAction <- struct{}{}
		}

		res.WriteHeader(http.StatusOK)
		res.Write([]byte(runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{})))
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

func newController(url string, batchPeriod time.Duration) *endpointController {
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: url, ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	informerFactory := informers.NewSharedInformerFactory(client, controllerpkg.NoResyncPeriodFunc())
	endpoints := NewEndpointController(informerFactory.Core().V1().Pods(), informerFactory.Core().V1().Services(),
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

func TestSyncEndpointsItemsPreserveNoSelector(t *testing.T) {
	ns := metav1.NamespaceDefault
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()
	endpoints := newController(testServer.URL, 0*time.Second)
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
	endpoints.syncService(ns + "/foo")
	endpointsHandler.ValidateRequestCount(t, 0)
}

func TestSyncEndpointsExistingNilSubsets(t *testing.T) {
	ns := metav1.NamespaceDefault
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()
	endpoints := newController(testServer.URL, 0*time.Second)
	endpoints.endpointsStore.Add(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
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
	endpoints.syncService(ns + "/foo")
	endpointsHandler.ValidateRequestCount(t, 0)
}

func TestSyncEndpointsExistingEmptySubsets(t *testing.T) {
	ns := metav1.NamespaceDefault
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()
	endpoints := newController(testServer.URL, 0*time.Second)
	endpoints.endpointsStore.Add(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
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
	endpoints.syncService(ns + "/foo")
	endpointsHandler.ValidateRequestCount(t, 0)
}

func TestSyncEndpointsNewNoSubsets(t *testing.T) {
	ns := metav1.NamespaceDefault
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()
	endpoints := newController(testServer.URL, 0*time.Second)
	endpoints.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: v1.ServiceSpec{
			Selector: map[string]string{"foo": "bar"},
			Ports:    []v1.ServicePort{{Port: 80}},
		},
	})
	endpoints.syncService(ns + "/foo")
	endpointsHandler.ValidateRequestCount(t, 1)
}

func TestCheckLeftoverEndpoints(t *testing.T) {
	ns := metav1.NamespaceDefault
	testServer, _ := makeTestServer(t, ns)
	defer testServer.Close()
	endpoints := newController(testServer.URL, 0*time.Second)
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
	endpoints := newController(testServer.URL, 0*time.Second)
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
			Selector: map[string]string{},
			Ports:    []v1.ServicePort{{Port: 80, TargetPort: intstr.FromInt(8080), Protocol: "TCP"}},
		},
	})
	endpoints.syncService(ns + "/foo")

	endpointsHandler.ValidateRequestCount(t, 1)
	data := runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
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
	endpoints := newController(testServer.URL, 0*time.Second)
	endpoints.endpointsStore.Add(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
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
	endpoints.syncService(ns + "/foo")
	endpointsHandler.ValidateRequestCount(t, 0)
}

func TestSyncEndpointsProtocolUDP(t *testing.T) {
	ns := "other"
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()
	endpoints := newController(testServer.URL, 0*time.Second)
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
			Selector: map[string]string{},
			Ports:    []v1.ServicePort{{Port: 80, TargetPort: intstr.FromInt(8080), Protocol: "UDP"}},
		},
	})
	endpoints.syncService(ns + "/foo")

	endpointsHandler.ValidateRequestCount(t, 1)
	data := runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
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
	endpoints := newController(testServer.URL, 0*time.Second)
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
			Selector: map[string]string{},
			Ports:    []v1.ServicePort{{Port: 80, TargetPort: intstr.FromInt(8080), Protocol: "SCTP"}},
		},
	})
	endpoints.syncService(ns + "/foo")

	endpointsHandler.ValidateRequestCount(t, 1)
	data := runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
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
	endpoints := newController(testServer.URL, 0*time.Second)
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
			Selector: map[string]string{},
			Ports:    []v1.ServicePort{{Port: 80, Protocol: "TCP", TargetPort: intstr.FromInt(8080)}},
		},
	})
	endpoints.syncService(ns + "/foo")

	data := runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
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
	endpoints := newController(testServer.URL, 0*time.Second)
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
			Selector: map[string]string{},
			Ports:    []v1.ServicePort{{Port: 80, Protocol: "TCP", TargetPort: intstr.FromInt(8080)}},
		},
	})
	endpoints.syncService(ns + "/foo")

	data := runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
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
	endpoints := newController(testServer.URL, 0*time.Second)
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
			Selector: map[string]string{},
			Ports:    []v1.ServicePort{{Port: 80, Protocol: "TCP", TargetPort: intstr.FromInt(8080)}},
		},
	})
	endpoints.syncService(ns + "/foo")

	data := runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
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
	endpoints := newController(testServer.URL, 0*time.Second)
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
	addPods(endpoints.podStore, ns, 1, 1, 0, ipv4only)
	endpoints.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: v1.ServiceSpec{
			Selector: map[string]string{"foo": "bar"},
			Ports:    []v1.ServicePort{{Port: 80, Protocol: "TCP", TargetPort: intstr.FromInt(8080)}},
		},
	})
	endpoints.syncService(ns + "/foo")

	data := runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
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
	endpoints := newController(testServer.URL, 0*time.Second)
	endpoints.endpointsStore.Add(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			ResourceVersion: "1",
			Name:            "foo",
			Namespace:       ns,
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
			Selector: map[string]string{"foo": "bar"},
			Ports:    []v1.ServicePort{{Port: 80, Protocol: "TCP", TargetPort: intstr.FromInt(8080)}},
		},
	})
	endpoints.syncService(ns + "/foo")
	endpointsHandler.ValidateRequestCount(t, 0)
}

func TestSyncEndpointsItems(t *testing.T) {
	ns := "other"
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()
	endpoints := newController(testServer.URL, 0*time.Second)
	addPods(endpoints.podStore, ns, 3, 2, 0, ipv4only)
	addPods(endpoints.podStore, "blah", 5, 2, 0, ipv4only) // make sure these aren't found!

	endpoints.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: v1.ServiceSpec{
			Selector: map[string]string{"foo": "bar"},
			Ports: []v1.ServicePort{
				{Name: "port0", Port: 80, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
				{Name: "port1", Port: 88, Protocol: "TCP", TargetPort: intstr.FromInt(8088)},
			},
		},
	})
	endpoints.syncService("other/foo")

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
	endpoints := newController(testServer.URL, 0*time.Second)
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
				{Name: "port0", Port: 80, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
				{Name: "port1", Port: 88, Protocol: "TCP", TargetPort: intstr.FromInt(8088)},
			},
		},
	})
	endpoints.syncService(ns + "/foo")

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
	endpoints := newController(testServer.URL, 0*time.Second)
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
			Selector: map[string]string{"foo": "bar"},
			Ports:    []v1.ServicePort{{Port: 80, Protocol: "TCP", TargetPort: intstr.FromInt(8080)}},
		},
	})
	endpoints.syncService(ns + "/foo")

	serviceLabels[v1.IsHeadlessService] = ""
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
			endpoints := newController(testServer.URL, 0*time.Second)
			addPods(endpoints.podStore, ns, 1, 1, 0, ipv4only)

			service := &v1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{},
					Ports:    []v1.ServicePort{{Port: 80, TargetPort: intstr.FromInt(8080), Protocol: "TCP"}},
				},
			}
			endpoints.serviceStore.Add(service)
			endpoints.onServiceUpdate(service)
			endpoints.podsSynced = test.podsSynced
			endpoints.servicesSynced = test.servicesSynced
			endpoints.endpointsSynced = test.endpointsSynced
			endpoints.workerLoopPeriod = 10 * time.Millisecond
			stopCh := make(chan struct{})
			defer close(stopCh)
			go endpoints.Run(1, stopCh)

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
	endpoints := newController(testServer.URL, 0*time.Second)
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
			Selector:  map[string]string{},
			ClusterIP: api.ClusterIPNone,
			Ports:     []v1.ServicePort{},
		},
	}
	originalService := service.DeepCopy()
	endpoints.serviceStore.Add(service)
	endpoints.syncService(ns + "/foo")
	data := runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
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
		t.Fatalf("syncing endpoints changed service: %s", diff.ObjectReflectDiff(service, originalService))
	}
	endpointsHandler.ValidateRequestCount(t, 1)
	endpointsHandler.ValidateRequest(t, "/api/v1/namespaces/"+ns+"/endpoints/foo", "PUT", &data)
}

func TestSyncEndpointsItemsExcludeNotReadyPodsWithRestartPolicyNeverAndPhaseFailed(t *testing.T) {
	ns := "other"
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()
	endpoints := newController(testServer.URL, 0*time.Second)
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
	addNotReadyPodsWithSpecifiedRestartPolicyAndPhase(endpoints.podStore, ns, 1, 1, v1.RestartPolicyNever, v1.PodFailed)
	endpoints.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: v1.ServiceSpec{
			Selector: map[string]string{"foo": "bar"},
			Ports:    []v1.ServicePort{{Port: 80, Protocol: "TCP", TargetPort: intstr.FromInt(8080)}},
		},
	})
	endpoints.syncService(ns + "/foo")
	data := runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
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
	endpoints := newController(testServer.URL, 0*time.Second)
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
			Selector: map[string]string{"foo": "bar"},
			Ports:    []v1.ServicePort{{Port: 80, Protocol: "TCP", TargetPort: intstr.FromInt(8080)}},
		},
	})
	endpoints.syncService(ns + "/foo")
	data := runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
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
	endpoints := newController(testServer.URL, 0*time.Second)
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
			Selector: map[string]string{"foo": "bar"},
			Ports:    []v1.ServicePort{{Port: 80, Protocol: "TCP", TargetPort: intstr.FromInt(8080)}},
		},
	})
	endpoints.syncService(ns + "/foo")
	data := runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
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
	endpoints := newController(testServer.URL, 0*time.Second)
	endpoints.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: v1.ServiceSpec{
			Selector:  map[string]string{"foo": "bar"},
			ClusterIP: "None",
			Ports:     nil,
		},
	})
	addPods(endpoints.podStore, ns, 1, 1, 0, ipv4only)
	endpoints.syncService(ns + "/foo")
	endpointsHandler.ValidateRequestCount(t, 1)
	data := runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
			Labels: map[string]string{
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

// There are 3*5 possibilities(3 types of RestartPolicy by 5 types of PodPhase). Not list them all here.
// Just list all of the 3 false cases and 3 of the 12 true cases.
func TestShouldPodBeInEndpoints(t *testing.T) {
	testCases := []struct {
		name     string
		pod      *v1.Pod
		expected bool
	}{
		// Pod should not be in endpoints cases:
		{
			name: "Failed pod with Never RestartPolicy",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
				},
				Status: v1.PodStatus{
					Phase: v1.PodFailed,
				},
			},
			expected: false,
		},
		{
			name: "Succeeded pod with Never RestartPolicy",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
				},
				Status: v1.PodStatus{
					Phase: v1.PodSucceeded,
				},
			},
			expected: false,
		},
		{
			name: "Succeeded pod with OnFailure RestartPolicy",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyOnFailure,
				},
				Status: v1.PodStatus{
					Phase: v1.PodSucceeded,
				},
			},
			expected: false,
		},
		// Pod should be in endpoints cases:
		{
			name: "Failed pod with Always RestartPolicy",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyAlways,
				},
				Status: v1.PodStatus{
					Phase: v1.PodFailed,
				},
			},
			expected: true,
		},
		{
			name: "Pending pod with Never RestartPolicy",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
				},
				Status: v1.PodStatus{
					Phase: v1.PodPending,
				},
			},
			expected: true,
		},
		{
			name: "Unknown pod with OnFailure RestartPolicy",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyOnFailure,
				},
				Status: v1.PodStatus{
					Phase: v1.PodUnknown,
				},
			},
			expected: true,
		},
	}
	for _, test := range testCases {
		result := shouldPodBeInEndpoints(test.pod)
		if result != test.expected {
			t.Errorf("%s: expected : %t, got: %t", test.name, test.expected, result)
		}
	}
}

func TestPodToEndpointAddressForService(t *testing.T) {
	ipv4 := v1.IPv4Protocol
	ipv6 := v1.IPv6Protocol

	testCases := []struct {
		name string

		enableDualStack bool
		ipFamilies      []v1.IPFamily

		service v1.Service

		expectedEndpointFamily v1.IPFamily
		expectError            bool
	}{
		{
			name: "v4 service, in a single stack cluster",

			enableDualStack: false,
			ipFamilies:      ipv4only,

			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIP: "10.0.0.1",
				},
			},

			expectedEndpointFamily: ipv4,
		},
		{
			name: "v4 service, in a dual stack cluster",

			enableDualStack: true,
			ipFamilies:      ipv4ipv6,

			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIP: "10.0.0.1",
				},
			},

			expectedEndpointFamily: ipv4,
		},
		{
			name: "v4 service, in a dual stack ipv6-primary cluster",

			enableDualStack: true,
			ipFamilies:      ipv6ipv4,

			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIP: "10.0.0.1",
				},
			},

			expectedEndpointFamily: ipv4,
		},
		{
			name: "v4 headless service, in a single stack cluster",

			enableDualStack: false,
			ipFamilies:      ipv4only,

			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIP: v1.ClusterIPNone,
				},
			},

			expectedEndpointFamily: ipv4,
		},
		{
			name: "v4 headless service, in a dual stack cluster",

			enableDualStack: false,
			ipFamilies:      ipv4ipv6,

			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIP:  v1.ClusterIPNone,
					IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
				},
			},

			expectedEndpointFamily: ipv4,
		},
		{
			name: "v4 legacy headless service, in a dual stack cluster",

			enableDualStack: false,
			ipFamilies:      ipv4ipv6,

			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIP: v1.ClusterIPNone,
				},
			},

			expectedEndpointFamily: ipv4,
		},
		{
			name: "v4 legacy headless service, in a dual stack ipv6-primary cluster",

			enableDualStack: true,
			ipFamilies:      ipv6ipv4,

			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIP: v1.ClusterIPNone,
				},
			},

			expectedEndpointFamily: ipv6,
		},
		{
			name: "v6 service, in a dual stack cluster",

			enableDualStack: true,
			ipFamilies:      ipv4ipv6,

			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIP: "3000::1",
				},
			},

			expectedEndpointFamily: ipv6,
		},
		{
			name: "v6 headless service, in a single stack cluster",

			enableDualStack: false,
			ipFamilies:      ipv6only,

			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIP: v1.ClusterIPNone,
				},
			},

			expectedEndpointFamily: ipv6,
		},
		{
			name: "v6 headless service, in a dual stack cluster (connected to a new api-server)",

			enableDualStack: true,
			ipFamilies:      ipv4ipv6,

			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIP:  v1.ClusterIPNone,
					IPFamilies: []v1.IPFamily{v1.IPv6Protocol}, // <- set by a api-server defaulting logic
				},
			},

			expectedEndpointFamily: ipv6,
		},
		{
			name: "v6 legacy headless service, in a dual stack cluster  (connected to a old api-server)",

			enableDualStack: false,
			ipFamilies:      ipv4ipv6,

			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIP: v1.ClusterIPNone, // <- families are not set by api-server
				},
			},

			expectedEndpointFamily: ipv4,
		},

		// in reality this is a misconfigured cluster
		// i.e user is not using dual stack and have PodIP == v4 and ServiceIP==v6
		// we are testing that we will keep producing the expected behavior
		{
			name: "v6 service, in a v4 only cluster. dual stack disabled",

			enableDualStack: false,
			ipFamilies:      ipv4only,

			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIP: "3000::1",
				},
			},

			expectedEndpointFamily: ipv4,
		},
		// but this will actually give an error
		{
			name: "v6 service, in a v4 only cluster - dual stack enabled",

			enableDualStack: true,
			ipFamilies:      ipv4only,

			service: v1.Service{
				Spec: v1.ServiceSpec{
					ClusterIP: "3000::1",
				},
			},

			expectError: true,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IPv6DualStack, tc.enableDualStack)()
			podStore := cache.NewStore(cache.DeletionHandlingMetaNamespaceKeyFunc)
			ns := "test"
			addPods(podStore, ns, 1, 1, 0, tc.ipFamilies)
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
			if *(epa.NodeName) != pod.Spec.NodeName {
				t.Fatalf("NodeName: expected: %s, got: %s", pod.Spec.NodeName, *(epa.NodeName))
			}
			if epa.TargetRef.Kind != "Pod" {
				t.Fatalf("TargetRef.Kind: expected: %s, got: %s", "Pod", epa.TargetRef.Kind)
			}
			if epa.TargetRef.Namespace != pod.ObjectMeta.Namespace {
				t.Fatalf("TargetRef.Kind: expected: %s, got: %s", pod.ObjectMeta.Namespace, epa.TargetRef.Namespace)
			}
			if epa.TargetRef.Name != pod.ObjectMeta.Name {
				t.Fatalf("TargetRef.Kind: expected: %s, got: %s", pod.ObjectMeta.Name, epa.TargetRef.Name)
			}
			if epa.TargetRef.UID != pod.ObjectMeta.UID {
				t.Fatalf("TargetRef.Kind: expected: %s, got: %s", pod.ObjectMeta.UID, epa.TargetRef.UID)
			}
			if epa.TargetRef.ResourceVersion != pod.ObjectMeta.ResourceVersion {
				t.Fatalf("TargetRef.Kind: expected: %s, got: %s", pod.ObjectMeta.ResourceVersion, epa.TargetRef.ResourceVersion)
			}
		})
	}

}

func TestLastTriggerChangeTimeAnnotation(t *testing.T) {
	ns := "other"
	testServer, endpointsHandler := makeTestServer(t, ns)
	defer testServer.Close()
	endpoints := newController(testServer.URL, 0*time.Second)
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
			Selector: map[string]string{},
			Ports:    []v1.ServicePort{{Port: 80, TargetPort: intstr.FromInt(8080), Protocol: "TCP"}},
		},
	})
	endpoints.syncService(ns + "/foo")

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
	endpoints := newController(testServer.URL, 0*time.Second)
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
			Selector: map[string]string{},
			Ports:    []v1.ServicePort{{Port: 80, TargetPort: intstr.FromInt(8080), Protocol: "TCP"}},
		},
	})
	endpoints.syncService(ns + "/foo")

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
	endpoints := newController(testServer.URL, 0*time.Second)
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
			Selector: map[string]string{},
			Ports:    []v1.ServicePort{{Port: 80, TargetPort: intstr.FromInt(8080), Protocol: "TCP"}},
		},
	})
	endpoints.syncService(ns + "/foo")

	endpointsHandler.ValidateRequestCount(t, 1)
	data := runtime.EncodeOrDie(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
			Labels: map[string]string{
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
			endpoints := newController(testServer.URL, tc.batchPeriod)
			stopCh := make(chan struct{})
			defer close(stopCh)
			endpoints.podsSynced = alwaysReady
			endpoints.servicesSynced = alwaysReady
			endpoints.endpointsSynced = alwaysReady
			endpoints.workerLoopPeriod = 10 * time.Millisecond

			go endpoints.Run(1, stopCh)

			addPods(endpoints.podStore, ns, tc.podsCount, 1, 0, ipv4only)

			endpoints.serviceStore.Add(&v1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{"foo": "bar"},
					Ports:    []v1.ServicePort{{Port: 80}},
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
			endpoints := newController(testServer.URL, tc.batchPeriod)
			stopCh := make(chan struct{})
			defer close(stopCh)
			endpoints.podsSynced = alwaysReady
			endpoints.servicesSynced = alwaysReady
			endpoints.endpointsSynced = alwaysReady
			endpoints.workerLoopPeriod = 10 * time.Millisecond

			go endpoints.Run(1, stopCh)

			endpoints.serviceStore.Add(&v1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{"foo": "bar"},
					Ports:    []v1.ServicePort{{Port: 80}},
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
			endpoints := newController(testServer.URL, tc.batchPeriod)
			stopCh := make(chan struct{})
			defer close(stopCh)
			endpoints.podsSynced = alwaysReady
			endpoints.servicesSynced = alwaysReady
			endpoints.endpointsSynced = alwaysReady
			endpoints.workerLoopPeriod = 10 * time.Millisecond

			go endpoints.Run(1, stopCh)

			addPods(endpoints.podStore, ns, tc.podsCount, 1, 0, ipv4only)

			endpoints.serviceStore.Add(&v1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{"foo": "bar"},
					Ports:    []v1.ServicePort{{Port: 80}},
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
	endpoints := newController(testServer.URL, 0)
	endpoints.endpointsStore.Add(&v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
		},
	})
	endpoints.syncService(ns + "/foo")
	endpointsHandler.ValidateRequestCount(t, 1)
	endpointsHandler.ValidateRequest(t, "/api/v1/namespaces/"+ns+"/endpoints/foo", "DELETE", nil)
}

func TestEndpointPortFromServicePort(t *testing.T) {
	http := utilpointer.StringPtr("http")
	testCases := map[string]struct {
		featureGateEnabled           bool
		serviceAppProtocol           *string
		expectedEndpointsAppProtocol *string
	}{
		"feature gate disabled, empty app protocol": {
			featureGateEnabled:           false,
			serviceAppProtocol:           nil,
			expectedEndpointsAppProtocol: nil,
		},
		"feature gate disabled, http app protocol": {
			featureGateEnabled:           false,
			serviceAppProtocol:           http,
			expectedEndpointsAppProtocol: nil,
		},
		"feature gate enabled, empty app protocol": {
			featureGateEnabled:           true,
			serviceAppProtocol:           nil,
			expectedEndpointsAppProtocol: nil,
		},
		"feature gate enabled, http app protocol": {
			featureGateEnabled:           true,
			serviceAppProtocol:           http,
			expectedEndpointsAppProtocol: http,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ServiceAppProtocol, tc.featureGateEnabled)()

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
	testServer := makeBlockingEndpointDeleteTestServer(t, controller, endpoint, blockDelete, blockNextAction, ns)
	defer testServer.Close()

	*controller = *newController(testServer.URL, 0*time.Second)
	addPods(controller.podStore, ns, 1, 1, 0, ipv4only)

	go func() { controller.Run(1, stopChan) }()

	svc := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
		Spec: v1.ServiceSpec{
			Selector:  map[string]string{"foo": "bar"},
			ClusterIP: "None",
			Ports:     nil,
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

func TestEndpointsDeletionEvents(t *testing.T) {
	ns := metav1.NamespaceDefault
	testServer, _ := makeTestServer(t, ns)
	defer testServer.Close()
	controller := newController(testServer.URL, 0)
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
		t.Errorf(errorMsg)
	case <-receivingChan:
	}
}
