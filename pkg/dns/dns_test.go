/*
Copyright 2016 The Kubernetes Authors.

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

package dns

import (
	"encoding/json"
	"fmt"
	"net"
	"strings"
	"sync"
	"testing"

	etcd "github.com/coreos/etcd/client"
	"github.com/miekg/dns"
	skymsg "github.com/skynetservices/skydns/msg"
	skyServer "github.com/skynetservices/skydns/server"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	kapi "k8s.io/kubernetes/pkg/api"
	endpointsapi "k8s.io/kubernetes/pkg/api/endpoints"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/cache"
	fake "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/util/sets"
)

const (
	testDomain    = "cluster.local."
	testService   = "testservice"
	testNamespace = "default"
)

func newKubeDNS() *KubeDNS {
	kd := &KubeDNS{
		domain:              testDomain,
		endpointsStore:      cache.NewStore(cache.MetaNamespaceKeyFunc),
		servicesStore:       cache.NewStore(cache.MetaNamespaceKeyFunc),
		cache:               NewTreeCache(),
		reverseRecordMap:    make(map[string]*skymsg.Service),
		clusterIPServiceMap: make(map[string]*kapi.Service),
		cacheLock:           sync.RWMutex{},
		domainPath:          reverseArray(strings.Split(strings.TrimRight(testDomain, "."), ".")),
		nodesStore:          cache.NewStore(cache.MetaNamespaceKeyFunc),
	}
	return kd
}

func TestNewKubeDNS(t *testing.T) {
	// Verify that it returns an error for invalid federation names.
	_, err := NewKubeDNS(nil, "domainName", map[string]string{"invalid.name.with.dot": "example.come"})
	if err == nil {
		t.Errorf("Expected an error due to invalid federation name")
	}
}

func TestPodDns(t *testing.T) {
	const (
		testPodIP      = "1.2.3.4"
		sanitizedPodIP = "1-2-3-4"
	)
	kd := newKubeDNS()

	records, err := kd.Records(sanitizedPodIP+".default.pod."+kd.domain, false)
	require.NoError(t, err)
	assert.Equal(t, 1, len(records))
	assert.Equal(t, testPodIP, records[0].Host)
}

func TestUnnamedSinglePortService(t *testing.T) {
	kd := newKubeDNS()
	s := newService(testNamespace, testService, "1.2.3.4", "", 80)
	// Add the service
	kd.newService(s)
	assertDNSForClusterIP(t, kd, s)
	assertReverseRecord(t, kd, s)
	// Delete the service
	kd.removeService(s)
	assertNoDNSForClusterIP(t, kd, s)
	assertNoReverseRecord(t, kd, s)
}

func TestNamedSinglePortService(t *testing.T) {
	const (
		portName1 = "http1"
		portName2 = "http2"
	)
	kd := newKubeDNS()
	s := newService(testNamespace, testService, "1.2.3.4", portName1, 80)
	// Add the service
	kd.newService(s)
	assertDNSForClusterIP(t, kd, s)
	assertSRVForNamedPort(t, kd, s, portName1)

	newService := *s
	// update the portName of the service
	newService.Spec.Ports[0].Name = portName2
	kd.updateService(s, &newService)
	assertDNSForClusterIP(t, kd, s)
	assertSRVForNamedPort(t, kd, s, portName2)
	assertNoSRVForNamedPort(t, kd, s, portName1)

	// Delete the service
	kd.removeService(s)
	assertNoDNSForClusterIP(t, kd, s)
	assertNoSRVForNamedPort(t, kd, s, portName1)
	assertNoSRVForNamedPort(t, kd, s, portName2)
}

func assertARecordsMatchIPs(t *testing.T, records []dns.RR, ips ...string) {
	expectedEndpoints := sets.NewString(ips...)
	gotEndpoints := sets.NewString()
	for _, r := range records {
		if a, ok := r.(*dns.A); !ok {
			t.Errorf("Expected A record, got %#v", a)
		} else {
			gotEndpoints.Insert(a.A.String())
		}
	}
	if !gotEndpoints.Equal(expectedEndpoints) {
		t.Errorf("Expected %v got %v", expectedEndpoints, gotEndpoints)
	}
}

func assertSRVRecordsMatchTarget(t *testing.T, records []dns.RR, targets ...string) {
	expectedTargets := sets.NewString(targets...)
	gotTargets := sets.NewString()
	for _, r := range records {
		if srv, ok := r.(*dns.SRV); !ok {
			t.Errorf("Expected SRV record, got %+v", srv)
		} else {
			gotTargets.Insert(srv.Target)
		}
	}
	if !gotTargets.Equal(expectedTargets) {
		t.Errorf("Expected %v got %v", expectedTargets, gotTargets)
	}
}

func assertSRVRecordsMatchPort(t *testing.T, records []dns.RR, port ...int) {
	expectedPorts := sets.NewInt(port...)
	gotPorts := sets.NewInt()
	for _, r := range records {
		if srv, ok := r.(*dns.SRV); !ok {
			t.Errorf("Expected SRV record, got %+v", srv)
		} else {
			gotPorts.Insert(int(srv.Port))
			t.Logf("got %+v", srv)
		}
	}
	if !gotPorts.Equal(expectedPorts) {
		t.Errorf("Expected %v got %v", expectedPorts, gotPorts)
	}
}

func TestSkySimpleSRVLookup(t *testing.T) {
	kd := newKubeDNS()
	skydnsConfig := &skyServer.Config{Domain: testDomain, DnsAddr: "0.0.0.0:53"}
	skyServer.SetDefaults(skydnsConfig)
	s := skyServer.New(kd, skydnsConfig)

	service := newHeadlessService()
	endpointIPs := []string{"10.0.0.1", "10.0.0.2"}
	endpoints := newEndpoints(service, newSubsetWithOnePort("", 80, endpointIPs...))
	assert.NoError(t, kd.endpointsStore.Add(endpoints))
	kd.newService(service)

	name := strings.Join([]string{testService, testNamespace, "svc", testDomain}, ".")
	question := dns.Question{Name: name, Qtype: dns.TypeSRV, Qclass: dns.ClassINET}

	rec, extra, err := s.SRVRecords(question, name, 512, false)
	if err != nil {
		t.Fatalf("Failed srv record lookup on service with fqdn %v", name)
	}
	assertARecordsMatchIPs(t, extra, endpointIPs...)
	targets := []string{}
	for _, eip := range endpointIPs {
		// A portal service is always created with a port of '0'
		targets = append(targets, fmt.Sprintf("%v.%v", fmt.Sprintf("%x", hashServiceRecord(newServiceRecord(eip, 0))), name))
	}
	assertSRVRecordsMatchTarget(t, rec, targets...)
}

func TestSkyPodHostnameSRVLookup(t *testing.T) {
	kd := newKubeDNS()
	skydnsConfig := &skyServer.Config{Domain: testDomain, DnsAddr: "0.0.0.0:53"}
	skyServer.SetDefaults(skydnsConfig)
	s := skyServer.New(kd, skydnsConfig)

	service := newHeadlessService()
	endpointIPs := []string{"10.0.0.1", "10.0.0.2"}
	endpoints := newEndpoints(service, newSubsetWithOnePort("", 80, endpointIPs...))

	// The format of thes annotations is:
	// endpoints.beta.kubernetes.io/hostnames-map: '{"ep-ip":{"HostName":"pod request hostname"}}'
	epRecords := map[string]endpointsapi.HostRecord{}
	for i, ep := range endpointIPs {
		epRecords[ep] = endpointsapi.HostRecord{HostName: fmt.Sprintf("ep-%d", i)}
	}
	b, err := json.Marshal(epRecords)
	if err != nil {
		t.Fatalf("%v", err)
	}
	endpoints.Annotations = map[string]string{
		endpointsapi.PodHostnamesAnnotation: string(b),
	}
	assert.NoError(t, kd.endpointsStore.Add(endpoints))
	kd.newService(service)
	name := strings.Join([]string{testService, testNamespace, "svc", testDomain}, ".")
	question := dns.Question{Name: name, Qtype: dns.TypeSRV, Qclass: dns.ClassINET}

	rec, _, err := s.SRVRecords(question, name, 512, false)
	if err != nil {
		t.Fatalf("Failed srv record lookup on service with fqdn %v", name)
	}
	targets := []string{}
	for i := range endpointIPs {
		targets = append(targets, fmt.Sprintf("%v.%v", fmt.Sprintf("ep-%d", i), name))
	}
	assertSRVRecordsMatchTarget(t, rec, targets...)
}

func TestSkyNamedPortSRVLookup(t *testing.T) {
	kd := newKubeDNS()
	skydnsConfig := &skyServer.Config{Domain: testDomain, DnsAddr: "0.0.0.0:53"}
	skyServer.SetDefaults(skydnsConfig)
	s := skyServer.New(kd, skydnsConfig)

	service := newHeadlessService()
	eip := "10.0.0.1"
	endpoints := newEndpoints(service, newSubsetWithOnePort("http", 8081, eip))
	assert.NoError(t, kd.endpointsStore.Add(endpoints))
	kd.newService(service)

	name := strings.Join([]string{"_http", "_tcp", testService, testNamespace, "svc", testDomain}, ".")
	question := dns.Question{Name: name, Qtype: dns.TypeSRV, Qclass: dns.ClassINET}
	rec, extra, err := s.SRVRecords(question, name, 512, false)
	if err != nil {
		t.Fatalf("Failed srv record lookup on service with fqdn %v", name)
	}

	svcDomain := strings.Join([]string{testService, testNamespace, "svc", testDomain}, ".")
	assertARecordsMatchIPs(t, extra, eip)
	assertSRVRecordsMatchTarget(t, rec, fmt.Sprintf("%v.%v", fmt.Sprintf("%x", hashServiceRecord(newServiceRecord(eip, 0))), svcDomain))
	assertSRVRecordsMatchPort(t, rec, 8081)
}

func TestSimpleHeadlessService(t *testing.T) {
	kd := newKubeDNS()
	s := newHeadlessService()
	assert.NoError(t, kd.servicesStore.Add(s))
	endpoints := newEndpoints(s, newSubsetWithOnePort("", 80, "10.0.0.1", "10.0.0.2"), newSubsetWithOnePort("", 8080, "10.0.0.3", "10.0.0.4"))

	assert.NoError(t, kd.endpointsStore.Add(endpoints))
	kd.newService(s)
	assertDNSForHeadlessService(t, kd, endpoints)
	kd.removeService(s)
	assertNoDNSForHeadlessService(t, kd, s)
}

func TestHeadlessServiceWithNamedPorts(t *testing.T) {
	kd := newKubeDNS()
	service := newHeadlessService()
	// add service to store
	assert.NoError(t, kd.servicesStore.Add(service))
	endpoints := newEndpoints(service, newSubsetWithTwoPorts("http1", 80, "http2", 81, "10.0.0.1", "10.0.0.2"),
		newSubsetWithOnePort("https", 443, "10.0.0.3", "10.0.0.4"))

	// We expect 10 records. 6 SRV records. 4 POD records.
	// add endpoints
	assert.NoError(t, kd.endpointsStore.Add(endpoints))

	// add service
	kd.newService(service)
	assertDNSForHeadlessService(t, kd, endpoints)
	assertSRVForHeadlessService(t, kd, service, endpoints)

	// reduce endpoints
	endpoints.Subsets = endpoints.Subsets[:1]
	kd.handleEndpointAdd(endpoints)
	// We expect 6 records. 4 SRV records. 2 POD records.
	assertDNSForHeadlessService(t, kd, endpoints)
	assertSRVForHeadlessService(t, kd, service, endpoints)

	kd.removeService(service)
	assertNoDNSForHeadlessService(t, kd, service)
}

func TestHeadlessServiceEndpointsUpdate(t *testing.T) {
	kd := newKubeDNS()
	service := newHeadlessService()
	// add service to store
	assert.NoError(t, kd.servicesStore.Add(service))

	endpoints := newEndpoints(service, newSubsetWithOnePort("", 80, "10.0.0.1", "10.0.0.2"))
	// add endpoints to store
	assert.NoError(t, kd.endpointsStore.Add(endpoints))

	// add service
	kd.newService(service)
	assertDNSForHeadlessService(t, kd, endpoints)

	// increase endpoints
	endpoints.Subsets = append(endpoints.Subsets,
		newSubsetWithOnePort("", 8080, "10.0.0.3", "10.0.0.4"),
	)
	// expected DNSRecords = 4
	kd.handleEndpointAdd(endpoints)
	assertDNSForHeadlessService(t, kd, endpoints)

	// remove all endpoints
	endpoints.Subsets = []kapi.EndpointSubset{}
	kd.handleEndpointAdd(endpoints)
	assertNoDNSForHeadlessService(t, kd, service)

	// remove service
	kd.removeService(service)
	assertNoDNSForHeadlessService(t, kd, service)
}

func TestHeadlessServiceWithDelayedEndpointsAddition(t *testing.T) {
	kd := newKubeDNS()
	// create service
	service := newHeadlessService()

	// add service to store
	assert.NoError(t, kd.servicesStore.Add(service))

	// add service
	kd.newService(service)
	assertNoDNSForHeadlessService(t, kd, service)

	// create endpoints
	endpoints := newEndpoints(service, newSubsetWithOnePort("", 80, "10.0.0.1", "10.0.0.2"))

	// add endpoints to store
	assert.NoError(t, kd.endpointsStore.Add(endpoints))

	// add endpoints
	kd.handleEndpointAdd(endpoints)

	assertDNSForHeadlessService(t, kd, endpoints)

	// remove service
	kd.removeService(service)
	assertNoDNSForHeadlessService(t, kd, service)
}

// Verifies that a single record with host "a" is returned for query "q".
func verifyRecord(q, a string, t *testing.T, kd *KubeDNS) {
	records, err := kd.Records(q, false)
	require.NoError(t, err)
	assert.Equal(t, 1, len(records))
	assert.Equal(t, a, records[0].Host)
}

// Verifies that quering KubeDNS for a headless federation service returns the DNS hostname when a local service does not exist and returns the endpoint IP when a local service exists.
func TestFederationHeadlessService(t *testing.T) {
	kd := newKubeDNS()
	kd.federations = map[string]string{
		"myfederation": "example.com",
	}
	kd.kubeClient = fake.NewSimpleClientset(newNodes())

	// Verify that quering for federation service returns a federation domain name.
	verifyRecord("testservice.default.myfederation.svc.cluster.local.",
		"testservice.default.myfederation.svc.testcontinent-testreg-testzone.testcontinent-testreg.example.com.",
		t, kd)

	// Add a local service without any endpoint.
	s := newHeadlessService()
	assert.NoError(t, kd.servicesStore.Add(s))
	kd.newService(s)

	// Verify that quering for federation service still returns the federation domain name.
	verifyRecord(getFederationServiceFQDN(kd, s, "myfederation"),
		"testservice.default.myfederation.svc.testcontinent-testreg-testzone.testcontinent-testreg.example.com.",
		t, kd)

	// Now add an endpoint.
	endpoints := newEndpoints(s, newSubsetWithOnePort("", 80, "10.0.0.1"))
	assert.NoError(t, kd.endpointsStore.Add(endpoints))
	kd.updateService(s, s)

	// Verify that quering for federation service returns the local service domain name this time.
	verifyRecord(getFederationServiceFQDN(kd, s, "myfederation"), "testservice.default.svc.cluster.local.", t, kd)

	// Delete the endpoint.
	endpoints.Subsets = []kapi.EndpointSubset{}
	kd.handleEndpointAdd(endpoints)
	kd.updateService(s, s)

	// Verify that quering for federation service returns the federation domain name again.
	verifyRecord(getFederationServiceFQDN(kd, s, "myfederation"),
		"testservice.default.myfederation.svc.testcontinent-testreg-testzone.testcontinent-testreg.example.com.",
		t, kd)
}

// Verifies that quering KubeDNS for a federation service returns the DNS hostname if no endpoint exists and returns the local cluster IP if endpoints exist.
func TestFederationService(t *testing.T) {
	kd := newKubeDNS()
	kd.federations = map[string]string{
		"myfederation": "example.com",
	}
	kd.kubeClient = fake.NewSimpleClientset(newNodes())

	// Verify that quering for federation service returns the federation domain name.
	verifyRecord("testservice.default.myfederation.svc.cluster.local.",
		"testservice.default.myfederation.svc.testcontinent-testreg-testzone.testcontinent-testreg.example.com.",
		t, kd)

	// Add a local service without any endpoint.
	s := newService(testNamespace, testService, "1.2.3.4", "", 80)
	assert.NoError(t, kd.servicesStore.Add(s))
	kd.newService(s)

	// Verify that quering for federation service still returns the federation domain name.
	verifyRecord(getFederationServiceFQDN(kd, s, "myfederation"),
		"testservice.default.myfederation.svc.testcontinent-testreg-testzone.testcontinent-testreg.example.com.",
		t, kd)

	// Now add an endpoint.
	endpoints := newEndpoints(s, newSubsetWithOnePort("", 80, "10.0.0.1"))
	assert.NoError(t, kd.endpointsStore.Add(endpoints))
	kd.updateService(s, s)

	// Verify that quering for federation service returns the local service domain name this time.
	verifyRecord(getFederationServiceFQDN(kd, s, "myfederation"), "testservice.default.svc.cluster.local.", t, kd)

	// Remove the endpoint.
	endpoints.Subsets = []kapi.EndpointSubset{}
	kd.handleEndpointAdd(endpoints)
	kd.updateService(s, s)

	// Verify that quering for federation service returns the federation domain name again.
	verifyRecord(getFederationServiceFQDN(kd, s, "myfederation"),
		"testservice.default.myfederation.svc.testcontinent-testreg-testzone.testcontinent-testreg.example.com.",
		t, kd)
}

func TestFederationQueryWithoutCache(t *testing.T) {
	kd := newKubeDNS()
	kd.federations = map[string]string{
		"myfederation":     "example.com",
		"secondfederation": "second.example.com",
	}
	kd.kubeClient = fake.NewSimpleClientset(newNodes())

	testValidFederationQueries(t, kd)
	testInvalidFederationQueries(t, kd)
}

func TestFederationQueryWithCache(t *testing.T) {
	kd := newKubeDNS()
	kd.federations = map[string]string{
		"myfederation":     "example.com",
		"secondfederation": "second.example.com",
	}

	// Add a node to the cache.
	nodeList := newNodes()
	if err := kd.nodesStore.Add(&nodeList.Items[1]); err != nil {
		t.Errorf("failed to add the node to the cache: %v", err)
	}

	testValidFederationQueries(t, kd)
	testInvalidFederationQueries(t, kd)
}

func testValidFederationQueries(t *testing.T, kd *KubeDNS) {
	queries := []struct {
		q string
		a string
	}{
		// Federation suffix is just a domain.
		{
			q: "mysvc.myns.myfederation.svc.cluster.local.",
			a: "mysvc.myns.myfederation.svc.testcontinent-testreg-testzone.testcontinent-testreg.example.com.",
		},
		// Federation suffix is a subdomain.
		{
			q: "secsvc.default.secondfederation.svc.cluster.local.",
			a: "secsvc.default.secondfederation.svc.testcontinent-testreg-testzone.testcontinent-testreg.second.example.com.",
		},
	}

	for _, query := range queries {
		verifyRecord(query.q, query.a, t, kd)
	}
}

func testInvalidFederationQueries(t *testing.T, kd *KubeDNS) {
	noAnswerQueries := []string{
		"mysvc.myns.svc.cluster.local.",
		"mysvc.default.nofederation.svc.cluster.local.",
	}
	for _, q := range noAnswerQueries {
		records, err := kd.Records(q, false)
		if err == nil {
			t.Errorf("expected not found error, got nil")
		}
		if etcdErr, ok := err.(etcd.Error); !ok || etcdErr.Code != etcd.ErrorCodeKeyNotFound {
			t.Errorf("expected not found error, got %v", etcdErr)
		}
		assert.Equal(t, 0, len(records))
	}
}

func newNodes() *kapi.NodeList {
	return &kapi.NodeList{
		Items: []kapi.Node{
			// Node without annotation.
			{
				ObjectMeta: kapi.ObjectMeta{
					Name: "testnode-0",
				},
			},
			{
				ObjectMeta: kapi.ObjectMeta{
					Name: "testnode-1",
					Labels: map[string]string{
						// Note: The zone name here is an arbitrary string and doesn't exactly follow the
						// format used by the cloud providers to name their zones. But that shouldn't matter
						// for these tests here.
						unversioned.LabelZoneFailureDomain: "testcontinent-testreg-testzone",
						unversioned.LabelZoneRegion:        "testcontinent-testreg",
					},
				},
			},
		},
	}
}

func newService(namespace, serviceName, clusterIP, portName string, portNumber int32) *kapi.Service {
	service := kapi.Service{
		ObjectMeta: kapi.ObjectMeta{
			Name:      serviceName,
			Namespace: namespace,
		},
		Spec: kapi.ServiceSpec{
			ClusterIP: clusterIP,
			Ports: []kapi.ServicePort{
				{Port: portNumber, Name: portName, Protocol: "TCP"},
			},
		},
	}
	return &service
}

func newHeadlessService() *kapi.Service {
	service := kapi.Service{
		ObjectMeta: kapi.ObjectMeta{
			Name:      testService,
			Namespace: testNamespace,
		},
		Spec: kapi.ServiceSpec{
			ClusterIP: "None",
			Ports: []kapi.ServicePort{
				{Port: 0},
			},
		},
	}
	return &service
}

func newEndpoints(service *kapi.Service, subsets ...kapi.EndpointSubset) *kapi.Endpoints {
	endpoints := kapi.Endpoints{
		ObjectMeta: service.ObjectMeta,
		Subsets:    []kapi.EndpointSubset{},
	}

	endpoints.Subsets = append(endpoints.Subsets, subsets...)
	return &endpoints
}

func newSubsetWithOnePort(portName string, port int32, ips ...string) kapi.EndpointSubset {
	subset := newSubset()
	subset.Ports = append(subset.Ports, kapi.EndpointPort{Port: port, Name: portName, Protocol: "TCP"})
	for _, ip := range ips {
		subset.Addresses = append(subset.Addresses, kapi.EndpointAddress{IP: ip})
	}
	return subset
}

func newSubsetWithTwoPorts(portName1 string, portNumber1 int32, portName2 string, portNumber2 int32, ips ...string) kapi.EndpointSubset {
	subset := newSubsetWithOnePort(portName1, portNumber1, ips...)
	subset.Ports = append(subset.Ports, kapi.EndpointPort{Port: portNumber2, Name: portName2, Protocol: "TCP"})
	return subset
}

func newSubset() kapi.EndpointSubset {
	subset := kapi.EndpointSubset{
		Addresses: []kapi.EndpointAddress{},
		Ports:     []kapi.EndpointPort{},
	}
	return subset
}

func assertSRVForHeadlessService(t *testing.T, kd *KubeDNS, s *kapi.Service, e *kapi.Endpoints) {
	for _, subset := range e.Subsets {
		for _, port := range subset.Ports {
			records, err := kd.Records(getSRVFQDN(kd, s, port.Name), false)
			require.NoError(t, err)
			assertRecordPortsMatchPort(t, port.Port, records)
			assertCNameRecordsMatchEndpointIPs(t, kd, subset.Addresses, records)
		}
	}
}

func assertDNSForHeadlessService(t *testing.T, kd *KubeDNS, e *kapi.Endpoints) {
	records, err := kd.Records(getEndpointsFQDN(kd, e), false)
	require.NoError(t, err)
	endpoints := map[string]bool{}
	for _, subset := range e.Subsets {
		for _, endpointAddress := range subset.Addresses {
			endpoints[endpointAddress.IP] = true
		}
	}
	assert.Equal(t, len(endpoints), len(records))
	for _, record := range records {
		_, found := endpoints[record.Host]
		assert.True(t, found)
	}
}

func assertRecordPortsMatchPort(t *testing.T, port int32, records []skymsg.Service) {
	for _, record := range records {
		assert.Equal(t, port, int32(record.Port))
	}
}

func assertCNameRecordsMatchEndpointIPs(t *testing.T, kd *KubeDNS, e []kapi.EndpointAddress, records []skymsg.Service) {
	endpoints := map[string]bool{}
	for _, endpointAddress := range e {
		endpoints[endpointAddress.IP] = true
	}
	assert.Equal(t, len(e), len(records), "unexpected record count")
	for _, record := range records {
		_, found := endpoints[getIPForCName(t, kd, record.Host)]
		assert.True(t, found, "Did not find endpoint with address:%s", record.Host)
	}
}

func getIPForCName(t *testing.T, kd *KubeDNS, cname string) string {
	records, err := kd.Records(cname, false)
	require.NoError(t, err)
	assert.Equal(t, 1, len(records), "Could not get IP for CNAME record for %s", cname)
	assert.NotNil(t, net.ParseIP(records[0].Host), "Invalid IP address %q", records[0].Host)
	return records[0].Host
}

func assertNoDNSForHeadlessService(t *testing.T, kd *KubeDNS, s *kapi.Service) {
	records, err := kd.Records(getServiceFQDN(kd, s), false)
	require.Error(t, err)
	assert.Equal(t, 0, len(records))
}

func assertSRVForNamedPort(t *testing.T, kd *KubeDNS, s *kapi.Service, portName string) {
	records, err := kd.Records(getSRVFQDN(kd, s, portName), false)
	require.NoError(t, err)
	assert.Equal(t, 1, len(records))
	assert.Equal(t, getServiceFQDN(kd, s), records[0].Host)
}

func assertNoSRVForNamedPort(t *testing.T, kd *KubeDNS, s *kapi.Service, portName string) {
	records, err := kd.Records(getSRVFQDN(kd, s, portName), false)
	require.Error(t, err)
	assert.Equal(t, 0, len(records))
}

func assertNoDNSForClusterIP(t *testing.T, kd *KubeDNS, s *kapi.Service) {
	serviceFQDN := getServiceFQDN(kd, s)
	queries := getEquivalentQueries(serviceFQDN, s.Namespace)
	for _, query := range queries {
		records, err := kd.Records(query, false)
		require.Error(t, err)
		assert.Equal(t, 0, len(records))
	}
}

func assertDNSForClusterIP(t *testing.T, kd *KubeDNS, s *kapi.Service) {
	serviceFQDN := getServiceFQDN(kd, s)
	queries := getEquivalentQueries(serviceFQDN, s.Namespace)
	for _, query := range queries {
		records, err := kd.Records(query, false)
		require.NoError(t, err)
		assert.Equal(t, 1, len(records))
		assert.Equal(t, s.Spec.ClusterIP, records[0].Host)
	}
}

func assertReverseRecord(t *testing.T, kd *KubeDNS, s *kapi.Service) {
	segments := reverseArray(strings.Split(s.Spec.ClusterIP, "."))
	reverseLookup := fmt.Sprintf("%s%s", strings.Join(segments, "."), arpaSuffix)
	reverseRecord, err := kd.ReverseRecord(reverseLookup)
	require.NoError(t, err)
	assert.Equal(t, kd.getServiceFQDN(s), reverseRecord.Host)
}

func assertNoReverseRecord(t *testing.T, kd *KubeDNS, s *kapi.Service) {
	segments := reverseArray(strings.Split(s.Spec.ClusterIP, "."))
	reverseLookup := fmt.Sprintf("%s%s", strings.Join(segments, "."), arpaSuffix)
	reverseRecord, err := kd.ReverseRecord(reverseLookup)
	require.Error(t, err)
	require.Nil(t, reverseRecord)
}

func getEquivalentQueries(serviceFQDN, namespace string) []string {
	return []string{
		serviceFQDN,
		strings.Replace(serviceFQDN, ".svc.", ".*.", 1),
		strings.Replace(serviceFQDN, namespace, "*", 1),
		strings.Replace(strings.Replace(serviceFQDN, namespace, "*", 1), ".svc.", ".*.", 1),
		"*." + serviceFQDN,
	}
}

func getFederationServiceFQDN(kd *KubeDNS, s *kapi.Service, federationName string) string {
	return fmt.Sprintf("%s.%s.%s.svc.%s", s.Name, s.Namespace, federationName, kd.domain)
}

func getServiceFQDN(kd *KubeDNS, s *kapi.Service) string {
	return fmt.Sprintf("%s.%s.svc.%s", s.Name, s.Namespace, kd.domain)
}

func getEndpointsFQDN(kd *KubeDNS, e *kapi.Endpoints) string {
	return fmt.Sprintf("%s.%s.svc.%s", e.Name, e.Namespace, kd.domain)
}

func getSRVFQDN(kd *KubeDNS, s *kapi.Service, portName string) string {
	return fmt.Sprintf("_%s._tcp.%s.%s.svc.%s", portName, s.Name, s.Namespace, kd.domain)
}
