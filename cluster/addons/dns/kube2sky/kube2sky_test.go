/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"path"
	"strings"
	"testing"
	"time"

	"github.com/coreos/go-etcd/etcd"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	kapi "k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/unversioned/cache"
)

type fakeEtcdClient struct {
	// TODO: Convert this to real fs to better simulate etcd behavior.
	writes map[string]string
}

func (ec *fakeEtcdClient) Set(key, value string, ttl uint64) (*etcd.Response, error) {
	ec.writes[key] = value
	return nil, nil
}

func (ec *fakeEtcdClient) Delete(key string, recursive bool) (*etcd.Response, error) {
	for p := range ec.writes {
		if (recursive && strings.HasPrefix(p, key)) || (!recursive && p == key) {
			delete(ec.writes, p)
		}
	}
	return nil, nil
}

func (ec *fakeEtcdClient) RawGet(key string, sort, recursive bool) (*etcd.RawResponse, error) {
	values := ec.Get(key)
	if len(values) == 0 {
		return &etcd.RawResponse{StatusCode: http.StatusNotFound}, nil
	}
	return &etcd.RawResponse{StatusCode: http.StatusOK}, nil
}

func (ec *fakeEtcdClient) Get(key string) []string {
	values := make([]string, 0, 10)
	minSeparatorCount := 0
	key = strings.ToLower(key)
	for path := range ec.writes {
		if strings.HasPrefix(path, key) {
			separatorCount := strings.Count(path, "/")
			if minSeparatorCount == 0 || separatorCount < minSeparatorCount {
				minSeparatorCount = separatorCount
				values = values[:0]
				values = append(values, ec.writes[path])
			} else if separatorCount == minSeparatorCount {
				values = append(values, ec.writes[path])
			}
		}
	}
	return values
}

const (
	testDomain       = "cluster.local."
	basePath         = "/skydns/local/cluster"
	serviceSubDomain = "svc"
)

func newKube2Sky(ec etcdClient) *kube2sky {
	return &kube2sky{
		etcdClient:          ec,
		domain:              testDomain,
		etcdMutationTimeout: time.Second,
		endpointsStore:      cache.NewStore(cache.MetaNamespaceKeyFunc),
		servicesStore:       cache.NewStore(cache.MetaNamespaceKeyFunc),
	}
}

func getEtcdPathForA(name, namespace string) string {
	return path.Join(basePath, serviceSubDomain, namespace, name)
}

func getEtcdPathForSRV(portName, protocol, name, namespace string) string {
	return path.Join(basePath, serviceSubDomain, namespace, name, fmt.Sprintf("_%s", strings.ToLower(protocol)), fmt.Sprintf("_%s", strings.ToLower(portName)))
}

type hostPort struct {
	Host string `json:"host"`
	Port int    `json:"port"`
}

func getHostPort(service *kapi.Service) *hostPort {
	return &hostPort{
		Host: service.Spec.ClusterIP,
		Port: service.Spec.Ports[0].Port,
	}
}

func getHostPortFromString(data string) (*hostPort, error) {
	var res hostPort
	err := json.Unmarshal([]byte(data), &res)
	return &res, err
}

func assertDnsServiceEntryInEtcd(t *testing.T, ec *fakeEtcdClient, serviceName, namespace string, expectedHostPort *hostPort) {
	key := getEtcdPathForA(serviceName, namespace)
	values := ec.Get(key)
	//require.True(t, exists)
	require.True(t, len(values) > 0, "entry not found.")
	actualHostPort, err := getHostPortFromString(values[0])
	require.NoError(t, err)
	assert.Equal(t, expectedHostPort.Host, actualHostPort.Host)
}

func assertSRVEntryInEtcd(t *testing.T, ec *fakeEtcdClient, portName, protocol, serviceName, namespace string, expectedPortNumber, expectedEntriesCount int) {
	srvKey := getEtcdPathForSRV(portName, protocol, serviceName, namespace)
	values := ec.Get(srvKey)
	assert.Equal(t, expectedEntriesCount, len(values))
	for i := range values {
		actualHostPort, err := getHostPortFromString(values[i])
		require.NoError(t, err)
		assert.Equal(t, expectedPortNumber, actualHostPort.Port)
	}
}

func newHeadlessService(namespace, serviceName string) kapi.Service {
	service := kapi.Service{
		ObjectMeta: kapi.ObjectMeta{
			Name:      serviceName,
			Namespace: namespace,
		},
		Spec: kapi.ServiceSpec{
			ClusterIP: "None",
			Ports: []kapi.ServicePort{
				{Port: 0},
			},
		},
	}
	return service
}

func newService(namespace, serviceName, clusterIP, portName string, portNumber int) kapi.Service {
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
	return service
}

func newSubset() kapi.EndpointSubset {
	subset := kapi.EndpointSubset{
		Addresses: []kapi.EndpointAddress{},
		Ports:     []kapi.EndpointPort{},
	}
	return subset
}

func newSubsetWithOnePort(portName string, port int, ips ...string) kapi.EndpointSubset {
	subset := newSubset()
	subset.Ports = append(subset.Ports, kapi.EndpointPort{Port: port, Name: portName, Protocol: "TCP"})
	for _, ip := range ips {
		subset.Addresses = append(subset.Addresses, kapi.EndpointAddress{IP: ip})
	}
	return subset
}

func newSubsetWithTwoPorts(portName1 string, portNumber1 int, portName2 string, portNumber2 int, ips ...string) kapi.EndpointSubset {
	subset := newSubsetWithOnePort(portName1, portNumber1, ips...)
	subset.Ports = append(subset.Ports, kapi.EndpointPort{Port: portNumber2, Name: portName2, Protocol: "TCP"})
	return subset
}

func newEndpoints(service kapi.Service, subsets ...kapi.EndpointSubset) kapi.Endpoints {
	endpoints := kapi.Endpoints{
		ObjectMeta: service.ObjectMeta,
		Subsets:    []kapi.EndpointSubset{},
	}

	for _, subset := range subsets {
		endpoints.Subsets = append(endpoints.Subsets, subset)
	}
	return endpoints
}

func TestHeadlessService(t *testing.T) {
	const (
		testService   = "testservice"
		testNamespace = "default"
	)
	ec := &fakeEtcdClient{make(map[string]string)}
	k2s := newKube2Sky(ec)
	service := newHeadlessService(testNamespace, testService)
	assert.NoError(t, k2s.servicesStore.Add(&service))
	endpoints := newEndpoints(service, newSubsetWithOnePort("", 80, "10.0.0.1", "10.0.0.2"), newSubsetWithOnePort("", 8080, "10.0.0.3", "10.0.0.4"))

	// We expect 4 records.
	expectedDNSRecords := 4
	assert.NoError(t, k2s.endpointsStore.Add(&endpoints))
	k2s.newService(&service)
	assert.Equal(t, expectedDNSRecords, len(ec.writes))
	k2s.removeService(&service)
	assert.Empty(t, ec.writes)
}

func TestHeadlessServiceWithNamedPorts(t *testing.T) {
	const (
		testService   = "testservice"
		testNamespace = "default"
	)
	ec := &fakeEtcdClient{make(map[string]string)}
	k2s := newKube2Sky(ec)
	service := newHeadlessService(testNamespace, testService)
	assert.NoError(t, k2s.servicesStore.Add(&service))
	endpoints := newEndpoints(service, newSubsetWithTwoPorts("http1", 80, "http2", 81, "10.0.0.1", "10.0.0.2"), newSubsetWithOnePort("https", 443, "10.0.0.3", "10.0.0.4"))

	// We expect 10 records. 6 SRV records. 4 POD records.
	expectedDNSRecords := 10
	assert.NoError(t, k2s.endpointsStore.Add(&endpoints))
	k2s.newService(&service)
	assert.Equal(t, expectedDNSRecords, len(ec.writes))
	assertSRVEntryInEtcd(t, ec, "http1", "tcp", testService, testNamespace, 80, 2)
	assertSRVEntryInEtcd(t, ec, "http2", "tcp", testService, testNamespace, 81, 2)
	assertSRVEntryInEtcd(t, ec, "https", "tcp", testService, testNamespace, 443, 2)

	endpoints.Subsets = endpoints.Subsets[:1]
	k2s.handleEndpointAdd(&endpoints)
	// We expect 6 records. 4 SRV records. 2 POD records.
	expectedDNSRecords = 6
	assert.Equal(t, expectedDNSRecords, len(ec.writes))
	assertSRVEntryInEtcd(t, ec, "http1", "tcp", testService, testNamespace, 80, 2)
	assertSRVEntryInEtcd(t, ec, "http2", "tcp", testService, testNamespace, 81, 2)

	k2s.removeService(&service)
	assert.Empty(t, ec.writes)
}

func TestHeadlessServiceEndpointsUpdate(t *testing.T) {
	const (
		testService   = "testservice"
		testNamespace = "default"
	)
	ec := &fakeEtcdClient{make(map[string]string)}
	k2s := newKube2Sky(ec)
	service := newHeadlessService(testNamespace, testService)
	assert.NoError(t, k2s.servicesStore.Add(&service))
	endpoints := newEndpoints(service, newSubsetWithOnePort("", 80, "10.0.0.1", "10.0.0.2"))

	expectedDNSRecords := 2
	assert.NoError(t, k2s.endpointsStore.Add(&endpoints))
	k2s.newService(&service)
	assert.Equal(t, expectedDNSRecords, len(ec.writes))
	endpoints.Subsets = append(endpoints.Subsets,
		newSubsetWithOnePort("", 8080, "10.0.0.3", "10.0.0.4"),
	)
	expectedDNSRecords = 4
	k2s.handleEndpointAdd(&endpoints)

	assert.Equal(t, expectedDNSRecords, len(ec.writes))
	k2s.removeService(&service)
	assert.Empty(t, ec.writes)
}

func TestHeadlessServiceWithDelayedEndpointsAddition(t *testing.T) {
	const (
		testService   = "testservice"
		testNamespace = "default"
	)
	ec := &fakeEtcdClient{make(map[string]string)}
	k2s := newKube2Sky(ec)
	service := newHeadlessService(testNamespace, testService)
	assert.NoError(t, k2s.servicesStore.Add(&service))
	// Headless service DNS records should not be created since
	// corresponding endpoints object doesn't exist.
	k2s.newService(&service)
	assert.Empty(t, ec.writes)

	// Add an endpoints object for the service.
	endpoints := newEndpoints(service, newSubsetWithOnePort("", 80, "10.0.0.1", "10.0.0.2"), newSubsetWithOnePort("", 8080, "10.0.0.3", "10.0.0.4"))
	// We expect 4 records.
	expectedDNSRecords := 4
	k2s.handleEndpointAdd(&endpoints)
	assert.Equal(t, expectedDNSRecords, len(ec.writes))
}

// TODO: Test service updates for headless services.
// TODO: Test headless service addition with delayed endpoints addition

func TestAddSinglePortService(t *testing.T) {
	const (
		testService   = "testservice"
		testNamespace = "default"
	)
	ec := &fakeEtcdClient{make(map[string]string)}
	k2s := newKube2Sky(ec)
	service := newService(testNamespace, testService, "1.2.3.4", "", 0)
	k2s.newService(&service)
	expectedValue := getHostPort(&service)
	assertDnsServiceEntryInEtcd(t, ec, testService, testNamespace, expectedValue)
}

func TestUpdateSinglePortService(t *testing.T) {
	const (
		testService   = "testservice"
		testNamespace = "default"
	)
	ec := &fakeEtcdClient{make(map[string]string)}
	k2s := newKube2Sky(ec)
	service := newService(testNamespace, testService, "1.2.3.4", "", 0)
	k2s.newService(&service)
	assert.Len(t, ec.writes, 1)
	newService := service
	newService.Spec.ClusterIP = "0.0.0.0"
	k2s.updateService(&service, &newService)
	expectedValue := getHostPort(&newService)
	assertDnsServiceEntryInEtcd(t, ec, testService, testNamespace, expectedValue)
}

func TestDeleteSinglePortService(t *testing.T) {
	const (
		testService   = "testservice"
		testNamespace = "default"
	)
	ec := &fakeEtcdClient{make(map[string]string)}
	k2s := newKube2Sky(ec)
	service := newService(testNamespace, testService, "1.2.3.4", "", 80)
	// Add the service
	k2s.newService(&service)
	assert.Len(t, ec.writes, 1)
	// Delete the service
	k2s.removeService(&service)
	assert.Empty(t, ec.writes)
}

func TestServiceWithNamePort(t *testing.T) {
	const (
		testService   = "testservice"
		testNamespace = "default"
	)
	ec := &fakeEtcdClient{make(map[string]string)}
	k2s := newKube2Sky(ec)

	// create service
	service := newService(testNamespace, testService, "1.2.3.4", "http1", 80)
	k2s.newService(&service)
	expectedValue := getHostPort(&service)
	assertDnsServiceEntryInEtcd(t, ec, testService, testNamespace, expectedValue)
	assertSRVEntryInEtcd(t, ec, "http1", "tcp", testService, testNamespace, 80, 1)
	assert.Len(t, ec.writes, 2)

	// update service
	newService := service
	newService.Spec.Ports[0].Name = "http2"
	k2s.updateService(&service, &newService)
	expectedValue = getHostPort(&newService)
	assertDnsServiceEntryInEtcd(t, ec, testService, testNamespace, expectedValue)
	assertSRVEntryInEtcd(t, ec, "http2", "tcp", testService, testNamespace, 80, 1)
	assert.Len(t, ec.writes, 2)

	// Delete the service
	k2s.removeService(&service)
	assert.Empty(t, ec.writes)
}

func TestBuildDNSName(t *testing.T) {
	expectedDNSName := "name.ns.svc.cluster.local."
	assert.Equal(t, expectedDNSName, buildDNSNameString("local.", "cluster", "svc", "ns", "name"))
	newExpectedDNSName := "00.name.ns.svc.cluster.local."
	assert.Equal(t, newExpectedDNSName, buildDNSNameString(expectedDNSName, "00"))
}
