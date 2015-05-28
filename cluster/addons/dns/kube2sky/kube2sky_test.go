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
	"net/http"
	"path"
	"strings"
	"testing"
	"time"

	kapi "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/coreos/go-etcd/etcd"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
	count := 0
	for path := range ec.writes {
		if strings.HasPrefix(path, key) {
			count++
		}
	}
	if count == 0 {
		return &etcd.RawResponse{StatusCode: http.StatusNotFound}, nil
	}
	return &etcd.RawResponse{StatusCode: http.StatusOK}, nil
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

func getEtcdOldStylePath(name, namespace string) string {
	return path.Join(basePath, namespace, name)
}

func getEtcdNewStylePath(name, namespace string) string {
	return path.Join(basePath, serviceSubDomain, namespace, name)
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
	oldStyleKey := getEtcdOldStylePath(serviceName, namespace)
	val, exists := ec.writes[oldStyleKey]
	require.True(t, exists)
	actualHostPort, err := getHostPortFromString(val)
	require.NoError(t, err)
	assert.Equal(t, actualHostPort, expectedHostPort)

	newStyleKey := getEtcdNewStylePath(serviceName, namespace)
	val, exists = ec.writes[newStyleKey]
	require.True(t, exists)
	actualHostPort, err = getHostPortFromString(val)
	require.NoError(t, err)
	assert.Equal(t, actualHostPort, expectedHostPort)
}

func TestHeadlessService(t *testing.T) {
	const (
		testService   = "testService"
		testNamespace = "default"
	)
	ec := &fakeEtcdClient{make(map[string]string)}
	k2s := newKube2Sky(ec)
	service := kapi.Service{
		ObjectMeta: kapi.ObjectMeta{
			Name:      testService,
			Namespace: testNamespace,
		},
		Spec: kapi.ServiceSpec{
			ClusterIP: "None",
			Ports: []kapi.ServicePort{
				{Port: 80},
			},
		},
	}
	assert.NoError(t, k2s.servicesStore.Add(&service))
	endpoints := kapi.Endpoints{
		ObjectMeta: service.ObjectMeta,
		Subsets: []kapi.EndpointSubset{
			{
				Addresses: []kapi.EndpointAddress{
					{IP: "10.0.0.1"},
					{IP: "10.0.0.2"},
				},
				Ports: []kapi.EndpointPort{
					{Port: 80},
				},
			},
			{
				Addresses: []kapi.EndpointAddress{
					{IP: "10.0.0.3"},
					{IP: "10.0.0.4"},
				},
				Ports: []kapi.EndpointPort{
					{Port: 8080},
				},
			},
		},
	}
	// We expect 4 records with "svc" subdomain and 4 records without
	// "svc" subdomain.
	expectedDNSRecords := 8
	assert.NoError(t, k2s.endpointsStore.Add(&endpoints))
	k2s.newService(&service)
	assert.Equal(t, expectedDNSRecords, len(ec.writes))
	k2s.removeService(&service)
	assert.Empty(t, ec.writes)
}

func TestHeadlessServiceEndpointsUpdate(t *testing.T) {
	const (
		testService   = "testService"
		testNamespace = "default"
	)
	ec := &fakeEtcdClient{make(map[string]string)}
	k2s := newKube2Sky(ec)
	service := kapi.Service{
		ObjectMeta: kapi.ObjectMeta{
			Name:      testService,
			Namespace: testNamespace,
		},
		Spec: kapi.ServiceSpec{
			ClusterIP: "None",
			Ports: []kapi.ServicePort{
				{Port: 80},
			},
		},
	}
	assert.NoError(t, k2s.servicesStore.Add(&service))
	endpoints := kapi.Endpoints{
		ObjectMeta: service.ObjectMeta,
		Subsets: []kapi.EndpointSubset{
			{
				Addresses: []kapi.EndpointAddress{
					{IP: "10.0.0.1"},
					{IP: "10.0.0.2"},
				},
				Ports: []kapi.EndpointPort{
					{Port: 80},
				},
			},
		},
	}
	expectedDNSRecords := 4
	assert.NoError(t, k2s.endpointsStore.Add(&endpoints))
	k2s.newService(&service)
	assert.Equal(t, expectedDNSRecords, len(ec.writes))
	endpoints.Subsets = append(endpoints.Subsets,
		kapi.EndpointSubset{
			Addresses: []kapi.EndpointAddress{
				{IP: "10.0.0.3"},
				{IP: "10.0.0.4"},
			},
			Ports: []kapi.EndpointPort{
				{Port: 8080},
			},
		},
	)
	expectedDNSRecords = 8
	k2s.handleEndpointAdd(&endpoints)

	assert.Equal(t, expectedDNSRecords, len(ec.writes))
	k2s.removeService(&service)
	assert.Empty(t, ec.writes)
}

func TestHeadlessServiceWithDelayedEndpointsAddition(t *testing.T) {
	const (
		testService   = "testService"
		testNamespace = "default"
	)
	ec := &fakeEtcdClient{make(map[string]string)}
	k2s := newKube2Sky(ec)
	service := kapi.Service{
		ObjectMeta: kapi.ObjectMeta{
			Name:      testService,
			Namespace: testNamespace,
		},
		Spec: kapi.ServiceSpec{
			ClusterIP: "None",
			Ports: []kapi.ServicePort{
				{Port: 80},
			},
		},
	}
	assert.NoError(t, k2s.servicesStore.Add(&service))
	// Headless service DNS records should not be created since
	// corresponding endpoints object doesn't exist.
	k2s.newService(&service)
	assert.Empty(t, ec.writes)

	// Add an endpoints object for the service.
	endpoints := kapi.Endpoints{
		ObjectMeta: service.ObjectMeta,
		Subsets: []kapi.EndpointSubset{
			{
				Addresses: []kapi.EndpointAddress{
					{IP: "10.0.0.1"},
					{IP: "10.0.0.2"},
				},
				Ports: []kapi.EndpointPort{
					{Port: 80},
				},
			},
			{
				Addresses: []kapi.EndpointAddress{
					{IP: "10.0.0.3"},
					{IP: "10.0.0.4"},
				},
				Ports: []kapi.EndpointPort{
					{Port: 8080},
				},
			},
		},
	}
	// We expect 4 records with "svc" subdomain and 4 records without
	// "svc" subdomain.
	expectedDNSRecords := 8
	k2s.handleEndpointAdd(&endpoints)
	assert.Equal(t, expectedDNSRecords, len(ec.writes))
}

// TODO: Test service updates for headless services.
// TODO: Test headless service addition with delayed endpoints addition

func TestAddSinglePortService(t *testing.T) {
	const (
		testService   = "testService"
		testNamespace = "default"
	)
	ec := &fakeEtcdClient{make(map[string]string)}
	k2s := newKube2Sky(ec)
	service := kapi.Service{
		ObjectMeta: kapi.ObjectMeta{
			Name:      testService,
			Namespace: testNamespace,
		},
		Spec: kapi.ServiceSpec{
			Ports: []kapi.ServicePort{
				{
					Port: 80,
				},
			},
			ClusterIP: "1.2.3.4",
		},
	}
	k2s.newService(&service)
	expectedValue := getHostPort(&service)
	assertDnsServiceEntryInEtcd(t, ec, testService, testNamespace, expectedValue)
}

func TestUpdateSinglePortService(t *testing.T) {
	const (
		testService   = "testService"
		testNamespace = "default"
	)
	ec := &fakeEtcdClient{make(map[string]string)}
	k2s := newKube2Sky(ec)
	service := kapi.Service{
		ObjectMeta: kapi.ObjectMeta{
			Name:      testService,
			Namespace: testNamespace,
		},
		Spec: kapi.ServiceSpec{
			Ports: []kapi.ServicePort{
				{
					Port: 80,
				},
			},
			ClusterIP: "1.2.3.4",
		},
	}
	k2s.newService(&service)
	assert.Len(t, ec.writes, 2)
	service.Spec.ClusterIP = "0.0.0.0"
	k2s.newService(&service)
	expectedValue := getHostPort(&service)
	assertDnsServiceEntryInEtcd(t, ec, testService, testNamespace, expectedValue)
}

func TestDeleteSinglePortService(t *testing.T) {
	const (
		testService   = "testService"
		testNamespace = "default"
	)
	ec := &fakeEtcdClient{make(map[string]string)}
	k2s := newKube2Sky(ec)
	service := kapi.Service{
		ObjectMeta: kapi.ObjectMeta{
			Name:      testService,
			Namespace: testNamespace,
		},
		Spec: kapi.ServiceSpec{
			Ports: []kapi.ServicePort{
				{
					Port: 80,
				},
			},
			ClusterIP: "1.2.3.4",
		},
	}
	// Add the service
	k2s.newService(&service)
	// two entries should get created, one with the svc subdomain (new-style)
	// , and one without the svc subdomain (old-style)
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
