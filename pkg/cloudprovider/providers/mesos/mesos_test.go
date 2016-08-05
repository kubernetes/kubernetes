/*
Copyright 2015 The Kubernetes Authors.

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

package mesos

import (
	"bytes"
	"net"
	"reflect"
	"testing"
	"time"

	log "github.com/golang/glog"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

func TestIPAddress(t *testing.T) {
	expected4 := net.IPv4(127, 0, 0, 1)
	ip, err := ipAddress("127.0.0.1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(ip, expected4) {
		t.Fatalf("expected %#v instead of %#v", expected4, ip)
	}

	expected6 := net.ParseIP("::1")
	if expected6 == nil {
		t.Fatalf("failed to parse ipv6 ::1")
	}
	ip, err = ipAddress("::1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(ip, expected6) {
		t.Fatalf("expected %#v instead of %#v", expected6, ip)
	}

	ip, err = ipAddress("localhost")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(ip, expected4) && !reflect.DeepEqual(ip, expected6) {
		t.Fatalf("expected %#v or %#v instead of %#v", expected4, expected6, ip)
	}

	_, err = ipAddress("")
	if err != noHostNameSpecified {
		t.Fatalf("expected error noHostNameSpecified but got none")
	}
}

// test mesos.newMesosCloud with no config
func Test_newMesosCloud_NoConfig(t *testing.T) {
	defer log.Flush()
	mesosCloud, err := newMesosCloud(nil)

	if err != nil {
		t.Fatalf("Creating a new Mesos cloud provider without config does not yield an error: %#v", err)
	}

	if mesosCloud.client.httpClient.Timeout != DefaultHttpClientTimeout {
		t.Fatalf("Creating a new Mesos cloud provider without config does not yield an error: %#v", err)
	}

	if mesosCloud.client.state.ttl != DefaultStateCacheTTL {
		t.Fatalf("Mesos client with default config has the expected state cache TTL value")
	}
}

// test mesos.newMesosCloud with custom config
func Test_newMesosCloud_WithConfig(t *testing.T) {
	defer log.Flush()

	configString := `
[mesos-cloud]
	http-client-timeout = 500ms
	state-cache-ttl = 1h`

	reader := bytes.NewBufferString(configString)

	mesosCloud, err := newMesosCloud(reader)

	if err != nil {
		t.Fatalf("Creating a new Mesos cloud provider with a custom config does not yield an error: %#v", err)
	}

	if mesosCloud.client.httpClient.Timeout != time.Duration(500)*time.Millisecond {
		t.Fatalf("Mesos client with a custom config has the expected HTTP client timeout value")
	}

	if mesosCloud.client.state.ttl != time.Duration(1)*time.Hour {
		t.Fatalf("Mesos client with a custom config has the expected state cache TTL value")
	}
}

// tests for capability reporting functions

// test mesos.Instances
func Test_Instances(t *testing.T) {
	defer log.Flush()
	mesosCloud, _ := newMesosCloud(nil)

	instances, supports_instances := mesosCloud.Instances()

	if !supports_instances || instances == nil {
		t.Fatalf("MesosCloud provides an implementation of Instances")
	}
}

// test mesos.LoadBalancer
func Test_TcpLoadBalancer(t *testing.T) {
	defer log.Flush()
	mesosCloud, _ := newMesosCloud(nil)

	lb, supports_lb := mesosCloud.LoadBalancer()

	if supports_lb || lb != nil {
		t.Fatalf("MesosCloud does not provide an implementation of LoadBalancer")
	}
}

// test mesos.Zones
func Test_Zones(t *testing.T) {
	defer log.Flush()
	mesosCloud, _ := newMesosCloud(nil)

	zones, supports_zones := mesosCloud.Zones()

	if supports_zones || zones != nil {
		t.Fatalf("MesosCloud does not provide an implementation of Zones")
	}
}

// test mesos.Clusters
func Test_Clusters(t *testing.T) {
	defer log.Flush()
	mesosCloud, _ := newMesosCloud(nil)

	clusters, supports_clusters := mesosCloud.Clusters()

	if !supports_clusters || clusters == nil {
		t.Fatalf("MesosCloud does not provide an implementation of Clusters")
	}
}

// test mesos.MasterURI
func Test_MasterURI(t *testing.T) {
	defer log.Flush()
	mesosCloud, _ := newMesosCloud(nil)

	uri := mesosCloud.MasterURI()

	if uri != DefaultMesosMaster {
		t.Fatalf("MasterURI returns the expected master URI (expected \"localhost\", actual \"%s\"", uri)
	}
}

// test mesos.ListClusters
func Test_ListClusters(t *testing.T) {
	defer log.Flush()
	md := FakeMasterDetector{}
	httpServer, httpClient, httpTransport := makeHttpMocks()
	defer httpServer.Close()
	cacheTTL := 500 * time.Millisecond
	mesosClient, err := createMesosClient(md, httpClient, httpTransport, cacheTTL)
	mesosCloud := &MesosCloud{client: mesosClient, config: createDefaultConfig()}

	clusters, err := mesosCloud.ListClusters()

	if err != nil {
		t.Fatalf("ListClusters does not yield an error: %#v", err)
	}

	if len(clusters) != 1 {
		t.Fatalf("ListClusters should return a list of size 1: (actual: %#v)", clusters)
	}

	expectedClusterNames := []string{"mesos"}

	if !reflect.DeepEqual(clusters, expectedClusterNames) {
		t.Fatalf("ListClusters should return the expected list of names: (expected: %#v, actual: %#v)",
			expectedClusterNames,
			clusters)
	}
}

// test mesos.Master
func Test_Master(t *testing.T) {
	defer log.Flush()
	md := FakeMasterDetector{}
	httpServer, httpClient, httpTransport := makeHttpMocks()
	defer httpServer.Close()
	cacheTTL := 500 * time.Millisecond
	mesosClient, err := createMesosClient(md, httpClient, httpTransport, cacheTTL)
	mesosCloud := &MesosCloud{client: mesosClient, config: createDefaultConfig()}

	clusters, err := mesosCloud.ListClusters()
	clusterName := clusters[0]
	master, err := mesosCloud.Master(clusterName)

	if err != nil {
		t.Fatalf("Master does not yield an error: %#v", err)
	}

	expectedMaster := unpackIPv4(TEST_MASTER_IP)

	if master != expectedMaster {
		t.Fatalf("Master returns the unexpected value: (expected: %#v, actual: %#v", expectedMaster, master)
	}
}

// test mesos.List
func Test_List(t *testing.T) {
	defer log.Flush()
	md := FakeMasterDetector{}
	httpServer, httpClient, httpTransport := makeHttpMocks()
	defer httpServer.Close()
	cacheTTL := 500 * time.Millisecond
	mesosClient, err := createMesosClient(md, httpClient, httpTransport, cacheTTL)
	mesosCloud := &MesosCloud{client: mesosClient, config: createDefaultConfig()}

	clusters, err := mesosCloud.List(".*") // recognizes the language of all strings

	if err != nil {
		t.Fatalf("List does not yield an error: %#v", err)
	}

	if len(clusters) != 3 {
		t.Fatalf("List with a catch-all filter should return a list of size 3: (actual: %#v)", clusters)
	}

	clusters, err = mesosCloud.List("$^") // end-of-string followed by start-of-string: recognizes the empty language

	if err != nil {
		t.Fatalf("List does not yield an error: %#v", err)
	}

	if len(clusters) != 0 {
		t.Fatalf("List with a reject-all filter should return a list of size 0: (actual: %#v)", clusters)
	}
}

func Test_ExternalID(t *testing.T) {
	defer log.Flush()
	md := FakeMasterDetector{}
	httpServer, httpClient, httpTransport := makeHttpMocks()
	defer httpServer.Close()
	cacheTTL := 500 * time.Millisecond
	mesosClient, err := createMesosClient(md, httpClient, httpTransport, cacheTTL)
	mesosCloud := &MesosCloud{client: mesosClient, config: createDefaultConfig()}

	_, err = mesosCloud.ExternalID("unknown")
	if err != cloudprovider.InstanceNotFound {
		t.Fatalf("ExternalID did not return InstanceNotFound on an unknown instance")
	}

	slaveName := "mesos3.internal.company.com"
	id, err := mesosCloud.ExternalID(slaveName)
	if id != "" {
		t.Fatalf("ExternalID should not be able to resolve %q", slaveName)
	}
	if err == cloudprovider.InstanceNotFound {
		t.Fatalf("ExternalID should find %q", slaveName)
	}
}
