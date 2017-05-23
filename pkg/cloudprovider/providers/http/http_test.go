/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package http_cloud

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/util"
)

// This is a test server which would respond to any incoming HTTP requests
func startHTTPTestServer() *httptest.Server {
	http_handler := func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case "GET":
			switch r.URL.Path {
			// This one is just to test a basic HTTP request can be responded to
			case "/testing":
				w.Write([]byte(`Testing works`))
			case "/cloudprovider/v1aplha1/instances/my-cloud/nodeAddresses":
				resp := &InstanceNodeAddress{
					Node: []api.NodeAddress{
						{Type: "InternalIP", Address: "internalIP"},
						{Type: "ExternalIP", Address: "externalIP"},
					},
				}
				NodeAddressesResponse, _ := json.Marshal(resp)
				w.Write(NodeAddressesResponse)
			case "/cloudprovider/v1aplha1/instances/my-cloud/ID":
				resp := &InstanceID{
					ID: "my-cloud-1",
				}
				InstanceIDResponse, _ := json.Marshal(resp)
				w.Write(InstanceIDResponse)
			case "/cloudprovider/v1aplha1/instances/my-":
				resp := &InstanceList{
					List: []InstanceName{
						{Name: "my-cloud-1"},
						{Name: "my-cloud-2"},
					},
				}
				InstanceListResponse, _ := json.Marshal(resp)
				w.Write(InstanceListResponse)
			case "/cloudprovider/v1aplha1/instances/node/my-host":
				resp := &InstanceNodeName{
					Name: "my-node",
				}
				InstanceIDResponse, _ := json.Marshal(resp)
				w.Write(InstanceIDResponse)
			case "/cloudprovider/v1aplha1/providerName":
				resp := &ProviderName{
					Name: "my-cloud",
				}
				ProviderNameResponse, _ := json.Marshal(resp)
				w.Write(ProviderNameResponse)
			case "/cloudprovider/v1aplha1/zones":
				resp := &Zone{
					Domain: "my-zone-2",
					Region: "USA",
				}
				ZoneResponse, _ := json.Marshal(resp)
				w.Write(ZoneResponse)
			case "/cloudprovider/v1aplha1/clusters":
				resp := &ClusterList{
					ClusterNames: []ClusterName{
						{Name: "my-cluster-1"},
						{Name: "my-cluster-2"},
					},
				}
				ClusterListResponse, _ := json.Marshal(resp)
				w.Write(ClusterListResponse)
			case "/cloudprovider/v1aplha1/clusters/my-cluster-1/master":
				resp := &ClusterMaster{
					Address: "127.0.0.1",
				}
				ClusterMasterResponse, _ := json.Marshal(resp)
				w.Write(ClusterMasterResponse)
			case "/cloudprovider/v1aplha1/routes/my-cluster-1":
				resp := &RouteList{
					List: []Route{
						{Name: "my-route-1", Target: "a1-small", DestinationCIDR: "192.168.1.0/24"},
						{Name: "my-route-2", Target: "a1-big", DestinationCIDR: "192.168.1.1/24"},
					},
				}
				RoutesListResponse, _ := json.Marshal(resp)
				w.Write(RoutesListResponse)
			case "/cloudprovider/v1aplha1/tcpLoadBalancers/USA/my-tcp-balancer":
				resp := &TCPLoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{
						{IP: "127.0.0.1", Hostname: "my-cloud-host1"},
						{Hostname: "my-cloud-host2"},
						{IP: "127.0.0.2"},
					},
					Exists: true,
				}
				TCPLoadBalancerResponse, _ := json.Marshal(resp)
				w.Write(TCPLoadBalancerResponse)
			default:
				http.NotFound(w, r)
				return
			}
		case "POST":
			switch r.URL.Path {
			case "/cloudprovider/v1aplha1/instances/sshKey":
				var resp InstanceSSHKey
				body, err := ioutil.ReadAll(r.Body)
				if err != nil {
					glog.Errorf("Error reading the request")
				}
				if err = json.Unmarshal(body, &resp); err != nil {
					glog.Errorf("Error in reading JSON response for %s %s: %v The response was: %s", "POST", r.URL.Path, err, body)
				}
				var respBody *InstanceSSHKeyResponse
				if string(resp.Key[:]) == "@my-cloud-1SampleKey1" {
					respBody = &InstanceSSHKeyResponse{
						KeyAdded: true,
					}
				} else {
					respBody = &InstanceSSHKeyResponse{
						KeyAdded: false,
					}
				}
				SSHKeyAddedResponse, _ := json.Marshal(respBody)
				w.Write(SSHKeyAddedResponse)
			case "/cloudprovider/v1aplha1/routes":
				var resp CreateRoute
				body, err := ioutil.ReadAll(r.Body)
				if err != nil {
					glog.Errorf("Error reading the request")
				}
				if err = json.Unmarshal(body, &resp); err != nil {
					glog.Errorf("Error in reading JSON response for %s %s: %v The response was: %s", "POST", r.URL.Path, err, body)
				}
				var respBody RouteCreatedStatus
				if resp.NewRoute.Name == "my-route-1" {
					respBody = RouteCreatedStatus{
						RouteCreated: true,
					}
				} else {
					respBody = RouteCreatedStatus{
						RouteCreated: false,
					}
				}
				RouteCreatedResponse, _ := json.Marshal(respBody)
				w.Write(RouteCreatedResponse)
			case "/cloudprovider/v1aplha1/tcpLoadBalancers":
				resp := &TCPLoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{
						{Hostname: "my-cloud-host1"},
						{Hostname: "my-cloud-host2"},
					},
					Exists: true,
				}
				TCPLoadBalancerResponse, _ := json.Marshal(resp)
				w.Write(TCPLoadBalancerResponse)
			default:
				http.NotFound(w, r)
				return
			}
		case "PUT":
			switch r.URL.Path {
			case "/cloudprovider/v1aplha1/tcpLoadBalancers/USA/my-tcp-balancer":
				w.WriteHeader(http.StatusNoContent)
			default:
				http.NotFound(w, r)
				return
			}
		case "DELETE":
			switch r.URL.Path {
			case "/cloudprovider/v1aplha1/routes/my-cluster-1/my-route-1":
				w.WriteHeader(http.StatusNoContent)
			case "/cloudprovider/v1aplha1/tcpLoadBalancers/USA/my-tcp-balancer":
				w.WriteHeader(http.StatusNoContent)
			default:
				http.NotFound(w, r)
				return
			}
		case "OPTIONS":
			switch r.URL.Path {
			case "/cloudprovider/v1aplha1/instances":
				w.WriteHeader(http.StatusOK)
			case "/cloudprovider/v1aplha1/tcpLoadBalancers":
				w.WriteHeader(http.StatusOK)
			case "/cloudprovider/v1aplha1/zones":
				w.WriteHeader(http.StatusOK)
			case "/cloudprovider/v1aplha1/clusters":
				w.WriteHeader(http.StatusOK)
			case "/cloudprovider/v1aplha1/routes":
				w.WriteHeader(http.StatusOK)
			default:
				w.WriteHeader(http.StatusNotImplemented)
				return
			}
		}
	}
	return httptest.NewServer(http.HandlerFunc(http_handler))
}

func TestSendHTTPRequests(t *testing.T) {
	server := startHTTPTestServer()
	defer server.Close()

	http_cloud := &httpCloud{
		clientURL: server.URL,
	}
	resp, err := http_cloud.sendHTTPRequest("GET", "/testing", nil)
	if err != nil {
		t.Fatalf("HTTP Request not send: %s", err)
	}
	expectedResponse := []byte(`Testing works`)
	if !bytes.Equal(expectedResponse, resp) {
		t.Fatalf("Expected %s, got %s", expectedResponse, resp)
	}
}

func TestNewHttpCloud(t *testing.T) {
	config := []byte(`{
		"clientURL": "http://127.0.0.1"}`)
	r := bytes.NewReader(config)
	expectedAddress := "http://127.0.0.1"
	if http_cloud, err := newHTTPCloud(r); err != nil {
		t.Fatalf("New Http Cloud returned an error: %s", err)
	} else if http_cloud.clientURL != expectedAddress {
		t.Fatalf("Client URL not correct got %s, expected %s", http_cloud.clientURL, expectedAddress)
	}
}

func TestNewHttpCloudEmptyURL(t *testing.T) {
	config := []byte(`{
		"clientURL": ""}`)
	r := bytes.NewReader(config)
	if _, err := newHTTPCloud(r); err == nil {
		t.Fatalf("New Http Cloud doesn't return an error on empty string")
	}
}

func TestNewHttpCloudWrongURL(t *testing.T) {
	config := []byte(`{
		"clientURL": "hello"}`)
	r := bytes.NewReader(config)
	if _, err := newHTTPCloud(r); err == nil {
		t.Fatalf("New Http Cloud doesn't return an error on wrong url ")
	}
}

func TestNewHttpCloudSubResources(t *testing.T) {
	config := []byte(`{
		"clientURL": "http://127.0.0.1/kube-api"}`)
	r := bytes.NewReader(config)
	expectedAddress := "http://127.0.0.1/kube-api"
	if http_cloud, err := newHTTPCloud(r); err != nil {
		t.Fatalf("New Http Cloud returned an error: %s", err)
	} else if http_cloud.clientURL != expectedAddress {
		t.Fatalf("Client URL not correct got %s, expected %s", http_cloud.clientURL, expectedAddress)
	}
}

func TestNewHttpCloudTrailingSlashes(t *testing.T) {
	config := []byte(`{
		"clientURL": "http://127.0.0.1/"}`)
	r := bytes.NewReader(config)
	expectedAddress := "http://127.0.0.1"
	if http_cloud, err := newHTTPCloud(r); err != nil {
		t.Fatalf("New Http Cloud returned an error: %s", err)
	} else if http_cloud.clientURL != expectedAddress {
		t.Fatalf("Client URL not correct got %s, expected %s", http_cloud.clientURL, expectedAddress)
	}
}
func TestInstancesSupport(t *testing.T) {
	server := startHTTPTestServer()
	defer server.Close()

	http_cloud := &httpCloud{
		clientURL: server.URL,
	}
	instance, supported := http_cloud.Instances()
	if instance != http_cloud && supported {
		t.Fatalf("Instance returned is not the same calling instance")
	}
	if !supported {
		t.Fatalf("Wrong value returned, expected true, got false")
	}
}
func TestZonesSupport(t *testing.T) {
	server := startHTTPTestServer()
	defer server.Close()

	http_cloud := &httpCloud{
		clientURL: server.URL,
	}
	zones, supported := http_cloud.Zones()
	if zones != http_cloud && supported {
		t.Fatalf("Instance returned is not the same calling instance")
	}
	if !supported {
		t.Fatalf("Wrong value returned, expected true, got false")
	}
}
func TestClustersSupport(t *testing.T) {
	server := startHTTPTestServer()
	defer server.Close()

	http_cloud := &httpCloud{
		clientURL: server.URL,
	}
	clusters, supported := http_cloud.Clusters()
	if clusters != http_cloud && supported {
		t.Fatalf("Instance returned is not the same calling instance")
	}
	if !supported {
		t.Fatalf("Wrong value returned, expected true, got false")
	}
}
func TestRoutesSupport(t *testing.T) {
	server := startHTTPTestServer()
	defer server.Close()

	http_cloud := &httpCloud{
		clientURL: server.URL,
	}
	routes, supported := http_cloud.Routes()
	if routes != http_cloud && supported {
		t.Fatalf("Instance returned is not the same calling instance")
	}
	if !supported {
		t.Fatalf("Wrong value returned, expected true, got false")
	}
}
func TestTCPLoadBalancerSupport(t *testing.T) {
	server := startHTTPTestServer()
	defer server.Close()

	http_cloud := &httpCloud{
		clientURL: server.URL,
	}
	balancer, supported := http_cloud.TCPLoadBalancer()
	if balancer != http_cloud && supported {
		t.Fatalf("Instance returned is not the same calling instance")
	}
	if !supported {
		t.Fatalf("Wrong value returned, expected true, got false")
	}
}

func TestInstancesNodeAddresses(t *testing.T) {
	server := startHTTPTestServer()
	defer server.Close()

	http_cloud := &httpCloud{
		clientURL: server.URL,
	}
	nodeAddresses, err := http_cloud.NodeAddresses("my-cloud")
	if err != nil {
		t.Fatalf("Node Addresses returns an error: %s", err)
	}
	if len(nodeAddresses) != 2 {
		t.Fatalf("Node Addresses size is too small, node addresses not returned correctly")
	}
	expectedType := "InternalIP"
	if string(nodeAddresses[0].Type) != expectedType {
		t.Fatalf("Node Addresses value is wrong, expected %s, got %s", expectedType, nodeAddresses[0].Type)
	}
}

func TestInstancesID(t *testing.T) {
	server := startHTTPTestServer()
	defer server.Close()

	http_cloud := &httpCloud{
		clientURL: server.URL,
	}
	instanceID, err := http_cloud.InstanceID("my-cloud")
	if err != nil {
		t.Fatalf("InstanceID returns an error: %s", err)
	}
	expectedID := "my-cloud-1"
	if instanceID != expectedID {
		t.Fatalf("Instance ID value is wrong, expected %s, got %s", expectedID, instanceID)
	}
}

func TestInstancesList(t *testing.T) {
	server := startHTTPTestServer()
	defer server.Close()

	http_cloud := &httpCloud{
		clientURL: server.URL,
	}
	instanceNames, err := http_cloud.List("my-")
	if err != nil {
		t.Fatalf("List returns an error: %s", err)
	}
	expectedNames := []string{"my-cloud-1", "my-cloud-2"}
	for i, name := range instanceNames {
		if expectedNames[i] != name {
			t.Fatalf("Instance name is wrong, expected %s, got %s", expectedNames[i], name)
		}
	}
}

func TestSSHKeyAdded(t *testing.T) {
	server := startHTTPTestServer()
	defer server.Close()

	http_cloud := &httpCloud{
		clientURL: server.URL,
	}
	body := []byte(`@my-cloud-1SampleKey1`)
	err := http_cloud.AddSSHKeyToAllInstances("user", body)
	if err != nil {
		t.Fatalf("List returns an error: %s", err)
	}
}

func TestCurrentNodeName(t *testing.T) {
	server := startHTTPTestServer()
	defer server.Close()

	http_cloud := &httpCloud{
		clientURL: server.URL,
	}
	nodeName, err := http_cloud.CurrentNodeName("my-host")
	if err != nil {
		t.Fatalf("CurrentNodeName returns an error: %s", err)
	}
	expectedNodeName := "my-node"
	if nodeName != expectedNodeName {
		t.Fatalf("Current node name value is wrong, expected %s, got %s", expectedNodeName, nodeName)
	}
}

func TestProviderName(t *testing.T) {
	server := startHTTPTestServer()
	defer server.Close()

	http_cloud := &httpCloud{
		clientURL: server.URL,
	}
	providerName := http_cloud.ProviderName()
	expectedReturn := "my-cloud"
	if providerName != expectedReturn {
		t.Fatalf("Wrong cloud provider name, expected %s, got %s", expectedReturn, providerName)
	}
}

func TestGetZone(t *testing.T) {
	server := startHTTPTestServer()
	defer server.Close()

	http_cloud := &httpCloud{
		clientURL: server.URL,
	}
	zone, err := http_cloud.GetZone()
	if err != nil {
		t.Fatalf("GetZone returns an error: %s", err)
	}
	expectedDomain := "my-zone-2"
	expectedRegion := "USA"
	if zone.FailureDomain != expectedDomain {
		t.Fatalf("Wrong failure domain, expected %s, got %s", expectedDomain, zone.FailureDomain)
	}
	if zone.Region != expectedRegion {
		t.Fatalf("Wrong region, expected %s, got %s", expectedRegion, zone.Region)
	}
}

func TestListClusters(t *testing.T) {
	server := startHTTPTestServer()
	defer server.Close()

	http_cloud := &httpCloud{
		clientURL: server.URL,
	}
	clusterList, err := http_cloud.ListClusters()
	if err != nil {
		t.Fatalf("ListCluster returns an error: %s", err)
	}
	expectedList := []string{"my-cluster-1", "my-cluster-2"}
	for i, name := range clusterList {
		if expectedList[i] != name {
			t.Fatalf("Cluster name is wrong, expected %s, got %s", expectedList[i], name)
		}
	}
}

func TestClusterMaster(t *testing.T) {
	server := startHTTPTestServer()
	defer server.Close()

	http_cloud := &httpCloud{
		clientURL: server.URL,
	}
	masterAddress, err := http_cloud.Master("my-cluster-1")
	if err != nil {
		t.Fatalf("Master returns an error: %s", err)
	}
	expectedAddress := "127.0.0.1"
	if masterAddress != expectedAddress {
		t.Fatalf("Wrong master address, expected %s, got %s", expectedAddress, masterAddress)
	}
}

func TestListRoute(t *testing.T) {
	server := startHTTPTestServer()
	defer server.Close()

	http_cloud := &httpCloud{
		clientURL: server.URL,
	}
	routeList, err := http_cloud.ListRoutes("my-cluster-1")
	if err != nil {
		t.Fatalf("ListRoutes returns an error: %s", err)
	}
	expectedList := []string{"my-route-1", "my-route-2"}
	for i, list := range routeList {
		if expectedList[i] != (*list).Name {
			t.Fatalf("Route name is wrong, expected %s, got %s", expectedList[i], (*list).Name)
		}
	}
}

func TestCreateRoute(t *testing.T) {
	server := startHTTPTestServer()
	defer server.Close()

	http_cloud := &httpCloud{
		clientURL: server.URL,
	}
	route := &cloudprovider.Route{Name: "my-route-1", TargetInstance: "a1-small", DestinationCIDR: "192.168.1.0/24"}
	err := http_cloud.CreateRoute("my-cluster-1", "hint", route)
	if err != nil {
		t.Fatalf("CreateRoute returns an error: %s", err)
	}
}

func TestDeleteRoute(t *testing.T) {
	server := startHTTPTestServer()
	defer server.Close()

	http_cloud := &httpCloud{
		clientURL: server.URL,
	}
	route := &cloudprovider.Route{Name: "my-route-1", TargetInstance: "a1-small", DestinationCIDR: "192.168.1.0/24"}
	err := http_cloud.DeleteRoute("my-cluster-1", route)
	if err != nil {
		t.Fatalf("CreateRoute returns an error: %s", err)
	}
}

func TestGetTCPLoadBalancer(t *testing.T) {
	server := startHTTPTestServer()
	defer server.Close()

	http_cloud := &httpCloud{
		clientURL: server.URL,
	}
	status, exists, err := http_cloud.GetTCPLoadBalancer("my-tcp-balancer", "USA")
	if err != nil {
		t.Fatalf("GetTCPLoadBalancer returns an error: %s", err)
	}
	if !exists {
		t.Fatalf("TCP Load Balancer does not exist")
	}
	if len((*status).Ingress) != 3 {
		t.Fatalf("TCP Load Balancer size expected 3 got %d", len((*status).Ingress))
	}
	expectedHostList := []string{"my-cloud-host1", "my-cloud-host2", ""}
	expectedIPList := []string{"127.0.0.1", "", "127.0.0.2"}
	IngressList := (*status).Ingress
	for i, list := range IngressList {
		if expectedHostList[i] != list.Hostname {
			t.Fatalf("Route name is wrong, expected %s, got %s", expectedHostList[i], list.Hostname)
		}
		if expectedIPList[i] != list.IP {
			t.Fatalf("Route name is wrong, expected %s, got %s", expectedIPList[i], list.IP)
		}
	}
}

func TestEnsureTCPLoadBalancer(t *testing.T) {
	server := startHTTPTestServer()
	defer server.Close()

	http_cloud := &httpCloud{
		clientURL: server.URL,
	}
	name := "my-tcp-balancer"
	region := "USA"
	externalIP := net.IP([]byte(`127.0.0.1`))
	var ports []*api.ServicePort
	port1 := &api.ServicePort{
		Name:       "SMTP",
		Protocol:   api.ProtocolTCP,
		Port:       1234,
		TargetPort: util.NewIntOrStringFromInt(1234),
		NodePort:   1234,
	}
	port2 := &api.ServicePort{
		Name:       "SMTP",
		Protocol:   api.ProtocolTCP,
		Port:       1234,
		TargetPort: util.NewIntOrStringFromString("1234"),
		NodePort:   1234,
	}
	ports = append(ports, port1)
	ports = append(ports, port2)
	hosts := []string{"my-cloud-host1", "my-cloud-host2"}
	affinityType := api.ServiceAffinityClientIP
	status, err := http_cloud.EnsureTCPLoadBalancer(name, region, externalIP, ports, hosts, affinityType)
	if err != nil {
		t.Fatalf("Error in CreateTCPLoadBalancer: %s", err)
	}
	if len((*status).Ingress) != 2 {
		t.Fatalf("TCP Load Balancer size expected 3 got %d", len((*status).Ingress))
	}
	expectedHostList := []string{"my-cloud-host1", "my-cloud-host2"}
	IngressList := (*status).Ingress
	for i, list := range IngressList {
		if expectedHostList[i] != list.Hostname {
			t.Fatalf("Route name is wrong, expected %s, got %s", expectedHostList[i], list.Hostname)
		}
	}
}

func TestUpdateTCPLoadBalancer(t *testing.T) {
	server := startHTTPTestServer()
	defer server.Close()

	http_cloud := &httpCloud{
		clientURL: server.URL,
	}
	hosts := []string{"my-cloud-host1", "my-cloud-host2"}
	err := http_cloud.UpdateTCPLoadBalancer("my-tcp-balancer", "USA", hosts)
	if err != nil {
		t.Fatalf("UpdateTCPLoadBalancer returns an error: %s", err)
	}
}

func TestDeleteTCPLoadBalancer(t *testing.T) {
	server := startHTTPTestServer()
	defer server.Close()

	http_cloud := &httpCloud{
		clientURL: server.URL,
	}
	err := http_cloud.EnsureTCPLoadBalancerDeleted("my-tcp-balancer", "USA")
	if err != nil {
		t.Fatalf("CreateRoute returns an error: %s", err)
	}
}
