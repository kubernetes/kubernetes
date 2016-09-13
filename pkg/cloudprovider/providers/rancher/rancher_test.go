package rancher

import (
	"fmt"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/rancher/go-rancher/client"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
)

var (
	testClient    *client.RancherClient
	cloudProvider *CloudProvider

	hostTestSerializer *sync.Mutex
	lbTestSerializer   *sync.Mutex

	hostList                *client.HostCollection
	loadBalancerServiceList *client.LoadBalancerServiceCollection
	serviceList             *client.ServiceCollection
	externalServiceList     *client.ExternalServiceCollection

	ipAddressLinks            map[string]*client.IpAddressCollection
	lbConsumedServicesLinks   map[string]*client.ServiceCollection
	lbConsumedByServicesLinks map[string]*client.ServiceCollection
	lbServiceLinks            map[string]*client.SetLoadBalancerServiceLinksInput
)

type fakeRancherBaseClient struct {
	client.RancherBaseClientImpl
}

func (f *fakeRancherBaseClient) GetLink(resource client.Resource, link string, respObject interface{}) error {
	switch link {
	case "ipAddresses":
		if tmp, ok := ipAddressLinks[resource.Id]; ok {
			respObject.(*client.IpAddressCollection).Data = tmp.Data
		}
	case "consumedservices":
		if tmp, ok := lbConsumedServicesLinks[resource.Id]; ok {
			respObject.(*client.ServiceCollection).Data = tmp.Data
		}
	case "consumedbyservices":
		if tmp, ok := lbConsumedByServicesLinks[resource.Id]; ok {
			respObject.(*client.ServiceCollection).Data = tmp.Data
		}
	}
	return nil
}

type fakeHostClient struct{}

func (f *fakeHostClient) List(opts *client.ListOpts) (*client.HostCollection, error) {
	return hostList, nil
}

func (f *fakeHostClient) Create(opts *client.Host) (*client.Host, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeHostClient) Update(existing *client.Host, updates interface{}) (*client.Host, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeHostClient) ById(id string) (*client.Host, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeHostClient) Delete(container *client.Host) error {
	return fmt.Errorf("not implemented")
}

func (f *fakeHostClient) ActionActivate(*client.Host) (*client.Host, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeHostClient) ActionCreate(*client.Host) (*client.Host, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeHostClient) ActionDeactivate(*client.Host) (*client.Host, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeHostClient) ActionDockersocket(*client.Host) (*client.HostAccess, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeHostClient) ActionPurge(*client.Host) (*client.Host, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeHostClient) ActionRemove(*client.Host) (*client.Host, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeHostClient) ActionRestore(*client.Host) (*client.Host, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeHostClient) ActionUpdate(*client.Host) (*client.Host, error) {
	return nil, fmt.Errorf("not implemented")
}

type fakeLoadBalancerServiceClient struct{}

func (f *fakeLoadBalancerServiceClient) List(opts *client.ListOpts) (*client.LoadBalancerServiceCollection, error) {
	return loadBalancerServiceList, nil
}

func (f *fakeLoadBalancerServiceClient) Create(opts *client.LoadBalancerService) (*client.LoadBalancerService, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeLoadBalancerServiceClient) Update(existing *client.LoadBalancerService, updates interface{}) (*client.LoadBalancerService, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeLoadBalancerServiceClient) ById(id string) (*client.LoadBalancerService, error) {
	for _, lbserv := range loadBalancerServiceList.Data {
		if lbserv.Id == id {
			return &lbserv, nil
		}
	}
	return nil, fmt.Errorf("Could not find lb service")
}

func (f *fakeLoadBalancerServiceClient) Delete(lb *client.LoadBalancerService) error {
	for pos, lbserv := range loadBalancerServiceList.Data {
		if lbserv.Id == lb.Id {
			//remove the element
			loadBalancerServiceList.Data = append(loadBalancerServiceList.Data[:pos], loadBalancerServiceList.Data[pos+1:]...)
			return nil
		}
	}
	return fmt.Errorf("Could not find lb service")
}

func (f *fakeLoadBalancerServiceClient) ActionActivate(*client.LoadBalancerService) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeLoadBalancerServiceClient) ActionAddservicelink(*client.LoadBalancerService, *client.AddRemoveLoadBalancerServiceLinkInput) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeLoadBalancerServiceClient) ActionCancelrollback(*client.LoadBalancerService) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeLoadBalancerServiceClient) ActionCancelupgrade(*client.LoadBalancerService) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeLoadBalancerServiceClient) ActionCreate(*client.LoadBalancerService) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeLoadBalancerServiceClient) ActionDeactivate(*client.LoadBalancerService) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeLoadBalancerServiceClient) ActionFinishupgrade(*client.LoadBalancerService) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeLoadBalancerServiceClient) ActionRemove(*client.LoadBalancerService) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeLoadBalancerServiceClient) ActionRemoveservicelink(*client.LoadBalancerService, *client.AddRemoveLoadBalancerServiceLinkInput) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeLoadBalancerServiceClient) ActionRestart(*client.LoadBalancerService, *client.ServiceRestart) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeLoadBalancerServiceClient) ActionRollback(*client.LoadBalancerService) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeLoadBalancerServiceClient) ActionSetservicelinks(lb *client.LoadBalancerService, links *client.SetLoadBalancerServiceLinksInput) (*client.Service, error) {
	lbServiceLinks[lb.Id] = links
	return nil, nil
}

func (f *fakeLoadBalancerServiceClient) ActionUpdate(*client.LoadBalancerService) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeLoadBalancerServiceClient) ActionUpgrade(*client.LoadBalancerService, *client.ServiceUpgrade) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

type fakeServiceClient struct{}

func (f *fakeServiceClient) List(opts *client.ListOpts) (*client.ServiceCollection, error) {
	return serviceList, nil
}

func (f *fakeServiceClient) Create(opts *client.Service) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeServiceClient) Update(existing *client.Service, updates interface{}) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeServiceClient) ById(id string) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeServiceClient) Delete(s *client.Service) error {
	for pos, serv := range serviceList.Data {
		if serv.Id == s.Id {
			//remove the element
			serviceList.Data = append(serviceList.Data[:pos], serviceList.Data[pos+1:]...)
			return nil
		}
	}
	return fmt.Errorf("Could not find service")
}

func (f *fakeServiceClient) ActionActivate(*client.Service) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeServiceClient) ActionAddservicelink(*client.Service, *client.AddRemoveServiceLinkInput) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeServiceClient) ActionCancelrollback(*client.Service) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeServiceClient) ActionCancelupgrade(*client.Service) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeServiceClient) ActionCreate(*client.Service) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeServiceClient) ActionDeactivate(*client.Service) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeServiceClient) ActionFinishupgrade(*client.Service) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeServiceClient) ActionRemove(*client.Service) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeServiceClient) ActionRemoveservicelink(*client.Service, *client.AddRemoveServiceLinkInput) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeServiceClient) ActionRestart(*client.Service, *client.ServiceRestart) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeServiceClient) ActionRollback(*client.Service) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeServiceClient) ActionSetservicelinks(*client.Service, *client.SetServiceLinksInput) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeServiceClient) ActionUpdate(*client.Service) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeServiceClient) ActionUpgrade(*client.Service, *client.ServiceUpgrade) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

type fakeExternalServiceClient struct{}

func (f *fakeExternalServiceClient) List(opts *client.ListOpts) (*client.ExternalServiceCollection, error) {
	return externalServiceList, nil
}

func (f *fakeExternalServiceClient) Create(opts *client.ExternalService) (*client.ExternalService, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeExternalServiceClient) Update(existing *client.ExternalService, updates interface{}) (*client.ExternalService, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeExternalServiceClient) ById(id string) (*client.ExternalService, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeExternalServiceClient) Delete(s *client.ExternalService) error {
	for pos, serv := range externalServiceList.Data {
		if serv.Id == s.Id {
			//remove the element
			externalServiceList.Data = append(externalServiceList.Data[:pos], externalServiceList.Data[pos+1:]...)
			return nil
		}
	}
	return fmt.Errorf("Could not find service")
}

func (f *fakeExternalServiceClient) ActionActivate(*client.ExternalService) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeExternalServiceClient) ActionCancelrollback(*client.ExternalService) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeExternalServiceClient) ActionCancelupgrade(*client.ExternalService) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeExternalServiceClient) ActionCreate(*client.ExternalService) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeExternalServiceClient) ActionDeactivate(*client.ExternalService) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeExternalServiceClient) ActionFinishupgrade(*client.ExternalService) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeExternalServiceClient) ActionRemove(*client.ExternalService) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeExternalServiceClient) ActionRestart(*client.ExternalService, *client.ServiceRestart) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeExternalServiceClient) ActionRollback(*client.ExternalService) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeExternalServiceClient) ActionUpdate(*client.ExternalService) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *fakeExternalServiceClient) ActionUpgrade(*client.ExternalService, *client.ServiceUpgrade) (*client.Service, error) {
	return nil, fmt.Errorf("not implemented")
}

func TestMain(m *testing.M) {
	hostTestSerializer = &sync.Mutex{}
	lbTestSerializer = &sync.Mutex{}

	testClient = &client.RancherClient{
		RancherBaseClient: &fakeRancherBaseClient{
			RancherBaseClientImpl: client.RancherBaseClientImpl{
				Types: map[string]client.Schema{},
			},
		},
	}

	hostClient := &fakeHostClient{}
	testClient.Host = hostClient

	loadBalancerServiceClient := &fakeLoadBalancerServiceClient{}
	testClient.LoadBalancerService = loadBalancerServiceClient

	serviceClient := &fakeServiceClient{}
	testClient.Service = serviceClient

	externalServiceClient := &fakeExternalServiceClient{}
	testClient.ExternalService = externalServiceClient

	ipAddressLinks = make(map[string]*client.IpAddressCollection)
	lbConsumedServicesLinks = make(map[string]*client.ServiceCollection)
	lbConsumedByServicesLinks = make(map[string]*client.ServiceCollection)
	lbServiceLinks = make(map[string]*client.SetLoadBalancerServiceLinksInput)

	cloudProvider = &CloudProvider{
		client:    testClient,
		conf:      &rConfig{},
		hostCache: cache.NewTTLStore(hostStoreKeyFunc, time.Duration(24)*time.Hour),
	}
	os.Exit(m.Run())
}

func TestList(t *testing.T) {
	hostTestSerializer.Lock()
	defer hostTestSerializer.Unlock()
	hostList = &client.HostCollection{
		Data: []client.Host{
			client.Host{
				Hostname: "test1",
			},
			client.Host{
				Hostname: "test2",
			},
			client.Host{
				Hostname: "notATest",
			},
		},
	}

	foundHosts, err := cloudProvider.List("^test.*")
	found1 := false
	found2 := false
	for _, host := range foundHosts {
		if host == "test1" {
			found1 = true
		}
		if host == "test2" {
			found2 = true
		}
	}

	if !(found1 && found2) {
		t.Errorf("expected [test1, test2] found [%v], err: [%v]", foundHosts, err)
	}
}

func TestInstanceID(t *testing.T) {
	hostTestSerializer.Lock()
	defer hostTestSerializer.Unlock()
	hostList = &client.HostCollection{
		Data: []client.Host{
			client.Host{
				Resource: client.Resource{
					Id: "1h1",
				},
				Hostname: "test1",
				Uuid:     "abcd",
			},
		},
	}
	coll := new(client.IpAddressCollection)
	coll.Data = make([]client.IpAddress, 1)
	coll.Data = append(coll.Data, client.IpAddress{Address: "127.0.0.1"})
	ipAddressLinks["1h1"] = coll

	instanceId, err := cloudProvider.InstanceID("test1")
	if instanceId != "abcd" {
		t.Errorf("expected instanceID abcd, found [%v], err: [%v]", instanceId, err)
	}
}

func TestNodeAddresses(t *testing.T) {
	hostTestSerializer.Lock()
	defer hostTestSerializer.Unlock()
	hostList = &client.HostCollection{
		Data: []client.Host{
			client.Host{
				Resource: client.Resource{
					Id: "1h2",
				},
				Hostname: "test2",
				Uuid:     "abcd",
			},
		},
	}
	coll := new(client.IpAddressCollection)
	coll.Data = make([]client.IpAddress, 1)
	coll.Data = append(coll.Data, client.IpAddress{Address: "192.168.1.1"})
	ipAddressLinks["1h2"] = coll

	addresses, err := cloudProvider.NodeAddresses("test2")

	if len(addresses) != 5 {
		t.Errorf("expected 5 addresses, found, [%+v], err: [%v]", addresses, err)
	}

	if addresses[2].Type != api.NodeExternalIP || addresses[2].Address != "192.168.1.1" {
		t.Errorf("expected address 0 to be 192.168.1.1, found %s", addresses[2].Address)
	}

	if addresses[3].Type != api.NodeLegacyHostIP || addresses[3].Address != "192.168.1.1" {
		t.Errorf("expected address 1 to be 192.168.1.1, found %s", addresses[3].Address)
	}

	if addresses[4].Type != api.NodeHostName || addresses[4].Address != "test2" {
		t.Errorf("expected address 2 to be test2, found %s", addresses[4].Address)
	}
}

func TestGetLoadBalancer(t *testing.T) {
	lbTestSerializer.Lock()
	defer lbTestSerializer.Unlock()
	loadBalancerServiceList = &client.LoadBalancerServiceCollection{
		Data: []client.LoadBalancerService{
			client.LoadBalancerService{
				Resource: client.Resource{
					Id: "1lb1",
				},
				PublicEndpoints: []interface{}{
					PublicEndpoint{
						IPAddress: "172.178.1.1",
						Port:      8080,
					},
				},
				Name: "atestlb1",
			},
		},
	}

	service := api.Service{
		ObjectMeta: api.ObjectMeta{
			Name: "test-lb-1",
			UID:  "test-lb-1",
		},
	}
	status, exists, err := cloudProvider.GetLoadBalancer("", &service)

	if !exists {
		t.Errorf("Did not find a load balancer, err: [%v]", err)
	}

	if len(status.Ingress) != 1 {
		t.Errorf("Expected to find 1 ingress, found [%d]", len(status.Ingress))
	}

	if status.Ingress[0].IP != "172.178.1.1" {
		t.Errorf("Expected to find IP 172.178.1.1, found %s", status.Ingress[0].IP)
	}
}

func TestEnsureLoadBalancerDeleted(t *testing.T) {
	lbTestSerializer.Lock()
	defer lbTestSerializer.Unlock()
	loadBalancerServiceList = &client.LoadBalancerServiceCollection{
		Data: []client.LoadBalancerService{
			client.LoadBalancerService{
				Resource: client.Resource{
					Id: "1lb1",
				},
				PublicEndpoints: []interface{}{
					PublicEndpoint{
						IPAddress: "172.178.1.1",
						Port:      8080,
					},
				},
				Name: "atestlb1",
			},
		},
	}

	service := api.Service{
		ObjectMeta: api.ObjectMeta{
			Name: "test-lb-1",
			UID:  "test-lb-1",
		},
	}
	serviceList = &client.ServiceCollection{
		Data: []client.Service{
			client.Service{
				Resource: client.Resource{
					Id: "1s1",
				},
			},
			client.Service{
				Resource: client.Resource{
					Id: "1s2",
				},
			},
		},
	}

	lbConsumedServicesLinks["1lb1"] = &client.ServiceCollection{
		Data: []client.Service{
			client.Service{
				Resource: client.Resource{
					Id: "1s1",
				},
			},
			client.Service{
				Resource: client.Resource{
					Id: "1s2",
				},
			},
		},
	}

	lbConsumedByServicesLinks["1s1"] = &client.ServiceCollection{
		Data: []client.Service{
			client.Service{
				Resource: client.Resource{
					Id: "1lb1",
				},
			},
		},
	}

	lbConsumedByServicesLinks["1s2"] = &client.ServiceCollection{
		Data: []client.Service{
			client.Service{
				Resource: client.Resource{
					Id: "1lb1",
				},
			},
			client.Service{
				Resource: client.Resource{
					Id: "1lb2",
				},
			},
		},
	}
	err := cloudProvider.EnsureLoadBalancerDeleted("", &service)

	if err != nil {
		t.Errorf("Error deleting load balancer, err: [%v]", err)
	}

	if len(loadBalancerServiceList.Data) != 0 {
		t.Errorf("expected load balancer to be deleted, but it persists")
	}

	if len(serviceList.Data) != 1 {
		t.Errorf("consumed services have not been deleted as expected, remaining services %+v", serviceList.Data)
	}

}

func TestUpdateLoadBalancer(t *testing.T) {
	lbTestSerializer.Lock()
	defer lbTestSerializer.Unlock()
	externalServiceList = &client.ExternalServiceCollection{
		Data: []client.ExternalService{
			client.ExternalService{
				Resource: client.Resource{
					Id: "1s2",
				},
				EnvironmentId: "1e1",
				Name:          "externalserv1",
				State:         "active",
			},
		},
	}

	loadBalancerServiceList = &client.LoadBalancerServiceCollection{
		Data: []client.LoadBalancerService{
			client.LoadBalancerService{
				Resource: client.Resource{
					Id: "1lb1",
					Actions: map[string]string{
						"setservicelinks": "setservicelinks",
					},
				},
				PublicEndpoints: []interface{}{
					PublicEndpoint{
						IPAddress: "172.178.1.1",
						Port:      8080,
					},
				},
				Name:          "atestlb1",
				EnvironmentId: "1e1",
			},
		},
	}

	service := api.Service{
		ObjectMeta: api.ObjectMeta{
			Name: "test-lb-1",
			UID:  "test-lb-1",
		},
	}
	serviceList = &client.ServiceCollection{
		Data: []client.Service{
			client.Service{
				Resource: client.Resource{
					Id: "1s1",
				},
			},
			client.Service{
				Resource: client.Resource{
					Id: "1s2",
				},
			},
		},
	}

	lbConsumedServicesLinks["1lb1"] = &client.ServiceCollection{
		Data: []client.Service{
			client.Service{
				Resource: client.Resource{
					Id: "1s1",
				},
			},
		},
	}

	lbConsumedByServicesLinks["1s1"] = &client.ServiceCollection{
		Data: []client.Service{
			client.Service{
				Resource: client.Resource{
					Id: "1lb1",
				},
			},
		},
	}

	err := cloudProvider.UpdateLoadBalancer("", &service, []string{"host1"})

	if err != nil {
		t.Errorf("Error deleting load balancer, err: [%v]", err)
	}

	if len(serviceList.Data) != 1 {
		t.Errorf("consumed services have not been deleted as expected, remaining services %+v", serviceList.Data)
	}

	lbServiceLink, ok := lbServiceLinks["1lb1"]
	if !ok {
		t.Errorf("lb service links not updated as expected")
	}

	if len(lbServiceLink.ServiceLinks) != 1 {
		t.Errorf("lb service links not updated as expected")
	}

	if lbServiceLink.ServiceLinks[0].(*client.LoadBalancerServiceLink).ServiceId != "1s2" {
		t.Errorf("lb service links not updated as expected")
	}
}

func TestEnsureLoadBalancer(t *testing.T) {
	lbTestSerializer.Lock()
	defer lbTestSerializer.Unlock()
	externalServiceList = &client.ExternalServiceCollection{
		Data: []client.ExternalService{
			client.ExternalService{
				Resource: client.Resource{
					Id: "1s2",
				},
				EnvironmentId: "1e1",
				Name:          "externalserv1",
				State:         "active",
			},
		},
	}

	loadBalancerServiceList = &client.LoadBalancerServiceCollection{
		Data: []client.LoadBalancerService{
			client.LoadBalancerService{
				Resource: client.Resource{
					Id: "1lb1",
					Actions: map[string]string{
						"setservicelinks": "setservicelinks",
						"deactivate":      "deactivate",
					},
				},
				PublicEndpoints: []interface{}{
					PublicEndpoint{
						IPAddress: "192.168.1.2",
						Port:      8080,
					},
				},
				Name:          "atestlb1",
				EnvironmentId: "1e1",
				LaunchConfig: &client.LaunchConfig{
					Ports: []string{
						"80:8000/tcp",
					},
				},
				State: "active",
			},
		},
	}

	service := api.Service{
		ObjectMeta: api.ObjectMeta{
			Name: "test-lb-1",
			UID:  "test-lb-1",
		},
		Spec: api.ServiceSpec{
			Ports: []api.ServicePort{
				api.ServicePort{
					Name:     "testPort",
					Protocol: "TCP",
					Port:     80,
					NodePort: 8000,
				},
			},
			SessionAffinity: api.ServiceAffinityNone,
		},
	}
	serviceList = &client.ServiceCollection{
		Data: []client.Service{
			client.Service{
				Resource: client.Resource{
					Id: "1s1",
				},
			},
			client.Service{
				Resource: client.Resource{
					Id: "1s2",
				},
			},
		},
	}

	lbConsumedServicesLinks["1lb1"] = &client.ServiceCollection{
		Data: []client.Service{
			client.Service{
				Resource: client.Resource{
					Id: "1s1",
				},
			},
		},
	}

	lbConsumedByServicesLinks["1s1"] = &client.ServiceCollection{
		Data: []client.Service{
			client.Service{
				Resource: client.Resource{
					Id: "1lb1",
				},
			},
		},
	}

	status, err := cloudProvider.EnsureLoadBalancer("", &service, []string{"host1"})

	if err != nil {
		t.Errorf("Error ensuring load balancer, err: [%v]", err)
	}

	lbServiceLink, ok := lbServiceLinks["1lb1"]
	if !ok {
		t.Errorf("lb service links not updated as expected")
	}

	if len(lbServiceLink.ServiceLinks) != 1 {
		t.Errorf("lb service links not updated as expected")
	}

	if lbServiceLink.ServiceLinks[0].(*client.LoadBalancerServiceLink).ServiceId != "1s2" {
		t.Errorf("lb service links not updated as expected")
	}

	if len(status.Ingress) != 1 {
		t.Errorf("Expected to find 1 ingress, found [%d]", len(status.Ingress))
	}

	if status.Ingress[0].IP != "192.168.1.2" {
		t.Errorf("Expected to find IP 192.168.1.2, found %s", status.Ingress[0].IP)
	}
}
