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

package fake

import (
	"context"
	"fmt"
	"net"
	"regexp"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	cloudprovider "k8s.io/cloud-provider"
)

const defaultProviderName = "fake"

// Balancer is a fake storage of balancer information
type Balancer struct {
	Name           string
	Region         string
	LoadBalancerIP string
	Ports          []v1.ServicePort
	Hosts          []*v1.Node
}

// UpdateBalancerCall represents a fake call to update load balancers
type UpdateBalancerCall struct {
	Service *v1.Service
	Hosts   []*v1.Node
}

var _ cloudprovider.Interface = (*Cloud)(nil)
var _ cloudprovider.Instances = (*Cloud)(nil)
var _ cloudprovider.LoadBalancer = (*Cloud)(nil)
var _ cloudprovider.Routes = (*Cloud)(nil)
var _ cloudprovider.Zones = (*Cloud)(nil)
var _ cloudprovider.PVLabeler = (*Cloud)(nil)
var _ cloudprovider.Clusters = (*Cloud)(nil)
var _ cloudprovider.InstancesV2 = (*Cloud)(nil)

// Cloud is a test-double implementation of Interface, LoadBalancer, Instances, and Routes. It is useful for testing.
type Cloud struct {
	DisableInstances     bool
	DisableRoutes        bool
	DisableLoadBalancers bool
	DisableZones         bool
	DisableClusters      bool

	Exists bool
	Err    error

	EnableInstancesV2       bool
	ExistsByProviderID      bool
	ErrByProviderID         error
	NodeShutdown            bool
	ErrShutdownByProviderID error
	MetadataErr             error

	Calls          []string
	Addresses      []v1.NodeAddress
	addressesMux   sync.Mutex
	ExtID          map[types.NodeName]string
	ExtIDErr       map[types.NodeName]error
	InstanceTypes  map[types.NodeName]string
	Machines       []types.NodeName
	NodeResources  v1.ResourceList
	ClusterList    []string
	MasterName     string
	ExternalIP     net.IP
	BalancerIPMode *v1.LoadBalancerIPMode
	Balancers      map[string]Balancer
	updateCallLock sync.Mutex
	UpdateCalls    []UpdateBalancerCall
	ensureCallLock sync.Mutex
	EnsureCalls    []UpdateBalancerCall
	EnsureCallCb   func(UpdateBalancerCall)
	UpdateCallCb   func(UpdateBalancerCall)
	RouteMap       map[string]*Route
	Lock           sync.Mutex
	Provider       string
	ProviderID     map[types.NodeName]string
	addCallLock    sync.Mutex
	cloudprovider.Zone
	VolumeLabelMap   map[string]map[string]string
	AdditionalLabels map[string]string

	OverrideInstanceMetadata func(ctx context.Context, node *v1.Node) (*cloudprovider.InstanceMetadata, error)

	RequestDelay time.Duration
}

// Route is a representation of an advanced routing rule.
type Route struct {
	ClusterName string
	Route       cloudprovider.Route
}

func (f *Cloud) addCall(desc string) {
	time.Sleep(f.RequestDelay)

	f.addCallLock.Lock()
	defer f.addCallLock.Unlock()
	f.Calls = append(f.Calls, desc)
}

// ClearCalls clears internal record of method calls to this Cloud.
func (f *Cloud) ClearCalls() {
	f.Calls = []string{}
}

// Initialize passes a Kubernetes clientBuilder interface to the cloud provider
func (f *Cloud) Initialize(clientBuilder cloudprovider.ControllerClientBuilder, stop <-chan struct{}) {
}

// ListClusters lists the names of the available clusters.
func (f *Cloud) ListClusters(ctx context.Context) ([]string, error) {
	return f.ClusterList, f.Err
}

// Master gets back the address (either DNS name or IP address) of the master node for the cluster.
func (f *Cloud) Master(ctx context.Context, name string) (string, error) {
	return f.MasterName, f.Err
}

// Clusters returns a clusters interface.  Also returns true if the interface is supported, false otherwise.
func (f *Cloud) Clusters() (cloudprovider.Clusters, bool) {
	return f, !f.DisableClusters
}

// ProviderName returns the cloud provider ID.
func (f *Cloud) ProviderName() string {
	if f.Provider == "" {
		return defaultProviderName
	}
	return f.Provider
}

// HasClusterID returns true if the cluster has a clusterID
func (f *Cloud) HasClusterID() bool {
	return true
}

// LoadBalancer returns a fake implementation of LoadBalancer.
// Actually it just returns f itself.
func (f *Cloud) LoadBalancer() (cloudprovider.LoadBalancer, bool) {
	return f, !f.DisableLoadBalancers
}

// Instances returns a fake implementation of Instances.
//
// Actually it just returns f itself.
func (f *Cloud) Instances() (cloudprovider.Instances, bool) {
	return f, !f.DisableInstances
}

// InstancesV2 returns a fake implementation of InstancesV2.
//
// Actually it just returns f itself.
func (f *Cloud) InstancesV2() (cloudprovider.InstancesV2, bool) {
	if f.EnableInstancesV2 {
		return f, true
	}
	return nil, false
}

// Zones returns a zones interface. Also returns true if the interface is supported, false otherwise.
func (f *Cloud) Zones() (cloudprovider.Zones, bool) {
	return f, !f.DisableZones
}

// Routes returns a routes interface along with whether the interface is supported.
func (f *Cloud) Routes() (cloudprovider.Routes, bool) {
	return f, !f.DisableRoutes
}

// GetLoadBalancer is a stub implementation of LoadBalancer.GetLoadBalancer.
func (f *Cloud) GetLoadBalancer(ctx context.Context, clusterName string, service *v1.Service) (*v1.LoadBalancerStatus, bool, error) {
	status := &v1.LoadBalancerStatus{}
	status.Ingress = []v1.LoadBalancerIngress{{IP: f.ExternalIP.String()}}

	return status, f.Exists, f.Err
}

// GetLoadBalancerName is a stub implementation of LoadBalancer.GetLoadBalancerName.
func (f *Cloud) GetLoadBalancerName(ctx context.Context, clusterName string, service *v1.Service) string {
	// TODO: replace DefaultLoadBalancerName to generate more meaningful loadbalancer names.
	return cloudprovider.DefaultLoadBalancerName(service)
}

// EnsureLoadBalancer is a test-spy implementation of LoadBalancer.EnsureLoadBalancer.
// It adds an entry "create" into the internal method call record.
func (f *Cloud) EnsureLoadBalancer(ctx context.Context, clusterName string, service *v1.Service, nodes []*v1.Node) (*v1.LoadBalancerStatus, error) {
	f.addCall("create")
	f.markEnsureCall(service, nodes)
	if f.Balancers == nil {
		f.Balancers = make(map[string]Balancer)
	}

	name := f.GetLoadBalancerName(ctx, clusterName, service)
	spec := service.Spec

	zone, err := f.GetZone(context.TODO())
	if err != nil {
		return nil, err
	}
	region := zone.Region

	f.Balancers[name] = Balancer{name, region, spec.LoadBalancerIP, spec.Ports, nodes}

	status := &v1.LoadBalancerStatus{}
	// process Ports
	portStatus := []v1.PortStatus{}
	for _, port := range spec.Ports {
		portStatus = append(portStatus, v1.PortStatus{
			Port:     port.Port,
			Protocol: port.Protocol,
		})
	}
	status.Ingress = []v1.LoadBalancerIngress{{IP: f.ExternalIP.String(), IPMode: f.BalancerIPMode, Ports: portStatus}}

	return status, f.Err
}

func (f *Cloud) markUpdateCall(service *v1.Service, nodes []*v1.Node) {
	f.updateCallLock.Lock()
	defer f.updateCallLock.Unlock()
	update := UpdateBalancerCall{service, nodes}
	f.UpdateCalls = append(f.UpdateCalls, update)
	if f.UpdateCallCb != nil {
		f.UpdateCallCb(update)
	}
}

func (f *Cloud) markEnsureCall(service *v1.Service, nodes []*v1.Node) {
	f.ensureCallLock.Lock()
	defer f.ensureCallLock.Unlock()
	update := UpdateBalancerCall{service, nodes}
	f.EnsureCalls = append(f.EnsureCalls, update)
	if f.EnsureCallCb != nil {
		f.EnsureCallCb(update)
	}
}

// UpdateLoadBalancer is a test-spy implementation of LoadBalancer.UpdateLoadBalancer.
// It adds an entry "update" into the internal method call record.
func (f *Cloud) UpdateLoadBalancer(ctx context.Context, clusterName string, service *v1.Service, nodes []*v1.Node) error {
	f.addCall("update")
	f.markUpdateCall(service, nodes)
	return f.Err
}

// EnsureLoadBalancerDeleted is a test-spy implementation of LoadBalancer.EnsureLoadBalancerDeleted.
// It adds an entry "delete" into the internal method call record.
func (f *Cloud) EnsureLoadBalancerDeleted(ctx context.Context, clusterName string, service *v1.Service) error {
	f.addCall("delete")
	return f.Err
}

// AddSSHKeyToAllInstances adds an SSH public key as a legal identity for all instances
// expected format for the key is standard ssh-keygen format: <protocol> <blob>
func (f *Cloud) AddSSHKeyToAllInstances(ctx context.Context, user string, keyData []byte) error {
	return cloudprovider.NotImplemented
}

// CurrentNodeName returns the name of the node we are currently running on
// On most clouds (e.g. GCE) this is the hostname, so we provide the hostname
func (f *Cloud) CurrentNodeName(ctx context.Context, hostname string) (types.NodeName, error) {
	return types.NodeName(hostname), nil
}

// NodeAddresses is a test-spy implementation of Instances.NodeAddresses.
// It adds an entry "node-addresses" into the internal method call record.
func (f *Cloud) NodeAddresses(ctx context.Context, instance types.NodeName) ([]v1.NodeAddress, error) {
	f.addCall("node-addresses")
	f.addressesMux.Lock()
	defer f.addressesMux.Unlock()
	return f.Addresses, f.Err
}

// SetNodeAddresses sets the addresses for a node
func (f *Cloud) SetNodeAddresses(nodeAddresses []v1.NodeAddress) {
	f.addressesMux.Lock()
	defer f.addressesMux.Unlock()
	f.Addresses = nodeAddresses
}

// NodeAddressesByProviderID is a test-spy implementation of Instances.NodeAddressesByProviderID.
// It adds an entry "node-addresses-by-provider-id" into the internal method call record.
func (f *Cloud) NodeAddressesByProviderID(ctx context.Context, providerID string) ([]v1.NodeAddress, error) {
	f.addCall("node-addresses-by-provider-id")
	f.addressesMux.Lock()
	defer f.addressesMux.Unlock()
	return f.Addresses, f.Err
}

// InstanceID returns the cloud provider ID of the node with the specified Name, unless an entry
// for the node exists in ExtIDError, in which case it returns the desired error (to facilitate
// testing of error handling).
func (f *Cloud) InstanceID(ctx context.Context, nodeName types.NodeName) (string, error) {
	f.addCall("instance-id")

	err, ok := f.ExtIDErr[nodeName]
	if ok {
		return "", err
	}

	return f.ExtID[nodeName], nil
}

// InstanceType returns the type of the specified instance.
func (f *Cloud) InstanceType(ctx context.Context, instance types.NodeName) (string, error) {
	f.addCall("instance-type")
	return f.InstanceTypes[instance], nil
}

// InstanceTypeByProviderID returns the type of the specified instance.
func (f *Cloud) InstanceTypeByProviderID(ctx context.Context, providerID string) (string, error) {
	f.addCall("instance-type-by-provider-id")
	return f.InstanceTypes[types.NodeName(providerID)], nil
}

// InstanceExistsByProviderID returns true if the instance with the given provider id still exists and is running.
// If false is returned with no error, the instance will be immediately deleted by the cloud controller manager.
func (f *Cloud) InstanceExistsByProviderID(ctx context.Context, providerID string) (bool, error) {
	f.addCall("instance-exists-by-provider-id")
	return f.ExistsByProviderID, f.ErrByProviderID
}

// InstanceShutdownByProviderID returns true if the instances is in safe state to detach volumes
func (f *Cloud) InstanceShutdownByProviderID(ctx context.Context, providerID string) (bool, error) {
	f.addCall("instance-shutdown-by-provider-id")

	if providerID == "" {
		return false, fmt.Errorf("cannot shutdown instance with empty providerID")
	}

	return f.NodeShutdown, f.ErrShutdownByProviderID
}

// InstanceExists returns true if the instance corresponding to a node still exists and is running.
// If false is returned with no error, the instance will be immediately deleted by the cloud controller manager.
func (f *Cloud) InstanceExists(ctx context.Context, node *v1.Node) (bool, error) {
	f.addCall("instance-exists")
	return f.ExistsByProviderID, f.ErrByProviderID
}

// InstanceShutdown returns true if the instances is in safe state to detach volumes
func (f *Cloud) InstanceShutdown(ctx context.Context, node *v1.Node) (bool, error) {
	f.addCall("instance-shutdown")
	return f.NodeShutdown, f.ErrShutdownByProviderID
}

// InstanceMetadata returns metadata of the specified instance.
func (f *Cloud) InstanceMetadata(ctx context.Context, node *v1.Node) (*cloudprovider.InstanceMetadata, error) {
	if f.OverrideInstanceMetadata != nil {
		return f.OverrideInstanceMetadata(ctx, node)
	}
	f.addCall("instance-metadata-by-provider-id")
	f.addressesMux.Lock()
	defer f.addressesMux.Unlock()

	providerID := ""
	id, ok := f.ProviderID[types.NodeName(node.Name)]
	if ok {
		providerID = id
	}

	return &cloudprovider.InstanceMetadata{
		ProviderID:       providerID,
		InstanceType:     f.InstanceTypes[types.NodeName(node.Spec.ProviderID)],
		NodeAddresses:    f.Addresses,
		Zone:             f.Zone.FailureDomain,
		Region:           f.Zone.Region,
		AdditionalLabels: f.AdditionalLabels,
	}, f.MetadataErr
}

// List is a test-spy implementation of Instances.List.
// It adds an entry "list" into the internal method call record.
func (f *Cloud) List(filter string) ([]types.NodeName, error) {
	f.addCall("list")
	result := []types.NodeName{}
	for _, machine := range f.Machines {
		if match, _ := regexp.MatchString(filter, string(machine)); match {
			result = append(result, machine)
		}
	}
	return result, f.Err
}

// GetZone returns the Zone containing the current failure zone and locality region that the program is running in
// In most cases, this method is called from the kubelet querying a local metadata service to acquire its zone.
// For the case of external cloud providers, use GetZoneByProviderID or GetZoneByNodeName since GetZone
// can no longer be called from the kubelets.
func (f *Cloud) GetZone(ctx context.Context) (cloudprovider.Zone, error) {
	f.addCall("get-zone")
	return f.Zone, f.Err
}

// GetZoneByProviderID implements Zones.GetZoneByProviderID
// This is particularly useful in external cloud providers where the kubelet
// does not initialize node data.
func (f *Cloud) GetZoneByProviderID(ctx context.Context, providerID string) (cloudprovider.Zone, error) {
	f.addCall("get-zone-by-provider-id")
	return f.Zone, f.Err
}

// GetZoneByNodeName implements Zones.GetZoneByNodeName
// This is particularly useful in external cloud providers where the kubelet
// does not initialize node data.
func (f *Cloud) GetZoneByNodeName(ctx context.Context, nodeName types.NodeName) (cloudprovider.Zone, error) {
	f.addCall("get-zone-by-node-name")
	return f.Zone, f.Err
}

// ListRoutes lists all managed routes that belong to the specified clusterName
func (f *Cloud) ListRoutes(ctx context.Context, clusterName string) ([]*cloudprovider.Route, error) {
	f.Lock.Lock()
	defer f.Lock.Unlock()
	f.addCall("list-routes")
	var routes []*cloudprovider.Route
	for _, fakeRoute := range f.RouteMap {
		if clusterName == fakeRoute.ClusterName {
			routeCopy := fakeRoute.Route
			routes = append(routes, &routeCopy)
		}
	}
	return routes, f.Err
}

// CreateRoute creates the described managed route
// route.Name will be ignored, although the cloud-provider may use nameHint
// to create a more user-meaningful name.
func (f *Cloud) CreateRoute(ctx context.Context, clusterName string, nameHint string, route *cloudprovider.Route) error {
	f.Lock.Lock()
	defer f.Lock.Unlock()
	f.addCall("create-route")
	name := clusterName + "-" + string(route.TargetNode) + "-" + route.DestinationCIDR
	if _, exists := f.RouteMap[name]; exists {
		f.Err = fmt.Errorf("route %q already exists", name)
		return f.Err
	}
	fakeRoute := Route{}
	fakeRoute.Route = *route
	fakeRoute.Route.Name = name
	fakeRoute.ClusterName = clusterName
	f.RouteMap[name] = &fakeRoute
	return nil
}

// DeleteRoute deletes the specified managed route
// Route should be as returned by ListRoutes
func (f *Cloud) DeleteRoute(ctx context.Context, clusterName string, route *cloudprovider.Route) error {
	f.Lock.Lock()
	defer f.Lock.Unlock()
	f.addCall("delete-route")
	name := ""
	for key, saved := range f.RouteMap {
		if route.DestinationCIDR == saved.Route.DestinationCIDR &&
			route.TargetNode == saved.Route.TargetNode &&
			clusterName == saved.ClusterName {
			name = key
			break
		}
	}

	if len(name) == 0 {
		f.Err = fmt.Errorf("no route found for node:%v with DestinationCIDR== %v", route.TargetNode, route.DestinationCIDR)
		return f.Err
	}

	delete(f.RouteMap, name)
	return nil
}

// GetLabelsForVolume returns the labels for a PersistentVolume
func (f *Cloud) GetLabelsForVolume(ctx context.Context, pv *v1.PersistentVolume) (map[string]string, error) {
	if val, ok := f.VolumeLabelMap[pv.Name]; ok {
		return val, nil
	}
	return nil, fmt.Errorf("label not found for volume")
}
