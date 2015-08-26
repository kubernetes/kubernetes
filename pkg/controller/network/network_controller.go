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

package networkcontroller

import (
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/cache"
	"k8s.io/kubernetes/pkg/client/unversioned/record"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/networkprovider"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

const (
	workerGoroutines = 10

	// We'll attempt to recompute EVERY service's endpoints at least this
	// often. Higher numbers = lower CPU/network load; lower numbers =
	// shorter amount of time before a mistaken endpoint is corrected.
	FullServiceResyncPeriod = 5 * time.Minute

	// We'll keep enpoint watches open up to this long
	EndpointRelistPeriod = 5 * time.Minute

	// We'll keep network watches open up to this long
	NetworkRelistPeriod = 5 * time.Minute

	clientRetryCount    = 5
	clientRetryInterval = 5 * time.Second
)

var (
	keyFunc = framework.DeletionHandlingMetaNamespaceKeyFunc
)

// NetworkController manages all networks
type NetworkController struct {
	client           *client.Client
	queue            *workqueue.Type
	netProvider      networkprovider.Interface
	eventRecorder    record.EventRecorder
	eventBroadcaster record.EventBroadcaster

	networkStore  cache.StoreToNetworksLister
	serviceStore  cache.StoreToServiceLister
	endpointStore cache.StoreToEndpointsLister

	networkController  *framework.Controller
	serviceController  *framework.Controller
	endpointController *framework.Controller
}

// NewNetworkController returns a new *NetworkController.
func NewNetworkController(client *client.Client, provider networkprovider.Interface) *NetworkController {
	broadcaster := record.NewBroadcaster()
	broadcaster.StartRecordingToSink(client.Events(""))
	recorder := broadcaster.NewRecorder(api.EventSource{Component: "network-controller"})

	e := &NetworkController{
		client:           client,
		queue:            workqueue.New(),
		netProvider:      provider,
		eventRecorder:    recorder,
		eventBroadcaster: broadcaster,
	}

	e.serviceStore.Store, e.serviceController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return e.client.Services(api.NamespaceAll).List(labels.Everything())
			},
			WatchFunc: func(rv string) (watch.Interface, error) {
				return e.client.Services(api.NamespaceAll).Watch(labels.Everything(), fields.Everything(), rv)
			},
		},
		&api.Service{},
		FullServiceResyncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc: e.enqueueService,
			UpdateFunc: func(old, cur interface{}) {
				e.enqueueService(cur)
			},
			DeleteFunc: e.enqueueService,
		},
	)

	e.networkStore.Store, e.networkController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return e.client.Networks().List(labels.Everything(), fields.Everything())
			},
			WatchFunc: func(rv string) (watch.Interface, error) {
				return e.client.Networks().Watch(labels.Everything(), fields.Everything(), rv)
			},
		},
		&api.Network{},
		NetworkRelistPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc:    e.addNetwork,
			UpdateFunc: e.updateNetwork,
			DeleteFunc: e.deleteNetwork,
		},
	)

	e.endpointStore.Store, e.endpointController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return e.client.Endpoints(api.NamespaceAll).List(labels.Everything())
			},
			WatchFunc: func(rv string) (watch.Interface, error) {
				return e.client.Endpoints(api.NamespaceAll).Watch(labels.Everything(), fields.Everything(), rv)
			},
		},
		&api.Endpoints{},
		EndpointRelistPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc:    e.addEndpoint,
			UpdateFunc: e.updateEndpoint,
			DeleteFunc: e.deleteEndpoint,
		},
	)

	return e
}

// Check network by networkprovider
// If network doesn't exist, then create one.
func (e *NetworkController) addNetwork(obj interface{}) {
	net := obj.(*api.Network)
	newNetwork := *net
	newNetworkStatus := api.NetworkActive

	glog.V(4).Infof("NetworkController: add network %s", net.Name)

	// Check if tenant id exist
	check, err := e.netProvider.CheckTenantID(net.Spec.TenantID)
	if err != nil {
		glog.Errorf("NetworkController: check tenantID failed: %v", err)
	}
	if !check {
		glog.Warningf("NetworkController: tenantID %s doesn't exit in network provider", net.Spec.TenantID)
		newNetwork.Status = api.NetworkStatus{Phase: api.NetworkFailed}
		_, err := e.client.Networks().Status(&newNetwork)
		if err != nil {
			glog.Errorf("NetworkController: failed to update network status: %v", err)
		}
		return
	}

	// Check if provider network id exist
	if net.Spec.ProviderNetworkID != "" {
		_, err := e.netProvider.Networks().GetNetworkByID(net.Spec.ProviderNetworkID)
		if err != nil {
			glog.Warningf("NetworkController: network %s doesn't exit in network provider", net.Spec.ProviderNetworkID)
			newNetworkStatus = api.NetworkFailed
		}
	} else {
		if len(net.Spec.Subnets) == 0 {
			glog.Warningf("NetworkController: subnets of %s is null", net.Name)
			newNetworkStatus = api.NetworkFailed
		} else {
			// Check if provider network has already created
			networkName := networkprovider.BuildNetworkName(net.Name, net.Spec.TenantID)
			_, err := e.netProvider.Networks().GetNetwork(networkName)
			if err == nil {
				glog.Infof("NetworkController: network %s has already created", networkName)
			} else if err.Error() == networkprovider.ErrNotFound.Error() {
				// Create a new network by network provider
				err := e.netProvider.Networks().CreateNetwork(networkprovider.ApiNetworkToProviderNetwork(net))
				if err != nil {
					glog.Warningf("NetworkController: create network %s failed: %v", net.Name, err)
					newNetworkStatus = api.NetworkFailed
				}
			} else {
				glog.Warningf("NetworkController: get network failed: %v", err)
				newNetworkStatus = api.NetworkFailed
			}
		}
	}

	newNetwork.Status = api.NetworkStatus{Phase: newNetworkStatus}
	_, err = e.client.Networks().Status(&newNetwork)
	if err != nil {
		glog.Errorf("NetworkController: failed to update network status: %v", err)
	}

}

// UpdateNetwork update network'by networkprovider
func (e *NetworkController) updateNetwork(old, cur interface{}) {
	if api.Semantic.DeepEqual(old, cur) {
		return
	}

	var failed bool
	oldNetwork := old.(*api.Network)
	newNetwork := cur.(*api.Network)

	if newNetwork.Spec.ProviderNetworkID != "" {
		// If oldNetwork has created network, delete it
		if oldNetwork.Spec.ProviderNetworkID == "" {
			// delete old tenant network
			networkName := networkprovider.BuildNetworkName(oldNetwork.Name, oldNetwork.Spec.TenantID)
			err := e.netProvider.Networks().DeleteNetwork(networkName)
			if err != nil {
				glog.Errorf("NetworkController: delete old network %s from networkprovider failed: %v", oldNetwork.Name, err)
				failed = true
			}
		}
	} else {
		// Update network's subnet
		net := networkprovider.ApiNetworkToProviderNetwork(newNetwork)
		err := e.netProvider.Networks().UpdateNetwork(net)
		if err != nil {
			glog.Errorf("NetworkController: update network %s failed: %v", newNetwork.Name, err)
			failed = true
		}
	}

	// If updated failed, update network status
	if failed {
		oldNetwork.Status = api.NetworkStatus{Phase: api.NetworkFailed}
		_, err := e.client.Networks().Status(oldNetwork)
		if err != nil {
			glog.Errorf("NetworkController: failed to update network status: %v", err)
		}
	}
}

// DeleteNetwork deletes networks created by networkprovider
func (e *NetworkController) deleteNetwork(obj interface{}) {
	if net, ok := obj.(*api.Network); ok {
		glog.V(4).Infof("NetworkController: network %s deleted", net.Name)
		// Only delete network created by networkprovider
		if net.Spec.ProviderNetworkID == "" {
			networkName := networkprovider.BuildNetworkName(net.Name, net.Spec.TenantID)
			err := e.netProvider.Networks().DeleteNetwork(networkName)
			if err != nil {
				glog.Errorf("NetworkController: delete network %s failed in networkprovider: %v", networkName, err)
			} else {
				glog.V(4).Infof("NetworkController: network %s deleted in networkprovider", networkName)
			}
		}
		return
	}

	key, err := keyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", obj, err)
	}
	glog.V(5).Infof("Network %q was deleted but we don't have a record of its final state", key)
}

// obj could be an *api.Service, or a DeletionFinalStateUnknown marker item.
func (e *NetworkController) enqueueService(obj interface{}) {
	if svc, ok := obj.(*api.Service); ok {
		if svc.Spec.Type != api.ServiceTypeNetworkProvider {
			return
		}
	}

	key, err := keyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", obj, err)
	}

	e.queue.Add(key)
}

// worker runs a worker thread that just dequeues items, processes them, and
// marks them done. You may run as many of these in parallel as you wish; the
// workqueue guarantees that they will not end up processing the same service
// at the same time.
func (e *NetworkController) worker() {
	for {
		func() {
			key, quit := e.queue.Get()
			if quit {
				return
			}
			// Use defer: in the unlikely event that there's a
			// panic, we'd still like this to get marked done--
			// otherwise the controller will not be able to sync
			// this service again until it is restarted.
			defer e.queue.Done(key)
			e.syncService(key.(string))
		}()
	}
}

// Runs e; will not return until stopCh is closed. workers determines how many
// endpoints will be handled in parallel.
func (e *NetworkController) Run(stopCh <-chan struct{}) error {
	if e.netProvider == nil {
		return fmt.Errorf("NetController should not be run without a networkprovider.")
	}

	defer util.HandleCrash()
	go e.serviceController.Run(stopCh)
	go e.networkController.Run(stopCh)
	go e.endpointController.Run(stopCh)

	for i := 0; i < workerGoroutines; i++ {
		go util.Until(e.worker, time.Second, stopCh)
	}

	go func() {
		defer util.HandleCrash()
		time.Sleep(5 * time.Minute) // give time for our cache to fill
		e.startUp()
	}()
	<-stopCh
	e.queue.ShutDown()
	return nil
}

func (e *NetworkController) syncService(key string) {
	glog.V(4).Infof("NetworkController: processing service %v", key)

	obj, exists, err := e.serviceStore.Store.GetByKey(key)
	if err != nil || !exists {
		// Delete the corresponding loadbalancer, as the service has been deleted.
		namespace, name, err := cache.SplitMetaNamespaceKey(key)
		if err != nil {
			glog.Errorf("NetworkController: couldn't understand the key %s: %v", key, err)
			return
		}

		loadBalancerFullName := networkprovider.BuildLoadBalancerName(name, namespace)
		deleteError := e.netProvider.LoadBalancers().DeleteLoadBalancer(loadBalancerFullName)
		if deleteError != nil {
			glog.Errorf("NetworkController: delete loadbalancer %s failed: %v", loadBalancerFullName, err)
		}

		return
	}

	service := obj.(*api.Service)
	if service.Spec.Selector == nil {
		// services without a selector receive no endpoints from this controller;
		// these services will receive the endpoints that are created out-of-band via the REST API.
		return
	}

	// check if loadbalancer already created
	var status *api.LoadBalancerStatus
	loadBalancerFullName := networkprovider.BuildLoadBalancerName(service.Name, service.Namespace)
	loadBalancer, err := e.netProvider.LoadBalancers().GetLoadBalancer(loadBalancerFullName)
	if err != nil && err.Error() == networkprovider.ErrNotFound.Error() {
		// create new loadbalancer
		status, _ = e.createLoadBalancer(service)
	} else if err != nil {
		glog.Errorf("NetworkController: couldn't get loadbalancer from networkprovider: %v", err)
		return
	} else {
		// update loadbalancer
		status, _ = e.updateLoadBalancer(service, loadBalancer)
	}

	if status != nil {
		service.Status.LoadBalancer = *status
		err := e.updateService(service)
		if err != nil {
			e.eventRecorder.Event(service, "created loadbalancer", "created loadbalancer")
		}
	}
}

func (e *NetworkController) getEndpointHosts(service *api.Service) ([]*networkprovider.HostPort, error) {
	hosts := make([]*networkprovider.HostPort, 0, 1)
	// get service's endpoints
	// Endpoints may be delayed since they are created/updated from another controller
	endpoint, err := e.client.Endpoints(service.Namespace).Get(service.Name)
	if err != nil {
		glog.Errorf("NetworkController: couldn't get endpoint for service %s: %v", service.Name, err)
		return hosts, err
	}

	for _, host := range endpoint.Subsets {
		for _, ip := range host.Addresses {
			for _, port := range host.Ports {
				hostport := networkprovider.HostPort{
					Name:       port.Name,
					IPAddress:  ip.IP,
					TargetPort: port.Port,
				}

				hosts = append(hosts, &hostport)
			}
		}
	}

	for _, svc := range service.Spec.Ports {
		var targetPort int
		if svc.TargetPort.String() == "" {
			targetPort = svc.Port
		} else if svc.TargetPort.Kind == util.IntstrInt {
			targetPort = svc.TargetPort.IntVal
		}

		for _, hostport := range hosts {
			if hostport.TargetPort == targetPort || svc.TargetPort.StrVal == hostport.Name {
				hostport.ServicePort = svc.Port
			}
		}
	}

	return hosts, nil
}

func (e *NetworkController) updateService(service *api.Service) error {
	var err error
	for i := 0; i < clientRetryCount; i++ {
		_, err = e.client.Services(service.Namespace).Update(service)
		if err == nil {
			return nil
		}
		// If the object no longer exists, we don't want to recreate it.
		if errors.IsNotFound(err) {
			glog.Infof("NetworkController: service not updated since no longer exists: %v", err)
			return nil
		}
		// If conflicted, wait another update
		if errors.IsConflict(err) {
			glog.Infof("NetworkController: service not updated because it has been changed since we received it: %v", err)
			return nil
		}
		glog.Warningf("NetworkController: Failed to persist updated LoadBalancerStatus to service %s: %v",
			service.Name, err)
		time.Sleep(clientRetryInterval)
	}
	return err
}

func (e *NetworkController) hostPortsEqual(old, new []*networkprovider.HostPort) bool {
	if len(old) != len(new) {
		return false
	}

	for _, o := range old {
		var found bool
		for _, n := range new {
			if n.ServicePort == o.ServicePort && n.IPAddress == o.IPAddress && n.TargetPort == o.TargetPort {
				found = true
			}
		}
		if !found {
			return false
		}
	}

	return true
}

func (e *NetworkController) updateLoadBalancer(service *api.Service, lb *networkprovider.LoadBalancer) (*api.LoadBalancerStatus, error) {
	loadBalancerFullName := networkprovider.BuildLoadBalancerName(service.Name, service.Namespace)

	newHosts, _ := e.getEndpointHosts(service)
	if len(newHosts) == 0 {
		glog.V(4).Infof("NetworkController: no endpoints on service %s", service.Name)
		return nil, nil
	}

	if e.hostPortsEqual(lb.Hosts, newHosts) {
		return nil, nil
	}

	vip, err := e.netProvider.LoadBalancers().UpdateLoadBalancer(loadBalancerFullName, newHosts, service.Spec.ExternalIPs)
	if err != nil {
		glog.Errorf("NetworkController: couldn't update loadlbalancer %s:%v", loadBalancerFullName, err)
		return nil, err
	}

	glog.V(4).Infof("NetworkController: loadbalancer %s (vip: %s) updated", loadBalancerFullName, vip)

	status := &api.LoadBalancerStatus{}
	status.Ingress = []api.LoadBalancerIngress{{IP: vip}}
	return status, nil
}

func (e *NetworkController) createLoadBalancer(service *api.Service) (*api.LoadBalancerStatus, error) {
	newHosts, _ := e.getEndpointHosts(service)
	if len(newHosts) == 0 {
		glog.V(4).Infof("NetworkController: no endpoints on service %s", service.Name)
		return nil, nil
	}

	// get namespace of svc
	namespace, err := e.client.Namespaces().Get(service.Namespace)
	if err != nil {
		glog.Errorf("NetworkController: couldn't get namespace for service %s: %v", service.Name, err)
		return nil, err
	}
	if namespace.Spec.Network == "" {
		glog.Warningf("NetworkController: there is no network associated with namespace %s", namespace.Name)
		return nil, nil
	}

	// get network of namespace
	network, err := e.client.Networks().Get(namespace.Spec.Network)
	if err != nil {
		glog.Errorf("NetworkController: couldn't get network for namespace %s: %v", namespace.Name, err)
		return nil, err
	}

	var networkInfo *networkprovider.Network
	if network.Spec.ProviderNetworkID != "" {
		networkInfo, err = e.netProvider.Networks().GetNetworkByID(network.Spec.ProviderNetworkID)
	} else {
		networkName := networkprovider.BuildNetworkName(network.Name, network.Spec.TenantID)
		networkInfo, err = e.netProvider.Networks().GetNetwork(networkName)
	}
	if err != nil {
		glog.Errorf("NetworkController: couldn't get network from networkprovider: %v", err)
		return nil, err
	}

	// create loadbalancer for service
	loadBalancerFullName := networkprovider.BuildLoadBalancerName(service.Name, service.Namespace)
	providerLoadBalancer := networkprovider.LoadBalancer{
		Name: loadBalancerFullName,
		// TODO: support more loadbalancer type
		Type:        networkprovider.LoadBalancerTypeTCP,
		TenantID:    networkInfo.TenantID,
		Subnets:     networkInfo.Subnets,
		Hosts:       newHosts,
		ExternalIPs: service.Spec.ExternalIPs,
	}

	if service.Spec.ClusterIP != "" {
		providerLoadBalancer.Vip = service.Spec.ClusterIP
	}

	vip, err := e.netProvider.LoadBalancers().CreateLoadBalancer(&providerLoadBalancer, service.Spec.SessionAffinity)
	if err != nil {
		glog.Errorf("NetworkController: create load balancer %s failed: %v", loadBalancerFullName, err)
		return nil, err
	}

	glog.V(4).Infof("NetworkController: load balancer for service %s created", service.Name)

	status := &api.LoadBalancerStatus{}
	status.Ingress = []api.LoadBalancerIngress{{IP: vip}}
	return status, nil
}

// In order to process services deleted while controller is down, fill the queue on startup
func (e *NetworkController) startUp() {
	svcList, err := e.client.Services(api.NamespaceAll).List(labels.Everything())
	if err != nil {
		glog.Errorf("Unable to list services: %v", err)
		return
	}

	for _, svc := range svcList.Items {
		if svc.Spec.Type == api.ServiceTypeNetworkProvider {
			key, err := keyFunc(svc)
			if err != nil {
				glog.Errorf("Unable to get key for svc %s", svc.Name)
				continue
			}
			e.queue.Add(key)
		}
	}

	endpointList, err := e.client.Endpoints(api.NamespaceAll).List(labels.Everything())
	if err != nil {
		glog.Errorf("Unable to list endpoints: %v", err)
		return
	}

	for _, ep := range endpointList.Items {
		e.addEndpoint(&ep)
	}
}

// When an endpoint is added, figure out its service and enqueue it
func (e *NetworkController) addEndpoint(obj interface{}) {
	if ep, ok := obj.(*api.Endpoints); ok {
		svc, err := e.client.Services(ep.Namespace).Get(ep.Name)
		if err != nil {
			glog.Errorf("NetworkController: service %s can not be found: %v", ep.Name, err)
		} else {
			e.enqueueService(svc)
		}
	}

	glog.V(4).Infof("NetworkController: endpoint %v added", obj)
}

// When an endpoint is updated, figure out its service and enqueue it
func (e *NetworkController) updateEndpoint(old, cur interface{}) {
	oldEP := old.(*api.Endpoints)
	newEP := cur.(*api.Endpoints)

	if api.Semantic.DeepEqual(oldEP.Subsets, newEP.Subsets) {
		return
	}

	e.addEndpoint(cur)

	glog.Infof("NetworkController: endpoint %v updated to %v", old, cur)
}

// When an endpoint is deleted, enqueue the services with same name
func (e *NetworkController) deleteEndpoint(obj interface{}) {
	if _, ok := obj.(*api.Endpoints); ok {
		e.addEndpoint(obj)
	}

	key, err := keyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", obj, err)
	}

	glog.V(4).Infof("NetworkController: endpoint %v deleted", key)
}
