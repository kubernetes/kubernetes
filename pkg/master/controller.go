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

package master

import (
	"fmt"
	"net"
	"time"

	"github.com/golang/glog"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	genericapiserver "k8s.io/apiserver/pkg/server"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	api "k8s.io/kubernetes/pkg/apis/core"
	coreclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/master/reconcilers"
	"k8s.io/kubernetes/pkg/registry/core/rangeallocation"
	corerest "k8s.io/kubernetes/pkg/registry/core/rest"
	servicecontroller "k8s.io/kubernetes/pkg/registry/core/service/ipallocator/controller"
	portallocatorcontroller "k8s.io/kubernetes/pkg/registry/core/service/portallocator/controller"
	"k8s.io/kubernetes/pkg/util/async"
)

const kubernetesServiceName = "kubernetes"

// Controller is the controller manager for the core bootstrap Kubernetes
// controller loops, which manage creating the "kubernetes" service, the
// "default", "kube-system" and "kube-public" namespaces, and provide the IP
// repair check on service IPs
type Controller struct {
	ServiceClient   coreclient.ServicesGetter
	NamespaceClient coreclient.NamespacesGetter
	EventClient     coreclient.EventsGetter

	ServiceClusterIPRegistry rangeallocation.RangeRegistry
	ServiceClusterIPInterval time.Duration
	ServiceClusterIPRange    net.IPNet

	ServiceNodePortRegistry rangeallocation.RangeRegistry
	ServiceNodePortInterval time.Duration
	ServiceNodePortRange    utilnet.PortRange

	EndpointReconciler reconcilers.EndpointReconciler
	EndpointInterval   time.Duration

	SystemNamespaces         []string
	SystemNamespacesInterval time.Duration

	PublicIP net.IP

	// ServiceIP indicates where the kubernetes service will live.  It may not be nil.
	ServiceIP                 net.IP
	ServicePort               int
	ExtraServicePorts         []api.ServicePort
	ExtraEndpointPorts        []api.EndpointPort
	PublicServicePort         int
	KubernetesServiceNodePort int

	runner *async.Runner
}

// NewBootstrapController returns a controller for watching the core capabilities of the master
func (c *completedConfig) NewBootstrapController(legacyRESTStorage corerest.LegacyRESTStorage, serviceClient coreclient.ServicesGetter, nsClient coreclient.NamespacesGetter, eventClient coreclient.EventsGetter) *Controller {
	_, publicServicePort, err := c.GenericConfig.SecureServing.HostPort()
	if err != nil {
		glog.Fatalf("failed to get listener address: %v", err)
	}

	systemNamespaces := []string{metav1.NamespaceSystem, metav1.NamespacePublic}
	if utilfeature.DefaultFeatureGate.Enabled(features.NodeLease) {
		systemNamespaces = append(systemNamespaces, corev1.NamespaceNodeLease)
	}

	return &Controller{
		ServiceClient:   serviceClient,
		NamespaceClient: nsClient,
		EventClient:     eventClient,

		EndpointReconciler: c.ExtraConfig.EndpointReconcilerConfig.Reconciler,
		EndpointInterval:   c.ExtraConfig.EndpointReconcilerConfig.Interval,

		SystemNamespaces:         systemNamespaces,
		SystemNamespacesInterval: 1 * time.Minute,

		ServiceClusterIPRegistry: legacyRESTStorage.ServiceClusterIPAllocator,
		ServiceClusterIPRange:    c.ExtraConfig.ServiceIPRange,
		ServiceClusterIPInterval: 3 * time.Minute,

		ServiceNodePortRegistry: legacyRESTStorage.ServiceNodePortAllocator,
		ServiceNodePortRange:    c.ExtraConfig.ServiceNodePortRange,
		ServiceNodePortInterval: 3 * time.Minute,

		PublicIP: c.GenericConfig.PublicAddress,

		ServiceIP:                 c.ExtraConfig.APIServerServiceIP,
		ServicePort:               c.ExtraConfig.APIServerServicePort,
		ExtraServicePorts:         c.ExtraConfig.ExtraServicePorts,
		ExtraEndpointPorts:        c.ExtraConfig.ExtraEndpointPorts,
		PublicServicePort:         publicServicePort,
		KubernetesServiceNodePort: c.ExtraConfig.KubernetesServiceNodePort,
	}
}

func (c *Controller) PostStartHook(hookContext genericapiserver.PostStartHookContext) error {
	c.Start()
	return nil
}

func (c *Controller) PreShutdownHook() error {
	c.Stop()
	return nil
}

// Start begins the core controller loops that must exist for bootstrapping
// a cluster.
func (c *Controller) Start() {
	if c.runner != nil {
		return
	}

	repairClusterIPs := servicecontroller.NewRepair(c.ServiceClusterIPInterval, c.ServiceClient, c.EventClient, &c.ServiceClusterIPRange, c.ServiceClusterIPRegistry)
	repairNodePorts := portallocatorcontroller.NewRepair(c.ServiceNodePortInterval, c.ServiceClient, c.EventClient, c.ServiceNodePortRange, c.ServiceNodePortRegistry)

	// run all of the controllers once prior to returning from Start.
	if err := repairClusterIPs.RunOnce(); err != nil {
		// If we fail to repair cluster IPs apiserver is useless. We should restart and retry.
		glog.Fatalf("Unable to perform initial IP allocation check: %v", err)
	}
	if err := repairNodePorts.RunOnce(); err != nil {
		// If we fail to repair node ports apiserver is useless. We should restart and retry.
		glog.Fatalf("Unable to perform initial service nodePort check: %v", err)
	}
	// Service definition is reconciled during first run to correct port and type per expectations.
	if err := c.UpdateKubernetesService(true); err != nil {
		glog.Errorf("Unable to perform initial Kubernetes service initialization: %v", err)
	}

	c.runner = async.NewRunner(c.RunKubernetesNamespaces, c.RunKubernetesService, repairClusterIPs.RunUntil, repairNodePorts.RunUntil)
	c.runner.Start()
}

func (c *Controller) Stop() {
	if c.runner != nil {
		c.runner.Stop()
	}
	endpointPorts := createEndpointPortSpec(c.PublicServicePort, "https", c.ExtraEndpointPorts)
	finishedReconciling := make(chan struct{})
	go func() {
		defer close(finishedReconciling)
		glog.Infof("Shutting down kubernetes service endpoint reconciler")
		if err := c.EndpointReconciler.StopReconciling("kubernetes", c.PublicIP, endpointPorts); err != nil {
			glog.Error(err)
		}
	}()

	select {
	case <-finishedReconciling:
		// done
	case <-time.After(2 * c.EndpointInterval):
		// don't block server shutdown forever if we can't reach etcd to remove ourselves
		glog.Warning("StopReconciling() timed out")
	}
}

// RunKubernetesNamespaces periodically makes sure that all internal namespaces exist
func (c *Controller) RunKubernetesNamespaces(ch chan struct{}) {
	wait.Until(func() {
		// Loop the system namespace list, and create them if they do not exist
		for _, ns := range c.SystemNamespaces {
			if err := createNamespaceIfNeeded(c.NamespaceClient, ns); err != nil {
				runtime.HandleError(fmt.Errorf("unable to create required kubernetes system namespace %s: %v", ns, err))
			}
		}
	}, c.SystemNamespacesInterval, ch)
}

// RunKubernetesService periodically updates the kubernetes service
func (c *Controller) RunKubernetesService(ch chan struct{}) {
	wait.Until(func() {
		// Service definition is not reconciled after first
		// run, ports and type will be corrected only during
		// start.
		if err := c.UpdateKubernetesService(false); err != nil {
			runtime.HandleError(fmt.Errorf("unable to sync kubernetes service: %v", err))
		}
	}, c.EndpointInterval, ch)
}

// UpdateKubernetesService attempts to update the default Kube service.
func (c *Controller) UpdateKubernetesService(reconcile bool) error {
	// Update service & endpoint records.
	// TODO: when it becomes possible to change this stuff,
	// stop polling and start watching.
	// TODO: add endpoints of all replicas, not just the elected master.
	if err := createNamespaceIfNeeded(c.NamespaceClient, metav1.NamespaceDefault); err != nil {
		return err
	}

	servicePorts, serviceType := createPortAndServiceSpec(c.ServicePort, c.PublicServicePort, c.KubernetesServiceNodePort, "https", c.ExtraServicePorts)
	if err := c.CreateOrUpdateMasterServiceIfNeeded(kubernetesServiceName, c.ServiceIP, servicePorts, serviceType, reconcile); err != nil {
		return err
	}
	endpointPorts := createEndpointPortSpec(c.PublicServicePort, "https", c.ExtraEndpointPorts)
	if err := c.EndpointReconciler.ReconcileEndpoints(kubernetesServiceName, c.PublicIP, endpointPorts, reconcile); err != nil {
		return err
	}
	return nil
}

// createPortAndServiceSpec creates an array of service ports.
// If the NodePort value is 0, just the servicePort is used, otherwise, a node port is exposed.
func createPortAndServiceSpec(servicePort int, targetServicePort int, nodePort int, servicePortName string, extraServicePorts []api.ServicePort) ([]api.ServicePort, api.ServiceType) {
	//Use the Cluster IP type for the service port if NodePort isn't provided.
	//Otherwise, we will be binding the master service to a NodePort.
	servicePorts := []api.ServicePort{{Protocol: api.ProtocolTCP,
		Port:       int32(servicePort),
		Name:       servicePortName,
		TargetPort: intstr.FromInt(targetServicePort)}}
	serviceType := api.ServiceTypeClusterIP
	if nodePort > 0 {
		servicePorts[0].NodePort = int32(nodePort)
		serviceType = api.ServiceTypeNodePort
	}
	if extraServicePorts != nil {
		servicePorts = append(servicePorts, extraServicePorts...)
	}
	return servicePorts, serviceType
}

// createEndpointPortSpec creates an array of endpoint ports
func createEndpointPortSpec(endpointPort int, endpointPortName string, extraEndpointPorts []api.EndpointPort) []api.EndpointPort {
	endpointPorts := []api.EndpointPort{{Protocol: api.ProtocolTCP,
		Port: int32(endpointPort),
		Name: endpointPortName,
	}}
	if extraEndpointPorts != nil {
		endpointPorts = append(endpointPorts, extraEndpointPorts...)
	}
	return endpointPorts
}

// CreateMasterServiceIfNeeded will create the specified service if it
// doesn't already exist.
func (c *Controller) CreateOrUpdateMasterServiceIfNeeded(serviceName string, serviceIP net.IP, servicePorts []api.ServicePort, serviceType api.ServiceType, reconcile bool) error {
	if s, err := c.ServiceClient.Services(metav1.NamespaceDefault).Get(serviceName, metav1.GetOptions{}); err == nil {
		// The service already exists.
		if reconcile {
			if svc, updated := reconcilers.GetMasterServiceUpdateIfNeeded(s, servicePorts, serviceType); updated {
				glog.Warningf("Resetting master service %q to %#v", serviceName, svc)
				_, err := c.ServiceClient.Services(metav1.NamespaceDefault).Update(svc)
				return err
			}
		}
		return nil
	}
	svc := &api.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      serviceName,
			Namespace: metav1.NamespaceDefault,
			Labels:    map[string]string{"provider": "kubernetes", "component": "apiserver"},
		},
		Spec: api.ServiceSpec{
			Ports: servicePorts,
			// maintained by this code, not by the pod selector
			Selector:        nil,
			ClusterIP:       serviceIP.String(),
			SessionAffinity: api.ServiceAffinityNone,
			Type:            serviceType,
		},
	}

	_, err := c.ServiceClient.Services(metav1.NamespaceDefault).Create(svc)
	if errors.IsAlreadyExists(err) {
		return c.CreateOrUpdateMasterServiceIfNeeded(serviceName, serviceIP, servicePorts, serviceType, reconcile)
	}
	return err
}
