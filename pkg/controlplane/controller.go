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

package controlplane

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/storage"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	eventsv1client "k8s.io/client-go/kubernetes/typed/events/v1"
	"k8s.io/client-go/rest"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controlplane/reconcilers"
	"k8s.io/kubernetes/pkg/registry/core/rangeallocation"
	corerest "k8s.io/kubernetes/pkg/registry/core/rest"
	servicecontroller "k8s.io/kubernetes/pkg/registry/core/service/ipallocator/controller"
	portallocatorcontroller "k8s.io/kubernetes/pkg/registry/core/service/portallocator/controller"
	"k8s.io/kubernetes/pkg/util/async"
	netutils "k8s.io/utils/net"
)

const (
	kubernetesServiceName = "kubernetes"
)

// Controller is the controller manager for the core bootstrap Kubernetes
// controller loops, which manage creating the "kubernetes" service, the
// "default", "kube-system" and "kube-public" namespaces, and provide the IP
// repair check on service IPs
type Controller struct {
	ServiceClient   corev1client.ServicesGetter
	NamespaceClient corev1client.NamespacesGetter
	EventClient     eventsv1client.EventsV1Interface
	readyzClient    rest.Interface

	ServiceClusterIPRegistry          rangeallocation.RangeRegistry
	ServiceClusterIPRange             net.IPNet
	SecondaryServiceClusterIPRegistry rangeallocation.RangeRegistry
	SecondaryServiceClusterIPRange    net.IPNet

	ServiceClusterIPInterval time.Duration

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
	PublicServicePort         int
	KubernetesServiceNodePort int

	runner *async.Runner
}

// NewBootstrapController returns a controller for watching the core capabilities of the master
func (c *completedConfig) NewBootstrapController(legacyRESTStorage corerest.LegacyRESTStorage, serviceClient corev1client.ServicesGetter, nsClient corev1client.NamespacesGetter, eventClient eventsv1client.EventsV1Interface, readyzClient rest.Interface) (*Controller, error) {
	_, publicServicePort, err := c.GenericConfig.SecureServing.HostPort()
	if err != nil {
		return nil, fmt.Errorf("failed to get listener address: %w", err)
	}

	// The "kubernetes.default" Service is SingleStack based on the configured ServiceIPRange.
	// If the bootstrap controller reconcile the kubernetes.default Service and Endpoints, it must
	// guarantee that the Service ClusterIP and the associated Endpoints have the same IP family, or
	// it will not work for clients because of the IP family mismatch.
	// TODO: revisit for dual-stack https://github.com/kubernetes/enhancements/issues/2438
	if c.ExtraConfig.EndpointReconcilerType != reconcilers.NoneEndpointReconcilerType {
		if netutils.IsIPv4CIDR(&c.ExtraConfig.ServiceIPRange) != netutils.IsIPv4(c.GenericConfig.PublicAddress) {
			return nil, fmt.Errorf("service IP family %q must match public address family %q", c.ExtraConfig.ServiceIPRange.String(), c.GenericConfig.PublicAddress.String())
		}
	}

	systemNamespaces := []string{metav1.NamespaceSystem, metav1.NamespacePublic, corev1.NamespaceNodeLease}

	return &Controller{
		ServiceClient:   serviceClient,
		NamespaceClient: nsClient,
		EventClient:     eventClient,
		readyzClient:    readyzClient,

		EndpointReconciler: c.ExtraConfig.EndpointReconcilerConfig.Reconciler,
		EndpointInterval:   c.ExtraConfig.EndpointReconcilerConfig.Interval,

		SystemNamespaces:         systemNamespaces,
		SystemNamespacesInterval: 1 * time.Minute,

		ServiceClusterIPRegistry:          legacyRESTStorage.ServiceClusterIPAllocator,
		ServiceClusterIPRange:             c.ExtraConfig.ServiceIPRange,
		SecondaryServiceClusterIPRegistry: legacyRESTStorage.SecondaryServiceClusterIPAllocator,
		SecondaryServiceClusterIPRange:    c.ExtraConfig.SecondaryServiceIPRange,

		ServiceClusterIPInterval: c.ExtraConfig.RepairServicesInterval,

		ServiceNodePortRegistry: legacyRESTStorage.ServiceNodePortAllocator,
		ServiceNodePortRange:    c.ExtraConfig.ServiceNodePortRange,
		ServiceNodePortInterval: c.ExtraConfig.RepairServicesInterval,

		PublicIP: c.GenericConfig.PublicAddress,

		ServiceIP:                 c.ExtraConfig.APIServerServiceIP,
		ServicePort:               c.ExtraConfig.APIServerServicePort,
		PublicServicePort:         publicServicePort,
		KubernetesServiceNodePort: c.ExtraConfig.KubernetesServiceNodePort,
	}, nil
}

// PostStartHook initiates the core controller loops that must exist for bootstrapping.
func (c *Controller) PostStartHook(hookContext genericapiserver.PostStartHookContext) error {
	c.Start()
	return nil
}

// PreShutdownHook triggers the actions needed to shut down the API Server cleanly.
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

	// Reconcile during first run removing itself until server is ready.
	endpointPorts := createEndpointPortSpec(c.PublicServicePort, "https")
	if err := c.EndpointReconciler.RemoveEndpoints(kubernetesServiceName, c.PublicIP, endpointPorts); err == nil {
		klog.Error("Found stale data, removed previous endpoints on kubernetes service, apiserver didn't exit successfully previously")
	} else if !storage.IsNotFound(err) {
		klog.Errorf("Error removing old endpoints from kubernetes service: %v", err)
	}

	repairClusterIPs := servicecontroller.NewRepair(c.ServiceClusterIPInterval, c.ServiceClient, c.EventClient, &c.ServiceClusterIPRange, c.ServiceClusterIPRegistry, &c.SecondaryServiceClusterIPRange, c.SecondaryServiceClusterIPRegistry)
	repairNodePorts := portallocatorcontroller.NewRepair(c.ServiceNodePortInterval, c.ServiceClient, c.EventClient, c.ServiceNodePortRange, c.ServiceNodePortRegistry)

	// We start both repairClusterIPs and repairNodePorts to ensure repair
	// loops of ClusterIPs and NodePorts.
	// We run both repair loops using RunUntil public interface.
	// However, we want to fail liveness/readiness until the first
	// successful repair loop, so we basically pass appropriate
	// callbacks to RunUtil methods.
	// Additionally, we ensure that we don't wait for it for longer
	// than 1 minute for backward compatibility of failing the whole
	// apiserver if we can't repair them.
	wg := sync.WaitGroup{}
	wg.Add(2)

	runRepairClusterIPs := func(stopCh chan struct{}) {
		repairClusterIPs.RunUntil(wg.Done, stopCh)
	}
	runRepairNodePorts := func(stopCh chan struct{}) {
		repairNodePorts.RunUntil(wg.Done, stopCh)
	}

	c.runner = async.NewRunner(c.RunKubernetesNamespaces, c.RunKubernetesService, runRepairClusterIPs, runRepairNodePorts)
	c.runner.Start()

	// For backward compatibility, we ensure that if we never are able
	// to repair clusterIPs and/or nodeports, we not only fail the liveness
	// and/or readiness, but also explicitly fail.
	done := make(chan struct{})
	go func() {
		defer close(done)
		wg.Wait()
	}()
	select {
	case <-done:
	case <-time.After(time.Minute):
		klog.Fatalf("Unable to perform initial IP and Port allocation check")
	}
}

// Stop cleans up this API Servers endpoint reconciliation leases so another master can take over more quickly.
func (c *Controller) Stop() {
	if c.runner != nil {
		c.runner.Stop()
	}
	endpointPorts := createEndpointPortSpec(c.PublicServicePort, "https")
	finishedReconciling := make(chan struct{})
	go func() {
		defer close(finishedReconciling)
		klog.Infof("Shutting down kubernetes service endpoint reconciler")
		c.EndpointReconciler.StopReconciling()
		if err := c.EndpointReconciler.RemoveEndpoints(kubernetesServiceName, c.PublicIP, endpointPorts); err != nil {
			klog.Errorf("Unable to remove endpoints from kubernetes service: %v", err)
		}
		c.EndpointReconciler.Destroy()
	}()

	select {
	case <-finishedReconciling:
		// done
	case <-time.After(2 * c.EndpointInterval):
		// don't block server shutdown forever if we can't reach etcd to remove ourselves
		klog.Warning("RemoveEndpoints() timed out")
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
	// wait until process is ready
	wait.PollImmediateUntil(100*time.Millisecond, func() (bool, error) {
		var code int
		c.readyzClient.Get().AbsPath("/readyz").Do(context.TODO()).StatusCode(&code)
		return code == http.StatusOK, nil
	}, ch)

	wait.NonSlidingUntil(func() {
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

	servicePorts, serviceType := createPortAndServiceSpec(c.ServicePort, c.PublicServicePort, c.KubernetesServiceNodePort, "https")
	if err := c.CreateOrUpdateMasterServiceIfNeeded(kubernetesServiceName, c.ServiceIP, servicePorts, serviceType, reconcile); err != nil {
		return err
	}
	endpointPorts := createEndpointPortSpec(c.PublicServicePort, "https")
	if err := c.EndpointReconciler.ReconcileEndpoints(kubernetesServiceName, c.PublicIP, endpointPorts, reconcile); err != nil {
		return err
	}
	return nil
}

// createPortAndServiceSpec creates an array of service ports.
// If the NodePort value is 0, just the servicePort is used, otherwise, a node port is exposed.
func createPortAndServiceSpec(servicePort int, targetServicePort int, nodePort int, servicePortName string) ([]corev1.ServicePort, corev1.ServiceType) {
	// Use the Cluster IP type for the service port if NodePort isn't provided.
	// Otherwise, we will be binding the master service to a NodePort.
	servicePorts := []corev1.ServicePort{{
		Protocol:   corev1.ProtocolTCP,
		Port:       int32(servicePort),
		Name:       servicePortName,
		TargetPort: intstr.FromInt(targetServicePort),
	}}
	serviceType := corev1.ServiceTypeClusterIP
	if nodePort > 0 {
		servicePorts[0].NodePort = int32(nodePort)
		serviceType = corev1.ServiceTypeNodePort
	}
	return servicePorts, serviceType
}

// createEndpointPortSpec creates the endpoint ports
func createEndpointPortSpec(endpointPort int, endpointPortName string) []corev1.EndpointPort {
	return []corev1.EndpointPort{{
		Protocol: corev1.ProtocolTCP,
		Port:     int32(endpointPort),
		Name:     endpointPortName,
	}}
}

// CreateOrUpdateMasterServiceIfNeeded will create the specified service if it
// doesn't already exist.
func (c *Controller) CreateOrUpdateMasterServiceIfNeeded(serviceName string, serviceIP net.IP, servicePorts []corev1.ServicePort, serviceType corev1.ServiceType, reconcile bool) error {
	if s, err := c.ServiceClient.Services(metav1.NamespaceDefault).Get(context.TODO(), serviceName, metav1.GetOptions{}); err == nil {
		// The service already exists.
		if reconcile {
			if svc, updated := getMasterServiceUpdateIfNeeded(s, servicePorts, serviceType); updated {
				klog.Warningf("Resetting master service %q to %#v", serviceName, svc)
				_, err := c.ServiceClient.Services(metav1.NamespaceDefault).Update(context.TODO(), svc, metav1.UpdateOptions{})
				return err
			}
		}
		return nil
	}
	singleStack := corev1.IPFamilyPolicySingleStack
	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      serviceName,
			Namespace: metav1.NamespaceDefault,
			Labels:    map[string]string{"provider": "kubernetes", "component": "apiserver"},
		},
		Spec: corev1.ServiceSpec{
			Ports: servicePorts,
			// maintained by this code, not by the pod selector
			Selector:        nil,
			ClusterIP:       serviceIP.String(),
			IPFamilyPolicy:  &singleStack,
			SessionAffinity: corev1.ServiceAffinityNone,
			Type:            serviceType,
		},
	}

	_, err := c.ServiceClient.Services(metav1.NamespaceDefault).Create(context.TODO(), svc, metav1.CreateOptions{})
	if errors.IsAlreadyExists(err) {
		return c.CreateOrUpdateMasterServiceIfNeeded(serviceName, serviceIP, servicePorts, serviceType, reconcile)
	}
	return err
}

// getMasterServiceUpdateIfNeeded sets service attributes for the given apiserver service.
func getMasterServiceUpdateIfNeeded(svc *corev1.Service, servicePorts []corev1.ServicePort, serviceType corev1.ServiceType) (s *corev1.Service, updated bool) {
	// Determine if the service is in the format we expect
	// (servicePorts are present and service type matches)
	formatCorrect := checkServiceFormat(svc, servicePorts, serviceType)
	if formatCorrect {
		return svc, false
	}
	svc.Spec.Ports = servicePorts
	svc.Spec.Type = serviceType
	return svc, true
}

// Determine if the service is in the correct format
// getMasterServiceUpdateIfNeeded expects (servicePorts are correct
// and service type matches).
func checkServiceFormat(s *corev1.Service, ports []corev1.ServicePort, serviceType corev1.ServiceType) (formatCorrect bool) {
	if s.Spec.Type != serviceType {
		return false
	}
	if len(ports) != len(s.Spec.Ports) {
		return false
	}
	for i, port := range ports {
		if port != s.Spec.Ports[i] {
			return false
		}
	}
	return true
}
