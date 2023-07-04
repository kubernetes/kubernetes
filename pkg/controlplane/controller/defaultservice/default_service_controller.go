/*
Copyright 2023 The Kubernetes Authors.

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

package defaultservice

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/storage"
	v1informers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	v1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controlplane/reconcilers"
	"k8s.io/kubernetes/pkg/util/async"
	netutils "k8s.io/utils/net"
)

const (
	controllerName = "default-service-controller"
	serviceName    = "kubernetes"
)

// New returns a new *Controller.
func NewController(
	serviceIP net.IP,
	servicePort int,
	serviceNodePort int,
	endpointIP net.IP,
	endpointPort int,
	endpointsReconciler reconcilers.EndpointReconciler,
	endpointInterval time.Duration,
	endpointReconcilerType reconcilers.Type,
	client clientset.Interface,
) (*Controller, error) {
	// The "kubernetes.default" Service is SingleStack based on the configured ServiceIPRange.
	// If the bootstrap controller reconcile the kubernetes.default Service and Endpoints, it must
	// guarantee that the Service ClusterIP and the associated Endpoints have the same IP family, or
	// it will not work for clients because of the IP family mismatch.
	// TODO: revisit for dual-stack https://github.com/kubernetes/enhancements/issues/2438
	if endpointReconcilerType != reconcilers.NoneEndpointReconcilerType {
		if netutils.IsIPv4(serviceIP) != netutils.IsIPv4(endpointIP) {
			return nil, fmt.Errorf("service IP family %q must match endpoint (apiserver) address family %q", serviceIP.String(), endpointIP.String())
		}
	}

	broadcaster := record.NewBroadcaster()
	recorder := broadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: controllerName})

	c := &Controller{
		endpointReconciler: endpointsReconciler,
		serviceIP:          serviceIP,
		servicePort:        servicePort,
		serviceNodePort:    serviceNodePort,
		endpointIP:         endpointIP,
		endpointPort:       endpointPort,
		client:             client,
		endpointInterval:   endpointInterval,
	}
	// we construct our own informer because we need such a small subset of the information available.
	// The kubernetes.default service
	serviceInformer := v1informers.NewFilteredServiceInformer(client, metav1.NamespaceDefault, 12*time.Hour,
		cache.Indexers{},
		func(options *metav1.ListOptions) {
			options.FieldSelector = fields.OneTermEqualSelector("metadata.name", serviceName).String()
		})

	c.serviceInformer = serviceInformer
	c.serviceLister = v1listers.NewServiceLister(serviceInformer.GetIndexer())
	c.servicesSynced = serviceInformer.HasSynced

	c.eventBroadcaster = broadcaster
	c.eventRecorder = recorder

	return c, nil
}

// Controller manages selector-based service ipAddress.
type Controller struct {
	client           clientset.Interface
	eventBroadcaster record.EventBroadcaster
	eventRecorder    record.EventRecorder

	// service frontend
	serviceIP       net.IP
	servicePort     int
	serviceNodePort int

	// published apiserver IP and Port
	endpointIP   net.IP
	endpointPort int

	serviceInformer cache.SharedIndexInformer
	serviceLister   v1listers.ServiceLister
	servicesSynced  cache.InformerSynced

	endpointReconciler reconcilers.EndpointReconciler
	endpointInterval   time.Duration

	runner *async.Runner

	stopOnce sync.Once
}

// Run will not return until ctx.Done() is closed.
func (c *Controller) start(stopCh <-chan struct{}) {
	if c.runner != nil {
		return
	}
	defer utilruntime.HandleCrash()

	c.eventBroadcaster.StartStructuredLogging(0)
	klog.Infof("Sending events to API server.")
	c.eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: c.client.CoreV1().Events("")})
	defer c.eventBroadcaster.Shutdown()

	klog.Infof("Starting %s", controllerName)
	defer klog.Infof("Shutting down %s", controllerName)

	// Reconcile during first run removing itself until server is ready.
	endpointPorts := createEndpointPortSpec(c.endpointPort, "https")
	if err := c.endpointReconciler.RemoveEndpoints(serviceName, c.endpointIP, endpointPorts); err == nil {
		klog.Error("Found stale data, removed previous endpoints for \"kubernetes\" Service. API server didn't exit successfully previously.")
	} else if !storage.IsNotFound(err) {
		klog.Errorf("Error removing old endpoints from \"kubernetes\" Service: %v", err)
	}

	go c.serviceInformer.Run(stopCh)

	if !cache.WaitForNamedCacheSync(controllerName, stopCh, c.servicesSynced) {
		klog.Error("Error waiting for syncing \"kubernetes\" Service")
		return
	}

	c.runner = async.NewRunner(c.RunKubernetesService)
	c.runner.Start()
}

// Stop cleans up this API server's Endpoints reconciliation leases, so that another API server can take over more quickly.
func (c *Controller) stop() {
	if c.runner != nil {
		c.runner.Stop()
	}
	endpointPorts := createEndpointPortSpec(c.endpointPort, "https")
	finishedReconciling := make(chan struct{})
	go func() {
		defer close(finishedReconciling)
		klog.Infof("Shutting down kubernetes service endpoint reconciler")
		c.endpointReconciler.StopReconciling()
		if err := c.endpointReconciler.RemoveEndpoints(serviceName, c.endpointIP, endpointPorts); err != nil {
			klog.Errorf("Unable to remove endpoints from kubernetes service: %v", err)
		}
		c.endpointReconciler.Destroy()
	}()

	select {
	case <-finishedReconciling:
		// done
	case <-time.After(2 * c.endpointInterval):
		// don't block server shutdown forever if we can't reach etcd to remove ourselves
		klog.Warning("RemoveEndpoints() timed out")
	}
}

// PostStartHook initiates the core controller loops that must exist for bootstrapping.
func (c *Controller) PostStartHook(hookContext genericapiserver.PostStartHookContext) error {
	go c.start(hookContext.StopCh)
	return nil
}

// PreShutdownHook triggers the actions needed to shut down the API Server cleanly.
func (c *Controller) PreShutdownHook() error {
	c.stopOnce.Do(c.stop)
	return nil
}

// RunKubernetesService periodically updates the kubernetes service
func (c *Controller) RunKubernetesService(ch chan struct{}) {
	// wait until process is ready
	wait.PollImmediateUntil(100*time.Millisecond, func() (bool, error) {
		var code int
		c.client.CoreV1().RESTClient().Get().AbsPath("/readyz").Do(context.TODO()).StatusCode(&code)
		return code == http.StatusOK, nil
	}, ch)

	wait.NonSlidingUntil(func() {
		// Service definition is not reconciled after first
		// run, ports and type will be corrected only during
		// start.
		if err := c.UpdateKubernetesService(false); err != nil {
			runtime.HandleError(fmt.Errorf("unable to sync kubernetes service: %v", err))
		}
	}, c.endpointInterval, ch)
}

// UpdateKubernetesService attempts to update the default Kube service.
func (c *Controller) UpdateKubernetesService(reconcile bool) error {
	// Update service & endpoint records.
	// TODO: when it becomes possible to change this stuff,
	// stop polling and start watching.
	// TODO: add endpoints of all replicas, not just the elected master.
	if err := createNamespaceIfNeeded(c.client.CoreV1(), metav1.NamespaceDefault); err != nil {
		return err
	}

	servicePorts, serviceType := createPortAndServiceSpec(c.servicePort, c.servicePort, c.serviceNodePort, "https")
	if err := c.createOrUpdateMasterServiceIfNeeded(serviceName, c.serviceIP, servicePorts, serviceType, reconcile); err != nil {
		return err
	}
	endpointPorts := createEndpointPortSpec(c.endpointPort, "https")
	if err := c.endpointReconciler.ReconcileEndpoints(serviceName, c.endpointIP, endpointPorts, reconcile); err != nil {
		return err
	}
	return nil
}

// createPortAndServiceSpec creates an array of service ports.
// If the NodePort value is 0, just the servicePort is used, otherwise, a node port is exposed.
func createPortAndServiceSpec(servicePort int, targetServicePort int, nodePort int, servicePortName string) ([]v1.ServicePort, v1.ServiceType) {
	// Use the Cluster IP type for the service port if NodePort isn't provided.
	// Otherwise, we will be binding the master service to a NodePort.
	servicePorts := []v1.ServicePort{{
		Protocol:   v1.ProtocolTCP,
		Port:       int32(servicePort),
		Name:       servicePortName,
		TargetPort: intstr.FromInt(targetServicePort),
	}}
	serviceType := v1.ServiceTypeClusterIP
	if nodePort > 0 {
		servicePorts[0].NodePort = int32(nodePort)
		serviceType = v1.ServiceTypeNodePort
	}
	return servicePorts, serviceType
}

// createEndpointPortSpec creates the endpoint ports
func createEndpointPortSpec(endpointPort int, endpointPortName string) []v1.EndpointPort {
	return []v1.EndpointPort{{
		Protocol: v1.ProtocolTCP,
		Port:     int32(endpointPort),
		Name:     endpointPortName,
	}}
}

// CreateOrUpdateMasterServiceIfNeeded will create the specified service if it
// doesn't already exist.
func (c *Controller) createOrUpdateMasterServiceIfNeeded(serviceName string, serviceIP net.IP, servicePorts []v1.ServicePort, serviceType v1.ServiceType, reconcile bool) error {
	if s, err := c.serviceLister.Services(metav1.NamespaceDefault).Get(serviceName); err == nil {
		// The service already exists.
		if reconcile {
			if svc, updated := getMasterServiceUpdateIfNeeded(s, servicePorts, serviceType); updated {
				klog.Warningf("Resetting master service %q to %#v", serviceName, svc)
				_, err := c.client.CoreV1().Services(metav1.NamespaceDefault).Update(context.TODO(), svc, metav1.UpdateOptions{})
				return err
			}
		}
		return nil
	}
	singleStack := v1.IPFamilyPolicySingleStack
	svc := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      serviceName,
			Namespace: metav1.NamespaceDefault,
			Labels:    map[string]string{"provider": "kubernetes", "component": "apiserver"},
		},
		Spec: v1.ServiceSpec{
			Ports: servicePorts,
			// maintained by this code, not by the pod selector
			Selector:        nil,
			ClusterIP:       serviceIP.String(),
			IPFamilyPolicy:  &singleStack,
			SessionAffinity: v1.ServiceAffinityNone,
			Type:            serviceType,
		},
	}

	_, err := c.client.CoreV1().Services(metav1.NamespaceDefault).Create(context.TODO(), svc, metav1.CreateOptions{})
	if errors.IsAlreadyExists(err) {
		return c.createOrUpdateMasterServiceIfNeeded(serviceName, serviceIP, servicePorts, serviceType, reconcile)
	}
	return err
}

// getMasterServiceUpdateIfNeeded sets service attributes for the given apiserver service.
func getMasterServiceUpdateIfNeeded(svc *v1.Service, servicePorts []v1.ServicePort, serviceType v1.ServiceType) (s *v1.Service, updated bool) {
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
func checkServiceFormat(s *v1.Service, ports []v1.ServicePort, serviceType v1.ServiceType) (formatCorrect bool) {
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
