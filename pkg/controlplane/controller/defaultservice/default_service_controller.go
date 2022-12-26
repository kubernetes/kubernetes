/*
Copyright 2022 The Kubernetes Authors.

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
	"reflect"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/storage"
	corev1informers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controlplane/reconcilers"
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
	serviceInformer := corev1informers.NewFilteredServiceInformer(client, metav1.NamespaceDefault, 12*time.Hour,
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
		func(options *metav1.ListOptions) {
			options.FieldSelector = fields.OneTermEqualSelector("metadata.name", serviceName).String()
		})

	c.serviceInformer = serviceInformer
	c.serviceLister = corev1listers.NewServiceLister(serviceInformer.GetIndexer())
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
	serviceLister   corev1listers.ServiceLister
	servicesSynced  cache.InformerSynced

	endpointReconciler reconcilers.EndpointReconciler
	endpointInterval   time.Duration

	stopOnce sync.Once
}

// Run will not return until ctx.Done() is closed.
func (c *Controller) start(stopCh <-chan struct{}) {
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
		return
	}

	// wait to publish the endpoints until the process is ready so we don't receive traffic
	wait.PollImmediateUntil(100*time.Millisecond, func() (bool, error) {
		var code int
		c.client.CoreV1().RESTClient().Get().AbsPath("/healthz").Do(context.Background()).StatusCode(&code)
		return code == http.StatusOK, nil
	}, stopCh)

	wait.NonSlidingUntil(func() {
		if err := c.sync(); err != nil {
			utilruntime.HandleError(fmt.Errorf("unable to sync kubernetes.default Service: %v", err))
		}
	}, c.endpointInterval, stopCh)

	c.stopOnce.Do(c.stop)
}

// Stop cleans up this API server's Endpoints reconciliation leases, so that another API server can take over more quickly.
func (c *Controller) stop() {
	endpointPorts := createEndpointPortSpec(c.endpointPort, "https")
	finishedReconciling := make(chan struct{})
	go func() {
		defer close(finishedReconciling)
		klog.Infof("Shutting down kubernetes Service endpoint reconciler")
		c.endpointReconciler.StopReconciling()
		if err := c.endpointReconciler.RemoveEndpoints(serviceName, c.endpointIP, endpointPorts); err != nil {
			klog.Errorf("Unable to remove endpoints from kubernetes Service: %v", err)
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

// sync attempts to update the default Kube service and the endpoint records
func (c *Controller) sync() error {
	if err := c.createNamespaceIfNeeded(); err != nil {
		return err
	}

	if err := c.createOrUpdateMasterServiceIfNeeded(); err != nil {
		return err
	}
	endpointPorts := createEndpointPortSpec(c.endpointPort, "https")
	if err := c.endpointReconciler.ReconcileEndpoints(serviceName, c.endpointIP, endpointPorts, true); err != nil {
		return err
	}
	return nil
}

// createPortAndServiceSpec creates an array of service ports.
// If the NodePort value is 0, just the servicePort is used, otherwise, a node port is exposed.
func createPortAndServiceSpec(servicePort int, targetServicePort int, nodePort int, servicePortName string) ([]v1.ServicePort, v1.ServiceType) {
	// Use the Cluster IP type for the service port if NodePort isn't provided.
	// Otherwise, we will be binding the API server service to a NodePort.
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

func (c *Controller) createNamespaceIfNeeded() error {
	if _, err := c.client.CoreV1().Namespaces().Get(context.TODO(), metav1.NamespaceDefault, metav1.GetOptions{}); err == nil {
		// the namespace already exists
		return nil
	}
	newNs := &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name:      metav1.NamespaceDefault,
			Namespace: "",
		},
	}
	_, err := c.client.CoreV1().Namespaces().Create(context.TODO(), newNs, metav1.CreateOptions{})
	if err != nil && errors.IsAlreadyExists(err) {
		err = nil
	}
	return err
}

// createOrUpdateMasterServiceIfNeeded will create the specified service if it doesn't already exist.
// TODO: rename to avoid use of term "master"
func (c *Controller) createOrUpdateMasterServiceIfNeeded() error {
	servicePorts, serviceType := createPortAndServiceSpec(c.servicePort, c.endpointPort, c.serviceNodePort, "https")
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
			ClusterIP:       c.serviceIP.String(),
			IPFamilyPolicy:  &singleStack,
			SessionAffinity: v1.ServiceAffinityNone,
			Type:            serviceType,
		},
	}

	s, err := c.serviceLister.Services(metav1.NamespaceDefault).Get(serviceName)
	// Service does not exist
	if err != nil && errors.IsNotFound(err) {
		_, err := c.client.CoreV1().Services(metav1.NamespaceDefault).Create(context.TODO(), svc, metav1.CreateOptions{})
		return err
	}
	// unexpected error
	if err != nil {
		return err
	}
	// Service exists; reconcile if necessary
	if serviceNeedsUpdate(svc, s) {
		klog.Warningf("Resetting API server Service %q to %#v", serviceName, svc)
		c.eventRecorder.Eventf(svc, v1.EventTypeWarning, "ReconcilingKubernetesDefaultService", "Reconciling Service default/kubernetes")
		_, err := c.client.CoreV1().Services(metav1.NamespaceDefault).Update(context.TODO(), svc, metav1.UpdateOptions{})
		return err
	}
	return nil
}

// checkServicePort check that the Service Port
func serviceNeedsUpdate(s1, s2 *v1.Service) bool {
	if s1.Name != s2.Name {
		return true
	}

	if s1.Namespace != s2.Namespace {
		return true
	}

	if s1.Spec.ClusterIP != s2.Spec.ClusterIP {
		return true
	}

	if s1.Spec.Type != s2.Spec.Type {
		return true
	}
	if s1.Spec.SessionAffinity != s2.Spec.SessionAffinity {
		return true
	}

	if len(s1.Spec.Ports) != len(s2.Spec.Ports) {
		return true
	}
	for i, port := range s1.Spec.Ports {
		if port != s2.Spec.Ports[i] {
			return true
		}
	}
	if !reflect.DeepEqual(s1.Labels, s2.Labels) {
		return true
	}

	if !reflect.DeepEqual(s1.Spec.Selector, s2.Spec.Selector) {
		return true
	}

	return false
}
