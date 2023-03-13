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

package defaultservicecidr

import (
	"context"
	"net"
	"time"

	v1 "k8s.io/api/core/v1"
	networkingapiv1alpha1 "k8s.io/api/networking/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	metav1apply "k8s.io/client-go/applyconfigurations/meta/v1"
	networkingapiv1alpha1apply "k8s.io/client-go/applyconfigurations/networking/v1alpha1"
	networkingv1alpha1informers "k8s.io/client-go/informers/networking/v1alpha1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	networkingv1alpha1listers "k8s.io/client-go/listers/networking/v1alpha1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	netutils "k8s.io/utils/net"
)

const (
	controllerName         = "kubernetes-service-cidr-controller"
	DefaultServiceCIDRName = "kubernetes"
)

// NewController returns a new *Controller that generates the default ServiceCIDR
// from the `--service-cluster-ip-range` flag and recreates it if necessary,
// but doesn't update it if is different.
// It follows the same logic that the kubernetes.default Service.
func NewController(
	primaryRange net.IPNet,
	secondaryRange net.IPNet,
	client clientset.Interface,
) *Controller {
	broadcaster := record.NewBroadcaster()
	recorder := broadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: controllerName})

	c := &Controller{
		client:   client,
		interval: 10 * time.Second, // same as DefaultEndpointReconcilerInterval
	}

	// obtain configuration from flags
	if netutils.IsIPv4CIDR(&primaryRange) {
		c.ipv4CIDR = primaryRange.String()
	} else if netutils.IsIPv4CIDR(&secondaryRange) {
		c.ipv4CIDR = secondaryRange.String()
	}
	if netutils.IsIPv6CIDR(&primaryRange) {
		c.ipv6CIDR = primaryRange.String()
	} else if netutils.IsIPv6CIDR(&secondaryRange) {
		c.ipv6CIDR = secondaryRange.String()
	}
	// instead of using the shared informers from the controlplane instance, we construct our own informer
	// because we need such a small subset of the information available, only the kubernetes.default ServiceCIDR
	c.serviceCIDRInformer = networkingv1alpha1informers.NewFilteredServiceCIDRInformer(client, 12*time.Hour,
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
		func(options *metav1.ListOptions) {
			options.FieldSelector = fields.OneTermEqualSelector("metadata.name", DefaultServiceCIDRName).String()
		})

	c.serviceCIDRLister = networkingv1alpha1listers.NewServiceCIDRLister(c.serviceCIDRInformer.GetIndexer())
	c.serviceCIDRsSynced = c.serviceCIDRInformer.HasSynced

	c.eventBroadcaster = broadcaster
	c.eventRecorder = recorder

	c.readyCh = make(chan struct{})

	return c
}

// Controller manages selector-based service ipAddress.
type Controller struct {
	ipv4CIDR string
	ipv6CIDR string

	client           clientset.Interface
	eventBroadcaster record.EventBroadcaster
	eventRecorder    record.EventRecorder

	serviceCIDRInformer cache.SharedIndexInformer
	serviceCIDRLister   networkingv1alpha1listers.ServiceCIDRLister
	serviceCIDRsSynced  cache.InformerSynced

	readyCh chan struct{} // channel to block until the default ServiceCIDR exists

	interval time.Duration
}

// Start will not return until the default ServiceCIDR exists or stopCh is closed.
func (c *Controller) Start(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()

	c.eventBroadcaster.StartStructuredLogging(0)
	c.eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: c.client.CoreV1().Events("")})
	defer c.eventBroadcaster.Shutdown()

	klog.Infof("Starting %s", controllerName)
	defer klog.Infof("Shutting down %s", controllerName)

	go c.serviceCIDRInformer.Run(stopCh)
	if !cache.WaitForNamedCacheSync(controllerName, stopCh, c.serviceCIDRsSynced) {
		return
	}

	go wait.Until(c.sync, c.interval, stopCh)

	select {
	case <-stopCh:
	case <-c.readyCh:
	}
}

func (c *Controller) sync() {
	// check if the default ServiceCIDR already exist
	serviceCIDR, err := c.serviceCIDRLister.Get(DefaultServiceCIDRName)
	// if exists
	if err == nil {
		c.setReady()
		c.syncStatus(serviceCIDR)
		return
	}

	// unknown error
	if !apierrors.IsNotFound(err) {
		klog.Infof("error trying to obtain the default ServiceCIDR: %v", err)
		return
	}

	// default ServiceCIDR does not exist
	klog.Infof("Creating default ServiceCIDR, ipv4: %q ipv6: %q", c.ipv4CIDR, c.ipv6CIDR)
	serviceCIDR = &networkingapiv1alpha1.ServiceCIDR{
		ObjectMeta: metav1.ObjectMeta{
			Name: DefaultServiceCIDRName,
		},
		Spec: networkingapiv1alpha1.ServiceCIDRSpec{
			IPv4: c.ipv4CIDR,
			IPv6: c.ipv6CIDR,
		},
	}
	serviceCIDR, err = c.client.NetworkingV1alpha1().ServiceCIDRs().Create(context.Background(), serviceCIDR, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		klog.Infof("error creating default ServiceCIDR: %v", err)
		c.eventRecorder.Eventf(serviceCIDR, v1.EventTypeWarning, "KubernetesDefaultServiceCIDRError", "The default ServiceCIDR can not be created")
		return
	}

	c.setReady()
	c.syncStatus(serviceCIDR)
}

func (c *Controller) setReady() {
	select {
	case <-c.readyCh:
	default:
		close(c.readyCh)
	}
}

func (c *Controller) syncStatus(serviceCIDR *networkingapiv1alpha1.ServiceCIDR) {
	// don't sync the status of the ServiceCIDR if is being deleted,
	// deletion must be handled by the controller-manager
	if !serviceCIDR.GetDeletionTimestamp().IsZero() {
		return
	}

	// This controller will set the Ready condition to true if the Ready condition
	// does not exist and the CIDR values match this controller CIDR values.
	for _, condition := range serviceCIDR.Status.Conditions {
		if condition.Type == networkingapiv1alpha1.ServiceCIDRConditionReady {
			if condition.Status == metav1.ConditionTrue {
				return
			}
			klog.Infof("default ServiceCIDR condition Ready is not True: %v", condition.Status)
			c.eventRecorder.Eventf(serviceCIDR, v1.EventTypeWarning, condition.Reason, condition.Message)
			return
		}
	}
	// set status to ready if the ServiceCIDR matches this configuration
	if c.ipv4CIDR == serviceCIDR.Spec.IPv4 &&
		c.ipv6CIDR == serviceCIDR.Spec.IPv6 {
		klog.Infof("Setting default ServiceCIDR condition Ready to True")
		svcApplyStatus := networkingapiv1alpha1apply.ServiceCIDRStatus().WithConditions(
			metav1apply.Condition().
				WithType(networkingapiv1alpha1.ServiceCIDRConditionReady).
				WithStatus(metav1.ConditionTrue).
				WithMessage("Kubernetes default Service CIDR is ready").
				WithLastTransitionTime(metav1.Now()))
		svcApply := networkingapiv1alpha1apply.ServiceCIDR(DefaultServiceCIDRName).WithStatus(svcApplyStatus)
		if _, errApply := c.client.NetworkingV1alpha1().ServiceCIDRs().ApplyStatus(context.Background(), svcApply, metav1.ApplyOptions{FieldManager: controllerName, Force: true}); errApply != nil {
			klog.Infof("error updating default ServiceCIDR status: %v", errApply)
			c.eventRecorder.Eventf(serviceCIDR, v1.EventTypeWarning, "KubernetesDefaultServiceCIDRError", "The default ServiceCIDR Status can not be set to Ready=True")
		}
	}
}
