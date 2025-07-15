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
	"reflect"
	"time"

	v1 "k8s.io/api/core/v1"
	networkingapiv1 "k8s.io/api/networking/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	metav1apply "k8s.io/client-go/applyconfigurations/meta/v1"
	networkingapiv1apply "k8s.io/client-go/applyconfigurations/networking/v1"
	networkingv1informers "k8s.io/client-go/informers/networking/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	networkingv1listers "k8s.io/client-go/listers/networking/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
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
	c := &Controller{
		client:   client,
		interval: 10 * time.Second, // same as DefaultEndpointReconcilerInterval
	}

	// obtain configuration from flags
	c.cidrs = append(c.cidrs, primaryRange.String())
	if secondaryRange.IP != nil {
		c.cidrs = append(c.cidrs, secondaryRange.String())
	}
	// instead of using the shared informers from the controlplane instance, we construct our own informer
	// because we need such a small subset of the information available, only the kubernetes.default ServiceCIDR
	c.serviceCIDRInformer = networkingv1informers.NewFilteredServiceCIDRInformer(client, 12*time.Hour,
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
		func(options *metav1.ListOptions) {
			options.FieldSelector = fields.OneTermEqualSelector("metadata.name", DefaultServiceCIDRName).String()
		})

	c.serviceCIDRLister = networkingv1listers.NewServiceCIDRLister(c.serviceCIDRInformer.GetIndexer())
	c.serviceCIDRsSynced = c.serviceCIDRInformer.HasSynced

	return c
}

// Controller manages selector-based service ipAddress.
type Controller struct {
	cidrs []string // order matters, first cidr defines the default IP family

	client           clientset.Interface
	eventBroadcaster record.EventBroadcaster
	eventRecorder    record.EventRecorder

	serviceCIDRInformer cache.SharedIndexInformer
	serviceCIDRLister   networkingv1listers.ServiceCIDRLister
	serviceCIDRsSynced  cache.InformerSynced

	interval                  time.Duration
	reportedMismatchedCIDRs   bool
	reportedNotReadyCondition bool
}

// Start will not return until the default ServiceCIDR exists or stopCh is closed.
func (c *Controller) Start(ctx context.Context) {
	defer utilruntime.HandleCrash()
	stopCh := ctx.Done()

	c.eventBroadcaster = record.NewBroadcaster(record.WithContext(ctx))
	c.eventRecorder = c.eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: controllerName})
	c.eventBroadcaster.StartStructuredLogging(0)
	c.eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: c.client.CoreV1().Events("")})
	defer c.eventBroadcaster.Shutdown()

	klog.Infof("Starting %s", controllerName)
	defer klog.Infof("Shutting down %s", controllerName)

	go c.serviceCIDRInformer.Run(stopCh)
	if !cache.WaitForNamedCacheSync(controllerName, stopCh, c.serviceCIDRsSynced) {
		return
	}

	// wait until first successfully sync
	// this blocks apiserver startup so poll with a short interval
	err := wait.PollUntilContextCancel(ctx, 100*time.Millisecond, true, func(ctx context.Context) (bool, error) {
		syncErr := c.sync()
		return syncErr == nil, nil
	})
	if err != nil {
		klog.Infof("error initializing the default ServiceCIDR: %v", err)

	}

	// run the sync loop in the background with the defined interval
	go wait.Until(func() {
		err := c.sync()
		if err != nil {
			klog.Infof("error trying to sync the default ServiceCIDR: %v", err)
		}
	}, c.interval, stopCh)
}

func (c *Controller) sync() error {
	// check if the default ServiceCIDR already exist
	serviceCIDR, err := c.serviceCIDRLister.Get(DefaultServiceCIDRName)
	// if exists
	if err == nil {
		// single to dual stack upgrade
		if len(c.cidrs) == 2 && len(serviceCIDR.Spec.CIDRs) == 1 && c.cidrs[0] == serviceCIDR.Spec.CIDRs[0] {
			klog.Infof("Updating default ServiceCIDR from single-stack (%v) to dual-stack (%v)", serviceCIDR.Spec.CIDRs, c.cidrs)
			serviceCIDRcopy := serviceCIDR.DeepCopy()
			serviceCIDRcopy.Spec.CIDRs = c.cidrs
			_, err := c.client.NetworkingV1().ServiceCIDRs().Update(context.Background(), serviceCIDRcopy, metav1.UpdateOptions{})
			if err != nil {
				klog.Infof("The default ServiceCIDR can not be updated from %s to dual stack %v : %v", c.cidrs[0], c.cidrs, err)
				c.eventRecorder.Eventf(serviceCIDR, v1.EventTypeWarning, "KubernetesDefaultServiceCIDRError", "The default ServiceCIDR can not be upgraded from %s to dual stack %v : %v", c.cidrs[0], c.cidrs, err)
			}
		} else {
			c.syncStatus(serviceCIDR)
		}
		return nil
	}

	// unknown error
	if !apierrors.IsNotFound(err) {
		return err
	}

	// default ServiceCIDR does not exist
	klog.Infof("Creating default ServiceCIDR with CIDRs: %v", c.cidrs)
	serviceCIDR = &networkingapiv1.ServiceCIDR{
		ObjectMeta: metav1.ObjectMeta{
			Name: DefaultServiceCIDRName,
		},
		Spec: networkingapiv1.ServiceCIDRSpec{
			CIDRs: c.cidrs,
		},
	}
	serviceCIDR, err = c.client.NetworkingV1().ServiceCIDRs().Create(context.Background(), serviceCIDR, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		c.eventRecorder.Eventf(serviceCIDR, v1.EventTypeWarning, "KubernetesDefaultServiceCIDRError", "The default ServiceCIDR can not be created")
		return err
	}
	c.syncStatus(serviceCIDR)
	return nil
}

func (c *Controller) syncStatus(serviceCIDR *networkingapiv1.ServiceCIDR) {
	// don't sync the status of the ServiceCIDR if is being deleted,
	// deletion must be handled by the controller-manager
	if !serviceCIDR.GetDeletionTimestamp().IsZero() {
		klog.V(6).Infof("ServiceCIDR %s is being deleted, skipping status sync", serviceCIDR.Name)
		return
	}

	// This controller will set the Ready condition to true if the Ready condition
	// does not exist and the CIDR values match this controller CIDR values.
	sameConfig := reflect.DeepEqual(c.cidrs, serviceCIDR.Spec.CIDRs)
	currentReadyCondition := apimeta.FindStatusCondition(serviceCIDR.Status.Conditions, networkingapiv1.ServiceCIDRConditionReady)

	// Handle inconsistent configuration
	if !sameConfig {
		if !c.reportedMismatchedCIDRs {
			klog.Infof("Inconsistent ServiceCIDR status for %s, controller configuration: %v, ServiceCIDR configuration: %v. Configure the flags to match current ServiceCIDR or manually delete it.", serviceCIDR.Name, c.cidrs, serviceCIDR.Spec.CIDRs)
			c.eventRecorder.Eventf(serviceCIDR, v1.EventTypeWarning, "KubernetesDefaultServiceCIDRInconsistent", "The default ServiceCIDR %v does not match the controller flag configurations %s", serviceCIDR.Spec.CIDRs, c.cidrs)
			c.reportedMismatchedCIDRs = true
		}
		// Regardless of the current Ready condition, inconsistent config is a problem.
		// We don't try to change the Ready status in this case.
		return
	}

	// Configuration is consistent (sameConfig)
	switch {
	// Current Ready=False, should not happen state.
	// Don't try to overwrite Ready=False set by another component to avoid hotlooping.
	// The default ServiceCIDR should never have this condition set to False, if this
	// is the case, then it will require an intervention by the cluster administrator.
	case currentReadyCondition != nil && currentReadyCondition.Status == metav1.ConditionFalse:
		if !c.reportedNotReadyCondition {
			klog.InfoS("Default ServiceCIDR condition Ready is False, but controller configuration matches. Please validate your cluster's network configuration.", "serviceCIDR", klog.KObj(serviceCIDR), "status", currentReadyCondition.Status, "reason", currentReadyCondition.Reason, "message", currentReadyCondition.Message)
			c.eventRecorder.Eventf(serviceCIDR, v1.EventTypeWarning, currentReadyCondition.Reason, "Configuration matches, but "+currentReadyCondition.Message)
			c.reportedNotReadyCondition = true
		}

	// Current Ready=True and config matches, nothing to do.
	case currentReadyCondition != nil && currentReadyCondition.Status == metav1.ConditionTrue:
		klog.V(6).Infof("ServiceCIDR %s is Ready and configuration matches. No status update needed.", serviceCIDR.Name)

	// No condition set and ServiceCIDR matches this apiserver configuration, set condition to True
	case currentReadyCondition == nil || currentReadyCondition.Status == metav1.ConditionUnknown:
		klog.Infof("Setting default ServiceCIDR condition Ready to True")
		svcApplyStatus := networkingapiv1apply.ServiceCIDRStatus().WithConditions(
			metav1apply.Condition().
				WithType(networkingapiv1.ServiceCIDRConditionReady).
				WithStatus(metav1.ConditionTrue).
				WithMessage("Kubernetes default Service CIDR is ready").
				WithLastTransitionTime(metav1.Now()))
		svcApply := networkingapiv1apply.ServiceCIDR(DefaultServiceCIDRName).WithStatus(svcApplyStatus)
		if _, errApply := c.client.NetworkingV1().ServiceCIDRs().ApplyStatus(context.Background(), svcApply, metav1.ApplyOptions{FieldManager: controllerName, Force: true}); errApply != nil {
			klog.Infof("error updating default ServiceCIDR status: %v", errApply)
			c.eventRecorder.Eventf(serviceCIDR, v1.EventTypeWarning, "KubernetesDefaultServiceCIDRError", "The default ServiceCIDR Status can not be set to Ready=True")
		}
	default:
	}
}
