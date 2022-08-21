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

package servicecidrs

import (
	"context"
	"reflect"
	"time"

	v1 "k8s.io/api/core/v1"
	networkingapiv1alpha1 "k8s.io/api/networking/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	netutils "k8s.io/utils/net"
)

func (c *Controller) addService(obj interface{}) {
	svc := obj.(*v1.Service)
	c.enqueue(svc)
}

// Process only if the IPs has changed
func (c *Controller) updateService(oldObj, obj interface{}) {
	oldSvc := oldObj.(*v1.Service)
	svc := obj.(*v1.Service)
	if sameStringSlice(oldSvc.Spec.ClusterIPs, svc.Spec.ClusterIPs) {
		return
	}
	c.enqueue(svc)
}

func (c *Controller) deleteService(obj interface{}) {
	svc, ok := obj.(*v1.Service)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			klog.V(2).Infof("Couldn't get object from tombstone %#v", obj)
			return
		}
		svc, ok = tombstone.Obj.(*v1.Service)
		if !ok {
			klog.V(2).Infof("Tombstone contained unexpected object: %#v", obj)
			return
		}
	}
	c.enqueue(svc)
}

func (c *Controller) svcWorker() {
	for c.processNextWorkSvc() {
	}
}

func (c *Controller) processNextWorkSvc() bool {
	eKey, quit := c.svcQueue.Get()
	if quit {
		return false
	}
	defer c.svcQueue.Done(eKey)

	err := c.syncService(eKey.(string))
	c.handleSvcErr(err, eKey)

	return true
}

func (c *Controller) handleSvcErr(err error, key interface{}) {
	if err == nil {
		c.svcQueue.Forget(key)
		return
	}

	if c.svcQueue.NumRequeues(key) < maxRetries {
		klog.V(2).InfoS("Error syncing Service, retrying", "Service", key, "err", err)
		c.svcQueue.AddRateLimited(key)
		return
	}

	klog.Warningf("Dropping Service %q out of the queue: %v", key, err)
	c.svcQueue.Forget(key)
	utilruntime.HandleError(err)
}

func (c *Controller) syncService(key string) error {
	startTime := time.Now()
	defer func() {
		klog.V(4).Infof("Finished syncing Service %q . (%v)", key, time.Since(startTime))
	}()

	klog.V(4).Infof("syncing Service %q . (%v)", key, time.Since(startTime))
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}
	svc, err := c.serviceLister.Services(namespace).Get(name)
	// Service deleted, the IPAddresses were enqueued on the handler and will be
	// processed as part of the IPAddress reconciler
	if err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	}
	// Verify that the IPAddress object exist and the Reference matches
	for _, clusterIP := range svc.Spec.ClusterIPs {
		ip := netutils.ParseIPSloppy(clusterIP)
		if ip == nil {
			continue
		}

		ipObj, err := c.ipAddressLister.Get(ip.String())
		if err != nil && !apierrors.IsNotFound(err) {
			return err
		}
		// IPAddress doesn't exist, create one and associate to the Service
		if apierrors.IsNotFound(err) {
			klog.Warningf("Service %s without IPAddress associated, creating the corresponding one", key)
			err := c.createIPAddress(name, svc)
			if err != nil {
				return err
			}
			continue
		}
		// ParentReference is enforced in the API, this can not happen
		if ipObj.Spec.ParentRef == nil {
			panic("parentRef nil")
		}
		// Check the ipAddress ParentReference matches the current Service
		svcRef := serviceToRef(svc)
		if !reflect.DeepEqual(svcRef, ipObj.Spec.ParentRef) {
			// Delete the IPAddress, the IPAddress reconcile loop should handle the deletion
			err = c.client.NetworkingV1alpha1().IPAddresses().Delete(context.TODO(), ipObj.Name, metav1.DeleteOptions{})
			if err != nil && !apierrors.IsNotFound(err) {
				return err
			}
		}
	}

	return nil
}

func serviceToRef(svc *v1.Service) *networkingapiv1alpha1.ParentReference {
	if svc == nil {
		return nil
	}

	return &networkingapiv1alpha1.ParentReference{
		Group:     "",
		Resource:  "services",
		Namespace: svc.Namespace,
		Name:      svc.Name,
		UID:       svc.UID,
	}
}
