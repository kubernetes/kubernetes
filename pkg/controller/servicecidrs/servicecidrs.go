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
	"encoding/json"
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	networkingapiv1alpha1 "k8s.io/api/networking/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
)

func (c *Controller) addServiceCIDR(obj interface{}) {
	cidr := obj.(*networkingapiv1alpha1.ServiceCIDR)

	c.muTree.Lock()
	c.treeV4.Insert(cidr.Spec.IPv4, cidr.Name)
	c.treeV6.Insert(cidr.Spec.IPv6, cidr.Name)
	c.muTree.Unlock()

	if needToAddFinalizer(cidr, networkingapiv1alpha1.ServiceCIDRProtectionFinalizer) ||
		needToRemoveFinalizer(cidr, networkingapiv1alpha1.ServiceCIDRProtectionFinalizer) {
		c.enqueue(obj)
	}
}

// service CIDRs are inmutable, check finalizer and labels weren't changed
func (c *Controller) updateServiceCIDR(oldObj, obj interface{}) {
	old := oldObj.(*networkingapiv1alpha1.ServiceCIDR)
	new := obj.(*networkingapiv1alpha1.ServiceCIDR)

	if old.Labels[networkingapiv1alpha1.LabelServiceCIDRFromFlags] != new.Labels[networkingapiv1alpha1.LabelServiceCIDRFromFlags] ||
		needToAddFinalizer(new, networkingapiv1alpha1.ServiceCIDRProtectionFinalizer) ||
		needToRemoveFinalizer(new, networkingapiv1alpha1.ServiceCIDRProtectionFinalizer) {
		c.enqueue(obj)
	}
}

// someone force deleted the object
func (c *Controller) deleteServiceCIDR(obj interface{}) {
	cidr, ok := obj.(*networkingapiv1alpha1.ServiceCIDR)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			klog.V(2).Infof("Couldn't get object from tombstone %#v", obj)
			return
		}
		cidr, ok = tombstone.Obj.(*networkingapiv1alpha1.ServiceCIDR)
		if !ok {
			klog.V(2).Infof("Tombstone contained unexpected object: %#v", obj)
			return
		}
	}
	c.muTree.Lock()
	c.treeV4.Delete(cidr.Spec.IPv4)
	c.treeV6.Delete(cidr.Spec.IPv6)
	c.muTree.Unlock()

	c.enqueue(cidr)
}

func (c *Controller) cidrWorker() {
	for c.processNextWorkCIDR() {
	}
}

func (c *Controller) processNextWorkCIDR() bool {
	eKey, quit := c.cidrQueue.Get()
	if quit {
		return false
	}
	defer c.cidrQueue.Done(eKey)

	err := c.syncServiceCIDR(eKey.(string))
	c.handleCIDRErr(err, eKey)

	return true
}

func (c *Controller) handleCIDRErr(err error, key interface{}) {
	if err == nil {
		c.cidrQueue.Forget(key)
		return
	}

	if c.cidrQueue.NumRequeues(key) < maxRetries {
		klog.V(2).InfoS("Error syncing ServiceCIDR, retrying", "ServiceCIDR", key, "err", err)
		c.cidrQueue.AddRateLimited(key)
		return
	}

	klog.Warningf("Dropping ServiceCIDR %q out of the queue: %v", key, err)
	c.cidrQueue.Forget(key)
	utilruntime.HandleError(err)
}

func (c *Controller) syncServiceCIDR(key string) error {
	startTime := time.Now()
	defer func() {
		klog.V(4).Infof("Finished syncing ServiceCIDR %q . (%v)", key, time.Since(startTime))
	}()

	klog.V(4).Infof("syncing ServiceCIDR %q", key)
	cidr, err := c.serviceCIDRLister.Get(key)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	}

	// Deleting ....
	if cidr.GetDeletionTimestamp() != nil {
		ipAddresses, err := c.ipAddressLister.List(labels.Everything())
		if err != nil {
			return err
		}
		if len(ipAddresses) == 0 {
			return c.removeServiceCIDRFinalizer(key)
		}
		hasParentV4 := false
		hasParentV6 := false
		if cidr.Spec.IPv4 != "" {
			_, oldCidr, ok := c.treeV4.LongestPrefixMatch(cidr.Spec.IPv4)
			// if has a parent that is not ourselves
			if ok && oldCidr != cidr.Name {
				hasParentV4 = true
			}
		}
		if cidr.Spec.IPv6 != "" {
			_, oldCidr, ok := c.treeV4.LongestPrefixMatch(cidr.Spec.IPv6)
			// if has a parent that is not ourselves
			if ok && oldCidr != cidr.Name {
				hasParentV6 = true
			}
		}
		if hasParentV4 && hasParentV6 {
			return c.removeServiceCIDRFinalizer(key)
		}

		klog.Warningf("ServiceCIDR %s can not leave orphan IPs", key)
		return fmt.Errorf("ServiceCIDR %s can not leave orphan IPs", key)
	}

	// Created or Updated
	// Reconcile all the IPAddresses to match the CIDR with the longest prefix match
	// TODO reconcile all or only the ones belonging to this CIDR
	errs := []error{}
	if cidr.Spec.IPv4 != "" {
		ipLabelSelector := labels.Set(map[string]string{
			networkingapiv1alpha1.LabelServiceIPAddressFamily: string(v1.IPv4Protocol),
		}).AsSelectorPreValidated()
		ips, err := c.ipAddressLister.List(ipLabelSelector)
		if err != nil {
			return err
		}
		for _, ip := range ips {
			subnet, _, ok := c.treeV4.LongestPrefixMatch(ip.Name + "/32")
			if ok && subnet == cidr.Spec.IPv4 {
				// TODO
			}
		}
	}

	if cidr.Spec.IPv6 != "" {
		ipLabelSelector := labels.Set(map[string]string{
			networkingapiv1alpha1.LabelServiceIPAddressFamily: string(v1.IPv6Protocol),
		}).AsSelectorPreValidated()
		ips, err := c.ipAddressLister.List(ipLabelSelector)
		if err != nil {
			return err
		}
		for _, ip := range ips {
			subnet, _, ok := c.treeV6.LongestPrefixMatch(ip.Name + "/128")
			if ok && subnet == cidr.Spec.IPv6 {
				// TODO
			}
		}
	}

	if needToAddFinalizer(cidr, networkingapiv1alpha1.ServiceCIDRProtectionFinalizer) {
		err := c.addServiceCIDRFinalizer(cidr)
		if err != nil {
			errs = append(errs, err)
		}
	}

	return utilerrors.NewAggregate(errs)
}
func (c *Controller) addServiceCIDRFinalizer(cidr *networkingapiv1alpha1.ServiceCIDR) error {
	clone := cidr.DeepCopy()
	clone.ObjectMeta.Finalizers = append(clone.ObjectMeta.Finalizers, networkingapiv1alpha1.ServiceCIDRProtectionFinalizer)
	_, err := c.client.NetworkingV1alpha1().ServiceCIDRs().Update(context.TODO(), clone, metav1.UpdateOptions{})
	if err != nil {
		return err
	}
	klog.V(4).Infof("Added protection finalizer to ServiceCIDR %s", cidr.Name)
	return nil
}

func (c *Controller) removeServiceCIDRFinalizer(name string) error {
	patch := map[string]interface{}{
		"metadata": map[string]interface{}{
			"$deleteFromPrimitiveList/finalizers": []string{networkingapiv1alpha1.ServiceCIDRProtectionFinalizer},
		},
	}
	patchBytes, err := json.Marshal(patch)
	if err != nil {
		return err
	}
	_, err = c.client.NetworkingV1alpha1().ServiceCIDRs().Patch(context.TODO(), name, types.StrategicMergePatchType, patchBytes, metav1.PatchOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	klog.V(4).Infof("Removed protection finalizer to ServiceCIDR %s", name)
	return nil
}
