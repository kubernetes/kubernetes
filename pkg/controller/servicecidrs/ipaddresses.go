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
	"time"

	v1 "k8s.io/api/core/v1"
	networkingapiv1alpha1 "k8s.io/api/networking/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	netutils "k8s.io/utils/net"
)

func (c *Controller) addIPAddress(obj interface{}) {
	ip := obj.(*networkingapiv1alpha1.IPAddress)
	if needToAddFinalizer(ip, networkingapiv1alpha1.ServiceCIDRProtectionFinalizer) ||
		needToRemoveFinalizer(ip, networkingapiv1alpha1.ServiceCIDRProtectionFinalizer) {
		c.enqueue(obj)
		return
	}

	family, ok := ip.Labels[networkingapiv1alpha1.LabelServiceIPAddressFamily]
	if !ok {
		c.enqueue(obj)
		return
	}
	if netutils.IsIPv6String(ip.Name) && family != string(v1.IPv6Protocol) ||
		netutils.IsIPv4String(ip.Name) && family != string(v1.IPv4Protocol) {
		c.enqueue(obj)
	}

}

func (c *Controller) updateIPAddress(oldObj, obj interface{}) {
	oldIP := oldObj.(*networkingapiv1alpha1.IPAddress)
	ip := obj.(*networkingapiv1alpha1.IPAddress)

	if oldIP.Labels[networkingapiv1alpha1.LabelServiceIPAddressFamily] != ip.Labels[networkingapiv1alpha1.LabelServiceIPAddressFamily] ||
		needToAddFinalizer(ip, networkingapiv1alpha1.ServiceCIDRProtectionFinalizer) ||
		needToRemoveFinalizer(ip, networkingapiv1alpha1.ServiceCIDRProtectionFinalizer) {
		c.enqueue(obj)
	}
}

// someone force deleted the object
func (c *Controller) deleteIPAddress(obj interface{}) {
	ip, ok := obj.(*networkingapiv1alpha1.IPAddress)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			klog.V(2).Infof("Couldn't get object from tombstone %#v", obj)
			return
		}
		ip, ok = tombstone.Obj.(*networkingapiv1alpha1.IPAddress)
		if !ok {
			klog.V(2).Infof("Tombstone contained unexpected object: %#v", obj)
			return
		}
	}
	c.enqueue(ip)
}

func (c *Controller) ipWorker() {
	for c.processNextWorkIP() {
	}
}

func (c *Controller) processNextWorkIP() bool {
	eKey, quit := c.ipQueue.Get()
	if quit {
		return false
	}
	defer c.ipQueue.Done(eKey)

	err := c.syncIPAddress(eKey.(string))
	c.handleIPErr(err, eKey)

	return true
}

func (c *Controller) handleIPErr(err error, key interface{}) {
	if err == nil {
		c.ipQueue.Forget(key)
		return
	}

	if c.ipQueue.NumRequeues(key) < maxRetries {
		klog.V(2).InfoS("Error syncing IPAddress, retrying", "IPAddress", key, "err", err)
		c.ipQueue.AddRateLimited(key)
		return
	}

	klog.Warningf("Dropping IPAddress %q out of the queue: %v", key, err)
	c.ipQueue.Forget(key)
	utilruntime.HandleError(err)
}

func (c *Controller) syncIPAddress(key string) error {
	startTime := time.Now()
	defer func() {
		klog.V(4).Infof("Finished syncing %q ipAddress. (%v)", key, time.Since(startTime))
	}()
	klog.V(4).Infof("syncing %q ipAddress. (%v)", key, time.Since(startTime))

	ip, err := c.ipAddressLister.Get(key)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	}

	if ip.Spec.ParentRef.Group != "" &&
		ip.Spec.ParentRef.Resource != "services" {
		klog.V(4).Infof("IPAddress %s not handled by this controller, parent not a Service", ip.String())
		return nil
	}

	// Remove finalizer if the Service has been deleted
	if !ip.GetDeletionTimestamp().IsZero() {
		svc, err := c.serviceLister.Services(ip.Spec.ParentRef.Namespace).Get(ip.Spec.ParentRef.Name)
		if err != nil {
			// remove finalizer
			if apierrors.IsNotFound(err) {
				return c.removeIPAddressFinalizer(ip.Name)
			}
			return err
		}
		// wait for the service to disappear before deleting the IP
		// otherwise a new Service can take over the IP and 2 Services
		// will be sharing the same IP
		if !svc.GetDeletionTimestamp().IsZero() {
			c.ipQueue.AddAfter(key, time.Second)
			return nil
		} else {
			klog.Warningf("IPAddress %s being deleted but Service %s still alive", ip.Name, svc.String())
		}

	}

	return nil

}

func (c *Controller) createIPAddress(name string, svc *v1.Service) error {
	family := string(v1.IPv4Protocol)
	if netutils.IsIPv6String(name) {
		family = string(v1.IPv6Protocol)
	}
	ipAddress := networkingapiv1alpha1.IPAddress{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				networkingapiv1alpha1.LabelServiceIPAddressFamily: family,
			},
			Finalizers: []string{networkingapiv1alpha1.IPAddressProtectionFinalizer},
		},
		Spec: networkingapiv1alpha1.IPAddressSpec{
			ParentRef: serviceToRef(svc),
		},
	}
	_, err := c.client.NetworkingV1alpha1().IPAddresses().Create(context.TODO(), &ipAddress, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		return err
	}
	return nil
}

func (c *Controller) removeIPAddressFinalizer(name string) error {
	patch := map[string]interface{}{
		"metadata": map[string]interface{}{
			"$deleteFromPrimitiveList/finalizers": []string{networkingapiv1alpha1.IPAddressProtectionFinalizer},
		},
	}
	patchBytes, err := json.Marshal(patch)
	if err != nil {
		return err
	}
	_, err = c.client.NetworkingV1alpha1().IPAddresses().Patch(context.TODO(), name, types.StrategicMergePatchType, patchBytes, metav1.PatchOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	return nil
}

func (c *Controller) terminateIPAddress(name string) error {
	err := c.removeIPAddressFinalizer(name)
	if err != nil {
		return err
	}

	err = c.client.NetworkingV1alpha1().IPAddresses().Delete(context.TODO(), name, metav1.DeleteOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	return nil
}
