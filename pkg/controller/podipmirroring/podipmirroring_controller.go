/*
Copyright 2025 The Kubernetes Authors.

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

package podipmirroring

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coreinformers "k8s.io/client-go/informers/core/v1"
	networkingv1informers "k8s.io/client-go/informers/networking/v1"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	networkinglisters "k8s.io/client-go/listers/networking/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
)

const (
	// ControllerName is the name of this controller
	ControllerName      = "podipmirroring-controller"
	podLabelsAnnotation = "pod-ip-controller/pod-labels"
)

// NewPodIPMirroringController returns a new *Controller.
func NewPodIPMirroringController(ctx context.Context, podInformer coreinformers.PodInformer, ipAddressInformer networkingv1informers.IPAddressInformer,
	client clientset.Interface) *Controller {

	c := &Controller{
		clientset:       client,
		podLister:       podInformer.Lister(),
		podsSynced:      podInformer.Informer().HasSynced,
		ipAddressLister: ipAddressInformer.Lister(),
		ipAddressSynced: ipAddressInformer.Informer().HasSynced,
		workqueue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "PodIPs"},
		),
	}

	klog.Infoln("Setting up event handlers")
	// We still use event handlers to trigger reconciliation, but the core logic
	// will now do a full sync instead of processing just the single item.
	podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    c.enqueueReconcile,
		UpdateFunc: func(old, new interface{}) { c.enqueueReconcile(new) },
		DeleteFunc: c.enqueueReconcile,
	})

	ipAddressInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: c.enqueueReconcile,
		UpdateFunc: func(old, new interface{}) {
			// If an IPAddress is updated, we might need to reconcile.
			c.enqueueReconcile(new)
		},
		DeleteFunc: c.enqueueReconcile,
	})

	return c
}

type Controller struct {
	clientset       clientset.Interface
	podLister       corelisters.PodLister
	podsSynced      cache.InformerSynced
	ipAddressLister networkinglisters.IPAddressLister
	ipAddressSynced cache.InformerSynced
	workqueue       workqueue.TypedRateLimitingInterface[string]
}

// enqueueReconcile triggers a full reconciliation by adding a static key to the queue.
func (c *Controller) enqueueReconcile(obj interface{}) {
	// We use a static key because the sync handler will perform a full reconciliation
	// of all pods and IPs, not just for the object that triggered the event.
	// Aggregate all changes for 3 seconds.
	c.workqueue.AddAfter("reconcile-all", 3*time.Second)
}

// Run starts the controller's main loop.
func (c *Controller) Run(ctx context.Context) error {
	defer runtime.HandleCrash()
	defer c.workqueue.ShutDown()

	klog.Infoln("Starting Pod IP controller")

	klog.Infoln("Waiting for informer caches to sync")
	if ok := cache.WaitForCacheSync(ctx.Done(), c.podsSynced, c.ipAddressSynced); !ok {
		return fmt.Errorf("failed to wait for caches to sync")
	}

	klog.Infoln("Starting workers")
	wait.UntilWithContext(ctx, c.runWorker, time.Second)
	klog.Infoln("Shutting down workers")

	return nil
}

// runWorker is a long-running function that will continually call the
// processNextWorkItem function in order to read and process a message on the queue.
func (c *Controller) runWorker(ctx context.Context) {
	for c.processNextWorkItem() {
	}
}

// processNextWorkItem will read a single work item off the workqueue and
// attempt to process it, by calling the syncHandler.
func (c *Controller) processNextWorkItem() bool {
	key, shutdown := c.workqueue.Get()
	if shutdown {
		return false
	}

	// We wrap this block in a func so we can defer c.workqueue.Done.
	err := func(key string) error {
		defer c.workqueue.Done(key)

		// The key is now static ("reconcile-all"), so we call our main sync function.
		if err := c.reconcileAll(); err != nil {
			// We had a failure, re-queue the item to retry later.
			c.workqueue.AddRateLimited(key)
			return fmt.Errorf("error during reconciliation: %s, requeuing", err.Error())
		}
		// Finally, if no error occurs we Forget this item so it does not
		// get queued again until another change happens.
		c.workqueue.Forget(key)
		klog.Infoln("Successfully completed full reconciliation cycle")
		return nil
	}(key)

	if err != nil {
		if c.workqueue.NumRequeues(key) < 5 {
			klog.Infof("Error syncing: %v", err)
			// Re-enqueue the key rate limited. Based on the rate limiter on the
			// queue and the re-enqueue history, the key will be processed later again.
			c.workqueue.AddRateLimited(key)
		} else {
			runtime.HandleError(err)
			klog.Infof("Fail to reconcile after 5 times: %v", err)
		}
	}

	return true
}

// reconcileAll performs a full synchronization between all Pod IPs and IPAddress objects.
func (c *Controller) reconcileAll() error {
	// 1. List all pods from the informer's cache
	pods, err := c.podLister.List(labels.Everything())
	if err != nil {
		return fmt.Errorf("failed to list pods: %w", err)
	}

	// 2. Build a map of required IPAddress objects based on current pods
	requiredIPs := make(map[string]*corev1.Pod)
	for _, pod := range pods {
		for _, ip := range pod.Status.PodIPs {
			requiredIPs[ip.IP] = pod
		}
	}

	// 3. List all existing IPAddress objects from the informer's cache
	ipAddresses, err := c.ipAddressLister.List(labels.Everything())
	if err != nil {
		return fmt.Errorf("failed to list IPAddresses: %w", err)
	}

	// 4. Create missing IPAddress objects
	var errorList []error
	for ip, pod := range requiredIPs {
		// Marshal pod labels to JSON to create the desired annotation.
		var expectedAnnotation string
		podLabels := pod.GetLabels()
		if len(podLabels) > 0 {
			labelsJSON, err := json.Marshal(podLabels)
			if err != nil {
				runtime.HandleError(fmt.Errorf("failed to marshal labels for pod %s/%s: %w", pod.Namespace, pod.Name, err))
				continue // Skip this pod if labels can't be processed
			}
			expectedAnnotation = string(labelsJSON)
		}
		// Check if an IPAddress object for this IP already exists
		ipAddr, err := c.ipAddressLister.Get(ip)
		if apierrors.IsNotFound(err) {
			// It doesn't exist, so create it
			newIPAddress := &networkingv1.IPAddress{
				ObjectMeta: metav1.ObjectMeta{
					Name: ip,
					OwnerReferences: []metav1.OwnerReference{
						*metav1.NewControllerRef(pod, corev1.SchemeGroupVersion.WithKind("Pod")),
					},
					Labels: map[string]string{
						networkingv1.LabelManagedBy: ControllerName,
						"pod-name":                  pod.Name,
						"pod-namespace":             pod.Namespace,
					},
					Annotations: make(map[string]string),
				},
				Spec: networkingv1.IPAddressSpec{
					ParentRef: &networkingv1.ParentReference{
						Group:     "", // Core group
						Resource:  "pods",
						Namespace: pod.Namespace,
						Name:      pod.Name,
					},
				},
			}
			if expectedAnnotation != "" {
				newIPAddress.Annotations[podLabelsAnnotation] = expectedAnnotation
			}
			klog.V(2).Infof("Creating IPAddress object '%s' for Pod '%s/%s'", ip, pod.Namespace, pod.Name)
			_, createErr := c.clientset.NetworkingV1().IPAddresses().Create(context.TODO(), newIPAddress, metav1.CreateOptions{})
			if createErr != nil {
				// Don't let a single failure stop the entire loop, just log it.
				runtime.HandleError(fmt.Errorf("failed to create IPAddress %s: %w", ip, createErr))
				errorList = append(errorList, createErr)
				continue
			}
		} else if err != nil {
			runtime.HandleError(fmt.Errorf("failed to get IPAddress %s: %w", ip, err))
			errorList = append(errorList, err)
			continue
		}
		// Case 2: IPAddress exists. Check if the annotation needs an update.
		currentAnnotation := ipAddr.Annotations[podLabelsAnnotation]
		if currentAnnotation != expectedAnnotation {
			ipAddrCopy := ipAddr.DeepCopy()
			if ipAddrCopy.Annotations == nil {
				ipAddrCopy.Annotations = make(map[string]string)
			}
			if expectedAnnotation == "" {
				delete(ipAddrCopy.Annotations, podLabelsAnnotation)
			} else {
				ipAddrCopy.Annotations[podLabelsAnnotation] = expectedAnnotation
			}
			klog.V(2).Infof("Updating IPAddress object '%s' with new pod labels annotation", ipAddr.Name)
			_, updateErr := c.clientset.NetworkingV1().IPAddresses().Update(context.TODO(), ipAddrCopy, metav1.UpdateOptions{})
			if updateErr != nil {
				runtime.HandleError(fmt.Errorf("failed to update IPAddress %s: %w", ipAddr.Name, updateErr))
				errorList = append(errorList, updateErr)
				continue
			}
		}
		if ipAddr.Spec.ParentRef == nil ||
			ipAddr.Spec.ParentRef.Group != "" ||
			ipAddr.Spec.ParentRef.Resource != "pods" ||
			ipAddr.Spec.ParentRef.Name != pod.Name ||
			ipAddr.Spec.ParentRef.Namespace != pod.Namespace {
			runtime.HandleError(fmt.Errorf("wrong reference for IPAddress %s: %v", ipAddr.Name, ipAddr.Spec.ParentRef))
			errorList = append(errorList, fmt.Errorf("wrong reference for IPAddress %s: %v", ipAddr.Name, ipAddr.Spec.ParentRef))
			continue
		}
	}

	// 5. Delete stale IPAddress objects.
	// The owner reference should handle most cases, but this provides self-healing.
	for _, ipAddr := range ipAddresses {
		// Only check objects we are supposed to manage
		if val, ok := ipAddr.Labels[networkingv1.LabelManagedBy]; !ok || val != ControllerName {
			continue
		}

		if _, needed := requiredIPs[ipAddr.Name]; !needed {
			// This IPAddress is no longer required by any running pod.
			klog.V(2).Infof("Deleting stale IPAddress object '%s'", ipAddr.Name)
			err := c.clientset.NetworkingV1().IPAddresses().Delete(context.TODO(), ipAddr.Name, metav1.DeleteOptions{})
			if err != nil && !apierrors.IsNotFound(err) {
				runtime.HandleError(fmt.Errorf("failed to delete stale IPAddress %s: %w", ipAddr.Name, err))
				errorList = append(errorList, err)
			}
		}
	}

	return errors.Join(errorList...)
}
