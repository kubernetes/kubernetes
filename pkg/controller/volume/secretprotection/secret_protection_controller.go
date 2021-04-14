/*
Copyright 2021 The Kubernetes Authors.

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

package secretprotection

import (
	"context"
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/component-base/metrics/prometheus/ratelimiter"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller/volume/protectionutil"
	"k8s.io/kubernetes/pkg/util/slice"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

// Controller is controller that removes SecretProtectionFinalizer
// from secrets that are used by no other resources.
type Controller struct {
	client clientset.Interface

	secretLister       corelisters.SecretLister
	secretListerSynced cache.InformerSynced

	podLister       corelisters.PodLister
	podListerSynced cache.InformerSynced

	pvLister       corelisters.PersistentVolumeLister
	pvListerSynced cache.InformerSynced

	queue workqueue.RateLimitingInterface

	// allows overriding of StorageObjectInUseProtection feature Enabled/Disabled for testing
	storageObjectInUseProtectionEnabled bool
}

// NewSecretProtectionController returns a new instance of SecretProtectionController.
func NewSecretProtectionController(secretInformer coreinformers.SecretInformer, podInformer coreinformers.PodInformer, pvInformer coreinformers.PersistentVolumeInformer, cl clientset.Interface, storageObjectInUseProtectionFeatureEnabled bool) *Controller {
	e := &Controller{
		client:                              cl,
		queue:                               workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "secretprotection"),
		storageObjectInUseProtectionEnabled: storageObjectInUseProtectionFeatureEnabled,
	}
	if cl != nil && cl.CoreV1().RESTClient().GetRateLimiter() != nil {
		ratelimiter.RegisterMetricAndTrackRateLimiterUsage("secret_protection_controller", cl.CoreV1().RESTClient().GetRateLimiter())
	}

	e.secretLister = secretInformer.Lister()
	e.secretListerSynced = secretInformer.Informer().HasSynced
	secretInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: e.secretAddedUpdated,
		UpdateFunc: func(old, new interface{}) {
			e.secretAddedUpdated(new)
		},
	})

	e.podLister = podInformer.Lister()
	e.podListerSynced = podInformer.Informer().HasSynced
	podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			e.podAddedDeletedUpdated(nil, obj, false)
		},
		DeleteFunc: func(obj interface{}) {
			e.podAddedDeletedUpdated(nil, obj, true)
		},
		UpdateFunc: func(old, new interface{}) {
			e.podAddedDeletedUpdated(old, new, false)
		},
	})

	e.pvLister = pvInformer.Lister()
	e.pvListerSynced = pvInformer.Informer().HasSynced
	pvInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			e.pvAddedDeletedUpdated(nil, obj, false)
		},
		DeleteFunc: func(obj interface{}) {
			e.pvAddedDeletedUpdated(nil, obj, true)
		},
		UpdateFunc: func(old, new interface{}) {
			e.pvAddedDeletedUpdated(old, new, false)
		},
	})

	return e
}

// Run runs the controller goroutines.
func (c *Controller) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	klog.InfoS("Starting secret protection controller")
	defer klog.InfoS("Shutting down secret protection controller")

	if !cache.WaitForNamedCacheSync("secret protection", stopCh, c.secretListerSynced, c.podListerSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.Until(c.runWorker, time.Second, stopCh)
	}

	<-stopCh
}

func (c *Controller) runWorker() {
	for c.processNextWorkItem() {
	}
}

// processNextWorkItem deals with one secretKey off the queue.  It returns false when it's time to quit.
func (c *Controller) processNextWorkItem() bool {
	secretKey, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(secretKey)

	secretNamespace, secretName, err := cache.SplitMetaNamespaceKey(secretKey.(string))
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("error parsing secret key %q: %v", secretKey, err))
		return true
	}

	err = c.processSecret(secretNamespace, secretName)
	if err == nil {
		c.queue.Forget(secretKey)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("secret %v failed with : %v", secretKey, err))
	c.queue.AddRateLimited(secretKey)

	return true
}

func (c *Controller) processSecret(secretNamespace, secretName string) error {
	klog.V(4).InfoS("Processing secret", "secret", klog.KRef(secretNamespace, secretName))
	startTime := time.Now()
	defer func() {
		klog.V(4).InfoS("Finished processing secret", "secret", klog.KRef(secretNamespace, secretName), "duration", time.Since(startTime))
	}()

	secret, err := c.secretLister.Secrets(secretNamespace).Get(secretName)
	if apierrors.IsNotFound(err) {
		klog.V(4).InfoS("Secret not found, ignoring", "secret", klog.KRef(secretNamespace, secretName))
		return nil
	}
	if err != nil {
		return err
	}

	if protectionutil.IsDeletionCandidate(secret, volumeutil.SecretProtectionFinalizer) {
		// secret should be deleted. Check if it's used and remove finalizer if
		// it's not.
		isUsed, err := c.isBeingUsed(secret)
		if err != nil {
			return err
		}
		if !isUsed {
			return c.removeFinalizer(secret)
		}
		klog.V(2).InfoS("Keeping secret because it is being used", "secret", klog.KObj(secret))
	}

	if protectionutil.NeedToAddFinalizer(secret, volumeutil.SecretProtectionFinalizer) {
		// secret is not being deleted -> it should have the finalizer. The
		// finalizer should be added by admission plugin, this is just to add
		// the finalizer to old secrets that were created before the admission
		// plugin was enabled.
		return c.addFinalizer(secret)
	}
	return nil
}

func (c *Controller) addFinalizer(secret *v1.Secret) error {
	// Skip adding Finalizer in case the StorageObjectInUseProtection feature is not enabled
	if !c.storageObjectInUseProtectionEnabled {
		return nil
	}
	secretClone := secret.DeepCopy()
	secretClone.ObjectMeta.Finalizers = append(secretClone.ObjectMeta.Finalizers, volumeutil.SecretProtectionFinalizer)
	_, err := c.client.CoreV1().Secrets(secretClone.Namespace).Update(context.TODO(), secretClone, metav1.UpdateOptions{})
	if err != nil {
		klog.ErrorS(err, "Error adding protection finalizer to secret", "secret", klog.KObj(secret))
		return err
	}
	klog.V(3).InfoS("Added protection finalizer to secret", "secret", klog.KObj(secret))
	return nil
}

func (c *Controller) removeFinalizer(secret *v1.Secret) error {
	secretClone := secret.DeepCopy()
	secretClone.ObjectMeta.Finalizers = slice.RemoveString(secretClone.ObjectMeta.Finalizers, volumeutil.SecretProtectionFinalizer, nil)
	_, err := c.client.CoreV1().Secrets(secretClone.Namespace).Update(context.TODO(), secretClone, metav1.UpdateOptions{})
	if err != nil {
		klog.ErrorS(err, "Error removing protection finalizer from secret", "secret", klog.KObj(secret))
		return err
	}
	klog.V(3).InfoS("Removed protection finalizer from secret", "secret", klog.KObj(secret))
	return nil
}

func (c *Controller) isBeingUsed(secret *v1.Secret) (bool, error) {
	// Look for Pods and PVs using secret in the Informer's cache. If one is found the
	// correct decision to keep secret is taken without doing an expensive live
	// list.
	if inUse, err := c.askInformer(secret); err != nil {
		// No need to return because a live list will follow.
		klog.Error(err)
	} else if inUse {
		return true, nil
	}

	// Even if no Pod and PV using secret was found in the Informer's cache it doesn't
	// mean such a Pod or a PV doesn't exist: it might just not be in the cache yet. To
	// be 100% confident that it is safe to delete secret make sure no Pod is using
	// it among those returned by a live list.
	if inUse, err := c.askAPIServer(secret); err != nil {
		klog.Error(err)
	} else if inUse {
		return true, nil
	}

	return false, nil
}

func (c *Controller) askInformer(secret *v1.Secret) (bool, error) {
	klog.V(4).InfoS("Looking for Pods using secret in the Informer's cache", "secret", klog.KObj(secret))

	pods, err := c.podLister.List(labels.NewSelector())
	if err != nil {
		return false, fmt.Errorf("cache-based list of pods failed while processing %s/%s: %s", secret.Namespace, secret.Name, err.Error())
	}
	for _, pod := range pods {
		if c.podUsesSecret(pod, secret) {
			return true, nil
		}
	}

	klog.V(4).InfoS("Looking for PVs using secret in the Informer's cache", "secret", klog.KObj(secret))

	pvs, err := c.pvLister.List(labels.NewSelector())
	if err != nil {
		return false, fmt.Errorf("cache-based list of PVs failed while processing %s/%s: %s", secret.Namespace, secret.Name, err.Error())
	}
	for _, pv := range pvs {
		if c.pvUsesSecret(pv, secret) {
			return true, nil
		}
	}

	klog.V(4).InfoS("No Pod and PV using secret was found in the Informer's cache", "secret", klog.KObj(secret))
	return false, nil
}

func (c *Controller) askAPIServer(secret *v1.Secret) (bool, error) {
	klog.V(4).InfoS("Looking for Pods using secret with a live list", "secret", klog.KObj(secret))

	podsList, err := c.client.CoreV1().Pods(secret.Namespace).List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("live list of pods failed: %s", err.Error())
	}

	for _, pod := range podsList.Items {
		if c.podUsesSecret(&pod, secret) {
			return true, nil
		}
	}

	klog.V(4).InfoS("Looking for PVs using secret with a live list", "secret", klog.KObj(secret))
	pvsList, err := c.client.CoreV1().PersistentVolumes().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("live list of PVs failed: %s", err.Error())
	}

	for _, pv := range pvsList.Items {
		if c.pvUsesSecret(&pv, secret) {
			return true, nil
		}
	}

	klog.V(2).InfoS("secret is unused", "secret", klog.KObj(secret))
	return false, nil
}

func (c *Controller) podUsesSecret(pod *v1.Pod, secret *v1.Secret) bool {
	// Check whether secret is used by pod only if pod is scheduled, because
	// kubelet sees pods after they have been scheduled and it won't allow
	// starting a pod referencing a secret with a non-nil deletionTimestamp.
	// TODO: Check if above is also correct for secret.
	if pod.Spec.NodeName != "" {
		for _, volume := range pod.Spec.Volumes {
			if podIsShutDown(pod) {
				continue
			}
			// Check if referenced from volume.Secret
			if volume.Secret != nil && volume.Secret.SecretName == secret.Name {
				klog.V(2).InfoS("Pod uses Secret", "pod", klog.KObj(pod), "secret", klog.KObj(secret))
				return true
			}
		}
	}
	return false
}

func (c *Controller) pvUsesSecret(pv *v1.PersistentVolume, secret *v1.Secret) bool {
	secretKey := fmt.Sprintf("%s/%s", secret.Namespace, secret.Name)
	for _, secretKeyUsed := range getSecretsUsedByPV(pv) {
		if secretKey == secretKeyUsed {
			klog.V(2).InfoS("PV uses Secret", "pv", klog.KObj(pv), "secret", klog.KObj(secret))
			return true
		}
	}

	return false
}

// podIsShutDown returns true if kubelet is done with the pod or
// it was force-deleted.
func podIsShutDown(pod *v1.Pod) bool {
	// A pod that has a deletionTimestamp and a zero
	// deletionGracePeriodSeconds
	// a) has been processed by kubelet and was set up for deletion
	//    by the apiserver:
	//    - canBeDeleted has verified that volumes were unpublished
	//      https://github.com/kubernetes/kubernetes/blob/5404b5a28a2114299608bab00e4292960dd864a0/pkg/kubelet/kubelet_pods.go#L980
	//    - deletionGracePeriodSeconds was set via a delete
	//      with zero GracePeriodSeconds
	//      https://github.com/kubernetes/kubernetes/blob/5404b5a28a2114299608bab00e4292960dd864a0/pkg/kubelet/status/status_manager.go#L580-L592
	// or
	// b) was force-deleted.
	//
	// It's now just waiting for garbage collection. We could wait
	// for it to actually get removed, but that may be blocked by
	// finalizers for the pod and thus get delayed.
	//
	// Worse, it is possible that there is a cyclic dependency
	// (pod finalizer waits for secret to get removed, secret protection
	// controller waits for pod to get removed).  By considering
	// the secret unused in this case, we allow the secret to get
	// removed and break such a cycle.
	//
	// Therefore it is better to proceed with secret removal,
	// which is safe (case a) and/or desirable (case b).
	return pod.DeletionTimestamp != nil && pod.DeletionGracePeriodSeconds != nil && *pod.DeletionGracePeriodSeconds == 0
}

// secretAddedUpdated reacts to secret added/updated events
func (c *Controller) secretAddedUpdated(obj interface{}) {
	secret, ok := obj.(*v1.Secret)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("Secret informer returned non-secret object: %#v", obj))
		return
	}
	key, err := cache.MetaNamespaceKeyFunc(secret)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for secret %#v: %v", secret, err))
		return
	}
	klog.V(4).InfoS("Got event on secret", key)

	if protectionutil.NeedToAddFinalizer(secret, volumeutil.SecretProtectionFinalizer) || protectionutil.IsDeletionCandidate(secret, volumeutil.SecretProtectionFinalizer) {
		c.queue.Add(key)
	}
}

// podAddedDeletedUpdated reacts to Pod events
func (c *Controller) podAddedDeletedUpdated(old, new interface{}, deleted bool) {
	if pod := c.parsePod(new); pod != nil {
		c.enqueueSecretsForPod(pod, deleted)

		// An update notification might mask the deletion of a pod X and the
		// following creation of a pod Y with the same namespaced name as X. If
		// that's the case X needs to be processed as well to handle the case
		// where it is blocking deletion of a secret not referenced by Y, otherwise
		// such secret will never be deleted.
		if oldPod := c.parsePod(old); oldPod != nil && oldPod.UID != pod.UID {
			c.enqueueSecretsForPod(oldPod, true)
		}
	}
}

func (*Controller) parsePod(obj interface{}) *v1.Pod {
	if obj == nil {
		return nil
	}
	pod, ok := obj.(*v1.Pod)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("couldn't get object from tombstone %#v", obj))
			return nil
		}
		pod, ok = tombstone.Obj.(*v1.Pod)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("tombstone contained object that is not a Pod %#v", obj))
			return nil
		}
	}
	return pod
}

func (c *Controller) enqueueSecretsForPod(pod *v1.Pod, deleted bool) {
	// Filter out pods that can't help us to remove a finalizer on Secret
	if !deleted && !volumeutil.IsPodTerminated(pod, pod.Status) && pod.Spec.NodeName != "" {
		return
	}

	klog.V(4).InfoS("Enqueuing Secrets for Pod", "pod", klog.KObj(pod), "podUID", pod.UID)

	// Enqueue all Secrets that the pod uses
	for _, volume := range pod.Spec.Volumes {
		switch {
		case volume.Secret != nil:
			c.queue.Add(pod.Namespace + "/" + volume.Secret.SecretName)
		}
	}
}

// pvAddedDeletedUpdated reacts to PV events
func (c *Controller) pvAddedDeletedUpdated(old, new interface{}, deleted bool) {
	if pv := c.parsePV(new); pv != nil {
		c.enqueueSecretsForPV(pv, deleted)

		// An update notification might mask the deletion of a pv X and the
		// following creation of a pv Y with the same namespaced name as X. If
		// that's the case X needs to be processed as well to handle the case
		// where it is blocking deletion of a secret not referenced by Y, otherwise
		// such secret will never be deleted.
		if oldPV := c.parsePV(old); oldPV != nil && oldPV.UID != pv.UID {
			c.enqueueSecretsForPV(oldPV, true)
		}
	}
}

func (*Controller) parsePV(obj interface{}) *v1.PersistentVolume {
	if obj == nil {
		return nil
	}
	pv, ok := obj.(*v1.PersistentVolume)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("couldn't get object from tombstone %#v", obj))
			return nil
		}
		pv, ok = tombstone.Obj.(*v1.PersistentVolume)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("tombstone contained object that is not a PV %#v", obj))
			return nil
		}
	}
	return pv
}

func (c *Controller) enqueueSecretsForPV(pv *v1.PersistentVolume, deleted bool) {
	klog.V(4).InfoS("Enqueuing Secrets for PV", "pv", klog.KObj(pv), "pvUID", pv.UID)
	// Enqueue all Secrets that the PV uses
	for _, secretKey := range getSecretsUsedByPV(pv) {
		c.queue.Add(secretKey)
	}
}

func getSecretsUsedByPV(pv *v1.PersistentVolume) []string {
	secretKeys := []string{}

	switch {
	case pv.Spec.PersistentVolumeSource.CSI != nil:
		csi := pv.Spec.PersistentVolumeSource.CSI

		if csi.ControllerPublishSecretRef != nil {
			secretKeys = append(secretKeys, fmt.Sprintf("%s/%s",
				csi.ControllerPublishSecretRef.Namespace, csi.ControllerPublishSecretRef.Name))
		}

		if csi.NodeStageSecretRef != nil {
			secretKeys = append(secretKeys, fmt.Sprintf("%s/%s",
				csi.NodeStageSecretRef.Namespace, csi.NodeStageSecretRef.Name))
		}

		if csi.NodePublishSecretRef != nil {
			secretKeys = append(secretKeys, fmt.Sprintf("%s/%s",
				csi.NodePublishSecretRef.Namespace, csi.NodePublishSecretRef.Name))
		}

		if csi.ControllerExpandSecretRef != nil {
			secretKeys = append(secretKeys, fmt.Sprintf("%s/%s",
				csi.ControllerExpandSecretRef.Namespace, csi.ControllerExpandSecretRef.Name))
		}
	}

	return secretKeys
}
