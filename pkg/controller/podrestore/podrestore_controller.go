/*
Copyright 2026 The Kubernetes Authors.

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

package podrestore

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	checkpointv1alpha1 "k8s.io/api/checkpoint/v1alpha1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
)

var podRestoreGVR = schema.GroupVersionResource{
	Group:    "checkpoint.k8s.io",
	Version:  "v1alpha1",
	Resource: "podrestores",
}

var podCheckpointGVR = schema.GroupVersionResource{
	Group:    "checkpoint.k8s.io",
	Version:  "v1alpha1",
	Resource: "podcheckpoints",
}

// checkpointLocationResponse is the JSON structure returned by the kubelet checkpoint API.
type checkpointLocationResponse struct {
	Items []string `json:"items"`
}

// Controller watches PodRestore resources and creates restored pods.
type Controller struct {
	kubeClient    kubernetes.Interface
	dynamicClient dynamic.Interface
	queue         workqueue.TypedRateLimitingInterface[string]
	informer      cache.Controller
	store         cache.Store
}

// NewController creates a new PodRestore controller.
func NewController(
	kubeClient kubernetes.Interface,
	dynamicClient dynamic.Interface,
) *Controller {
	c := &Controller{
		kubeClient:    kubeClient,
		dynamicClient: dynamicClient,
		queue:         workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[string]()),
	}

	listWatcher := &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
			return dynamicClient.Resource(podRestoreGVR).Namespace("").List(context.TODO(), options)
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			return dynamicClient.Resource(podRestoreGVR).Namespace("").Watch(context.TODO(), options)
		},
	}

	store, controller := cache.NewInformer(
		listWatcher,
		&unstructured.Unstructured{},
		0,
		cache.ResourceEventHandlerFuncs{
			AddFunc:    c.onAdd,
			UpdateFunc: c.onUpdate,
		},
	)
	c.store = store
	c.informer = controller

	return c
}

func (c *Controller) onAdd(obj interface{}) {
	key, err := cache.MetaNamespaceKeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(err)
		return
	}
	c.queue.Add(key)
}

func (c *Controller) onUpdate(oldObj, newObj interface{}) {
	key, err := cache.MetaNamespaceKeyFunc(newObj)
	if err != nil {
		utilruntime.HandleError(err)
		return
	}
	c.queue.Add(key)
}

// Run starts the controller.
func (c *Controller) Run(ctx context.Context, workers int) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	klog.InfoS("Starting PodRestore controller")
	defer klog.InfoS("Shutting down PodRestore controller")

	go c.informer.Run(ctx.Done())

	if !cache.WaitForCacheSync(ctx.Done(), c.informer.HasSynced) {
		klog.ErrorS(nil, "Failed to sync PodRestore informer cache")
		return
	}

	for i := 0; i < workers; i++ {
		go wait.UntilWithContext(ctx, c.worker, time.Second)
	}

	<-ctx.Done()
}

func (c *Controller) worker(ctx context.Context) {
	for c.processNextWorkItem(ctx) {
	}
}

func (c *Controller) processNextWorkItem(ctx context.Context) bool {
	key, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(key)

	err := c.syncHandler(ctx, key)
	if err == nil {
		c.queue.Forget(key)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("sync %q failed: %v", key, err))
	c.queue.AddRateLimited(key)
	return true
}

func (c *Controller) syncHandler(ctx context.Context, key string) error {
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}

	obj, err := c.dynamicClient.Resource(podRestoreGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		if errors.IsNotFound(err) {
			return nil
		}
		return err
	}

	var pr checkpointv1alpha1.PodRestore
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(obj.Object, &pr); err != nil {
		return fmt.Errorf("failed to convert unstructured to PodRestore: %v", err)
	}

	// Skip if already processed
	if pr.Status.Phase == checkpointv1alpha1.PodRestoreCompleted ||
		pr.Status.Phase == checkpointv1alpha1.PodRestoreFailed ||
		pr.Status.Phase == checkpointv1alpha1.PodRestoreRestoring {
		return nil
	}

	// Look up the referenced PodCheckpoint
	cpObj, err := c.dynamicClient.Resource(podCheckpointGVR).Namespace(namespace).Get(ctx, pr.Spec.CheckpointName, metav1.GetOptions{})
	if err != nil {
		return c.updateStatus(ctx, obj, string(checkpointv1alpha1.PodRestoreFailed), "",
			fmt.Sprintf("failed to get PodCheckpoint %q: %v", pr.Spec.CheckpointName, err))
	}

	var pc checkpointv1alpha1.PodCheckpoint
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(cpObj.Object, &pc); err != nil {
		return c.updateStatus(ctx, obj, string(checkpointv1alpha1.PodRestoreFailed), "",
			fmt.Sprintf("failed to convert PodCheckpoint: %v", err))
	}

	// Verify checkpoint is Ready
	if pc.Status.Phase != checkpointv1alpha1.PodCheckpointReady {
		return c.updateStatus(ctx, obj, string(checkpointv1alpha1.PodRestoreFailed), "",
			fmt.Sprintf("PodCheckpoint %q is not ready (phase: %s)", pr.Spec.CheckpointName, pc.Status.Phase))
	}

	// Parse checkpoint location to get the archive path
	restoreFrom, err := parseCheckpointLocation(pc.Status.CheckpointLocation)
	if err != nil {
		return c.updateStatus(ctx, obj, string(checkpointv1alpha1.PodRestoreFailed), "",
			fmt.Sprintf("failed to parse checkpoint location: %v", err))
	}

	// Update status to Restoring
	if err := c.updateStatus(ctx, obj, string(checkpointv1alpha1.PodRestoreRestoring), "", "creating restored pod"); err != nil {
		return err
	}

	// Re-fetch to get updated resourceVersion
	obj, err = c.dynamicClient.Resource(podRestoreGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return err
	}

	// Create the restored pod
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: name + "-",
			Namespace:    namespace,
			Labels:       pr.Spec.PodTemplate.Labels,
			Annotations:  pr.Spec.PodTemplate.Annotations,
		},
		Spec: pr.Spec.PodTemplate.Spec,
	}

	// Set RestoreFrom to the checkpoint archive path
	pod.Spec.RestoreFrom = &restoreFrom

	// Schedule on the same node where the checkpoint was created
	if pc.Status.NodeName != "" {
		pod.Spec.NodeName = pc.Status.NodeName
	}

	createdPod, err := c.kubeClient.CoreV1().Pods(namespace).Create(ctx, pod, metav1.CreateOptions{})
	if err != nil {
		return c.updateStatus(ctx, obj, string(checkpointv1alpha1.PodRestoreFailed), "",
			fmt.Sprintf("failed to create restored pod: %v", err))
	}

	return c.updateStatus(ctx, obj, string(checkpointv1alpha1.PodRestoreCompleted), createdPod.Name,
		fmt.Sprintf("restored pod %q created successfully", createdPod.Name))
}

func (c *Controller) updateStatus(ctx context.Context, obj *unstructured.Unstructured, phase, restoredPodName, message string) error {
	status := map[string]interface{}{
		"phase":   phase,
		"message": message,
	}
	if restoredPodName != "" {
		status["restoredPodName"] = restoredPodName
	}

	if err := unstructured.SetNestedField(obj.Object, status, "status"); err != nil {
		return fmt.Errorf("failed to set status: %v", err)
	}

	_, err := c.dynamicClient.Resource(podRestoreGVR).Namespace(obj.GetNamespace()).UpdateStatus(ctx, obj, metav1.UpdateOptions{})
	if err != nil {
		return fmt.Errorf("failed to update PodRestore status: %v", err)
	}
	return nil
}

// parseCheckpointLocation parses the checkpoint location JSON and returns
// the first checkpoint archive path.
func parseCheckpointLocation(location string) (string, error) {
	if location == "" {
		return "", fmt.Errorf("checkpoint location is empty")
	}

	var resp checkpointLocationResponse
	if err := json.Unmarshal([]byte(location), &resp); err != nil {
		// If it's not JSON, treat the location as a direct path
		return location, nil
	}

	if len(resp.Items) == 0 {
		return "", fmt.Errorf("checkpoint location has no items")
	}

	return resp.Items[0], nil
}
