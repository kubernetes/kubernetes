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

package podcheckpoint

import (
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"net"
	"net/http"
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
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/transport"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
)

var podCheckpointGVR = schema.GroupVersionResource{
	Group:    "checkpoint.k8s.io",
	Version:  "v1alpha1",
	Resource: "podcheckpoints",
}

// Controller watches PodCheckpoint resources and triggers checkpoint operations.
type Controller struct {
	kubeClient    kubernetes.Interface
	dynamicClient dynamic.Interface
	restConfig    *restclient.Config
	queue         workqueue.TypedRateLimitingInterface[string]
	informer      cache.Controller
	store         cache.Store
}

// NewController creates a new PodCheckpoint controller.
func NewController(
	kubeClient kubernetes.Interface,
	dynamicClient dynamic.Interface,
	restConfig *restclient.Config,
) *Controller {
	c := &Controller{
		kubeClient:    kubeClient,
		dynamicClient: dynamicClient,
		restConfig:    restConfig,
		queue:         workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[string]()),
	}

	listWatcher := &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
			return dynamicClient.Resource(podCheckpointGVR).Namespace("").List(context.TODO(), options)
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			return dynamicClient.Resource(podCheckpointGVR).Namespace("").Watch(context.TODO(), options)
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

	klog.InfoS("Starting PodCheckpoint controller")
	defer klog.InfoS("Shutting down PodCheckpoint controller")

	go c.informer.Run(ctx.Done())

	if !cache.WaitForCacheSync(ctx.Done(), c.informer.HasSynced) {
		klog.ErrorS(nil, "Failed to sync PodCheckpoint informer cache")
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

	obj, err := c.dynamicClient.Resource(podCheckpointGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		if errors.IsNotFound(err) {
			return nil
		}
		return err
	}

	var pc checkpointv1alpha1.PodCheckpoint
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(obj.Object, &pc); err != nil {
		return fmt.Errorf("failed to convert unstructured to PodCheckpoint: %v", err)
	}

	// Skip if already processed
	if pc.Status.Phase == checkpointv1alpha1.PodCheckpointReady ||
		pc.Status.Phase == checkpointv1alpha1.PodCheckpointFailed ||
		pc.Status.Phase == checkpointv1alpha1.PodCheckpointInProgress {
		return nil
	}

	// Look up the source pod
	pod, err := c.kubeClient.CoreV1().Pods(namespace).Get(ctx, pc.Spec.SourcePodName, metav1.GetOptions{})
	if err != nil {
		return c.updateStatus(ctx, obj, string(checkpointv1alpha1.PodCheckpointFailed), "", "",
			fmt.Sprintf("failed to get source pod %q: %v", pc.Spec.SourcePodName, err))
	}

	if pod.Status.Phase != v1.PodRunning {
		return c.updateStatus(ctx, obj, string(checkpointv1alpha1.PodCheckpointFailed), "", "",
			fmt.Sprintf("source pod %q is not running (phase: %s)", pc.Spec.SourcePodName, pod.Status.Phase))
	}

	nodeName := pod.Spec.NodeName
	if nodeName == "" {
		return c.updateStatus(ctx, obj, string(checkpointv1alpha1.PodCheckpointFailed), "", "",
			fmt.Sprintf("source pod %q is not assigned to a node", pc.Spec.SourcePodName))
	}

	// Update status to InProgress
	if err := c.updateStatus(ctx, obj, string(checkpointv1alpha1.PodCheckpointInProgress), nodeName, "", "checkpoint in progress"); err != nil {
		return err
	}

	// Re-fetch to get updated resourceVersion
	obj, err = c.dynamicClient.Resource(podCheckpointGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return err
	}

	// Call the kubelet checkpoint API
	checkpointLocation, err := c.callKubeletCheckpoint(ctx, nodeName, namespace, pc.Spec.SourcePodName)
	if err != nil {
		return c.updateStatus(ctx, obj, string(checkpointv1alpha1.PodCheckpointFailed), nodeName, "",
			fmt.Sprintf("checkpoint failed: %v", err))
	}

	return c.updateStatus(ctx, obj, string(checkpointv1alpha1.PodCheckpointReady), nodeName, checkpointLocation, "checkpoint completed successfully")
}

func (c *Controller) updateStatus(ctx context.Context, obj *unstructured.Unstructured, phase, nodeName, checkpointLocation, message string) error {
	status := map[string]interface{}{
		"phase":   phase,
		"message": message,
	}
	if nodeName != "" {
		status["nodeName"] = nodeName
	}
	if checkpointLocation != "" {
		status["checkpointLocation"] = checkpointLocation
	}

	if err := unstructured.SetNestedField(obj.Object, status, "status"); err != nil {
		return fmt.Errorf("failed to set status: %v", err)
	}

	_, err := c.dynamicClient.Resource(podCheckpointGVR).Namespace(obj.GetNamespace()).UpdateStatus(ctx, obj, metav1.UpdateOptions{})
	if err != nil {
		return fmt.Errorf("failed to update PodCheckpoint status: %v", err)
	}
	return nil
}

func (c *Controller) callKubeletCheckpoint(ctx context.Context, nodeName, namespace, podName string) (string, error) {
	node, err := c.kubeClient.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
	if err != nil {
		return "", fmt.Errorf("failed to get node %q: %v", nodeName, err)
	}

	var nodeAddress string
	for _, addr := range node.Status.Addresses {
		if addr.Type == v1.NodeInternalIP {
			nodeAddress = addr.Address
			break
		}
	}
	if nodeAddress == "" {
		return "", fmt.Errorf("no internal IP found for node %q", nodeName)
	}

	url := fmt.Sprintf("https://%s/checkpointpod/%s/%s", net.JoinHostPort(nodeAddress, "10250"), namespace, podName)

	// Build an HTTP client that carries the same auth credentials as the
	// controller's API server client (bearer token or client certs).
	// We skip kubelet server certificate verification (like kubectl does
	// for the checkpoint curl command).
	transportConfig, err := c.restConfig.TransportConfig()
	if err != nil {
		return "", fmt.Errorf("failed to get transport config: %v", err)
	}

	// Override TLS to skip kubelet server cert verification and target
	// the kubelet rather than the API server.
	transportConfig.TLS.Insecure = true
	transportConfig.TLS.ServerName = ""
	transportConfig.TLS.CAFile = ""
	transportConfig.TLS.CAData = nil

	tlsConfig, err := transport.TLSConfigFor(transportConfig)
	if err != nil {
		return "", fmt.Errorf("failed to create TLS config: %v", err)
	}
	if tlsConfig == nil {
		tlsConfig = &tls.Config{InsecureSkipVerify: true}
	}

	rt, err := transport.New(transportConfig)
	if err != nil {
		return "", fmt.Errorf("failed to create transport: %v", err)
	}

	client := &http.Client{
		Transport: rt,
		Timeout:   5 * time.Minute,
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, nil)
	if err != nil {
		return "", fmt.Errorf("failed to create request: %v", err)
	}

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("kubelet checkpoint request failed: %v", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("kubelet checkpoint failed with status %d: %s", resp.StatusCode, string(body))
	}

	return string(body), nil
}
