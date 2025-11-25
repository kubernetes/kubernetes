/*
Copyright 2024 The Kubernetes Authors.

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

package helmapplyset

import (
	"context"
	"fmt"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	corev1informers "k8s.io/client-go/informers/core/v1"
	"k8s.io/client-go/kubernetes"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"

	"k8s.io/kubernetes/pkg/controller/helmapplyset/events"
	"k8s.io/kubernetes/pkg/controller/helmapplyset/labeler"
	"k8s.io/kubernetes/pkg/controller/helmapplyset/metrics"
	"k8s.io/kubernetes/pkg/controller/helmapplyset/parent"
)

const (
	// ControllerName is the name of this controller
	ControllerName = "helm-applyset-controller"

	// DefaultResyncPeriod is the default resync period for informers
	DefaultResyncPeriod = 15 * time.Minute

	// MaxRetries is the maximum number of retries for a failed reconciliation
	MaxRetries = 5
)

// Controller reconciles Helm releases with ApplySet metadata
type Controller struct {
	// Kubernetes clients
	kubeClient    kubernetes.Interface
	dynamicClient dynamic.Interface

	// Informers
	secretInformer corev1informers.SecretInformer
	secretSynced   cache.InformerSynced

	// Components
	watcher       *Watcher
	parentManager *parent.Manager
	labeler       *labeler.Labeler
	mapper        meta.RESTMapper

	// Events and metrics
	eventGenerator  *events.EventGenerator
	metricsExporter *metrics.Exporter

	// Workqueue
	queue workqueue.TypedRateLimitingInterface[string]

	// Event recorder
	recorder record.EventRecorder

	// Logger
	logger klog.Logger
}

// NewController creates a new HelmApplySet controller
func NewController(
	ctx context.Context,
	kubeClient kubernetes.Interface,
	dynamicClient dynamic.Interface,
	secretInformer corev1informers.SecretInformer,
	mapper meta.RESTMapper,
) (*Controller, error) {
	logger := klog.FromContext(ctx)

	// Create event broadcaster
	eventBroadcaster := record.NewBroadcaster(record.WithContext(ctx))
	eventBroadcaster.StartStructuredLogging(0)
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: kubeClient.CoreV1().Events("")})
	recorder := eventBroadcaster.NewRecorder(runtime.NewScheme(), v1.EventSource{Component: ControllerName})

	// Create workqueue
	queue := workqueue.NewTypedRateLimitingQueueWithConfig(
		workqueue.DefaultTypedControllerRateLimiter[string](),
		workqueue.TypedRateLimitingQueueConfig[string]{
			Name: ControllerName,
		},
	)

	// Create parent manager
	parentManager := parent.NewManager(kubeClient, recorder, logger)

	// Create labeler
	labelerInstance := labeler.NewLabeler(dynamicClient, kubeClient, mapper, logger)

	// Create event generator
	eventGenerator := events.NewEventGenerator(recorder, logger)

	// Create metrics exporter
	metricsExporter := metrics.NewExporter(logger)

	// Register metrics
	metrics.Register()

	// Create watcher
	watcher := NewWatcher(secretInformer, func(obj interface{}) {
		key, err := cache.MetaNamespaceKeyFunc(obj)
		if err != nil {
			utilruntime.HandleError(err)
			return
		}
		queue.Add(key)
	}, logger)

	controller := &Controller{
		kubeClient:      kubeClient,
		dynamicClient:   dynamicClient,
		secretInformer:  secretInformer,
		secretSynced:    secretInformer.Informer().HasSynced,
		watcher:         watcher,
		parentManager:   parentManager,
		labeler:         labelerInstance,
		mapper:          mapper,
		eventGenerator:  eventGenerator,
		metricsExporter: metricsExporter,
		queue:           queue,
		recorder:        recorder,
		logger:          logger,
	}

	return controller, nil
}

// Run starts the controller and blocks until the context is cancelled
func (c *Controller) Run(ctx context.Context, workers int) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	logger := klog.FromContext(ctx)
	c.logger = logger

	c.logger.Info("Starting HelmApplySet controller", "workers", workers)
	defer c.logger.Info("Shutting down HelmApplySet controller")

	// Start watcher
	c.watcher.Start(ctx)

	// Wait for caches to sync
	if !cache.WaitForNamedCacheSyncWithContext(ctx, c.secretSynced) {
		c.logger.Error(nil, "Failed to wait for caches to sync")
		return
	}

	c.logger.Info("Caches synced, starting workers")

	// Start workers
	for i := 0; i < workers; i++ {
		go wait.UntilWithContext(ctx, c.runWorker, time.Second)
	}

	<-ctx.Done()
	c.logger.Info("Context cancelled, stopping workers")
}

// runWorker runs a worker that processes items from the workqueue
func (c *Controller) runWorker(ctx context.Context) {
	for c.processNextWorkItem(ctx) {
	}
}

// processNextWorkItem processes a single item from the workqueue
func (c *Controller) processNextWorkItem(ctx context.Context) bool {
	obj, shutdown := c.queue.Get()
	if shutdown {
		return false
	}

	defer c.queue.Done(obj)

	// Process the item
	err := c.syncHandler(ctx, obj)
	if err == nil {
		// Success - forget the item
		c.queue.Forget(obj)
		return true
	}

	// Check if it's a permanent error
	if isPermanentError(err) {
		c.logger.Error(err, "Permanent error, not retrying",
			"key", obj)
		c.queue.Forget(obj)
		return true
	}

	// Retry with backoff
	retries := c.queue.NumRequeues(obj)
	if retries < MaxRetries {
		c.logger.V(4).Info("Retrying reconciliation",
			"key", obj,
			"retries", retries,
			"error", err)
		c.queue.AddRateLimited(obj)
		return true
	}

	// Max retries exceeded
	c.logger.Error(err, "Max retries exceeded, giving up",
		"key", obj,
		"retries", retries)
	c.queue.Forget(obj)
	utilruntime.HandleError(err)
	return true
}

// syncHandler reconciles a Helm release Secret
func (c *Controller) syncHandler(ctx context.Context, key string) error {
	logger := klog.FromContext(ctx)
	c.logger = logger

	// Track reconciliation start time for metrics
	reconcileStart := time.Now()

	// Parse namespace/name from key
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("invalid resource key: %s", key))
		return nil
	}

	c.logger.V(4).Info("Reconciling Helm release Secret",
		"namespace", namespace,
		"name", name)

	// Get Helm release Secret from informer cache
	secret, err := c.secretInformer.Lister().Secrets(namespace).Get(name)
	if apierrors.IsNotFound(err) {
		// Secret was deleted - handle cleanup
		c.logger.V(4).Info("Helm release Secret not found, may have been deleted",
			"namespace", namespace,
			"name", name)
		// Extract release name from Secret name for cleanup
		releaseName, err := ExtractReleaseNameFromSecretName(name)
		if err != nil {
			c.logger.V(4).Info("Could not extract release name from Secret name, skipping cleanup",
				"secretName", name,
				"error", err)
			return nil
		}
		return c.handleReleaseDeletion(ctx, releaseName, namespace)
	}
	if err != nil {
		return fmt.Errorf("failed to get Helm release Secret %s/%s: %w", namespace, name, err)
	}

	// Check if it's a Helm release Secret
	if !IsHelmReleaseSecret(secret) {
		c.logger.V(5).Info("Secret is not a Helm release Secret, skipping",
			"namespace", namespace,
			"name", name)
		return nil
	}

	// Parse Helm release Secret
	releaseInfo, err := ParseHelmReleaseSecret(secret)
	if err != nil {
		return fmt.Errorf("failed to parse Helm release Secret %s/%s: %w", namespace, name, err)
	}

	c.logger.Info("Reconciling Helm release",
		"release", releaseInfo.Name,
		"namespace", releaseInfo.Namespace,
		"version", releaseInfo.Version,
		"status", releaseInfo.Status)

	// Step 1: Create or update ApplySet parent Secret
	applySetID := parent.ComputeApplySetID(releaseInfo.Name, releaseInfo.Namespace)
	parentSecret, err := c.parentManager.CreateOrUpdateParent(ctx, releaseInfo.Name, releaseInfo.Namespace, releaseInfo.GroupKinds, secret)
	if err != nil {
		return fmt.Errorf("failed to create/update ApplySet parent Secret: %w", err)
	}

	c.logger.V(4).Info("Created/updated ApplySet parent Secret",
		"parentSecret", parentSecret.Name,
		"applySetID", applySetID)

	// Step 2: Label all managed resources
	if err := c.labeler.LabelResources(ctx, releaseInfo.Name, releaseInfo.Namespace, releaseInfo.Manifest, releaseInfo.GroupKinds, applySetID); err != nil {
		// Log error but don't fail reconciliation - partial success is better than complete failure
		c.logger.Error(err, "Failed to label some resources, continuing",
			"release", releaseInfo.Name,
			"namespace", releaseInfo.Namespace)
		// Record event for visibility
		c.recorder.Eventf(secret, v1.EventTypeWarning, "LabelingFailed",
			"Failed to label some resources for Helm release %s: %v", releaseInfo.Name, err)
		// Return error to trigger retry, but allow partial success
		return fmt.Errorf("failed to label resources: %w", err)
	}

	c.logger.Info("Successfully reconciled Helm release",
		"release", releaseInfo.Name,
		"namespace", releaseInfo.Namespace,
		"version", releaseInfo.Version)

	c.recorder.Eventf(secret, v1.EventTypeNormal, "Reconciled",
		"Successfully reconciled Helm release %s with ApplySet metadata", releaseInfo.Name)

	// Update metrics
	reconcileDuration := time.Since(reconcileStart)
	// Note: For full metrics, we'd need to aggregate health status using the status aggregator
	// For now, metrics are registered and ready. Full integration with status aggregator
	// would update metrics here: c.metricsExporter.UpdateMetrics(health, secret.CreationTimestamp.Time, reconcileDuration)
	_ = reconcileDuration // Suppress unused variable warning until full integration

	return nil
}

// handleReleaseDeletion handles cleanup when a Helm release Secret is deleted
func (c *Controller) handleReleaseDeletion(ctx context.Context, releaseName, namespace string) error {
	logger := klog.FromContext(ctx)
	c.logger = logger

	c.logger.Info("Handling Helm release deletion",
		"release", releaseName,
		"namespace", namespace)

	// Try to get parent Secret to determine GroupKinds
	parentSecret, err := c.parentManager.GetParent(ctx, releaseName, namespace)
	if apierrors.IsNotFound(err) {
		// Parent Secret doesn't exist, nothing to clean up
		c.logger.V(4).Info("ApplySet parent Secret not found, nothing to clean up",
			"release", releaseName)
		return nil
	}
	if err != nil {
		return fmt.Errorf("failed to get ApplySet parent Secret: %w", err)
	}

	// Extract GroupKinds from parent Secret annotation
	groupKindsStr, ok := parentSecret.Annotations[parent.ApplySetGKsAnnotation]
	if !ok {
		c.logger.V(4).Info("Parent Secret missing GroupKinds annotation, skipping label removal",
			"release", releaseName)
	} else {
		// Parse GroupKinds and remove labels
		groupKinds, err := parent.ParseGroupKinds(groupKindsStr)
		if err != nil {
			c.logger.Error(err, "Failed to parse GroupKinds from parent Secret",
				"release", releaseName,
				"groupKindsStr", groupKindsStr)
			// Continue with parent deletion even if parsing fails
		} else if groupKinds.Len() > 0 {
			applySetID := parentSecret.Labels[parent.ApplySetParentIDLabel]
			if err := c.labeler.RemoveLabels(ctx, releaseName, namespace, applySetID, groupKinds); err != nil {
				c.logger.Error(err, "Failed to remove labels during cleanup",
					"release", releaseName)
				// Continue with parent deletion even if label removal fails
			}
		}
	}

	// Delete parent Secret
	if err := c.parentManager.DeleteParent(ctx, releaseName, namespace); err != nil {
		return fmt.Errorf("failed to delete ApplySet parent Secret: %w", err)
	}

	c.logger.Info("Successfully cleaned up Helm release",
		"release", releaseName,
		"namespace", namespace)

	return nil
}

// isPermanentError determines if an error is permanent and should not be retried
func isPermanentError(err error) bool {
	if err == nil {
		return false
	}

	// Invalid Secret format - can't be fixed by retrying
	if apierrors.IsInvalid(err) {
		return true
	}

	// Check for specific error messages that indicate permanent failures
	errStr := err.Error()
	permanentErrorPatterns := []string{
		"invalid Helm release Secret name format",
		"missing 'release' key",
		"failed to decode Helm release data",
		"failed to unmarshal Helm release JSON",
		"missing 'name' field",
		"missing 'namespace' field",
		"missing 'manifest' field",
	}

	for _, pattern := range permanentErrorPatterns {
		if strings.Contains(errStr, pattern) {
			return true
		}
	}

	return false
}
