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

package status

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"

	apps "k8s.io/api/apps/v1"
	batch "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
)

const (
	// ApplySetPartOfLabel is the label key for ApplySet membership
	ApplySetPartOfLabel = "applyset.kubernetes.io/part-of"
)

// HealthStatus represents the health of a resource
type HealthStatus struct {
	Healthy   bool
	Reason    string
	Message   string
	Timestamp metav1.Time
}

// ReleaseHealth represents aggregated health of a Helm release
type ReleaseHealth struct {
	ReleaseName string
	Namespace   string
	ApplySetID  string

	OverallStatus string // healthy, progressing, degraded, failed, unknown

	ResourceHealth map[string]HealthStatus // key: "gvk/namespace/name"

	TotalResources       int
	HealthyResources     int
	ProgressingResources int
	DegradedResources    int
	FailedResources      int

	Timestamp metav1.Time
}

// GetUnhealthyResourceNames returns a sorted list of unhealthy (degraded) resource names
func (h *ReleaseHealth) GetUnhealthyResourceNames() []string {
	resources := make([]string, 0)
	for key, status := range h.ResourceHealth {
		if !status.Healthy && !strings.Contains(status.Reason, "Progressing") &&
			!strings.Contains(status.Reason, "Failed") {
			resources = append(resources, extractResourceName(key))
		}
	}
	sort.Strings(resources)
	return resources
}

// GetFailedResourceNames returns a sorted list of failed resource names
func (h *ReleaseHealth) GetFailedResourceNames() []string {
	resources := make([]string, 0)
	for key, status := range h.ResourceHealth {
		if !status.Healthy && strings.Contains(status.Reason, "Failed") {
			resources = append(resources, extractResourceName(key))
		}
	}
	sort.Strings(resources)
	return resources
}

// GetProgressingResourceNames returns a sorted list of progressing resource names
func (h *ReleaseHealth) GetProgressingResourceNames() []string {
	resources := make([]string, 0)
	for key, status := range h.ResourceHealth {
		if !status.Healthy && strings.Contains(status.Reason, "Progressing") {
			resources = append(resources, extractResourceName(key))
		}
	}
	sort.Strings(resources)
	return resources
}

// extractResourceName extracts a readable resource name from a resource key
// key format: "gvk/namespace/name"
func extractResourceName(key string) string {
	parts := strings.Split(key, "/")
	if len(parts) >= 3 {
		return fmt.Sprintf("%s/%s", parts[len(parts)-2], parts[len(parts)-1])
	}
	return key
}

// HealthChecker computes health for a specific resource type
type HealthChecker interface {
	CheckHealth(ctx context.Context, obj *unstructured.Unstructured) (HealthStatus, error)
	GroupKind() schema.GroupKind
}

// Aggregator watches resources and computes health status
type Aggregator struct {
	kubeClient    kubernetes.Interface
	dynamicClient dynamic.Interface
	mapper        meta.RESTMapper

	// Health checkers by resource type
	healthCheckers map[schema.GroupKind]HealthChecker

	logger klog.Logger
	mu     sync.RWMutex
}

// NewAggregator creates a new status aggregator
func NewAggregator(
	kubeClient kubernetes.Interface,
	dynamicClient dynamic.Interface,
	mapper meta.RESTMapper,
	logger klog.Logger,
) *Aggregator {
	agg := &Aggregator{
		kubeClient:     kubeClient,
		dynamicClient:  dynamicClient,
		mapper:         mapper,
		healthCheckers: make(map[schema.GroupKind]HealthChecker),
		logger:         logger,
	}

	// Register built-in health checkers
	agg.registerHealthCheckers()

	return agg
}

// registerHealthCheckers registers all built-in health checkers
func (a *Aggregator) registerHealthCheckers() {
	// Deployment
	a.healthCheckers[schema.GroupKind{Group: "apps", Kind: "Deployment"}] = &DeploymentHealthChecker{
		logger: a.logger,
	}

	// StatefulSet
	a.healthCheckers[schema.GroupKind{Group: "apps", Kind: "StatefulSet"}] = &StatefulSetHealthChecker{
		logger: a.logger,
	}

	// DaemonSet
	a.healthCheckers[schema.GroupKind{Group: "apps", Kind: "DaemonSet"}] = &DaemonSetHealthChecker{
		logger: a.logger,
	}

	// Service
	a.healthCheckers[schema.GroupKind{Group: "", Kind: "Service"}] = &ServiceHealthChecker{
		kubeClient: a.kubeClient,
		logger:     a.logger,
	}

	// PersistentVolumeClaim
	a.healthCheckers[schema.GroupKind{Group: "", Kind: "PersistentVolumeClaim"}] = &PVCHealthChecker{
		logger: a.logger,
	}

	// Job
	a.healthCheckers[schema.GroupKind{Group: "batch", Kind: "Job"}] = &JobHealthChecker{
		logger: a.logger,
	}

	// Custom resources use generic checker
	a.healthCheckers[schema.GroupKind{Group: "", Kind: ""}] = &GenericHealthChecker{
		logger: a.logger,
	}
}

// AggregateHealth computes aggregated health for a Helm release
func (a *Aggregator) AggregateHealth(
	ctx context.Context,
	releaseName, namespace, applySetID string,
	groupKinds sets.Set[schema.GroupKind],
) (*ReleaseHealth, error) {
	logger := klog.FromContext(ctx)
	logger.Info("Aggregating health for Helm release",
		"release", releaseName,
		"namespace", namespace,
		"applySetID", applySetID)

	health := &ReleaseHealth{
		ReleaseName:    releaseName,
		Namespace:      namespace,
		ApplySetID:     applySetID,
		ResourceHealth: make(map[string]HealthStatus),
		Timestamp:      metav1.Now(),
	}

	// Query resources by ApplySet label
	labelSelector := fmt.Sprintf("%s=%s", ApplySetPartOfLabel, applySetID)

	var allErrors []error
	var wg sync.WaitGroup
	var mu sync.Mutex

	// Check health for each GroupKind
	for gk := range groupKinds {
		wg.Add(1)
		go func(groupKind schema.GroupKind) {
			defer wg.Done()

			resources, err := a.getResourcesByGroupKind(ctx, namespace, groupKind, labelSelector)
			if err != nil {
				logger.Error(err, "Failed to get resources",
					"groupKind", groupKind.String())
				mu.Lock()
				allErrors = append(allErrors, err)
				mu.Unlock()
				return
			}

			// Check health for each resource
			for _, resource := range resources {
				resourceHealth, err := a.checkResourceHealth(ctx, groupKind, resource)
				if err != nil {
					logger.Error(err, "Failed to check resource health",
						"groupKind", groupKind.String(),
						"name", resource.GetName())
					mu.Lock()
					allErrors = append(allErrors, err)
					mu.Unlock()
					continue
				}

				// Store health status
				key := fmt.Sprintf("%s/%s/%s", groupKind.String(), resource.GetNamespace(), resource.GetName())
				mu.Lock()
				health.ResourceHealth[key] = resourceHealth
				mu.Unlock()
			}
		}(gk)
	}

	wg.Wait()

	// Compute aggregate status
	health.computeAggregateStatus()

	if len(allErrors) > 0 {
		logger.V(4).Info("Some errors occurred during health aggregation",
			"errorCount", len(allErrors))
	}

	return health, nil
}

// getResourcesByGroupKind gets resources of a specific GroupKind with label selector
func (a *Aggregator) getResourcesByGroupKind(
	ctx context.Context,
	namespace string,
	gk schema.GroupKind,
	labelSelector string,
) ([]*unstructured.Unstructured, error) {
	// Get REST mapping
	mapping, err := a.mapper.RESTMapping(gk)
	if err != nil {
		return nil, fmt.Errorf("failed to get REST mapping for %s: %w", gk.String(), err)
	}

	// Get resource client
	var resourceClient dynamic.ResourceInterface
	if mapping.Scope.Name() == meta.RESTScopeNameRoot {
		// Cluster-scoped resource
		resourceClient = a.dynamicClient.Resource(mapping.Resource)
	} else {
		// Namespace-scoped resource
		resourceClient = a.dynamicClient.Resource(mapping.Resource).Namespace(namespace)
	}

	// List resources with label selector
	list, err := resourceClient.List(ctx, metav1.ListOptions{
		LabelSelector: labelSelector,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to list resources: %w", err)
	}

	resources := make([]*unstructured.Unstructured, 0, len(list.Items))
	for i := range list.Items {
		resources = append(resources, &list.Items[i])
	}

	return resources, nil
}

// checkResourceHealth checks health of a single resource
func (a *Aggregator) checkResourceHealth(
	ctx context.Context,
	gk schema.GroupKind,
	obj *unstructured.Unstructured,
) (HealthStatus, error) {
	// Get health checker for this GroupKind
	checker, ok := a.healthCheckers[gk]
	if !ok {
		// Use generic checker for unknown types
		checker = a.healthCheckers[schema.GroupKind{}]
	}

	return checker.CheckHealth(ctx, obj)
}

// computeAggregateStatus computes overall release status from individual resource health
func (h *ReleaseHealth) computeAggregateStatus() {
	h.TotalResources = len(h.ResourceHealth)
	h.HealthyResources = 0
	h.ProgressingResources = 0
	h.DegradedResources = 0
	h.FailedResources = 0

	for _, status := range h.ResourceHealth {
		if status.Healthy {
			h.HealthyResources++
		} else {
			// Determine if progressing or degraded based on reason
			if containsAny(status.Reason, "Progressing", "Updating", "Scaling") {
				h.ProgressingResources++
			} else if containsAny(status.Reason, "Failed", "Error", "CrashLoopBackOff") {
				h.FailedResources++
			} else {
				h.DegradedResources++
			}
		}
	}

	// Determine overall status
	if h.TotalResources == 0 {
		h.OverallStatus = "unknown"
	} else if h.FailedResources > 0 {
		h.OverallStatus = "failed"
	} else if h.DegradedResources > 0 {
		h.OverallStatus = "degraded"
	} else if h.ProgressingResources > 0 {
		h.OverallStatus = "progressing"
	} else if h.HealthyResources == h.TotalResources {
		h.OverallStatus = "healthy"
	} else {
		h.OverallStatus = "unknown"
	}
}

// containsAny checks if a string contains any of the given substrings
func containsAny(s string, substrings ...string) bool {
	for _, substr := range substrings {
		if strings.Contains(s, substr) {
			return true
		}
	}
	return false
}

// DeploymentHealthChecker checks health of Deployments
type DeploymentHealthChecker struct {
	logger klog.Logger
}

func (c *DeploymentHealthChecker) GroupKind() schema.GroupKind {
	return schema.GroupKind{Group: "apps", Kind: "Deployment"}
}

func (c *DeploymentHealthChecker) CheckHealth(ctx context.Context, obj *unstructured.Unstructured) (HealthStatus, error) {
	var deployment apps.Deployment
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(obj.Object, &deployment); err != nil {
		return HealthStatus{
			Healthy: false,
			Reason:  "ParseError",
			Message: fmt.Sprintf("Failed to parse Deployment: %v", err),
		}, nil
	}

	status := HealthStatus{
		Timestamp: metav1.Now(),
	}

	specReplicas := int32(1)
	if deployment.Spec.Replicas != nil {
		specReplicas = *deployment.Spec.Replicas
	}

	readyReplicas := deployment.Status.ReadyReplicas
	updatedReplicas := deployment.Status.UpdatedReplicas
	availableReplicas := deployment.Status.AvailableReplicas

	// Check if deployment is healthy
	if readyReplicas == specReplicas &&
		updatedReplicas == specReplicas &&
		availableReplicas >= specReplicas {
		status.Healthy = true
		status.Reason = "AllReplicasReady"
		status.Message = fmt.Sprintf("Deployment has %d/%d ready replicas", readyReplicas, specReplicas)
	} else if updatedReplicas < specReplicas {
		status.Healthy = false
		status.Reason = "Progressing"
		status.Message = fmt.Sprintf("Deployment is updating: %d/%d replicas updated", updatedReplicas, specReplicas)
	} else if readyReplicas < specReplicas {
		status.Healthy = false
		status.Reason = "ReplicasNotReady"
		status.Message = fmt.Sprintf("Deployment has %d/%d ready replicas", readyReplicas, specReplicas)
	} else {
		status.Healthy = false
		status.Reason = "Unknown"
		status.Message = fmt.Sprintf("Deployment status unclear: ready=%d, updated=%d, available=%d, spec=%d",
			readyReplicas, updatedReplicas, availableReplicas, specReplicas)
	}

	return status, nil
}

// StatefulSetHealthChecker checks health of StatefulSets
type StatefulSetHealthChecker struct {
	logger klog.Logger
}

func (c *StatefulSetHealthChecker) GroupKind() schema.GroupKind {
	return schema.GroupKind{Group: "apps", Kind: "StatefulSet"}
}

func (c *StatefulSetHealthChecker) CheckHealth(ctx context.Context, obj *unstructured.Unstructured) (HealthStatus, error) {
	var statefulSet apps.StatefulSet
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(obj.Object, &statefulSet); err != nil {
		return HealthStatus{
			Healthy: false,
			Reason:  "ParseError",
			Message: fmt.Sprintf("Failed to parse StatefulSet: %v", err),
		}, nil
	}

	status := HealthStatus{
		Timestamp: metav1.Now(),
	}

	specReplicas := int32(1)
	if statefulSet.Spec.Replicas != nil {
		specReplicas = *statefulSet.Spec.Replicas
	}

	readyReplicas := statefulSet.Status.ReadyReplicas
	currentReplicas := statefulSet.Status.CurrentReplicas

	if readyReplicas == specReplicas && currentReplicas == specReplicas {
		status.Healthy = true
		status.Reason = "AllReplicasReady"
		status.Message = fmt.Sprintf("StatefulSet has %d/%d ready replicas", readyReplicas, specReplicas)
	} else if currentReplicas < specReplicas {
		status.Healthy = false
		status.Reason = "Progressing"
		status.Message = fmt.Sprintf("StatefulSet is updating: %d/%d replicas current", currentReplicas, specReplicas)
	} else {
		status.Healthy = false
		status.Reason = "ReplicasNotReady"
		status.Message = fmt.Sprintf("StatefulSet has %d/%d ready replicas", readyReplicas, specReplicas)
	}

	return status, nil
}

// DaemonSetHealthChecker checks health of DaemonSets
type DaemonSetHealthChecker struct {
	logger klog.Logger
}

func (c *DaemonSetHealthChecker) GroupKind() schema.GroupKind {
	return schema.GroupKind{Group: "apps", Kind: "DaemonSet"}
}

func (c *DaemonSetHealthChecker) CheckHealth(ctx context.Context, obj *unstructured.Unstructured) (HealthStatus, error) {
	var daemonSet apps.DaemonSet
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(obj.Object, &daemonSet); err != nil {
		return HealthStatus{
			Healthy: false,
			Reason:  "ParseError",
			Message: fmt.Sprintf("Failed to parse DaemonSet: %v", err),
		}, nil
	}

	status := HealthStatus{
		Timestamp: metav1.Now(),
	}

	desiredNumberScheduled := daemonSet.Status.DesiredNumberScheduled
	numberReady := daemonSet.Status.NumberReady
	numberUnavailable := daemonSet.Status.NumberUnavailable

	if numberReady == desiredNumberScheduled && numberUnavailable == 0 {
		status.Healthy = true
		status.Reason = "AllPodsReady"
		status.Message = fmt.Sprintf("DaemonSet has %d/%d pods ready", numberReady, desiredNumberScheduled)
	} else if numberUnavailable > 0 {
		status.Healthy = false
		status.Reason = "PodsUnavailable"
		status.Message = fmt.Sprintf("DaemonSet has %d unavailable pods (%d/%d ready)",
			numberUnavailable, numberReady, desiredNumberScheduled)
	} else {
		status.Healthy = false
		status.Reason = "Progressing"
		status.Message = fmt.Sprintf("DaemonSet is progressing: %d/%d pods ready", numberReady, desiredNumberScheduled)
	}

	return status, nil
}

// ServiceHealthChecker checks health of Services
type ServiceHealthChecker struct {
	kubeClient kubernetes.Interface
	logger     klog.Logger
}

func (c *ServiceHealthChecker) GroupKind() schema.GroupKind {
	return schema.GroupKind{Group: "", Kind: "Service"}
}

func (c *ServiceHealthChecker) CheckHealth(ctx context.Context, obj *unstructured.Unstructured) (HealthStatus, error) {
	var service v1.Service
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(obj.Object, &service); err != nil {
		return HealthStatus{
			Healthy: false,
			Reason:  "ParseError",
			Message: fmt.Sprintf("Failed to parse Service: %v", err),
		}, nil
	}

	status := HealthStatus{
		Timestamp: metav1.Now(),
	}

	// Skip headless services or services without selectors
	if service.Spec.ClusterIP == v1.ClusterIPNone || len(service.Spec.Selector) == 0 {
		status.Healthy = true
		status.Reason = "NoEndpointsRequired"
		status.Message = "Service does not require endpoints"
		return status, nil
	}

	// Check endpoints
	endpoints, err := c.kubeClient.CoreV1().Endpoints(service.Namespace).Get(ctx, service.Name, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		status.Healthy = false
		status.Reason = "NoEndpoints"
		status.Message = "Service has no endpoints"
		return status, nil
	}
	if err != nil {
		return HealthStatus{
			Healthy: false,
			Reason:  "EndpointCheckError",
			Message: fmt.Sprintf("Failed to check endpoints: %v", err),
		}, err
	}

	// Check if endpoints have ready addresses
	readyAddresses := 0
	for _, subset := range endpoints.Subsets {
		readyAddresses += len(subset.Addresses)
	}

	if readyAddresses > 0 {
		status.Healthy = true
		status.Reason = "EndpointsReady"
		status.Message = fmt.Sprintf("Service has %d ready endpoint addresses", readyAddresses)
	} else {
		status.Healthy = false
		status.Reason = "NoReadyEndpoints"
		status.Message = "Service has no ready endpoint addresses"
	}

	return status, nil
}

// PVCHealthChecker checks health of PersistentVolumeClaims
type PVCHealthChecker struct {
	logger klog.Logger
}

func (c *PVCHealthChecker) GroupKind() schema.GroupKind {
	return schema.GroupKind{Group: "", Kind: "PersistentVolumeClaim"}
}

func (c *PVCHealthChecker) CheckHealth(ctx context.Context, obj *unstructured.Unstructured) (HealthStatus, error) {
	var pvc v1.PersistentVolumeClaim
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(obj.Object, &pvc); err != nil {
		return HealthStatus{
			Healthy: false,
			Reason:  "ParseError",
			Message: fmt.Sprintf("Failed to parse PVC: %v", err),
		}, nil
	}

	status := HealthStatus{
		Timestamp: metav1.Now(),
	}

	phase := pvc.Status.Phase
	if phase == v1.ClaimBound {
		status.Healthy = true
		status.Reason = "Bound"
		status.Message = "PVC is bound"
	} else if phase == v1.ClaimPending {
		status.Healthy = false
		status.Reason = "Pending"
		status.Message = "PVC is pending binding"
	} else if phase == v1.ClaimLost {
		status.Healthy = false
		status.Reason = "Lost"
		status.Message = "PVC is lost"
	} else {
		status.Healthy = false
		status.Reason = "Unknown"
		status.Message = fmt.Sprintf("PVC has unknown phase: %s", phase)
	}

	return status, nil
}

// JobHealthChecker checks health of Jobs
type JobHealthChecker struct {
	logger klog.Logger
}

func (c *JobHealthChecker) GroupKind() schema.GroupKind {
	return schema.GroupKind{Group: "batch", Kind: "Job"}
}

func (c *JobHealthChecker) CheckHealth(ctx context.Context, obj *unstructured.Unstructured) (HealthStatus, error) {
	var job batch.Job
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(obj.Object, &job); err != nil {
		return HealthStatus{
			Healthy: false,
			Reason:  "ParseError",
			Message: fmt.Sprintf("Failed to parse Job: %v", err),
		}, nil
	}

	status := HealthStatus{
		Timestamp: metav1.Now(),
	}

	completions := int32(1)
	if job.Spec.Completions != nil {
		completions = *job.Spec.Completions
	}

	succeeded := job.Status.Succeeded
	failed := job.Status.Failed
	active := job.Status.Active

	if succeeded >= completions {
		status.Healthy = true
		status.Reason = "Completed"
		status.Message = fmt.Sprintf("Job completed successfully: %d/%d completions", succeeded, completions)
	} else if failed > 0 && active == 0 {
		status.Healthy = false
		status.Reason = "Failed"
		status.Message = fmt.Sprintf("Job failed: %d failed pods", failed)
	} else if active > 0 {
		status.Healthy = false
		status.Reason = "Progressing"
		status.Message = fmt.Sprintf("Job is running: %d active pods, %d/%d completed", active, succeeded, completions)
	} else {
		status.Healthy = false
		status.Reason = "Unknown"
		status.Message = fmt.Sprintf("Job status unclear: succeeded=%d, failed=%d, active=%d, completions=%d",
			succeeded, failed, active, completions)
	}

	return status, nil
}

// GenericHealthChecker checks health of custom resources using conditions
type GenericHealthChecker struct {
	logger klog.Logger
}

func (c *GenericHealthChecker) GroupKind() schema.GroupKind {
	return schema.GroupKind{} // Empty means generic
}

func (c *GenericHealthChecker) CheckHealth(ctx context.Context, obj *unstructured.Unstructured) (HealthStatus, error) {
	status := HealthStatus{
		Timestamp: metav1.Now(),
	}

	// Try to find Ready condition
	conditions, found, err := unstructured.NestedSlice(obj.Object, "status", "conditions")
	if err != nil || !found {
		// No conditions - assume healthy if resource exists
		status.Healthy = true
		status.Reason = "NoConditions"
		status.Message = "Resource has no conditions, assuming healthy"
		return status, nil
	}

	// Look for Ready condition
	for _, cond := range conditions {
		condMap, ok := cond.(map[string]interface{})
		if !ok {
			continue
		}

		condType, _ := condMap["type"].(string)
		condStatus, _ := condMap["status"].(string)
		condReason, _ := condMap["reason"].(string)
		condMessage, _ := condMap["message"].(string)

		if condType == "Ready" || condType == "Available" {
			if condStatus == "True" {
				status.Healthy = true
				status.Reason = condReason
				status.Message = condMessage
				return status, nil
			} else {
				status.Healthy = false
				status.Reason = condReason
				status.Message = condMessage
				return status, nil
			}
		}
	}

	// No Ready condition found - check for any negative conditions
	for _, cond := range conditions {
		condMap, ok := cond.(map[string]interface{})
		if !ok {
			continue
		}

		condType, _ := condMap["type"].(string)
		condStatus, _ := condMap["status"].(string)
		condReason, _ := condMap["reason"].(string)
		condMessage, _ := condMap["message"].(string)

		if condStatus == "False" && (condType == "Degraded" || condType == "Failed") {
			status.Healthy = false
			status.Reason = condReason
			status.Message = condMessage
			return status, nil
		}
	}

	// Default to unknown
	status.Healthy = false
	status.Reason = "Unknown"
	status.Message = "Resource has no recognizable health conditions"
	return status, nil
}
