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

package labeler

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/retry"
	"k8s.io/klog/v2"
)

const (
	// ApplysetPartOfLabel is the label key for ApplySet membership
	ApplysetPartOfLabel = "applyset.kubernetes.io/part-of"

	// HelmManagedByLabel is the standard Helm label for managed-by
	HelmManagedByLabel = "app.kubernetes.io/managed-by"

	// HelmInstanceLabel is the standard Helm label for instance name
	HelmInstanceLabel = "app.kubernetes.io/instance"

	// DefaultRetryAttempts is the default number of retry attempts for labeling operations
	DefaultRetryAttempts = 3

	// DefaultRetryDelay is the default delay between retries
	DefaultRetryDelay = 100 * time.Millisecond
)

// Labeler adds ApplySet labels to Helm-managed resources
type Labeler struct {
	dynamicClient dynamic.Interface
	kubeClient    kubernetes.Interface
	mapper        meta.RESTMapper
	logger        klog.Logger
	retryAttempts int
	retryDelay    time.Duration
}

// NewLabeler creates a new ApplySet resource labeler
func NewLabeler(
	dynamicClient dynamic.Interface,
	kubeClient kubernetes.Interface,
	mapper meta.RESTMapper,
	logger klog.Logger,
) *Labeler {
	return &Labeler{
		dynamicClient: dynamicClient,
		kubeClient:    kubeClient,
		mapper:        mapper,
		logger:        logger,
		retryAttempts: DefaultRetryAttempts,
		retryDelay:    DefaultRetryDelay,
	}
}

// LabelResources labels all resources in a Helm release with ApplySet labels
// It uses both label-based discovery and manifest parsing to find resources
func (l *Labeler) LabelResources(
	ctx context.Context,
	releaseName string,
	releaseNamespace string,
	manifest string,
	groupKinds sets.Set[schema.GroupKind],
	applySetID string,
) error {
	logger := klog.FromContext(ctx)
	l.logger = logger

	l.logger.Info("Labeling resources for Helm release",
		"release", releaseName,
		"namespace", releaseNamespace,
		"applySetID", applySetID)

	var allErrors []error

	// Method 1: Parse manifest to get exact resource list
	manifestResources, err := l.parseManifestResources(manifest)
	if err != nil {
		l.logger.Error(err, "Failed to parse manifest, falling back to label-based discovery",
			"release", releaseName)
		// Continue with label-based discovery
	} else {
		// Label resources from manifest
		for _, res := range manifestResources {
			if err := l.labelResourceFromManifest(ctx, res, applySetID); err != nil {
				l.logger.Error(err, "Failed to label resource from manifest",
					"groupKind", res.GroupKind.String(),
					"name", res.Name,
					"namespace", res.Namespace)
				allErrors = append(allErrors, err)
				// Continue with other resources
			}
		}
	}

	// Method 2: Label-based discovery (fallback or supplement)
	// Query resources by Helm labels
	if err := l.labelResourcesByHelmLabels(ctx, releaseName, releaseNamespace, applySetID, groupKinds); err != nil {
		l.logger.Error(err, "Failed to label resources by Helm labels",
			"release", releaseName)
		allErrors = append(allErrors, err)
	}

	if len(allErrors) > 0 {
		return utilerrors.NewAggregate(allErrors)
	}

	l.logger.Info("Successfully labeled resources for Helm release",
		"release", releaseName,
		"namespace", releaseNamespace)
	return nil
}

// ResourceInfo contains information about a resource from the manifest
type ResourceInfo struct {
	GroupKind  schema.GroupKind
	Name       string
	Namespace  string
	APIVersion string
}

// parseManifestResources parses the Helm manifest to extract resource information
func (l *Labeler) parseManifestResources(manifest string) ([]ResourceInfo, error) {
	var resources []ResourceInfo

	if manifest == "" {
		return resources, nil
	}

	// Use YAML decoder to parse multi-document YAML
	decoder := yaml.NewYAMLOrJSONDecoder(strings.NewReader(manifest), 4096)

	for {
		var obj unstructured.Unstructured
		err := decoder.Decode(&obj)
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			// Skip invalid YAML documents
			continue
		}

		if obj.GetName() == "" {
			continue
		}

		// Extract GroupVersionKind
		gvk := obj.GroupVersionKind()
		if gvk.Kind == "" {
			continue
		}

		resources = append(resources, ResourceInfo{
			GroupKind:  schema.GroupKind{Group: gvk.Group, Kind: gvk.Kind},
			Name:       obj.GetName(),
			Namespace:  obj.GetNamespace(),
			APIVersion: gvk.GroupVersion().String(),
		})
	}

	return resources, nil
}

// labelResourceFromManifest labels a specific resource identified from the manifest
func (l *Labeler) labelResourceFromManifest(
	ctx context.Context,
	res ResourceInfo,
	applySetID string,
) error {
	// Get REST mapping
	mapping, err := l.mapper.RESTMapping(res.GroupKind)
	if err != nil {
		return fmt.Errorf("failed to get REST mapping for %s: %w", res.GroupKind.String(), err)
	}

	// Determine if resource is namespaced
	isNamespaced := mapping.Scope.Name() == meta.RESTScopeNameNamespace

	// Get dynamic client for this resource
	namespaceableClient := l.dynamicClient.Resource(mapping.Resource)
	var resourceClient dynamic.ResourceInterface
	if isNamespaced {
		if res.Namespace == "" {
			return fmt.Errorf("namespaced resource %s/%s missing namespace", res.GroupKind.String(), res.Name)
		}
		resourceClient = namespaceableClient.Namespace(res.Namespace)
	} else {
		resourceClient = namespaceableClient
	}

	// Label the resource with retry logic
	return l.labelResourceWithRetry(ctx, resourceClient, res.Name, applySetID)
}

// labelResourcesByHelmLabels labels resources discovered by Helm labels
func (l *Labeler) labelResourcesByHelmLabels(
	ctx context.Context,
	releaseName string,
	namespace string,
	applySetID string,
	groupKinds sets.Set[schema.GroupKind],
) error {
	// Build label selector for Helm resources
	selector := labels.SelectorFromSet(map[string]string{
		HelmManagedByLabel: "Helm",
		HelmInstanceLabel:  releaseName,
	})

	// Label each GroupKind
	for gk := range groupKinds {
		if err := l.labelGroupKind(ctx, gk, namespace, selector, applySetID); err != nil {
			l.logger.Error(err, "Failed to label GroupKind",
				"groupKind", gk.String(),
				"release", releaseName)
			// Continue with other GroupKinds
		}
	}

	return nil
}

// labelGroupKind labels all resources of a specific GroupKind matching the selector
func (l *Labeler) labelGroupKind(
	ctx context.Context,
	gk schema.GroupKind,
	namespace string,
	selector labels.Selector,
	applySetID string,
) error {
	// Get REST mapping
	mapping, err := l.mapper.RESTMapping(gk)
	if err != nil {
		return fmt.Errorf("failed to get REST mapping for %s: %w", gk.String(), err)
	}

	// Determine if resource is namespaced
	isNamespaced := mapping.Scope.Name() == meta.RESTScopeNameNamespace

	// Get dynamic client for this resource
	namespaceableClient := l.dynamicClient.Resource(mapping.Resource)
	var resourceClient dynamic.ResourceInterface
	if isNamespaced {
		resourceClient = namespaceableClient.Namespace(namespace)
	} else {
		resourceClient = namespaceableClient
	}

	// List resources with selector
	list, err := resourceClient.List(ctx, metav1.ListOptions{
		LabelSelector: selector.String(),
	})
	if err != nil {
		return fmt.Errorf("failed to list resources: %w", err)
	}

	// Label each resource
	for _, item := range list.Items {
		if err := l.labelResourceWithRetry(ctx, resourceClient, item.GetName(), applySetID); err != nil {
			l.logger.Error(err, "Failed to label resource",
				"name", item.GetName(),
				"groupKind", gk.String())
			// Continue with other resources
		}
	}

	return nil
}

// labelResourceWithRetry labels a resource with retry logic
func (l *Labeler) labelResourceWithRetry(
	ctx context.Context,
	resourceClient dynamic.ResourceInterface,
	name string,
	applySetID string,
) error {
	return retry.OnError(
		wait.Backoff{
			Steps:    l.retryAttempts,
			Duration: l.retryDelay,
			Factor:   2.0,
			Jitter:   0.1,
		},
		func(err error) bool {
			// Retry on transient errors
			// Check for server errors (5xx), timeouts, and rate limiting
			if apierrors.IsTimeout(err) || apierrors.IsTooManyRequests(err) {
				return true
			}
			// Check for server errors by status code
			if statusErr, ok := err.(apierrors.APIStatus); ok {
				status := statusErr.Status()
				if status.Code >= 500 && status.Code < 600 {
					return true
				}
			}
			return false
		},
		func() error {
			return l.labelResource(ctx, resourceClient, name, applySetID)
		},
	)
}

// labelResource labels a single resource with ApplySet label using JSON merge patch
func (l *Labeler) labelResource(
	ctx context.Context,
	resourceClient dynamic.ResourceInterface,
	name string,
	applySetID string,
) error {
	// Get current resource
	obj, err := resourceClient.Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		if apierrors.IsNotFound(err) {
			// Resource doesn't exist, skip
			return nil
		}
		return fmt.Errorf("failed to get resource %s: %w", name, err)
	}

	// Check if label already exists and matches
	labels := obj.GetLabels()
	if labels == nil {
		labels = make(map[string]string)
	}

	if existingID, ok := labels[ApplysetPartOfLabel]; ok && existingID == applySetID {
		// Already labeled correctly, skip
		return nil
	}

	// Create patch for adding label
	patchObj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"metadata": map[string]interface{}{
				"labels": map[string]interface{}{
					ApplysetPartOfLabel: applySetID,
				},
			},
		},
	}

	patchBytes, err := json.Marshal(patchObj.Object)
	if err != nil {
		return fmt.Errorf("failed to marshal patch: %w", err)
	}

	// Apply JSON merge patch (works better with unstructured objects)
	_, err = resourceClient.Patch(
		ctx,
		name,
		types.MergePatchType,
		patchBytes,
		metav1.PatchOptions{},
	)
	if err != nil {
		return fmt.Errorf("failed to patch resource %s: %w", name, err)
	}

	l.logger.V(4).Info("Labeled resource",
		"name", name,
		"applySetID", applySetID)

	return nil
}

// RemoveLabels removes ApplySet labels from resources (for cleanup)
func (l *Labeler) RemoveLabels(
	ctx context.Context,
	releaseName string,
	namespace string,
	applySetID string,
	groupKinds sets.Set[schema.GroupKind],
) error {
	logger := klog.FromContext(ctx)
	l.logger = logger

	l.logger.Info("Removing ApplySet labels from resources",
		"release", releaseName,
		"namespace", namespace,
		"applySetID", applySetID)

	// Build label selector for resources with this ApplySet ID
	selector := labels.SelectorFromSet(map[string]string{
		ApplysetPartOfLabel: applySetID,
	})

	var allErrors []error

	// Remove labels from each GroupKind
	for gk := range groupKinds {
		if err := l.removeLabelsFromGroupKind(ctx, gk, namespace, selector, applySetID); err != nil {
			l.logger.Error(err, "Failed to remove labels from GroupKind",
				"groupKind", gk.String(),
				"release", releaseName)
			allErrors = append(allErrors, err)
		}
	}

	if len(allErrors) > 0 {
		return utilerrors.NewAggregate(allErrors)
	}

	return nil
}

// removeLabelsFromGroupKind removes ApplySet labels from all resources of a GroupKind
func (l *Labeler) removeLabelsFromGroupKind(
	ctx context.Context,
	gk schema.GroupKind,
	namespace string,
	selector labels.Selector,
	applySetID string,
) error {
	// Get REST mapping
	mapping, err := l.mapper.RESTMapping(gk)
	if err != nil {
		return fmt.Errorf("failed to get REST mapping for %s: %w", gk.String(), err)
	}

	// Determine if resource is namespaced
	isNamespaced := mapping.Scope.Name() == meta.RESTScopeNameNamespace

	// Get dynamic client for this resource
	namespaceableClient := l.dynamicClient.Resource(mapping.Resource)
	var resourceClient dynamic.ResourceInterface
	if isNamespaced {
		resourceClient = namespaceableClient.Namespace(namespace)
	} else {
		resourceClient = namespaceableClient
	}

	// List resources with selector
	list, err := resourceClient.List(ctx, metav1.ListOptions{
		LabelSelector: selector.String(),
	})
	if err != nil {
		return fmt.Errorf("failed to list resources: %w", err)
	}

	// Remove label from each resource
	for _, item := range list.Items {
		if err := l.removeLabelFromResource(ctx, resourceClient, item.GetName()); err != nil {
			l.logger.Error(err, "Failed to remove label from resource",
				"name", item.GetName(),
				"groupKind", gk.String())
			// Continue with other resources
		}
	}

	return nil
}

// removeLabelFromResource removes ApplySet label from a single resource
func (l *Labeler) removeLabelFromResource(
	ctx context.Context,
	resourceClient dynamic.ResourceInterface,
	name string,
) error {
	// Get current resource
	obj, err := resourceClient.Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return fmt.Errorf("failed to get resource %s: %w", name, err)
	}

	// Check if label exists
	labels := obj.GetLabels()
	if labels == nil {
		return nil // No labels, nothing to remove
	}

	if _, ok := labels[ApplysetPartOfLabel]; !ok {
		return nil // Label doesn't exist, nothing to remove
	}

	// Create patch for removing label
	// Use null value in strategic merge patch to remove the label
	patchObj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"metadata": map[string]interface{}{
				"labels": map[string]interface{}{
					ApplysetPartOfLabel: nil,
				},
			},
		},
	}

	patchBytes, err := json.Marshal(patchObj.Object)
	if err != nil {
		return fmt.Errorf("failed to marshal patch: %w", err)
	}

	// Apply JSON merge patch (works better with unstructured objects)
	_, err = resourceClient.Patch(
		ctx,
		name,
		types.MergePatchType,
		patchBytes,
		metav1.PatchOptions{},
	)
	if err != nil {
		return fmt.Errorf("failed to patch resource %s: %w", name, err)
	}

	return nil
}
