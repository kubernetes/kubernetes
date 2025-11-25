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

package parent

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"sort"
	"strings"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
)

const (
	// ApplySetParentIDLabel is the label key for ApplySet parent ID
	// This matches the constant from kubectl applyset implementation
	ApplySetParentIDLabel = "applyset.kubernetes.io/id"

	// ApplySetToolingAnnotation is the annotation key for tooling identifier
	ApplySetToolingAnnotation = "applyset.kubernetes.io/tooling"

	// ApplySetGKsAnnotation is the annotation key for group-kinds list
	ApplySetGKsAnnotation = "applyset.kubernetes.io/contains-group-kinds"

	// V1ApplySetIdFormat is the format for ApplySet ID
	V1ApplySetIdFormat = "applyset-%s-v1"

	// HelmReleaseTooling is the tooling identifier for Helm releases
	HelmReleaseTooling = "helm/v3"

	// ParentSecretNamePrefix is the prefix for ApplySet parent Secret names
	ParentSecretNamePrefix = "applyset-"

	// applySetIDPartDelimiter is the delimiter used in ApplySet ID computation
	applySetIDPartDelimiter = "."
)

// Manager manages ApplySet parent Secrets for Helm releases
type Manager struct {
	client   kubernetes.Interface
	recorder record.EventRecorder
	logger   klog.Logger
}

// NewManager creates a new ApplySet parent manager
func NewManager(
	client kubernetes.Interface,
	recorder record.EventRecorder,
	logger klog.Logger,
) *Manager {
	return &Manager{
		client:   client,
		recorder: recorder,
		logger:   logger,
	}
}

// ComputeApplySetID computes the ApplySet ID from release name and namespace
// Format: base64(sha256(<name>.<namespace>.Secret.)) using URL-safe encoding
func ComputeApplySetID(releaseName, namespace string) string {
	// Format: <name>.<namespace>.<kind>.<group>
	// For ApplySet parent Secret: <name>.<namespace>.Secret.
	unencoded := strings.Join([]string{
		releaseName,
		namespace,
		"Secret",
		"", // Empty group for core v1
	}, applySetIDPartDelimiter)

	hashed := sha256.Sum256([]byte(unencoded))
	b64 := base64.RawURLEncoding.EncodeToString(hashed[:])
	return fmt.Sprintf(V1ApplySetIdFormat, b64)
}

// CreateOrUpdateParent creates or updates the ApplySet parent Secret for a Helm release
// This is idempotent - it will create if not exists, or update if labels/annotations changed
func (m *Manager) CreateOrUpdateParent(
	ctx context.Context,
	releaseName string,
	releaseNamespace string,
	groupKinds sets.Set[schema.GroupKind],
	helmReleaseSecret *v1.Secret,
) (*v1.Secret, error) {
	logger := klog.FromContext(ctx)
	m.logger = logger

	applySetID := ComputeApplySetID(releaseName, releaseNamespace)
	parentName := fmt.Sprintf("%s%s", ParentSecretNamePrefix, releaseName)

	// Format GroupKinds annotation
	groupKindsList := formatGroupKinds(groupKinds)

	// Build desired parent Secret
	desiredSecret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      parentName,
			Namespace: releaseNamespace,
			Labels: map[string]string{
				ApplySetParentIDLabel: applySetID,
			},
			Annotations: map[string]string{
				ApplySetToolingAnnotation: HelmReleaseTooling,
				ApplySetGKsAnnotation:     groupKindsList,
			},
			OwnerReferences: []metav1.OwnerReference{
				{
					APIVersion: "v1",
					Kind:       "Secret",
					Name:       helmReleaseSecret.Name,
					UID:        helmReleaseSecret.UID,
					Controller: func() *bool { b := true; return &b }(),
				},
			},
		},
		Type: v1.SecretTypeOpaque,
	}

	// Try to get existing parent Secret
	existingSecret, err := m.client.CoreV1().Secrets(releaseNamespace).Get(
		ctx,
		parentName,
		metav1.GetOptions{},
	)

	if apierrors.IsNotFound(err) {
		// Create new parent Secret
		created, createErr := m.client.CoreV1().Secrets(releaseNamespace).Create(
			ctx,
			desiredSecret,
			metav1.CreateOptions{},
		)
		if createErr != nil {
			m.logger.Error(createErr, "Failed to create ApplySet parent Secret",
				"name", parentName,
				"namespace", releaseNamespace,
				"applySetID", applySetID)
			if m.recorder != nil {
				m.recorder.Eventf(helmReleaseSecret, v1.EventTypeWarning, "FailedCreateApplySetParent",
					"Failed to create ApplySet parent Secret %s: %v", parentName, createErr)
			}
			return nil, fmt.Errorf("failed to create ApplySet parent Secret: %w", createErr)
		}

		m.logger.Info("Created ApplySet parent Secret",
			"name", parentName,
			"namespace", releaseNamespace,
			"applySetID", applySetID,
			"release", releaseName)
		if m.recorder != nil {
			m.recorder.Eventf(helmReleaseSecret, v1.EventTypeNormal, "CreatedApplySetParent",
				"Created ApplySet parent Secret %s", parentName)
		}
		return created, nil
	}

	if err != nil {
		m.logger.Error(err, "Failed to get existing ApplySet parent Secret",
			"name", parentName,
			"namespace", releaseNamespace)
		return nil, fmt.Errorf("failed to get existing ApplySet parent Secret: %w", err)
	}

	// Check if update is needed
	needsUpdate := false

	// Check labels
	if existingSecret.Labels == nil {
		existingSecret.Labels = make(map[string]string)
		needsUpdate = true
	}
	if existingSecret.Labels[ApplySetParentIDLabel] != applySetID {
		existingSecret.Labels[ApplySetParentIDLabel] = applySetID
		needsUpdate = true
	}

	// Check annotations
	if existingSecret.Annotations == nil {
		existingSecret.Annotations = make(map[string]string)
		needsUpdate = true
	}
	if existingSecret.Annotations[ApplySetToolingAnnotation] != HelmReleaseTooling {
		existingSecret.Annotations[ApplySetToolingAnnotation] = HelmReleaseTooling
		needsUpdate = true
	}
	if existingSecret.Annotations[ApplySetGKsAnnotation] != groupKindsList {
		existingSecret.Annotations[ApplySetGKsAnnotation] = groupKindsList
		needsUpdate = true
	}

	// Check owner reference
	hasOwnerRef := false
	for _, ref := range existingSecret.OwnerReferences {
		if ref.Kind == "Secret" && ref.Name == helmReleaseSecret.Name {
			hasOwnerRef = true
			// Update UID if it changed
			if ref.UID != helmReleaseSecret.UID {
				ref.UID = helmReleaseSecret.UID
				needsUpdate = true
			}
			break
		}
	}
	if !hasOwnerRef {
		existingSecret.OwnerReferences = desiredSecret.OwnerReferences
		needsUpdate = true
	}

	if !needsUpdate {
		m.logger.V(4).Info("ApplySet parent Secret already up to date",
			"name", parentName,
			"namespace", releaseNamespace,
			"applySetID", applySetID)
		return existingSecret, nil
	}

	// Update existing parent Secret
	updated, updateErr := m.client.CoreV1().Secrets(releaseNamespace).Update(
		ctx,
		existingSecret,
		metav1.UpdateOptions{},
	)
	if updateErr != nil {
		m.logger.Error(updateErr, "Failed to update ApplySet parent Secret",
			"name", parentName,
			"namespace", releaseNamespace,
			"applySetID", applySetID)
		if m.recorder != nil {
			m.recorder.Eventf(helmReleaseSecret, v1.EventTypeWarning, "FailedUpdateApplySetParent",
				"Failed to update ApplySet parent Secret %s: %v", parentName, updateErr)
		}
		return nil, fmt.Errorf("failed to update ApplySet parent Secret: %w", updateErr)
	}

	m.logger.Info("Updated ApplySet parent Secret",
		"name", parentName,
		"namespace", releaseNamespace,
		"applySetID", applySetID,
		"release", releaseName)
	if m.recorder != nil {
		m.recorder.Eventf(helmReleaseSecret, v1.EventTypeNormal, "UpdatedApplySetParent",
			"Updated ApplySet parent Secret %s", parentName)
	}
	return updated, nil
}

// DeleteParent deletes the ApplySet parent Secret for a Helm release
// This is idempotent - it will not error if the Secret doesn't exist
func (m *Manager) DeleteParent(
	ctx context.Context,
	releaseName string,
	namespace string,
) error {
	logger := klog.FromContext(ctx)
	m.logger = logger

	parentName := fmt.Sprintf("%s%s", ParentSecretNamePrefix, releaseName)

	err := m.client.CoreV1().Secrets(namespace).Delete(ctx, parentName, metav1.DeleteOptions{})
	if apierrors.IsNotFound(err) {
		m.logger.V(4).Info("ApplySet parent Secret already deleted",
			"name", parentName,
			"namespace", namespace)
		return nil // Already deleted, no error
	}
	if err != nil {
		m.logger.Error(err, "Failed to delete ApplySet parent Secret",
			"name", parentName,
			"namespace", namespace)
		return fmt.Errorf("failed to delete ApplySet parent Secret: %w", err)
	}

	m.logger.Info("Deleted ApplySet parent Secret",
		"name", parentName,
		"namespace", namespace,
		"release", releaseName)
	return nil
}

// GetParent retrieves the ApplySet parent Secret for a Helm release
func (m *Manager) GetParent(
	ctx context.Context,
	releaseName string,
	namespace string,
) (*v1.Secret, error) {
	logger := klog.FromContext(ctx)
	m.logger = logger

	parentName := fmt.Sprintf("%s%s", ParentSecretNamePrefix, releaseName)

	secret, err := m.client.CoreV1().Secrets(namespace).Get(ctx, parentName, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}

	return secret, nil
}

// formatGroupKinds formats a set of GroupKinds into a comma-separated string
// Format: <Kind>.<Group> (e.g., "Deployment.apps,Service,ConfigMap")
func formatGroupKinds(groupKinds sets.Set[schema.GroupKind]) string {
	if groupKinds.Len() == 0 {
		return ""
	}

	var parts []string
	for gk := range groupKinds {
		if gk.Group == "" {
			parts = append(parts, gk.Kind)
		} else {
			parts = append(parts, fmt.Sprintf("%s.%s", gk.Kind, gk.Group))
		}
	}

	// Sort for deterministic output
	sort.Strings(parts)
	return strings.Join(parts, ",")
}

// ParseGroupKinds parses a comma-separated string of GroupKinds back into a set
// This is useful for testing and validation
func ParseGroupKinds(groupKindsStr string) (sets.Set[schema.GroupKind], error) {
	groupKinds := sets.New[schema.GroupKind]()

	if groupKindsStr == "" {
		return groupKinds, nil
	}

	parts := strings.Split(groupKindsStr, ",")
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		// Parse format: <Kind>.<Group> or just <Kind>
		var gk schema.GroupKind
		if strings.Contains(part, ".") {
			dotIndex := strings.LastIndex(part, ".")
			gk.Kind = part[:dotIndex]
			gk.Group = part[dotIndex+1:]
		} else {
			gk.Kind = part
			gk.Group = "" // Core group
		}

		groupKinds.Insert(gk)
	}

	return groupKinds, nil
}

// ValidateParentSecret validates that a Secret has the correct structure for an ApplySet parent
func ValidateParentSecret(secret *v1.Secret) error {
	if secret == nil {
		return fmt.Errorf("secret is nil")
	}

	// Check required label
	if secret.Labels == nil {
		return fmt.Errorf("secret %s/%s missing labels", secret.Namespace, secret.Name)
	}

	applySetID, ok := secret.Labels[ApplySetParentIDLabel]
	if !ok {
		return fmt.Errorf("secret %s/%s missing label %s", secret.Namespace, secret.Name, ApplySetParentIDLabel)
	}

	// Validate ApplySet ID format
	if !strings.HasPrefix(applySetID, "applyset-") {
		return fmt.Errorf("secret %s/%s has invalid ApplySet ID format: %s", secret.Namespace, secret.Name, applySetID)
	}
	if !strings.HasSuffix(applySetID, "-v1") {
		return fmt.Errorf("secret %s/%s has invalid ApplySet ID format: %s", secret.Namespace, secret.Name, applySetID)
	}

	// Check required annotations
	if secret.Annotations == nil {
		return fmt.Errorf("secret %s/%s missing annotations", secret.Namespace, secret.Name)
	}

	tooling, ok := secret.Annotations[ApplySetToolingAnnotation]
	if !ok {
		return fmt.Errorf("secret %s/%s missing annotation %s", secret.Namespace, secret.Name, ApplySetToolingAnnotation)
	}
	if tooling != HelmReleaseTooling {
		return fmt.Errorf("secret %s/%s has incorrect tooling annotation: %s (expected %s)",
			secret.Namespace, secret.Name, tooling, HelmReleaseTooling)
	}

	// GroupKinds annotation is optional but should be present for Helm releases
	if _, ok := secret.Annotations[ApplySetGKsAnnotation]; !ok {
		// This is a warning, not an error, as the annotation is optional per spec
		// But we expect it for Helm releases
	}

	return nil
}
