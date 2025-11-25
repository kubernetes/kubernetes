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
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2/ktesting"
)

// testHelmReleaseInfo is a test-only struct to avoid import cycles
type testHelmReleaseInfo struct {
	Name       string
	Namespace  string
	Version    int
	GroupKinds sets.Set[schema.GroupKind]
}

func TestComputeApplySetID(t *testing.T) {
	tests := []struct {
		name         string
		releaseName  string
		namespace    string
		expectedHash string // We'll check the format, not exact hash
	}{
		{
			name:        "simple release",
			releaseName: "my-app",
			namespace:   "default",
		},
		{
			name:        "release with dash",
			releaseName: "my-long-app-name",
			namespace:   "production",
		},
		{
			name:        "release in custom namespace",
			releaseName: "test-release",
			namespace:   "custom-ns",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			id := ComputeApplySetID(tt.releaseName, tt.namespace)

			// Validate format: applyset-<base64-hash>-v1
			if !strings.HasPrefix(id, "applyset-") {
				t.Errorf("ApplySet ID should start with 'applyset-', got: %s", id)
			}
			if !strings.HasSuffix(id, "-v1") {
				t.Errorf("ApplySet ID should end with '-v1', got: %s", id)
			}

			// Check deterministic - same inputs should produce same ID
			id2 := ComputeApplySetID(tt.releaseName, tt.namespace)
			if id != id2 {
				t.Errorf("ApplySet ID should be deterministic, got different IDs: %s vs %s", id, id2)
			}

			// Check hash part length (base64 URL-safe encoding of SHA256 = 43 chars)
			hashPart := strings.TrimPrefix(strings.TrimSuffix(id, "-v1"), "applyset-")
			if len(hashPart) != 43 {
				t.Errorf("ApplySet ID hash part should be 43 characters, got %d: %s", len(hashPart), hashPart)
			}
		})
	}
}

func TestFormatGroupKinds(t *testing.T) {
	tests := []struct {
		name       string
		groupKinds sets.Set[schema.GroupKind]
		expected   string
	}{
		{
			name:       "empty set",
			groupKinds: sets.New[schema.GroupKind](),
			expected:   "",
		},
		{
			name: "core resources",
			groupKinds: sets.New[schema.GroupKind](
				schema.GroupKind{Group: "", Kind: "Service"},
				schema.GroupKind{Group: "", Kind: "ConfigMap"},
			),
			expected: "ConfigMap,Service", // Should be sorted
		},
		{
			name: "mixed core and apps",
			groupKinds: sets.New[schema.GroupKind](
				schema.GroupKind{Group: "apps", Kind: "Deployment"},
				schema.GroupKind{Group: "", Kind: "Service"},
				schema.GroupKind{Group: "apps", Kind: "StatefulSet"},
			),
			expected: "Deployment.apps,Service,StatefulSet.apps", // Should be sorted
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := formatGroupKinds(tt.groupKinds)
			if result != tt.expected {
				t.Errorf("formatGroupKinds() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestParseGroupKinds(t *testing.T) {
	tests := []struct {
		name        string
		input       string
		expected    sets.Set[schema.GroupKind]
		expectError bool
	}{
		{
			name:     "empty string",
			input:    "",
			expected: sets.New[schema.GroupKind](),
		},
		{
			name:  "core resources",
			input: "Service,ConfigMap",
			expected: sets.New[schema.GroupKind](
				schema.GroupKind{Group: "", Kind: "Service"},
				schema.GroupKind{Group: "", Kind: "ConfigMap"},
			),
		},
		{
			name:  "mixed core and apps",
			input: "Deployment.apps,Service,StatefulSet.apps",
			expected: sets.New[schema.GroupKind](
				schema.GroupKind{Group: "apps", Kind: "Deployment"},
				schema.GroupKind{Group: "", Kind: "Service"},
				schema.GroupKind{Group: "apps", Kind: "StatefulSet"},
			),
		},
		{
			name:  "with spaces",
			input: "Deployment.apps, Service , ConfigMap",
			expected: sets.New[schema.GroupKind](
				schema.GroupKind{Group: "apps", Kind: "Deployment"},
				schema.GroupKind{Group: "", Kind: "Service"},
				schema.GroupKind{Group: "", Kind: "ConfigMap"},
			),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := ParseGroupKinds(tt.input)
			if tt.expectError {
				if err == nil {
					t.Errorf("ParseGroupKinds() expected error but got none")
				}
				return
			}
			if err != nil {
				t.Errorf("ParseGroupKinds() error = %v", err)
				return
			}
			if !result.Equal(tt.expected) {
				t.Errorf("ParseGroupKinds() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestValidateParentSecret(t *testing.T) {
	validID := ComputeApplySetID("test-release", "default")

	tests := []struct {
		name        string
		secret      *v1.Secret
		expectError bool
	}{
		{
			name:        "nil secret",
			secret:      nil,
			expectError: true,
		},
		{
			name: "valid parent secret",
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "applyset-test-release",
					Namespace: "default",
					Labels: map[string]string{
						ApplySetParentIDLabel: validID,
					},
					Annotations: map[string]string{
						ApplySetToolingAnnotation: HelmReleaseTooling,
						ApplySetGKsAnnotation:     "Deployment.apps,Service",
					},
				},
				Type: v1.SecretTypeOpaque,
			},
			expectError: false,
		},
		{
			name: "missing labels",
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "applyset-test-release",
					Namespace: "default",
					Annotations: map[string]string{
						ApplySetToolingAnnotation: HelmReleaseTooling,
					},
				},
			},
			expectError: true,
		},
		{
			name: "missing ApplySet ID label",
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "applyset-test-release",
					Namespace: "default",
					Labels:    map[string]string{},
					Annotations: map[string]string{
						ApplySetToolingAnnotation: HelmReleaseTooling,
					},
				},
			},
			expectError: true,
		},
		{
			name: "invalid ApplySet ID format",
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "applyset-test-release",
					Namespace: "default",
					Labels: map[string]string{
						ApplySetParentIDLabel: "invalid-id",
					},
					Annotations: map[string]string{
						ApplySetToolingAnnotation: HelmReleaseTooling,
					},
				},
			},
			expectError: true,
		},
		{
			name: "missing annotations",
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "applyset-test-release",
					Namespace: "default",
					Labels: map[string]string{
						ApplySetParentIDLabel: validID,
					},
				},
			},
			expectError: true,
		},
		{
			name: "wrong tooling annotation",
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "applyset-test-release",
					Namespace: "default",
					Labels: map[string]string{
						ApplySetParentIDLabel: validID,
					},
					Annotations: map[string]string{
						ApplySetToolingAnnotation: "kubectl/v1.27",
					},
				},
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateParentSecret(tt.secret)
			if tt.expectError {
				if err == nil {
					t.Errorf("ValidateParentSecret() expected error but got none")
				}
				return
			}
			if err != nil {
				t.Errorf("ValidateParentSecret() unexpected error = %v", err)
			}
		})
	}
}

func TestManager_CreateOrUpdateParent_Create(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	client := fake.NewSimpleClientset()
	recorder := &record.FakeRecorder{}

	manager := NewManager(client, recorder, logger)

	groupKinds := sets.New[schema.GroupKind](
		schema.GroupKind{Group: "apps", Kind: "Deployment"},
		schema.GroupKind{Group: "", Kind: "Service"},
	)

	helmSecret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "sh.helm.release.v1.test-release.1",
			Namespace: "default",
			UID:       types.UID("test-uid"),
		},
		Type: v1.SecretType("helm.sh/release.v1"),
	}

	ctx := context.Background()
	parentSecret, err := manager.CreateOrUpdateParent(ctx, "test-release", "default", groupKinds, helmSecret)
	if err != nil {
		t.Fatalf("CreateOrUpdateParent() error = %v", err)
	}

	// Verify parent Secret structure
	if parentSecret.Name != "applyset-test-release" {
		t.Errorf("Parent Secret name = %v, want applyset-test-release", parentSecret.Name)
	}
	if parentSecret.Namespace != "default" {
		t.Errorf("Parent Secret namespace = %v, want default", parentSecret.Namespace)
	}

	// Verify labels
	applySetID := parentSecret.Labels[ApplySetParentIDLabel]
	if applySetID == "" {
		t.Error("Parent Secret missing ApplySet ID label")
	}
	expectedID := ComputeApplySetID("test-release", "default")
	if applySetID != expectedID {
		t.Errorf("ApplySet ID = %v, want %v", applySetID, expectedID)
	}

	// Verify annotations
	if parentSecret.Annotations[ApplySetToolingAnnotation] != HelmReleaseTooling {
		t.Errorf("Tooling annotation = %v, want %v",
			parentSecret.Annotations[ApplySetToolingAnnotation], HelmReleaseTooling)
	}

	groupKindsStr := parentSecret.Annotations[ApplySetGKsAnnotation]
	if groupKindsStr != "Deployment.apps,Service" {
		t.Errorf("GroupKinds annotation = %v, want Deployment.apps,Service", groupKindsStr)
	}

	// Verify owner reference
	if len(parentSecret.OwnerReferences) != 1 {
		t.Errorf("Expected 1 owner reference, got %d", len(parentSecret.OwnerReferences))
	}
	ownerRef := parentSecret.OwnerReferences[0]
	if ownerRef.Name != helmSecret.Name {
		t.Errorf("Owner reference name = %v, want %v", ownerRef.Name, helmSecret.Name)
	}
	if ownerRef.UID != helmSecret.UID {
		t.Errorf("Owner reference UID = %v, want %v", ownerRef.UID, helmSecret.UID)
	}
}

func TestManager_CreateOrUpdateParent_Update(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	// Create existing parent Secret
	existingParent := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "applyset-test-release",
			Namespace: "default",
			Labels: map[string]string{
				ApplySetParentIDLabel: ComputeApplySetID("test-release", "default"),
			},
			Annotations: map[string]string{
				ApplySetToolingAnnotation: HelmReleaseTooling,
				ApplySetGKsAnnotation:     "Deployment.apps", // Old value
			},
		},
		Type: v1.SecretTypeOpaque,
	}

	client := fake.NewSimpleClientset(existingParent)
	recorder := &record.FakeRecorder{}

	manager := NewManager(client, recorder, logger)

	groupKinds := sets.New[schema.GroupKind](
		schema.GroupKind{Group: "apps", Kind: "Deployment"},
		schema.GroupKind{Group: "", Kind: "Service"}, // New resource added
	)

	helmSecret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "sh.helm.release.v1.test-release.2",
			Namespace: "default",
			UID:       types.UID("test-uid-2"),
		},
		Type: v1.SecretType("helm.sh/release.v1"),
	}

	ctx := context.Background()
	parentSecret, err := manager.CreateOrUpdateParent(ctx, "test-release", "default", groupKinds, helmSecret)
	if err != nil {
		t.Fatalf("CreateOrUpdateParent() error = %v", err)
	}

	// Verify update
	if parentSecret.Annotations[ApplySetGKsAnnotation] != "Deployment.apps,Service" {
		t.Errorf("GroupKinds annotation = %v, want Deployment.apps,Service",
			parentSecret.Annotations[ApplySetGKsAnnotation])
	}

	// Verify owner reference updated
	if len(parentSecret.OwnerReferences) != 1 {
		t.Errorf("Expected 1 owner reference, got %d", len(parentSecret.OwnerReferences))
	}
	if parentSecret.OwnerReferences[0].Name != helmSecret.Name {
		t.Errorf("Owner reference name = %v, want %v",
			parentSecret.OwnerReferences[0].Name, helmSecret.Name)
	}
}

func TestManager_CreateOrUpdateParent_NoUpdateNeeded(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	applySetID := ComputeApplySetID("test-release", "default")
	existingParent := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "applyset-test-release",
			Namespace: "default",
			Labels: map[string]string{
				ApplySetParentIDLabel: applySetID,
			},
			Annotations: map[string]string{
				ApplySetToolingAnnotation: HelmReleaseTooling,
				ApplySetGKsAnnotation:     "Deployment.apps,Service",
			},
			OwnerReferences: []metav1.OwnerReference{
				{
					APIVersion: "v1",
					Kind:       "Secret",
					Name:       "sh.helm.release.v1.test-release.1",
					UID:        types.UID("test-uid"),
				},
			},
		},
		Type: v1.SecretTypeOpaque,
	}

	client := fake.NewSimpleClientset(existingParent)
	recorder := &record.FakeRecorder{}

	manager := NewManager(client, recorder, logger)

	groupKinds := sets.New[schema.GroupKind](
		schema.GroupKind{Group: "apps", Kind: "Deployment"},
		schema.GroupKind{Group: "", Kind: "Service"},
	)

	helmSecret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "sh.helm.release.v1.test-release.1",
			Namespace: "default",
			UID:       types.UID("test-uid"),
		},
		Type: v1.SecretType("helm.sh/release.v1"),
	}

	ctx := context.Background()

	// Track update calls
	updateCalled := false
	client.PrependReactor("update", "secrets", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		updateCalled = true
		return false, nil, nil
	})

	parentSecret, err := manager.CreateOrUpdateParent(ctx, "test-release", "default", groupKinds, helmSecret)
	if err != nil {
		t.Fatalf("CreateOrUpdateParent() error = %v", err)
	}

	// Should not call update if nothing changed
	if updateCalled {
		t.Error("Update should not be called when parent Secret is already up to date")
	}

	// Should return existing secret
	if parentSecret.ResourceVersion != existingParent.ResourceVersion {
		t.Error("Should return existing secret without modification")
	}
}

func TestManager_DeleteParent(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	existingParent := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "applyset-test-release",
			Namespace: "default",
		},
		Type: v1.SecretTypeOpaque,
	}

	client := fake.NewSimpleClientset(existingParent)
	recorder := &record.FakeRecorder{}

	manager := NewManager(client, recorder, logger)

	ctx := context.Background()
	err := manager.DeleteParent(ctx, "test-release", "default")
	if err != nil {
		t.Fatalf("DeleteParent() error = %v", err)
	}

	// Verify Secret was deleted
	_, err = client.CoreV1().Secrets("default").Get(ctx, "applyset-test-release", metav1.GetOptions{})
	if !apierrors.IsNotFound(err) {
		t.Errorf("Expected NotFound error, got: %v", err)
	}
}

func TestManager_DeleteParent_NotFound(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	client := fake.NewSimpleClientset()
	recorder := &record.FakeRecorder{}

	manager := NewManager(client, recorder, logger)

	ctx := context.Background()
	// Should not error if Secret doesn't exist
	err := manager.DeleteParent(ctx, "test-release", "default")
	if err != nil {
		t.Errorf("DeleteParent() should not error on NotFound, got: %v", err)
	}
}

func TestManager_GetParent(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	existingParent := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "applyset-test-release",
			Namespace: "default",
			Labels: map[string]string{
				ApplySetParentIDLabel: ComputeApplySetID("test-release", "default"),
			},
		},
		Type: v1.SecretTypeOpaque,
	}

	client := fake.NewSimpleClientset(existingParent)
	recorder := &record.FakeRecorder{}

	manager := NewManager(client, recorder, logger)

	ctx := context.Background()
	secret, err := manager.GetParent(ctx, "test-release", "default")
	if err != nil {
		t.Fatalf("GetParent() error = %v", err)
	}

	if secret.Name != "applyset-test-release" {
		t.Errorf("GetParent() name = %v, want applyset-test-release", secret.Name)
	}
}

func TestManager_GetParent_NotFound(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	client := fake.NewSimpleClientset()
	recorder := &record.FakeRecorder{}

	manager := NewManager(client, recorder, logger)

	ctx := context.Background()
	_, err := manager.GetParent(ctx, "test-release", "default")
	if !apierrors.IsNotFound(err) {
		t.Errorf("GetParent() expected NotFound error, got: %v", err)
	}
}
