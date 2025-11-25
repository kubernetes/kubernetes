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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes"

	"k8s.io/kubernetes/pkg/controller/helmapplyset/parent"
)

// GetApplySetParentSecret retrieves an ApplySet parent Secret by release name
func GetApplySetParentSecret(client kubernetes.Interface, releaseName, namespace string) (*v1.Secret, error) {
	parentName := fmt.Sprintf("%s%s", parent.ParentSecretNamePrefix, releaseName)
	secret, err := client.CoreV1().Secrets(namespace).Get(context.TODO(), parentName, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to get ApplySet parent Secret %s/%s: %w", namespace, parentName, err)
	}

	if !IsApplySetParentSecret(secret) {
		return nil, fmt.Errorf("Secret %s/%s is not an ApplySet parent Secret", namespace, parentName)
	}

	return secret, nil
}

// IsApplySetParentSecret checks if a Secret is an ApplySet parent Secret
func IsApplySetParentSecret(secret *v1.Secret) bool {
	if secret.Labels == nil {
		return false
	}
	_, hasLabel := secret.Labels[parent.ApplySetParentIDLabel]
	return hasLabel
}

// ExtractReleaseNameFromParentSecret extracts the Helm release name from a parent Secret name
func ExtractReleaseNameFromParentSecret(secretName string) string {
	return strings.TrimPrefix(secretName, parent.ParentSecretNamePrefix)
}

// GetHelmLabelSelector creates a label selector for Helm-managed resources
func GetHelmLabelSelector(releaseName string) string {
	return fmt.Sprintf("app.kubernetes.io/instance=%s", releaseName)
}

// CountResourcesByGroupKind counts resources by GroupKind for a given release
func CountResourcesByGroupKind(client kubernetes.Interface, releaseName, namespace string) (map[schema.GroupKind]int, error) {
	labelSelector := GetHelmLabelSelector(releaseName)
	counts := make(map[schema.GroupKind]int)

	// Count Deployments
	deployments, err := client.AppsV1().Deployments(namespace).List(context.TODO(), metav1.ListOptions{
		LabelSelector: labelSelector,
	})
	if err == nil {
		counts[schema.GroupKind{Group: "apps", Kind: "Deployment"}] = len(deployments.Items)
	}

	// Count Services
	services, err := client.CoreV1().Services(namespace).List(context.TODO(), metav1.ListOptions{
		LabelSelector: labelSelector,
	})
	if err == nil {
		counts[schema.GroupKind{Group: "", Kind: "Service"}] = len(services.Items)
	}

	// Count ConfigMaps
	configMaps, err := client.CoreV1().ConfigMaps(namespace).List(context.TODO(), metav1.ListOptions{
		LabelSelector: labelSelector,
	})
	if err == nil {
		counts[schema.GroupKind{Group: "", Kind: "ConfigMap"}] = len(configMaps.Items)
	}

	// Count Secrets (excluding parent Secret)
	secrets, err := client.CoreV1().Secrets(namespace).List(context.TODO(), metav1.ListOptions{
		LabelSelector: labelSelector,
	})
	if err == nil {
		parentName := fmt.Sprintf("%s%s", parent.ParentSecretNamePrefix, releaseName)
		count := 0
		for _, s := range secrets.Items {
			if s.Name != parentName {
				count++
			}
		}
		counts[schema.GroupKind{Group: "", Kind: "Secret"}] = count
	}

	return counts, nil
}

// FormatGroupKind formats a GroupKind as a string
func FormatGroupKind(gk schema.GroupKind) string {
	if gk.Group == "" {
		return gk.Kind
	}
	return fmt.Sprintf("%s.%s", gk.Kind, gk.Group)
}

// GetAge returns a human-readable age string from a time
func GetAge(t time.Time) string {
	if t.IsZero() {
		return "<unknown>"
	}
	duration := time.Since(t)
	if duration < time.Minute {
		return fmt.Sprintf("%ds", int(duration.Seconds()))
	}
	if duration < time.Hour {
		return fmt.Sprintf("%dm", int(duration.Minutes()))
	}
	if duration < 24*time.Hour {
		return fmt.Sprintf("%dh", int(duration.Hours()))
	}
	days := int(duration.Hours() / 24)
	if days < 30 {
		return fmt.Sprintf("%dd", days)
	}
	months := days / 30
	if months < 12 {
		return fmt.Sprintf("%dmo", months)
	}
	years := months / 12
	return fmt.Sprintf("%dy", years)
}
