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
	"bytes"
	"compress/gzip"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/yaml"
	corev1informers "k8s.io/client-go/informers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
)

const (
	// HelmReleaseSecretType is the Secret type for Helm releases
	HelmReleaseSecretType v1.SecretType = "helm.sh/release.v1"

	// HelmReleaseSecretPrefix is the prefix for Helm release Secret names
	HelmReleaseSecretPrefix = "sh.helm.release.v1."

	// HelmReleaseDataKey is the key in Secret.Data containing the release data
	HelmReleaseDataKey = "release"
)

// HelmReleaseInfo contains parsed information from a Helm release Secret
type HelmReleaseInfo struct {
	// Name is the Helm release name
	Name string

	// Namespace is the namespace where the release is installed
	Namespace string

	// Version is the release version (revision number)
	Version int

	// Chart is the chart name and version (e.g., "nginx-1.0.0")
	Chart string

	// ChartVersion is the chart version
	ChartVersion string

	// Manifest is the YAML manifest containing all resources in the release
	Manifest string

	// Status is the release status (e.g., "deployed", "failed", "pending-upgrade")
	Status string

	// GroupKinds is the set of GroupKinds present in the manifest
	GroupKinds sets.Set[schema.GroupKind]

	// SecretName is the name of the Helm release Secret
	SecretName string

	// SecretNamespace is the namespace of the Helm release Secret
	SecretNamespace string
}

// HelmReleaseJSON represents the JSON structure of a Helm release
// This matches the structure used by Helm v3
type HelmReleaseJSON struct {
	Name      string `json:"name"`
	Namespace string `json:"namespace"`
	Version   int    `json:"version"`
	Status    string `json:"status"`
	Chart     struct {
		Metadata struct {
			Name    string `json:"name"`
			Version string `json:"version"`
		} `json:"metadata"`
	} `json:"chart"`
	Manifest string `json:"manifest"`
	// Additional fields that may be present in different Helm versions
	Info struct {
		Status string `json:"status"`
	} `json:"info"`
}

// Watcher watches for Helm release Secrets and parses them
type Watcher struct {
	secretInformer corev1informers.SecretInformer
	enqueueFunc    func(interface{})
	logger         klog.Logger
}

// NewWatcher creates a new Helm release Secret watcher
func NewWatcher(
	secretInformer corev1informers.SecretInformer,
	enqueueFunc func(interface{}),
	logger klog.Logger,
) *Watcher {
	return &Watcher{
		secretInformer: secretInformer,
		enqueueFunc:    enqueueFunc,
		logger:         logger,
	}
}

// Start sets up event handlers for watching Helm release Secrets
func (w *Watcher) Start(ctx context.Context) {
	w.secretInformer.Informer().AddEventHandlerWithOptions(
		cache.FilteringResourceEventHandler{
			FilterFunc: func(obj interface{}) bool {
				secret, ok := obj.(*v1.Secret)
				if !ok {
					return false
				}
				return IsHelmReleaseSecret(secret)
			},
			Handler: cache.ResourceEventHandlerFuncs{
				AddFunc:    w.handleAdd,
				UpdateFunc: w.handleUpdate,
				DeleteFunc: w.handleDelete,
			},
		},
		cache.HandlerOptions{
			Logger: &w.logger,
		},
	)
}

// IsHelmReleaseSecret checks if a Secret is a Helm release Secret
func IsHelmReleaseSecret(secret *v1.Secret) bool {
	// Check Secret type
	if secret.Type != HelmReleaseSecretType {
		return false
	}

	// Check name pattern: sh.helm.release.v1.<name>.<version>
	return strings.HasPrefix(secret.Name, HelmReleaseSecretPrefix)
}

// handleAdd handles Secret add events
func (w *Watcher) handleAdd(obj interface{}) {
	secret := obj.(*v1.Secret)
	// Note: FilterFunc already ensures this is a Helm release Secret
	w.logger.V(4).Info("Helm release Secret added", "secret", klog.KObj(secret))
	w.enqueueFunc(secret)
}

// handleUpdate handles Secret update events
func (w *Watcher) handleUpdate(oldObj, newObj interface{}) {
	newSecret := newObj.(*v1.Secret)
	// Note: FilterFunc already ensures this is a Helm release Secret

	// Skip if ResourceVersion hasn't changed (periodic resync)
	oldSecret, ok := oldObj.(*v1.Secret)
	if ok && oldSecret.ResourceVersion == newSecret.ResourceVersion {
		return
	}

	w.logger.V(4).Info("Helm release Secret updated", "secret", klog.KObj(newSecret))
	w.enqueueFunc(newSecret)
}

// handleDelete handles Secret delete events
func (w *Watcher) handleDelete(obj interface{}) {
	secret, ok := obj.(*v1.Secret)
	if !ok {
		// Handle tombstone
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("couldn't get object from tombstone %#v", obj))
			return
		}
		secret, ok = tombstone.Obj.(*v1.Secret)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("tombstone contained object that is not a Secret %#v", tombstone.Obj))
			return
		}
	}
	// Note: FilterFunc already ensures this is a Helm release Secret

	w.logger.V(4).Info("Helm release Secret deleted", "secret", klog.KObj(secret))
	w.enqueueFunc(secret)
}

// ParseHelmReleaseSecret parses a Helm release Secret and extracts release information
func ParseHelmReleaseSecret(secret *v1.Secret) (*HelmReleaseInfo, error) {
	if !IsHelmReleaseSecret(secret) {
		return nil, fmt.Errorf("secret %s/%s is not a Helm release Secret", secret.Namespace, secret.Name)
	}

	// Extract release name and version from Secret name
	// Format: sh.helm.release.v1.<name>.<version>
	parts := strings.Split(secret.Name, ".")
	if len(parts) < 5 {
		return nil, fmt.Errorf("invalid Helm release Secret name format: %s (expected sh.helm.release.v1.<name>.<version>)", secret.Name)
	}

	// Release name is the 5th part (index 4)
	// Version is the last part (we'll use version from JSON if available)
	releaseName := parts[4]

	// Get release data from Secret
	releaseData, ok := secret.Data[HelmReleaseDataKey]
	if !ok {
		return nil, fmt.Errorf("missing '%s' key in Helm release Secret %s/%s", HelmReleaseDataKey, secret.Namespace, secret.Name)
	}

	// Decode and decompress Helm release data
	releaseJSON, err := decodeHelmRelease(releaseData)
	if err != nil {
		return nil, fmt.Errorf("failed to decode Helm release data: %w", err)
	}

	// Parse Helm release JSON
	var release HelmReleaseJSON
	if err := json.Unmarshal(releaseJSON, &release); err != nil {
		return nil, fmt.Errorf("failed to unmarshal Helm release JSON: %w", err)
	}

	// Validate required fields
	if release.Name == "" {
		return nil, fmt.Errorf("Helm release JSON missing 'name' field")
	}
	if release.Namespace == "" {
		return nil, fmt.Errorf("Helm release JSON missing 'namespace' field")
	}
	if release.Manifest == "" {
		return nil, fmt.Errorf("Helm release JSON missing 'manifest' field")
	}

	// Extract status (may be in different locations depending on Helm version)
	status := release.Status
	if status == "" {
		status = release.Info.Status
	}
	if status == "" {
		status = "unknown"
	}

	// Extract chart information
	chartName := release.Chart.Metadata.Name
	chartVersion := release.Chart.Metadata.Version
	chart := chartName
	if chartVersion != "" {
		chart = fmt.Sprintf("%s-%s", chartName, chartVersion)
	}

	// Extract GroupKinds from manifest
	groupKinds, err := extractGroupKindsFromManifest(release.Manifest)
	if err != nil {
		return nil, fmt.Errorf("failed to extract GroupKinds from manifest: %w", err)
	}

	// Use release name from JSON if available, otherwise use parsed name
	finalReleaseName := release.Name
	if finalReleaseName == "" {
		finalReleaseName = releaseName
	}

	// Use release namespace from JSON if available, otherwise use Secret namespace
	finalNamespace := release.Namespace
	if finalNamespace == "" {
		finalNamespace = secret.Namespace
	}

	// Use version from JSON, fallback to parsing from Secret name if needed
	finalVersion := release.Version
	if finalVersion == 0 {
		versionStr := parts[len(parts)-1]
		if parsedVersion, err := strconv.Atoi(versionStr); err == nil {
			finalVersion = parsedVersion
		}
	}

	return &HelmReleaseInfo{
		Name:            finalReleaseName,
		Namespace:       finalNamespace,
		Version:         finalVersion,
		Chart:           chart,
		ChartVersion:    chartVersion,
		Manifest:        release.Manifest,
		Status:          status,
		GroupKinds:      groupKinds,
		SecretName:      secret.Name,
		SecretNamespace: secret.Namespace,
	}, nil
}

// decodeHelmRelease decodes Helm release data from base64-encoded, gzipped format
func decodeHelmRelease(data []byte) ([]byte, error) {
	// Step 1: Base64 decode
	decoded, err := base64.StdEncoding.DecodeString(string(data))
	if err != nil {
		// Try URL encoding if standard encoding fails
		decoded, err = base64.URLEncoding.DecodeString(string(data))
		if err != nil {
			return nil, fmt.Errorf("base64 decode failed: %w", err)
		}
	}

	// Step 2: Gunzip
	reader, err := gzip.NewReader(bytes.NewReader(decoded))
	if err != nil {
		return nil, fmt.Errorf("gzip reader creation failed: %w", err)
	}
	defer reader.Close()

	decompressed, err := io.ReadAll(reader)
	if err != nil {
		return nil, fmt.Errorf("gzip decompression failed: %w", err)
	}

	return decompressed, nil
}

// extractGroupKindsFromManifest extracts GroupKinds from a Helm manifest YAML
func extractGroupKindsFromManifest(manifest string) (sets.Set[schema.GroupKind], error) {
	groupKinds := sets.New[schema.GroupKind]()

	if manifest == "" {
		return groupKinds, nil
	}

	// Use YAML decoder to parse multi-document YAML
	decoder := yaml.NewYAMLOrJSONDecoder(strings.NewReader(manifest), 4096)

	for {
		// Parse each YAML document
		var obj struct {
			APIVersion string `json:"apiVersion" yaml:"apiVersion"`
			Kind       string `json:"kind" yaml:"kind"`
		}

		err := decoder.Decode(&obj)
		if err == io.EOF {
			break
		}
		if err != nil {
			// Skip invalid YAML documents
			continue
		}

		if obj.Kind == "" {
			continue
		}

		// Default apiVersion to v1 if not specified
		apiVersion := obj.APIVersion
		if apiVersion == "" {
			apiVersion = "v1"
		}

		// Parse GroupVersion from apiVersion
		gv, err := schema.ParseGroupVersion(apiVersion)
		if err != nil {
			// Skip invalid apiVersion
			continue
		}

		gk := schema.GroupKind{
			Group: gv.Group,
			Kind:  obj.Kind,
		}
		groupKinds.Insert(gk)
	}

	return groupKinds, nil
}

// ExtractReleaseNameFromSecretName extracts the Helm release name from a Secret name
func ExtractReleaseNameFromSecretName(secretName string) (string, error) {
	if !strings.HasPrefix(secretName, HelmReleaseSecretPrefix) {
		return "", fmt.Errorf("secret name %s does not match Helm release pattern", secretName)
	}

	parts := strings.Split(secretName, ".")
	if len(parts) < 5 {
		return "", fmt.Errorf("invalid Helm release Secret name format: %s", secretName)
	}

	return parts[4], nil
}

// ExtractReleaseVersionFromSecretName extracts the Helm release version from a Secret name
func ExtractReleaseVersionFromSecretName(secretName string) (int, error) {
	if !strings.HasPrefix(secretName, HelmReleaseSecretPrefix) {
		return 0, fmt.Errorf("secret name %s does not match Helm release pattern", secretName)
	}

	parts := strings.Split(secretName, ".")
	if len(parts) < 5 {
		return 0, fmt.Errorf("invalid Helm release Secret name format: %s", secretName)
	}

	versionStr := parts[len(parts)-1]
	version, err := strconv.Atoi(versionStr)
	if err != nil {
		return 0, fmt.Errorf("invalid version in Secret name %s: %w", secretName, err)
	}

	return version, nil
}
