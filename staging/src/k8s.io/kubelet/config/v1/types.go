/*
Copyright 2022 The Kubernetes Authors.

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

package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// ServiceAccountTokenCacheType is the type of cache key used for caching credentials returned by the plugin
// when the service account token is used.
type ServiceAccountTokenCacheType string

const (
	// TokenServiceAccountTokenCacheType means the kubelet will cache returned credentials
	// on a per-token basis. This should be set if the returned credential's lifetime is limited
	// to the input service account token's lifetime.
	// For example, this must be used when returning the input service account token directly as a pull credential.
	TokenServiceAccountTokenCacheType ServiceAccountTokenCacheType = "Token"
	// ServiceAccountServiceAccountTokenCacheType means the kubelet will cache returned credentials
	// on a per-serviceaccount basis. This should be set if the plugin's credential retrieval logic
	// depends only on the service account and not on pod-specific claims.
	// Use this when the returned credential is valid for all pods using the same service account.
	ServiceAccountServiceAccountTokenCacheType ServiceAccountTokenCacheType = "ServiceAccount"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// CredentialProviderConfig is the configuration containing information about
// each exec credential provider. Kubelet reads this configuration from disk and enables
// each provider as specified by the CredentialProvider type.
type CredentialProviderConfig struct {
	metav1.TypeMeta `json:",inline"`

	// providers is a list of credential provider plugins that will be enabled by the kubelet.
	// Multiple providers may match against a single image, in which case credentials
	// from all providers will be returned to the kubelet. If multiple providers are called
	// for a single image, the results are combined. If providers return overlapping
	// auth keys, the value from the provider earlier in this list is attempted first.
	Providers []CredentialProvider `json:"providers"`
}

// CredentialProvider represents an exec plugin to be invoked by the kubelet. The plugin is only
// invoked when an image being pulled matches the images handled by the plugin (see matchImages).
type CredentialProvider struct {
	// name is the required name of the credential provider. It must match the name of the
	// provider executable as seen by the kubelet. The executable must be in the kubelet's
	// bin directory (set by the --image-credential-provider-bin-dir flag).
	// Required to be unique across all providers.
	Name string `json:"name"`

	// matchImages is a required list of strings used to match against images in order to
	// determine if this provider should be invoked. If one of the strings matches the
	// requested image from the kubelet, the plugin will be invoked and given a chance
	// to provide credentials. Images are expected to contain the registry domain
	// and URL path.
	//
	// Each entry in matchImages is a pattern which can optionally contain a port and a path.
	// Globs can be used in the domain, but not in the port or the path. Globs are supported
	// as subdomains like '*.k8s.io' or 'k8s.*.io', and top-level-domains such as 'k8s.*'.
	// Matching partial subdomains like 'app*.k8s.io' is also supported. Each glob can only match
	// a single subdomain segment, so *.io does not match *.k8s.io.
	//
	// A match exists between an image and a matchImage when all of the below are true:
	// - Both contain the same number of domain parts and each part matches.
	// - The URL path of an imageMatch must be a prefix of the target image URL path.
	// - If the imageMatch contains a port, then the port must match in the image as well.
	//
	// Example values of matchImages:
	//   - 123456789.dkr.ecr.us-east-1.amazonaws.com
	//   - *.azurecr.io
	//   - gcr.io
	//   - *.*.registry.io
	//   - registry.io:8080/path
	MatchImages []string `json:"matchImages"`

	// defaultCacheDuration is the default duration the plugin will cache credentials in-memory
	// if a cache duration is not provided in the plugin response. This field is required.
	DefaultCacheDuration *metav1.Duration `json:"defaultCacheDuration"`

	// Required input version of the exec CredentialProviderRequest. The returned CredentialProviderResponse
	// MUST use the same encoding version as the input. Current supported values are:
	// - credentialprovider.kubelet.k8s.io/v1
	APIVersion string `json:"apiVersion"`

	// Arguments to pass to the command when executing it.
	// +optional
	Args []string `json:"args,omitempty"`

	// Env defines additional environment variables to expose to the process. These
	// are unioned with the host's environment, as well as variables client-go uses
	// to pass argument to the plugin.
	// +optional
	Env []ExecEnvVar `json:"env,omitempty"`

	// tokenAttributes is the configuration for the service account token that will be passed to the plugin.
	// The credential provider opts in to using service account tokens for image pull by setting this field.
	// When this field is set, kubelet will generate a service account token bound to the pod for which the
	// image is being pulled and pass to the plugin as part of CredentialProviderRequest along with other
	// attributes required by the plugin.
	//
	// The service account metadata and token attributes will be used as a dimension to cache
	// the credentials in kubelet. The cache key is generated by combining the service account metadata
	// (namespace, name, UID, and annotations key+value for the keys defined in
	// serviceAccountTokenAttribute.requiredServiceAccountAnnotationKeys and serviceAccountTokenAttribute.optionalServiceAccountAnnotationKeys).
	// The pod metadata (namespace, name, UID) that are in the service account token are not used as a dimension
	// to cache the credentials in kubelet. This means workloads that are using the same service account
	// could end up using the same credentials for image pull. For plugins that don't want this behavior, or
	// plugins that operate in pass-through mode; i.e., they return the service account token as-is, they
	// can set the credentialProviderResponse.cacheDuration to 0. This will disable the caching of
	// credentials in kubelet and the plugin will be invoked for every image pull. This does result in
	// token generation overhead for every image pull, but it is the only way to ensure that the
	// credentials are not shared across pods (even if they are using the same service account).
	// +optional
	TokenAttributes *ServiceAccountTokenAttributes `json:"tokenAttributes,omitempty"`
}

// ServiceAccountTokenAttributes is the configuration for the service account token that will be passed to the plugin.
type ServiceAccountTokenAttributes struct {
	// serviceAccountTokenAudience is the intended audience for the projected service account token.
	// +required
	ServiceAccountTokenAudience string `json:"serviceAccountTokenAudience"`

	// cacheType indicates the type of cache key use for caching the credentials returned by the plugin
	// when the service account token is used.
	// The most conservative option is to set this to "Token", which means the kubelet will cache returned credentials
	// on a per-token basis. This should be set if the returned credential's lifetime is limited to the service account
	// token's lifetime.
	// If the plugin's credential retrieval logic depends only on the service account and not on pod-specific claims,
	// then the plugin can set this to "ServiceAccount". In this case, the kubelet will cache returned credentials
	// on a per-serviceaccount basis. Use this when the returned credential is valid for all pods using the same service account.
	// +required
	CacheType ServiceAccountTokenCacheType `json:"cacheType"`

	// requireServiceAccount indicates whether the plugin requires the pod to have a service account.
	// If set to true, kubelet will only invoke the plugin if the pod has a service account.
	// If set to false, kubelet will invoke the plugin even if the pod does not have a service account
	// and will not include a token in the CredentialProviderRequest in that scenario. This is useful for plugins that
	// are used to pull images for pods without service accounts (e.g., static pods).
	// +required
	RequireServiceAccount *bool `json:"requireServiceAccount"`

	// requiredServiceAccountAnnotationKeys is the list of annotation keys that the plugin is interested in
	// and that are required to be present in the service account.
	// The keys defined in this list will be extracted from the corresponding service account and passed
	// to the plugin as part of the CredentialProviderRequest. If any of the keys defined in this list
	// are not present in the service account, kubelet will not invoke the plugin and will return an error.
	// This field is optional and may be empty. Plugins may use this field to extract
	// additional information required to fetch credentials or allow workloads to opt in to
	// using service account tokens for image pull.
	// If non-empty, requireServiceAccount must be set to true.
	// Keys in this list must be unique.
	// This list needs to be mutually exclusive with optionalServiceAccountAnnotationKeys.
	// +optional
	// +listType=set
	RequiredServiceAccountAnnotationKeys []string `json:"requiredServiceAccountAnnotationKeys,omitempty"`

	// optionalServiceAccountAnnotationKeys is the list of annotation keys that the plugin is interested in
	// and that are optional to be present in the service account.
	// The keys defined in this list will be extracted from the corresponding service account and passed
	// to the plugin as part of the CredentialProviderRequest. The plugin is responsible for validating
	// the existence of annotations and their values.
	// This field is optional and may be empty. Plugins may use this field to extract
	// additional information required to fetch credentials.
	// Keys in this list must be unique.
	// +optional
	// +listType=set
	OptionalServiceAccountAnnotationKeys []string `json:"optionalServiceAccountAnnotationKeys,omitempty"`
}

// ExecEnvVar is used for setting environment variables when executing an exec-based
// credential plugin.
type ExecEnvVar struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}
