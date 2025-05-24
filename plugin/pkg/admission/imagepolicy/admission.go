/*
Copyright 2016 The Kubernetes Authors.

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

// Package imagepolicy contains an admission controller that configures a webhook to which policy
// decisions are delegated.
package imagepolicy

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"

	"k8s.io/klog/v2"

	"k8s.io/api/imagepolicy/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/cache"
	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/util/webhook"
	"k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"

	// install the clientgo image policy API for use with api registry
	_ "k8s.io/kubernetes/pkg/apis/imagepolicy/install"
)

// PluginName indicates name of admission plugin.
const PluginName = "ImagePolicyWebhook"
const ephemeralcontainers = "ephemeralcontainers"

// AuditKeyPrefix is used as the prefix for all audit keys handled by this
// pluggin. Some well known suffixes are listed below.
var AuditKeyPrefix = strings.ToLower(PluginName) + ".image-policy.k8s.io/"

const (
	// ImagePolicyFailedOpenKeySuffix in an annotation indicates the image
	// review failed open when the image policy webhook backend connection
	// failed.
	ImagePolicyFailedOpenKeySuffix string = "failed-open"

	// ImagePolicyAuditRequiredKeySuffix in an annotation indicates the pod
	// should be audited.
	ImagePolicyAuditRequiredKeySuffix string = "audit-required"
)

var (
	groupVersions = []schema.GroupVersion{v1alpha1.SchemeGroupVersion}
)

// DEPRECATED: The ImagePolicyWebhook admission plugin is deprecated and will be removed in a future release.
// Please migrate to ValidatingAdmissionWebhook or other supported mechanisms.
//
// TODO: Remove this plugin after deprecation period.
// Register registers a plugin
func Register(plugins *admission.Plugins) {
	klog.Warning("ImagePolicyWebhook admission plugin is DEPRECATED and will be removed in a future release. Please migrate to ValidatingAdmissionWebhook or other supported mechanisms.")
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		newImagePolicyWebhook, err := NewImagePolicyWebhook(config)
		if err != nil {
			return nil, err
		}
		return newImagePolicyWebhook, nil
	})
}

// Plugin is an implementation of admission.Interface.
type Plugin struct {
	*admission.Handler
	webhook       *webhook.GenericWebhook
	responseCache *cache.LRUExpireCache
	allowTTL      time.Duration
	denyTTL       time.Duration
	defaultAllow  bool
}

var _ admission.ValidationInterface = &Plugin{}

func (a *Plugin) statusTTL(status v1alpha1.ImageReviewStatus) time.Duration {
	if status.Allowed {
		return a.allowTTL
	}
	return a.denyTTL
}

// Filter out annotations that don't match *.image-policy.k8s.io/*
func (a *Plugin) filterAnnotations(allAnnotations map[string]string) map[string]string {
	annotations := make(map[string]string)
	for k, v := range allAnnotations {
		if strings.Contains(k, ".image-policy.k8s.io/") {
			annotations[k] = v
		}
	}
	return annotations
}

// Function to call on webhook failure; behavior determined by defaultAllow flag
func (a *Plugin) webhookError(pod *api.Pod, attributes admission.Attributes, err error) error {
	if err != nil {
		klog.V(2).Infof("error contacting webhook backend: %s", err)
		if a.defaultAllow {
			attributes.AddAnnotation(AuditKeyPrefix+ImagePolicyFailedOpenKeySuffix, "true")
			// TODO(wteiken): Remove the annotation code for the 1.13 release
			annotations := pod.GetAnnotations()
			if annotations == nil {
				annotations = make(map[string]string)
			}
			annotations[api.ImagePolicyFailedOpenKey] = "true"
			pod.ObjectMeta.SetAnnotations(annotations)

			klog.V(2).Infof("resource allowed in spite of webhook backend failure")
			return nil
		}
		klog.V(2).Infof("resource not allowed due to webhook backend failure ")
		return admission.NewForbidden(attributes, err)
	}
	return nil
}

// Validate makes an admission decision based on the request attributes
func (a *Plugin) Validate(ctx context.Context, attributes admission.Attributes, o admission.ObjectInterfaces) (err error) {
	// Ignore all calls to subresources other than ephemeralcontainers or calls to resources other than pods.
	subresource := attributes.GetSubresource()
	if (subresource != "" && subresource != ephemeralcontainers) || attributes.GetResource().GroupResource() != api.Resource("pods") {
		return nil
	}

	pod, ok := attributes.GetObject().(*api.Pod)
	if !ok {
		return apierrors.NewBadRequest("Resource was marked with kind Pod but was unable to be converted")
	}

	// Build list of ImageReviewContainerSpec
	var imageReviewContainerSpecs []v1alpha1.ImageReviewContainerSpec
	if subresource == "" {
		containers := make([]api.Container, 0, len(pod.Spec.Containers)+len(pod.Spec.InitContainers))
		containers = append(containers, pod.Spec.Containers...)
		containers = append(containers, pod.Spec.InitContainers...)
		for _, c := range containers {
			imageReviewContainerSpecs = append(imageReviewContainerSpecs, v1alpha1.ImageReviewContainerSpec{
				Image: c.Image,
			})
		}
	} else if subresource == ephemeralcontainers {
		for _, c := range pod.Spec.EphemeralContainers {
			imageReviewContainerSpecs = append(imageReviewContainerSpecs, v1alpha1.ImageReviewContainerSpec{
				Image: c.Image,
			})
		}
	}
	imageReview := v1alpha1.ImageReview{
		Spec: v1alpha1.ImageReviewSpec{
			Containers:  imageReviewContainerSpecs,
			Annotations: a.filterAnnotations(pod.Annotations),
			Namespace:   attributes.GetNamespace(),
		},
	}
	if err := a.admitPod(ctx, pod, attributes, &imageReview); err != nil {
		return admission.NewForbidden(attributes, err)
	}
	return nil
}

func (a *Plugin) admitPod(ctx context.Context, pod *api.Pod, attributes admission.Attributes, review *v1alpha1.ImageReview) error {
	cacheKey, err := json.Marshal(review.Spec)
	if err != nil {
		return err
	}
	if entry, ok := a.responseCache.Get(string(cacheKey)); ok {
		review.Status = entry.(v1alpha1.ImageReviewStatus)
	} else {
		result := a.webhook.WithExponentialBackoff(ctx, func() rest.Result {
			return a.webhook.RestClient.Post().Body(review).Do(ctx)
		})

		if err := result.Error(); err != nil {
			return a.webhookError(pod, attributes, err)
		}
		var statusCode int
		if result.StatusCode(&statusCode); statusCode < 200 || statusCode >= 300 {
			return a.webhookError(pod, attributes, fmt.Errorf("Error contacting webhook: %d", statusCode))
		}

		if err := result.Into(review); err != nil {
			return a.webhookError(pod, attributes, err)
		}

		a.responseCache.Add(string(cacheKey), review.Status, a.statusTTL(review.Status))
	}

	for k, v := range review.Status.AuditAnnotations {
		if err := attributes.AddAnnotation(AuditKeyPrefix+k, v); err != nil {
			klog.Warningf("failed to set admission audit annotation %s to %s: %v", AuditKeyPrefix+k, v, err)
		}
	}
	if !review.Status.Allowed {
		if len(review.Status.Reason) > 0 {
			return fmt.Errorf("image policy webhook backend denied one or more images: %s", review.Status.Reason)
		}
		return errors.New("one or more images rejected by webhook backend")
	}
	return nil
}

// Versioned config for ImagePolicyWebhook
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type ImagePolicyWebhookConfigurationV1Alpha1 struct {
	APIVersion  string                   `json:"apiVersion"`
	Kind        string                   `json:"kind"`
	ImagePolicy imagePolicyWebhookConfig `json:"imagePolicy"`
}

// DEPRECATED: The ImagePolicyWebhook admission plugin is deprecated and will be removed in a future release.
// Please migrate to ValidatingAdmissionWebhook or other supported mechanisms.
//
// TODO: Remove this plugin after deprecation period.
// NewImagePolicyWebhook a new ImagePolicyWebhook plugin from the provided config file.
// The config file is specified by --admission-control-config-file and has the
// following format for a webhook:
//
// Versioned format (recommended):
//
//	{
//	  "apiVersion": "imagepolicy.admission.k8s.io/v1alpha1",
//	  "kind": "ImagePolicyWebhookConfiguration",
//	  "imagePolicy": { ... }
//	}
//
// Legacy format (deprecated, still supported for backward compatibility):
//
//	{
//	  "imagePolicy": { ... }
//	}
//
// Register registers a plugin
func NewImagePolicyWebhook(configFile io.Reader) (*Plugin, error) {
	if configFile == nil {
		return nil, fmt.Errorf("no config specified")
	}

	// Require io.ReadSeeker for versioned config support
	readSeeker, ok := configFile.(io.ReadSeeker)
	if !ok {
		return nil, fmt.Errorf("ImagePolicyWebhook config reader must be an io.ReadSeeker for versioned config support")
	}

	// Support both versioned and legacy config formats
	var versioned struct {
		APIVersion string `json:"apiVersion" yaml:"apiVersion"`
		Kind       string `json:"kind" yaml:"kind"`
		AdmissionConfig
	}
	var legacy AdmissionConfig

	d := yaml.NewYAMLOrJSONDecoder(readSeeker, 4096)
	err := d.Decode(&versioned)
	if err == nil && versioned.APIVersion != "" && versioned.Kind != "" {
		// Versioned config detected
		legacy = versioned.AdmissionConfig
	} else {
		// Try legacy config
		_, errSeek := readSeeker.Seek(0, io.SeekStart)
		if errSeek != nil {
			return nil, errSeek
		}
		d = yaml.NewYAMLOrJSONDecoder(readSeeker, 4096)
		err = d.Decode(&legacy)
		if err != nil {
			return nil, err
		}
		klog.Warning("Using legacy (unversioned) configuration for ImagePolicyWebhook. Please migrate to the versioned format.")
	}

	whConfig := legacy.ImagePolicyWebhook
	if err := normalizeWebhookConfig(&whConfig); err != nil {
		return nil, err
	}

	clientConfig, err := webhook.LoadKubeconfig(whConfig.KubeConfigFile, nil)
	if err != nil {
		return nil, err
	}
	retryBackoff := webhook.DefaultRetryBackoffWithInitialDelay(whConfig.RetryBackoff)
	gw, err := webhook.NewGenericWebhook(legacyscheme.Scheme, legacyscheme.Codecs, clientConfig, groupVersions, retryBackoff)
	if err != nil {
		return nil, err
	}
	return &Plugin{
		Handler:       admission.NewHandler(admission.Create, admission.Update),
		webhook:       gw,
		responseCache: cache.NewLRUExpireCache(1024),
		allowTTL:      whConfig.AllowTTL,
		denyTTL:       whConfig.DenyTTL,
		defaultAllow:  whConfig.DefaultAllow,
	}, nil
}
