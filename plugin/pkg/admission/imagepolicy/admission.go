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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"strings"
	"time"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/imagepolicy/v1alpha1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/util/cache"
	"k8s.io/kubernetes/plugin/pkg/webhook"

	"k8s.io/kubernetes/pkg/admission"
)

var (
	groupVersions = []unversioned.GroupVersion{v1alpha1.SchemeGroupVersion}
)

const (
	defaultRetryBackoff = time.Duration(500) * time.Millisecond
	minRetryBackoff     = time.Duration(1)
	maxRetryBackoff     = time.Duration(5) * time.Minute
	defaultAllowTTL     = time.Duration(5) * time.Minute
	defaultDenyTTL      = time.Duration(30) * time.Second
	minAllowTTL         = time.Duration(1) * time.Second
	maxAllowTTL         = time.Duration(30) * time.Minute
	minDenyTTL          = time.Duration(1) * time.Second
	maxDenyTTL          = time.Duration(30) * time.Minute
	useDefault          = time.Duration(0)  //sentinel for using default TTL
	disableTTL          = time.Duration(-1) //sentinel for disabling a TTL
)

func init() {
	admission.RegisterPlugin("ImagePolicyWebhook", func(client clientset.Interface, config io.Reader) (admission.Interface, error) {
		newImagePolicyWebhook, err := NewImagePolicyWebhook(client, config)
		if err != nil {
			return nil, err
		}
		return newImagePolicyWebhook, nil
	})
}

// imagePolicyWebhook is an implementation of admission.Interface.
type imagePolicyWebhook struct {
	*admission.Handler
	webhook       *webhook.GenericWebhook
	responseCache *cache.LRUExpireCache
	allowTTL      time.Duration
	denyTTL       time.Duration
	retryBackoff  time.Duration
	defaultAllow  bool
}

// imagePolicyWebhookConfig holds config data for imagePolicyWebhook
type imagePolicyWebhookConfig struct {
	KubeConfigFile string        `json:"kubeConfigFile"`
	AllowTTL       time.Duration `json:"allowTTL"`
	DenyTTL        time.Duration `json:"denyTTL"`
	RetryBackoff   time.Duration `json:"retryBackoff"`
	DefaultAllow   bool          `json:"defaultAllow"`
}

// AdmissionConfig holds config data for admission controllers
type AdmissionConfig struct {
	ImagePolicyWebhook imagePolicyWebhookConfig `json:"ImagePolicy.webhook"`
}

func (a *imagePolicyWebhook) statusTTL(status v1alpha1.ImageReviewStatus) time.Duration {
	if status.Allowed {
		return a.allowTTL
	}
	return a.denyTTL
}

// Filter out annotations that don't match *.image-policy.k8s.io/*
func (a *imagePolicyWebhook) filterAnnotations(allAnnotations map[string]string) map[string]string {
	annotations := make(map[string]string)
	for k, v := range allAnnotations {
		if strings.Contains(k, ".image-policy.k8s.io/") {
			annotations[k] = v
		}
	}
	return annotations
}

// Function to call on webhook failure; behavior determined by defaultAllow flag
func (a *imagePolicyWebhook) webhookError(err error) error {
	if err != nil {
		glog.V(2).Infof("error contacting webhook backend: %s")
		if a.defaultAllow {
			glog.V(2).Infof("pod allowed in spite of webhook backend failure")
			return nil
		}
		glog.V(2).Infof("pod not allowed due to webhook backend failure ")
		return err
	}
	return nil
}

func (a *imagePolicyWebhook) Admit(attributes admission.Attributes) (err error) {
	// Ignore all calls to subresources or resources other than pods.
	allowedResources := map[unversioned.GroupResource]bool{
		api.Resource("pods"): true,
	}

	if len(attributes.GetSubresource()) != 0 || !allowedResources[attributes.GetResource().GroupResource()] {
		return nil
	}

	pod, ok := attributes.GetObject().(*api.Pod)
	if !ok {
		return apierrors.NewBadRequest("Resource was marked with kind Pod but was unable to be converted")
	}

	// Build list of ImageReviewContainerSpec
	var imageReviewContainerSpecs []v1alpha1.ImageReviewContainerSpec
	containers := make([]api.Container, len(pod.Spec.Containers), len(pod.Spec.Containers)+len(pod.Spec.InitContainers))
	copy(containers, pod.Spec.Containers)
	containers = append(containers, pod.Spec.InitContainers...)
	for _, c := range containers {
		imageReviewContainerSpecs = append(imageReviewContainerSpecs, v1alpha1.ImageReviewContainerSpec{
			Image: c.Image,
		})
	}
	imageReview := v1alpha1.ImageReview{
		Spec: v1alpha1.ImageReviewSpec{
			Containers:  imageReviewContainerSpecs,
			Annotations: a.filterAnnotations(pod.Annotations),
			Namespace:   attributes.GetNamespace(),
		},
	}

	return a.admitPod(&imageReview)
}

func (a *imagePolicyWebhook) admitPod(review *v1alpha1.ImageReview) error {
	cacheKey, err := json.Marshal(review.Spec)
	if err != nil {
		return err
	}
	if entry, ok := a.responseCache.Get(string(cacheKey)); ok {
		review.Status = entry.(v1alpha1.ImageReviewStatus)
	} else {
		result := a.webhook.WithExponentialBackoff(func() restclient.Result {
			return a.webhook.RestClient.Post().Body(review).Do()
		})

		if err := result.Error(); err != nil {
			return a.webhookError(err)
		}
		var statusCode int
		if result.StatusCode(&statusCode); statusCode < 200 || statusCode >= 300 {
			return a.webhookError(fmt.Errorf("Error contacting webhook: %d", statusCode))
		}

		if err := result.Into(review); err != nil {
			return a.webhookError(err)
		}

		a.responseCache.Add(string(cacheKey), review.Status, a.statusTTL(review.Status))
	}

	if !review.Status.Allowed {
		if len(review.Status.Reason) > 0 {
			return fmt.Errorf("image policy webook backend denied one or more images: %s", review.Status.Reason)
		}
		return errors.New("one or more images rejected by webhook backend")
	}

	return nil
}

// NewImagePolicyWebhook a new imagePolicyWebhook from the provided config file.
// The config file is specified by --admission-controller-config-file and has the
// following format for a webhook:
//
//   {
//     "ImagePolicy.webhook": {
//        "kubeConfigFile": "path/to/kubeconfig/for/backend",
//        "allowTTL": 30,           # time in s to cache approval
//        "denyTTL": 30,            # time in s to cache denial
//        "retryBackoff": 500,      # time in ms to wait between retries
//        "defaultAllow": true      # determines behavior if the webhook backend fails
//     }
//   }
//
// The kubeconfig property refers to another file in the kubeconfig format which
// specifies how to connect to the webhook backend.
//
// The kubeconfig's cluster field is used to refer to the remote service, user refers to the returned authorizer.
//
//     # clusters refers to the remote service.
//     clusters:
//     - name: name-of-remote-imagepolicy-service
//       cluster:
//         certificate-authority: /path/to/ca.pem      # CA for verifying the remote service.
//         server: https://images.example.com/policy # URL of remote service to query. Must use 'https'.
//
//     # users refers to the API server's webhook configuration.
//     users:
//     - name: name-of-api-server
//       user:
//         client-certificate: /path/to/cert.pem # cert for the webhook plugin to use
//         client-key: /path/to/key.pem          # key matching the cert
//
// For additional HTTP configuration, refer to the kubeconfig documentation
// http://kubernetes.io/v1.1/docs/user-guide/kubeconfig-file.html.
func NewImagePolicyWebhook(client clientset.Interface, config io.Reader) (admission.Interface, error) {
	configFile, err := ioutil.ReadAll(config)
	if err != nil {
		return nil, err
	}

	return newWithConfig(configFile)
}

func newWithConfig(configFile []byte) (*imagePolicyWebhook, error) {
	var config AdmissionConfig
	err := json.Unmarshal(configFile, &config)
	if err != nil {
		return nil, err
	}

	whConfig := config.ImagePolicyWebhook
	if err := normalizeWebhookConfig(&whConfig); err != nil {
		return nil, err
	}

	gw, err := webhook.NewGenericWebhook(whConfig.KubeConfigFile, groupVersions, whConfig.RetryBackoff)
	if err != nil {
		return nil, err
	}
	return &imagePolicyWebhook{
		Handler:       admission.NewHandler(admission.Create, admission.Update),
		webhook:       gw,
		responseCache: cache.NewLRUExpireCache(1024),
		allowTTL:      whConfig.AllowTTL,
		denyTTL:       whConfig.DenyTTL,
		defaultAllow:  whConfig.DefaultAllow,
	}, nil
}

func normalizeWebhookConfig(config *imagePolicyWebhookConfig) (err error) {
	config.RetryBackoff, err = normalizeConfigValue("backoff", time.Millisecond, config.RetryBackoff, minRetryBackoff, maxRetryBackoff, defaultRetryBackoff)
	if err != nil {
		return err
	}
	config.AllowTTL, err = normalizeConfigValue("allow cache", time.Second, config.AllowTTL, minAllowTTL, maxAllowTTL, defaultAllowTTL)
	if err != nil {
		return err
	}
	config.DenyTTL, err = normalizeConfigValue("deny cache", time.Second, config.DenyTTL, minDenyTTL, maxDenyTTL, defaultDenyTTL)
	if err != nil {
		return err
	}
	return nil
}

func normalizeConfigValue(name string, scale, value, min, max, defaultValue time.Duration) (time.Duration, error) {
	// disable with -1 sentinel
	if value == disableTTL {
		glog.V(2).Infof("image policy webhook %s disabled", name)
		return time.Duration(0), nil
	}

	if value == useDefault {
		glog.V(2).Infof("image policy webhook %s using default value", name)
		return defaultValue, nil
	}

	// convert to s; unmarshalling gives ns
	value *= scale

	// check value is within range
	if value <= min || value > max {
		return value, fmt.Errorf("valid value is between %v and %v, got %v", min, max, value)
	}
	return value, nil
}
