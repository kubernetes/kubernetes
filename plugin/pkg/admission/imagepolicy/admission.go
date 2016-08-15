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
	"k8s.io/kubernetes/pkg/apis/imagepolicy/v1beta1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/util/cache"
	"k8s.io/kubernetes/plugin/pkg/webhook"

	"k8s.io/kubernetes/pkg/admission"
)

var (
	groupVersions = []unversioned.GroupVersion{v1beta1.SchemeGroupVersion}
)

const retryBackoff = 500 * time.Millisecond

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
	*webhook.GenericWebhook
	responseCache *cache.LRUExpireCache
	allowTTL      time.Duration
	denyTTL       time.Duration
	defaultAllow  bool
}

// imagePolicyWebhookConfig holds config data for imagePolicyWebhook
type imagePolicyWebhookConfig struct {
	KubeConfigFile string        `json:"kubeConfigFile"`
	AllowTTL       time.Duration `json:"allowTTL"`
	DenyTTL        time.Duration `json:"denyTTL"`
	DefaultAllow   bool          `json:"defaultAllow"`
}

// AdmissionConfig holds config data for admission controllers
type AdmissionConfig struct {
	Webhook imagePolicyWebhookConfig `json:"webhook"`
}

func (a *imagePolicyWebhook) statusTTL(status v1beta1.ImageReviewStatus) time.Duration {
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
		glog.V(4).Infof("error contacting webhook backend: %s")
		if a.defaultAllow {
			glog.V(4).Infof("pod allowed in spite of webhook backend failure")
			return nil
		}
		glog.V(4).Infof("pod not allowed due to webhook backend failure ")
		return err
	}
	return nil
}

func (a *imagePolicyWebhook) Admit(attributes admission.Attributes) (err error) {
	// Ignore all calls to subresources or resources other than pods.
	if len(attributes.GetSubresource()) != 0 || attributes.GetResource().GroupResource() != api.Resource("pods") {
		return nil
	}
	pod, ok := attributes.GetObject().(*api.Pod)
	if !ok {
		return apierrors.NewBadRequest("Resource was marked with kind Pod but was unable to be converted")
	}

	annotations := a.filterAnnotations(pod.Annotations)
	namespace := attributes.GetNamespace()

	// Build list of ImageReviews to send to backend, one per image
	var imageReviews []v1beta1.ImageReview
	for _, c := range pod.Spec.Containers {
		imageReviews = append(imageReviews, v1beta1.ImageReview{
			Spec: v1beta1.ImageReviewSpec{
				Container: v1beta1.ImageReviewContainerSpec{
					Image: c.Image,
				},
				Annotations: annotations,
				Namespace:   namespace,
			},
		})
	}

	for _, r := range imageReviews {
		if err := a.admitContainer(&r); err != nil {
			return err
		}
	}

	return nil
}

func (a *imagePolicyWebhook) admitContainer(review *v1beta1.ImageReview) error {
	cacheKey, err := json.Marshal(review.Spec)
	if err != nil {
		return err
	}
	if entry, ok := a.responseCache.Get(string(cacheKey)); ok {
		review.Status = entry.(v1beta1.ImageReviewStatus)
	} else {
		result := a.WithExponentialBackoff(func() restclient.Result {
			return a.RestClient.Post().Body(review).Do()
		})

		if err := result.Error(); err != nil {
			return a.webhookError(err)
		}
		var statusCode int
		if result.StatusCode(&statusCode); statusCode < 200 || statusCode >= 300 {
			return a.webhookError(fmt.Errorf("Error contacting webhook: %d", statusCode))
		}

		raw, err := result.Raw()
		if err != nil {
			return a.webhookError(err)
		}
		json.Unmarshal(raw, review)

		// TODO: why doesn't this work?
		//if err := result.Into(review); err != nil {
		//	return a.webhookError(err)
		//}

		a.responseCache.Add(string(cacheKey), review.Status, a.statusTTL(review.Status))
	}

	if !review.Status.Allowed {
		return errors.New(review.Status.Reason)
	}

	return nil
}

// NewImagePolicyWebhook a new imagePolicyWebhook from the provided config file.
// The config file is specified by --admission-controller-config-file and has the
// following format for a webhook:
//
//   {
//     "webhook": {
//        "kubeConfigFile": "path/to/kubeconfig/for/backend",
//        "allowTTL": 500000,       # time in ns to cache approval
//				"denyTTL": 500000,        # time in ns to cache denial
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

	return newWithBackoff(configFile, retryBackoff)
}

// newWithBackoff allows tests to skip the sleep.
func newWithBackoff(configFile []byte, initialBackoff time.Duration) (*imagePolicyWebhook, error) {
	var config AdmissionConfig
	err := json.Unmarshal(configFile, &config)
	if err != nil {
		return nil, err
	}

	gw, err := webhook.NewGenericWebhook(config.Webhook.KubeConfigFile, groupVersions, initialBackoff)
	if err != nil {
		return nil, err
	}
	return &imagePolicyWebhook{
		Handler:        admission.NewHandler(admission.Create, admission.Update),
		GenericWebhook: gw,
		responseCache:  cache.NewLRUExpireCache(1024),
		allowTTL:       config.Webhook.AllowTTL,
		denyTTL:        config.Webhook.DenyTTL,
		defaultAllow:   config.Webhook.DefaultAllow,
	}, nil
}
