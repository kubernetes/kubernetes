/*
Copyright 2017 The Kubernetes Authors.

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

package authorizer

import (
	"fmt"
	"net"
	"time"

	"github.com/golang/glog"
	"github.com/pkg/errors"

	authorization "k8s.io/api/authorization/v1beta1"
	"k8s.io/apimachinery/pkg/apimachinery/registered"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilwebhook "k8s.io/apiserver/pkg/util/webhook"
	"k8s.io/apiserver/plugin/pkg/authorizer/webhook"
	"k8s.io/client-go/kubernetes/scheme"
	authorizationclient "k8s.io/client-go/kubernetes/typed/authorization/v1beta1"
	"k8s.io/client-go/rest"
)

// defaultRequestTimeout is the default timeout for the kube apiserver webhook
// requests.
const defaultRequestTimeout = 30 * time.Second

// This code is copied from the generic version at
// k8s.io/apiserver/plugin/pkg/authorizer/webhook/webhook.go
var (
	registry      = registered.NewOrDie("")
	groupVersions = []schema.GroupVersion{authorization.SchemeGroupVersion}
)

func init() {
	registry.RegisterVersions(groupVersions)
	if err := registry.EnableVersions(groupVersions...); err != nil {
		panic(fmt.Sprintf("failed to enable version %v", groupVersions))
	}
}

// NewWebhookAuthorizer creates a new kubeapiserver specific webhook
// authorizer.  configFilePath is the filesystem path to a kubeconfig file that
// holds the webhook configuration.  cacheAuthorizedTTL is the time for which
// the positive responses from the authorizer are cached.
// cacheUnauthorizedTTL is the time for which the negative responses from the
// authorizer are cached.  dialer is the custom dialer function for establishing
// the webhook server connections.
func NewWebhookAuthorizer(
	configFilePath string,
	cacheAuthorizedTTL time.Duration,
	cacheUnauthorizedTTL time.Duration,
	dialer func(network, address string) (net.Conn, error),
) (*webhook.WebhookAuthorizer, error) {
	// Load the configuration from a kubeconfig file.
	clientConfig, err := utilwebhook.NewRestConfig(
		registry, scheme.Codecs, configFilePath, groupVersions, defaultRequestTimeout)
	if err != nil {
		return nil, fmt.Errorf("invalid webhook config: %v", err)
	}
	clientConfig.Dial = dialer
	restClient, err := rest.UnversionedRESTClientFor(clientConfig)
	if err != nil {
		return nil, errors.Wrapf(err, "while creating REST client for webhook config: %v", configFilePath)
	}
	return webhook.NewFromInterface(&subjectAccessReviewClient{restClient},
		cacheAuthorizedTTL, cacheUnauthorizedTTL)
}

var _ authorizationclient.SubjectAccessReviewInterface = (*subjectAccessReviewClient)(nil)

// subjectAccessReview client is a kubeapiserver specific authorizer client.
type subjectAccessReviewClient struct {
	// The REST client used to issue authorization requests.
	restClient *rest.RESTClient
}

// Create calls out to the webhook authorizer with 'subjectAccessReview' to
// review.  Returns the authorization decision, or error in case of failure.
func (t *subjectAccessReviewClient) Create(
	subjectAccessReview *authorization.SubjectAccessReview,
) (*authorization.SubjectAccessReview, error) {
	glog.V(2).Infof("Create request: %+v", subjectAccessReview)
	result := &authorization.SubjectAccessReview{}
	err := t.restClient.Post().Body(subjectAccessReview).Do().Into(result)
	glog.V(4).Infof("Create response:\n\tresult=%+v\n\terr=%v", result, err)
	return result, err
}
