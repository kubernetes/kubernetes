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

// Package webhook implements a generic HTTP webhook plugin.
package webhook

import (
	"fmt"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
)

// defaultRequestTimeout is set for all webhook request. This is the absolute
// timeout of the HTTP request, including reading the response body.
const defaultRequestTimeout = 30 * time.Second

type GenericWebhook struct {
	RestClient     *rest.RESTClient
	initialBackoff time.Duration
}

// NewGenericWebhook creates a new GenericWebhook from the provided kubeconfig file.
func NewGenericWebhook(scheme *runtime.Scheme, codecFactory serializer.CodecFactory, kubeConfigFile string, groupVersions []schema.GroupVersion, initialBackoff time.Duration) (*GenericWebhook, error) {
	return newGenericWebhook(scheme, codecFactory, kubeConfigFile, groupVersions, initialBackoff, defaultRequestTimeout)
}

func newGenericWebhook(scheme *runtime.Scheme, codecFactory serializer.CodecFactory, kubeConfigFile string, groupVersions []schema.GroupVersion, initialBackoff, requestTimeout time.Duration) (*GenericWebhook, error) {
	for _, groupVersion := range groupVersions {
		if !scheme.IsVersionRegistered(groupVersion) {
			return nil, fmt.Errorf("webhook plugin requires enabling extension resource: %s", groupVersion)
		}
	}

	loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
	loadingRules.ExplicitPath = kubeConfigFile
	loader := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(loadingRules, &clientcmd.ConfigOverrides{})

	clientConfig, err := loader.ClientConfig()
	if err != nil {
		return nil, err
	}

	// Kubeconfigs can't set a timeout, this can only be set through a command line flag.
	//
	// https://github.com/kubernetes/client-go/blob/master/tools/clientcmd/overrides.go
	//
	// Set this to something reasonable so request to webhooks don't hang forever.
	clientConfig.Timeout = requestTimeout

	codec := codecFactory.LegacyCodec(groupVersions...)
	clientConfig.ContentConfig.NegotiatedSerializer = serializer.NegotiatedSerializerWrapper(runtime.SerializerInfo{Serializer: codec})

	restClient, err := rest.UnversionedRESTClientFor(clientConfig)
	if err != nil {
		return nil, err
	}

	return &GenericWebhook{restClient, initialBackoff}, nil
}

// WithExponentialBackoff will retry webhookFn() up to 5 times with exponentially increasing backoff when
// it returns an error for which apierrors.SuggestsClientDelay() or apierrors.IsInternalError() returns true.
func (g *GenericWebhook) WithExponentialBackoff(webhookFn func() rest.Result) rest.Result {
	var result rest.Result
	WithExponentialBackoff(g.initialBackoff, func() error {
		result = webhookFn()
		return result.Error()
	})
	return result
}

// WithExponentialBackoff will retry webhookFn() up to 5 times with exponentially increasing backoff when
// it returns an error for which apierrors.SuggestsClientDelay() or apierrors.IsInternalError() returns true.
func WithExponentialBackoff(initialBackoff time.Duration, webhookFn func() error) error {
	backoff := wait.Backoff{
		Duration: initialBackoff,
		Factor:   1.5,
		Jitter:   0.2,
		Steps:    5,
	}

	var err error
	wait.ExponentialBackoff(backoff, func() (bool, error) {
		err = webhookFn()
		// these errors indicate a transient error that should be retried.
		if net.IsConnectionReset(err) || apierrors.IsInternalError(err) || apierrors.IsTimeout(err) || apierrors.IsTooManyRequests(err) {
			return false, nil
		}
		// if the error sends the Retry-After header, we respect it as an explicit confirmation we should retry.
		if _, shouldRetry := apierrors.SuggestsClientDelay(err); shouldRetry {
			return false, nil
		}
		if err != nil {
			return false, err
		}
		return true, nil
	})
	return err
}
