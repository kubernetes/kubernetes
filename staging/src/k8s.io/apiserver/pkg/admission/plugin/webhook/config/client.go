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

package config

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/url"

	lru "github.com/hashicorp/golang-lru"
	admissionv1beta1 "k8s.io/api/admission/v1beta1"
	"k8s.io/api/admissionregistration/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	webhookerrors "k8s.io/apiserver/pkg/admission/plugin/webhook/errors"
	"k8s.io/client-go/rest"
)

const (
	defaultCacheSize = 200
)

var (
	ErrNeedServiceOrURL = errors.New("webhook configuration must have either service or URL")
)

// ClientManager builds REST clients to talk to webhooks. It caches the clients
// to avoid duplicate creation.
type ClientManager struct {
	authInfoResolver     AuthenticationInfoResolver
	serviceResolver      ServiceResolver
	negotiatedSerializer runtime.NegotiatedSerializer
	cache                *lru.Cache
}

// NewClientManager creates a clientManager.
func NewClientManager() (ClientManager, error) {
	cache, err := lru.New(defaultCacheSize)
	if err != nil {
		return ClientManager{}, err
	}
	admissionScheme := runtime.NewScheme()
	admissionv1beta1.AddToScheme(admissionScheme)
	return ClientManager{
		cache: cache,
		negotiatedSerializer: serializer.NegotiatedSerializerWrapper(runtime.SerializerInfo{
			Serializer: serializer.NewCodecFactory(admissionScheme).LegacyCodec(admissionv1beta1.SchemeGroupVersion),
		}),
	}, nil
}

// SetAuthenticationInfoResolverWrapper sets the
// AuthenticationInfoResolverWrapper.
func (cm *ClientManager) SetAuthenticationInfoResolverWrapper(wrapper AuthenticationInfoResolverWrapper) {
	if wrapper != nil {
		cm.authInfoResolver = wrapper(cm.authInfoResolver)
	}
}

// SetAuthenticationInfoResolver sets the AuthenticationInfoResolver.
func (cm *ClientManager) SetAuthenticationInfoResolver(resolver AuthenticationInfoResolver) {
	cm.authInfoResolver = resolver
}

// SetServiceResolver sets the ServiceResolver.
func (cm *ClientManager) SetServiceResolver(sr ServiceResolver) {
	if sr != nil {
		cm.serviceResolver = sr
	}
}

// Validate checks if ClientManager is properly set up.
func (cm *ClientManager) Validate() error {
	var errs []error
	if cm.negotiatedSerializer == nil {
		errs = append(errs, fmt.Errorf("the clientManager requires a negotiatedSerializer"))
	}
	if cm.serviceResolver == nil {
		errs = append(errs, fmt.Errorf("the clientManager requires a serviceResolver"))
	}
	if cm.authInfoResolver == nil {
		errs = append(errs, fmt.Errorf("the clientManager requires an authInfoResolver"))
	}
	return utilerrors.NewAggregate(errs)
}

// HookClient get a RESTClient from the cache, or constructs one based on the
// webhook configuration.
func (cm *ClientManager) HookClient(h *v1beta1.Webhook) (*rest.RESTClient, error) {
	cacheKey, err := json.Marshal(h.ClientConfig)
	if err != nil {
		return nil, err
	}
	if client, ok := cm.cache.Get(string(cacheKey)); ok {
		return client.(*rest.RESTClient), nil
	}

	complete := func(cfg *rest.Config) (*rest.RESTClient, error) {
		// Combine CAData from the config with any existing CA bundle provided
		if len(cfg.TLSClientConfig.CAData) > 0 {
			cfg.TLSClientConfig.CAData = append(cfg.TLSClientConfig.CAData, '\n')
		}
		cfg.TLSClientConfig.CAData = append(cfg.TLSClientConfig.CAData, h.ClientConfig.CABundle...)

		cfg.ContentConfig.NegotiatedSerializer = cm.negotiatedSerializer
		cfg.ContentConfig.ContentType = runtime.ContentTypeJSON
		client, err := rest.UnversionedRESTClientFor(cfg)
		if err == nil {
			cm.cache.Add(string(cacheKey), client)
		}
		return client, err
	}

	if svc := h.ClientConfig.Service; svc != nil {
		restConfig, err := cm.authInfoResolver.ClientConfigForService(svc.Name, svc.Namespace)
		if err != nil {
			return nil, err
		}
		cfg := rest.CopyConfig(restConfig)
		serverName := svc.Name + "." + svc.Namespace + ".svc"
		host := serverName + ":443"
		cfg.Host = "https://" + host
		if svc.Path != nil {
			cfg.APIPath = *svc.Path
		}
		// Set the server name if not already set
		if len(cfg.TLSClientConfig.ServerName) == 0 {
			cfg.TLSClientConfig.ServerName = serverName
		}

		delegateDialer := cfg.Dial
		if delegateDialer == nil {
			var d net.Dialer
			delegateDialer = d.DialContext
		}
		cfg.Dial = func(ctx context.Context, network, addr string) (net.Conn, error) {
			if addr == host {
				u, err := cm.serviceResolver.ResolveEndpoint(svc.Namespace, svc.Name)
				if err != nil {
					return nil, err
				}
				addr = u.Host
			}
			return delegateDialer(ctx, network, addr)
		}

		return complete(cfg)
	}

	if h.ClientConfig.URL == nil {
		return nil, &webhookerrors.ErrCallingWebhook{WebhookName: h.Name, Reason: ErrNeedServiceOrURL}
	}

	u, err := url.Parse(*h.ClientConfig.URL)
	if err != nil {
		return nil, &webhookerrors.ErrCallingWebhook{WebhookName: h.Name, Reason: fmt.Errorf("Unparsable URL: %v", err)}
	}

	restConfig, err := cm.authInfoResolver.ClientConfigFor(u.Host)
	if err != nil {
		return nil, err
	}

	cfg := rest.CopyConfig(restConfig)
	cfg.Host = u.Scheme + "://" + u.Host
	cfg.APIPath = u.Path

	return complete(cfg)
}
