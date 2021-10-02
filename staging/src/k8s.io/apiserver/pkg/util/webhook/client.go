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

package webhook

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/url"
	"strconv"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apiserver/pkg/util/x509metrics"
	"k8s.io/client-go/rest"
	"k8s.io/utils/lru"
)

const (
	defaultCacheSize = 200
)

// ClientConfig defines parameters required for creating a hook client.
type ClientConfig struct {
	Name     string
	URL      string
	CABundle []byte
	Service  *ClientConfigService
}

// ClientConfigService defines service discovery parameters of the webhook.
type ClientConfigService struct {
	Name      string
	Namespace string
	Path      string
	Port      int32
}

// ClientManager builds REST clients to talk to webhooks. It caches the clients
// to avoid duplicate creation.
type ClientManager struct {
	authInfoResolver     AuthenticationInfoResolver
	serviceResolver      ServiceResolver
	negotiatedSerializer runtime.NegotiatedSerializer
	cache                *lru.Cache
}

// NewClientManager creates a clientManager.
func NewClientManager(gvs []schema.GroupVersion, addToSchemaFuncs ...func(s *runtime.Scheme) error) (ClientManager, error) {
	cache := lru.New(defaultCacheSize)
	hookScheme := runtime.NewScheme()
	for _, addToSchemaFunc := range addToSchemaFuncs {
		if err := addToSchemaFunc(hookScheme); err != nil {
			return ClientManager{}, err
		}
	}
	return ClientManager{
		cache: cache,
		negotiatedSerializer: serializer.NegotiatedSerializerWrapper(runtime.SerializerInfo{
			Serializer: serializer.NewCodecFactory(hookScheme).LegacyCodec(gvs...),
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
func (cm *ClientManager) HookClient(cc ClientConfig) (*rest.RESTClient, error) {
	ccWithNoName := cc
	ccWithNoName.Name = ""
	cacheKey, err := json.Marshal(ccWithNoName)
	if err != nil {
		return nil, err
	}
	if client, ok := cm.cache.Get(string(cacheKey)); ok {
		return client.(*rest.RESTClient), nil
	}

	complete := func(cfg *rest.Config) (*rest.RESTClient, error) {
		// Avoid client-side rate limiting talking to the webhook backend.
		// Rate limiting should happen when deciding how many requests to serve.
		cfg.QPS = -1

		// Combine CAData from the config with any existing CA bundle provided
		if len(cfg.TLSClientConfig.CAData) > 0 {
			cfg.TLSClientConfig.CAData = append(cfg.TLSClientConfig.CAData, '\n')
		}
		cfg.TLSClientConfig.CAData = append(cfg.TLSClientConfig.CAData, cc.CABundle...)

		// Use http/1.1 instead of http/2.
		// This is a workaround for http/2-enabled clients not load-balancing concurrent requests to multiple backends.
		// See http://issue.k8s.io/75791 for details.
		cfg.NextProtos = []string{"http/1.1"}

		cfg.ContentConfig.NegotiatedSerializer = cm.negotiatedSerializer
		cfg.ContentConfig.ContentType = runtime.ContentTypeJSON

		// Add a transport wrapper that allows detection of TLS connections to
		// servers without SAN extension in their serving certificates
		cfg.Wrap(x509metrics.NewMissingSANRoundTripperWrapperConstructor(x509MissingSANCounter))

		client, err := rest.UnversionedRESTClientFor(cfg)
		if err == nil {
			cm.cache.Add(string(cacheKey), client)
		}
		return client, err
	}

	if cc.Service != nil {
		port := cc.Service.Port
		if port == 0 {
			// Default to port 443 if no service port is specified
			port = 443
		}

		restConfig, err := cm.authInfoResolver.ClientConfigForService(cc.Service.Name, cc.Service.Namespace, int(port))
		if err != nil {
			return nil, err
		}
		cfg := rest.CopyConfig(restConfig)
		serverName := cc.Service.Name + "." + cc.Service.Namespace + ".svc"

		host := net.JoinHostPort(serverName, strconv.Itoa(int(port)))
		cfg.Host = "https://" + host
		cfg.APIPath = cc.Service.Path
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
				u, err := cm.serviceResolver.ResolveEndpoint(cc.Service.Namespace, cc.Service.Name, port)
				if err != nil {
					return nil, err
				}
				addr = u.Host
			}
			return delegateDialer(ctx, network, addr)
		}

		return complete(cfg)
	}

	if cc.URL == "" {
		return nil, &ErrCallingWebhook{WebhookName: cc.Name, Reason: errors.New("webhook configuration must have either service or URL")}
	}

	u, err := url.Parse(cc.URL)
	if err != nil {
		return nil, &ErrCallingWebhook{WebhookName: cc.Name, Reason: fmt.Errorf("Unparsable URL: %v", err)}
	}

	hostPort := u.Host
	if len(u.Port()) == 0 {
		// Default to port 443 if no port is specified
		hostPort = net.JoinHostPort(hostPort, "443")
	}

	restConfig, err := cm.authInfoResolver.ClientConfigFor(hostPort)
	if err != nil {
		return nil, err
	}

	cfg := rest.CopyConfig(restConfig)
	cfg.Host = u.Scheme + "://" + u.Host
	cfg.APIPath = u.Path

	return complete(cfg)
}
