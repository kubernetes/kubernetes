/*
Copyright The Kubernetes Authors.

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

// Package webhookauth wires the API server's egress selector and tracing into
// the rest.Configs used to call webhooks.
//
// It is kept separate from k8s.io/apiserver/pkg/util/webhook so that packages
// which only need the generic webhook client, its retry helpers or its
// validation functions do not depend on the egress selector (gRPC,
// konnectivity) or the OpenTelemetry SDK and exporters.
package webhookauth

import (
	"net/http"

	"go.opentelemetry.io/otel/trace"

	corev1 "k8s.io/api/core/v1"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apiserver/pkg/server/egressselector"
	"k8s.io/apiserver/pkg/util/webhook"
	"k8s.io/client-go/rest"
	"k8s.io/component-base/tracing"
)

// NewDefaultAuthenticationInfoResolverWrapper builds a default authn resolver wrapper
func NewDefaultAuthenticationInfoResolverWrapper(
	proxyTransport *http.Transport,
	egressSelector *egressselector.EgressSelector,
	kubeapiserverClientConfig *rest.Config,
	tp trace.TracerProvider) webhook.AuthenticationInfoResolverWrapper {

	webhookAuthResolverWrapper := func(delegate webhook.AuthenticationInfoResolver) webhook.AuthenticationInfoResolver {
		return &webhook.AuthenticationInfoResolverDelegator{
			ClientConfigForFunc: func(hostPort string) (*rest.Config, error) {
				if hostPort == "kubernetes.default.svc:443" {
					return kubeapiserverClientConfig, nil
				}
				ret, err := delegate.ClientConfigFor(hostPort)
				if err != nil {
					return nil, err
				}
				ret.Wrap(tracing.WrapperFor(tp))

				if egressSelector != nil {
					networkContext := egressselector.ControlPlane.AsNetworkContext()
					var egressDialer utilnet.DialFunc
					egressDialer, err = egressSelector.Lookup(networkContext)

					if err != nil {
						return nil, err
					}

					ret.Dial = egressDialer
				}
				return ret, nil
			},
			ClientConfigForServiceFunc: func(serviceName, serviceNamespace string, servicePort int) (*rest.Config, error) {
				if serviceName == "kubernetes" && serviceNamespace == corev1.NamespaceDefault && servicePort == 443 {
					return kubeapiserverClientConfig, nil
				}
				ret, err := delegate.ClientConfigForService(serviceName, serviceNamespace, servicePort)
				if err != nil {
					return nil, err
				}
				ret.Wrap(tracing.WrapperFor(tp))

				if egressSelector != nil {
					networkContext := egressselector.Cluster.AsNetworkContext()
					var egressDialer utilnet.DialFunc
					egressDialer, err = egressSelector.Lookup(networkContext)
					if err != nil {
						return nil, err
					}

					ret.Dial = egressDialer
				} else if proxyTransport != nil && proxyTransport.DialContext != nil {
					ret.Dial = proxyTransport.DialContext
				}
				return ret, nil
			},
		}
	}
	return webhookAuthResolverWrapper
}
