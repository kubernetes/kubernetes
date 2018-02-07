/*
Copyright 2018 The Kubernetes Authors.

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

package testing

import (
	"sync/atomic"

	"k8s.io/apiserver/pkg/admission/plugin/webhook/config"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/testcerts"
	"k8s.io/client-go/rest"
)

// Wrapper turns an AuthenticationInfoResolver into a AuthenticationInfoResolverWrapper that unconditionally
// returns the given AuthenticationInfoResolver.
func Wrapper(r config.AuthenticationInfoResolver) func(config.AuthenticationInfoResolver) config.AuthenticationInfoResolver {
	return func(config.AuthenticationInfoResolver) config.AuthenticationInfoResolver {
		return r
	}
}

// NewAuthenticationInfoResolver creates a fake AuthenticationInfoResolver that counts cache misses on
// every call to its methods.
func NewAuthenticationInfoResolver(cacheMisses *int32) config.AuthenticationInfoResolver {
	return &authenticationInfoResolver{
		restConfig: &rest.Config{
			TLSClientConfig: rest.TLSClientConfig{
				CAData:   testcerts.CACert,
				CertData: testcerts.ClientCert,
				KeyData:  testcerts.ClientKey,
			},
		},
		cacheMisses: cacheMisses,
	}
}

type authenticationInfoResolver struct {
	restConfig  *rest.Config
	cacheMisses *int32
}

func (a *authenticationInfoResolver) ClientConfigFor(server string) (*rest.Config, error) {
	atomic.AddInt32(a.cacheMisses, 1)
	return a.restConfig, nil
}

func (a *authenticationInfoResolver) ClientConfigForService(serviceName, serviceNamespace string) (*rest.Config, error) {
	atomic.AddInt32(a.cacheMisses, 1)
	return a.restConfig, nil
}
