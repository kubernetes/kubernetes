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

package authenticator

import (
	"errors"
	"fmt"
	"time"

	"github.com/go-openapi/spec"

	"k8s.io/kubernetes/pkg/auth/authenticator"
	"k8s.io/kubernetes/pkg/auth/authenticator/bearertoken"
	"k8s.io/kubernetes/pkg/auth/group"
	"k8s.io/kubernetes/pkg/auth/user"
	authenticationclient "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/authentication/v1beta1"
	"k8s.io/kubernetes/pkg/util/cert"
	"k8s.io/kubernetes/plugin/pkg/auth/authenticator/request/anonymous"
	"k8s.io/kubernetes/plugin/pkg/auth/authenticator/request/headerrequest"
	unionauth "k8s.io/kubernetes/plugin/pkg/auth/authenticator/request/union"
	"k8s.io/kubernetes/plugin/pkg/auth/authenticator/request/x509"
	webhooktoken "k8s.io/kubernetes/plugin/pkg/auth/authenticator/token/webhook"
)

// DelegatingAuthenticatorConfig is the minimal configuration needed to create an authenticator
// built to delegate authentication to a kube API server
type DelegatingAuthenticatorConfig struct {
	Anonymous bool

	TokenAccessReviewClient authenticationclient.TokenReviewInterface

	// CacheTTL is the length of time that a token authentication answer will be cached.
	CacheTTL time.Duration

	// ClientCAFile is the CA bundle file used to authenticate client certificates
	ClientCAFile string

	RequestHeaderConfig *RequestHeaderConfig
}

func (c DelegatingAuthenticatorConfig) New() (authenticator.Request, *spec.SecurityDefinitions, error) {
	authenticators := []authenticator.Request{}
	securityDefinitions := spec.SecurityDefinitions{}

	// front-proxy first, then remote
	// Add the front proxy authenticator if requested
	if c.RequestHeaderConfig != nil {
		requestHeaderAuthenticator, err := headerrequest.NewSecure(
			c.RequestHeaderConfig.ClientCA,
			c.RequestHeaderConfig.AllowedClientNames,
			c.RequestHeaderConfig.UsernameHeaders,
			c.RequestHeaderConfig.GroupHeaders,
			c.RequestHeaderConfig.ExtraHeaderPrefixes,
		)
		if err != nil {
			return nil, nil, err
		}
		authenticators = append(authenticators, requestHeaderAuthenticator)
	}

	// x509 client cert auth
	if len(c.ClientCAFile) > 0 {
		clientCAs, err := cert.NewPool(c.ClientCAFile)
		if err != nil {
			return nil, nil, fmt.Errorf("unable to load client CA file %s: %v", c.ClientCAFile, err)
		}
		verifyOpts := x509.DefaultVerifyOptions()
		verifyOpts.Roots = clientCAs
		authenticators = append(authenticators, x509.New(verifyOpts, x509.CommonNameUserConversion))
	}

	if c.TokenAccessReviewClient != nil {
		tokenAuth, err := webhooktoken.NewFromInterface(c.TokenAccessReviewClient, c.CacheTTL)
		if err != nil {
			return nil, nil, err
		}
		authenticators = append(authenticators, bearertoken.New(tokenAuth))

		securityDefinitions["BearerToken"] = &spec.SecurityScheme{
			SecuritySchemeProps: spec.SecuritySchemeProps{
				Type:        "apiKey",
				Name:        "authorization",
				In:          "header",
				Description: "Bearer Token authentication",
			},
		}
	}

	if len(authenticators) == 0 {
		if c.Anonymous {
			return anonymous.NewAuthenticator(), &securityDefinitions, nil
		}
		return nil, nil, errors.New("No authentication method configured")
	}

	authenticator := group.NewGroupAdder(unionauth.New(authenticators...), []string{user.AllAuthenticated})
	if c.Anonymous {
		authenticator = unionauth.NewFailOnError(authenticator, anonymous.NewAuthenticator())
	}
	return authenticator, &securityDefinitions, nil
}
