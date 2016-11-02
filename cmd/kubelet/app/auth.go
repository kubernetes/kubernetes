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

package app

import (
	"errors"
	"fmt"
	"reflect"

	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/auth/authenticator"
	"k8s.io/kubernetes/pkg/auth/authenticator/bearertoken"
	"k8s.io/kubernetes/pkg/auth/authorizer"
	"k8s.io/kubernetes/pkg/auth/group"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	authenticationclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/authentication/internalversion"
	authorizationclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/authorization/internalversion"
	alwaysallowauthorizer "k8s.io/kubernetes/pkg/genericapiserver/authorizer"
	"k8s.io/kubernetes/pkg/kubelet/server"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/cert"
	"k8s.io/kubernetes/plugin/pkg/auth/authenticator/request/anonymous"
	unionauth "k8s.io/kubernetes/plugin/pkg/auth/authenticator/request/union"
	"k8s.io/kubernetes/plugin/pkg/auth/authenticator/request/x509"
	webhooktoken "k8s.io/kubernetes/plugin/pkg/auth/authenticator/token/webhook"
	webhooksar "k8s.io/kubernetes/plugin/pkg/auth/authorizer/webhook"
)

func buildAuth(nodeName types.NodeName, client internalclientset.Interface, config componentconfig.KubeletConfiguration) (server.AuthInterface, error) {
	// Get clients, if provided
	var (
		tokenClient authenticationclient.TokenReviewInterface
		sarClient   authorizationclient.SubjectAccessReviewInterface
	)
	if client != nil && !reflect.ValueOf(client).IsNil() {
		tokenClient = client.Authentication().TokenReviews()
		sarClient = client.Authorization().SubjectAccessReviews()
	}

	authenticator, err := buildAuthn(tokenClient, config.Authentication)
	if err != nil {
		return nil, err
	}

	attributes := server.NewNodeAuthorizerAttributesGetter(nodeName)

	authorizer, err := buildAuthz(sarClient, config.Authorization)
	if err != nil {
		return nil, err
	}

	return server.NewKubeletAuth(authenticator, attributes, authorizer), nil
}

func buildAuthn(client authenticationclient.TokenReviewInterface, authn componentconfig.KubeletAuthentication) (authenticator.Request, error) {
	authenticators := []authenticator.Request{}

	// x509 client cert auth
	if len(authn.X509.ClientCAFile) > 0 {
		clientCAs, err := cert.NewPool(authn.X509.ClientCAFile)
		if err != nil {
			return nil, fmt.Errorf("unable to load client CA file %s: %v", authn.X509.ClientCAFile, err)
		}
		verifyOpts := x509.DefaultVerifyOptions()
		verifyOpts.Roots = clientCAs
		authenticators = append(authenticators, x509.New(verifyOpts, x509.CommonNameUserConversion))
	}

	// bearer token auth that uses authentication.k8s.io TokenReview to determine userinfo
	if authn.Webhook.Enabled {
		if client == nil {
			return nil, errors.New("no client provided, cannot use webhook authentication")
		}
		tokenAuth, err := webhooktoken.NewFromInterface(client, authn.Webhook.CacheTTL.Duration)
		if err != nil {
			return nil, err
		}
		authenticators = append(authenticators, bearertoken.New(tokenAuth))
	}

	if len(authenticators) == 0 {
		if authn.Anonymous.Enabled {
			return anonymous.NewAuthenticator(), nil
		}
		return nil, errors.New("No authentication method configured")
	}

	authenticator := group.NewGroupAdder(unionauth.New(authenticators...), []string{"system:authenticated"})
	if authn.Anonymous.Enabled {
		authenticator = unionauth.NewFailOnError(authenticator, anonymous.NewAuthenticator())
	}
	return authenticator, nil
}

func buildAuthz(client authorizationclient.SubjectAccessReviewInterface, authz componentconfig.KubeletAuthorization) (authorizer.Authorizer, error) {
	switch authz.Mode {
	case componentconfig.KubeletAuthorizationModeAlwaysAllow:
		return alwaysallowauthorizer.NewAlwaysAllowAuthorizer(), nil

	case componentconfig.KubeletAuthorizationModeWebhook:
		if client == nil {
			return nil, errors.New("no client provided, cannot use webhook authorization")
		}
		return webhooksar.NewFromInterface(
			client,
			authz.Webhook.CacheAuthorizedTTL.Duration,
			authz.Webhook.CacheUnauthorizedTTL.Duration,
		)

	case "":
		return nil, fmt.Errorf("No authorization mode specified")

	default:
		return nil, fmt.Errorf("Unknown authorization mode %s", authz.Mode)

	}
}
