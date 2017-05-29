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

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/authenticatorfactory"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/authorization/authorizerfactory"
	clientset "k8s.io/client-go/kubernetes"
	authenticationclient "k8s.io/client-go/kubernetes/typed/authentication/v1beta1"
	authorizationclient "k8s.io/client-go/kubernetes/typed/authorization/v1beta1"

	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/kubelet/server"
)

// BuildAuth creates an authenticator, an authorizer, and a matching authorizer attributes getter compatible with the kubelet's needs
func BuildAuth(nodeName types.NodeName, client clientset.Interface, config componentconfig.KubeletConfiguration) (server.AuthInterface, error) {
	// Get clients, if provided
	var (
		tokenClient authenticationclient.TokenReviewInterface
		sarClient   authorizationclient.SubjectAccessReviewInterface
	)
	if client != nil && !reflect.ValueOf(client).IsNil() {
		tokenClient = client.AuthenticationV1beta1().TokenReviews()
		sarClient = client.AuthorizationV1beta1().SubjectAccessReviews()
	}

	authenticator, err := BuildAuthn(tokenClient, config.Authentication)
	if err != nil {
		return nil, err
	}

	attributes := server.NewNodeAuthorizerAttributesGetter(nodeName)

	authorizer, err := BuildAuthz(sarClient, config.Authorization)
	if err != nil {
		return nil, err
	}

	return server.NewKubeletAuth(authenticator, attributes, authorizer), nil
}

// BuildAuthn creates an authenticator compatible with the kubelet's needs
func BuildAuthn(client authenticationclient.TokenReviewInterface, authn componentconfig.KubeletAuthentication) (authenticator.Request, error) {
	authenticatorConfig := authenticatorfactory.DelegatingAuthenticatorConfig{
		Anonymous:    authn.Anonymous.Enabled,
		CacheTTL:     authn.Webhook.CacheTTL.Duration,
		ClientCAFile: authn.X509.ClientCAFile,
	}

	if authn.Webhook.Enabled {
		if client == nil {
			return nil, errors.New("no client provided, cannot use webhook authentication")
		}
		authenticatorConfig.TokenAccessReviewClient = client
	}

	authenticator, _, err := authenticatorConfig.New()
	return authenticator, err
}

// BuildAuthz creates an authorizer compatible with the kubelet's needs
func BuildAuthz(client authorizationclient.SubjectAccessReviewInterface, authz componentconfig.KubeletAuthorization) (authorizer.Authorizer, error) {
	switch authz.Mode {
	case componentconfig.KubeletAuthorizationModeAlwaysAllow:
		return authorizerfactory.NewAlwaysAllowAuthorizer(), nil

	case componentconfig.KubeletAuthorizationModeWebhook:
		if client == nil {
			return nil, errors.New("no client provided, cannot use webhook authorization")
		}
		authorizerConfig := authorizerfactory.DelegatingAuthorizerConfig{
			SubjectAccessReviewClient: client,
			AllowCacheTTL:             authz.Webhook.CacheAuthorizedTTL.Duration,
			DenyCacheTTL:              authz.Webhook.CacheUnauthorizedTTL.Duration,
		}
		return authorizerConfig.New()

	case "":
		return nil, fmt.Errorf("No authorization mode specified")

	default:
		return nil, fmt.Errorf("Unknown authorization mode %s", authz.Mode)

	}
}
