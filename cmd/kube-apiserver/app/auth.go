/*
Copyright 2014 The Kubernetes Authors.

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

// Package app does all of the work necessary to create a Kubernetes
// APIServer by binding together the API, master and APIServer infrastructure.
// It can be configured and called directly or via the hyperkube framework.
package app

import (
	"github.com/go-openapi/spec"

	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientgoinformers "k8s.io/client-go/informers"
	clientgoclientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	serviceaccountcontroller "k8s.io/kubernetes/pkg/controller/serviceaccount"
	"k8s.io/kubernetes/pkg/features"
	_ "k8s.io/kubernetes/pkg/util/workqueue/prometheus" // for workqueue metric registration
	"k8s.io/kubernetes/plugin/pkg/auth/authenticator/token/bootstrap"
)

// BuildAuthenticator constructs the authenticator
func BuildAuthenticator(s *options.ServerRunOptions, extclient clientgoclientset.Interface, versionedInformer clientgoinformers.SharedInformerFactory) (authenticator.Request, *spec.SecurityDefinitions, error) {
	authenticatorConfig := s.Authentication.ToAuthenticationConfig()
	if s.Authentication.ServiceAccounts.Lookup || utilfeature.DefaultFeatureGate.Enabled(features.TokenRequest) {
		authenticatorConfig.ServiceAccountTokenGetter = serviceaccountcontroller.NewGetterFromClient(
			extclient,
			versionedInformer.Core().V1().Secrets().Lister(),
			versionedInformer.Core().V1().ServiceAccounts().Lister(),
			versionedInformer.Core().V1().Pods().Lister(),
		)
	}
	authenticatorConfig.BootstrapTokenAuthenticator = bootstrap.NewTokenAuthenticator(
		versionedInformer.Core().V1().Secrets().Lister().Secrets(v1.NamespaceSystem),
	)

	return authenticatorConfig.New()
}

// BuildAuthorizer constructs the authorizer
func BuildAuthorizer(s *options.ServerRunOptions, versionedInformers clientgoinformers.SharedInformerFactory) (authorizer.Authorizer, authorizer.RuleResolver, error) {
	authorizationConfig := s.Authorization.ToAuthorizationConfig(versionedInformers)
	return authorizationConfig.New()
}
