/*
Copyright 2025 The Kubernetes Authors.

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
	"context"
	"fmt"

	"k8s.io/client-go/util/keyutil"
	"k8s.io/controller-manager/pkg/clientbuilder"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	serviceaccountcontroller "k8s.io/kubernetes/pkg/controller/serviceaccount"
	"k8s.io/kubernetes/pkg/serviceaccount"
)

// serviceAccountTokenController is special because it must run first to set up permissions for other controllers.
// It cannot use the "normal" client builder, so it tracks its own.
func newServiceAccountTokenControllerDescriptor(rootClientBuilder clientbuilder.ControllerClientBuilder) *ControllerDescriptor {
	return &ControllerDescriptor{
		name:    names.ServiceAccountTokenController,
		aliases: []string{"serviceaccount-token"},
		constructor: func(ctx context.Context, controllerContext ControllerContext, controllerName string) (Controller, error) {
			return newServiceAccountTokenController(ctx, controllerContext, controllerName, rootClientBuilder)
		},
		// This controller is started manually before any other controller.
		requiresSpecialHandling: true,
	}
}

func newServiceAccountTokenController(
	ctx context.Context, controllerContext ControllerContext, controllerName string,
	rootClientBuilder clientbuilder.ControllerClientBuilder,
) (Controller, error) {
	if len(controllerContext.ComponentConfig.SAController.ServiceAccountKeyFile) == 0 {
		klog.FromContext(ctx).Info("Controller is disabled because there is no private key", "controller", controllerName)
		return nil, nil
	}

	privateKey, err := keyutil.PrivateKeyFromFile(controllerContext.ComponentConfig.SAController.ServiceAccountKeyFile)
	if err != nil {
		return nil, fmt.Errorf("error reading key for service account token controller: %w", err)
	}

	var rootCA []byte
	if controllerContext.ComponentConfig.SAController.RootCAFile != "" {
		if rootCA, err = readCA(controllerContext.ComponentConfig.SAController.RootCAFile); err != nil {
			return nil, fmt.Errorf("error parsing root-ca-file at %s: %w", controllerContext.ComponentConfig.SAController.RootCAFile, err)
		}
	} else {
		config, err := rootClientBuilder.Config("tokens-controller")
		if err != nil {
			return nil, fmt.Errorf("failed to create Kubernetes client config for %q: %w", "tokens-controller", err)
		}
		rootCA = config.CAData
	}

	client, err := rootClientBuilder.Client("tokens-controller")
	if err != nil {
		return nil, fmt.Errorf("failed to create Kubernetes client for %q: %w", "tokens-controller", err)
	}

	tokenGenerator, err := serviceaccount.JWTTokenGenerator(serviceaccount.LegacyIssuer, privateKey)
	if err != nil {
		return nil, fmt.Errorf("failed to build token generator: %w", err)
	}
	tokenController, err := serviceaccountcontroller.NewTokensController(
		klog.FromContext(ctx),
		controllerContext.InformerFactory.Core().V1().ServiceAccounts(),
		controllerContext.InformerFactory.Core().V1().Secrets(),
		client,
		serviceaccountcontroller.TokensControllerOptions{
			TokenGenerator: tokenGenerator,
			RootCA:         rootCA,
		},
	)
	if err != nil {
		return nil, fmt.Errorf("error creating Tokens controller: %w", err)
	}

	return newControllerLoop(func(ctx context.Context) {
		tokenController.Run(ctx, int(controllerContext.ComponentConfig.SAController.ConcurrentSATokenSyncs))
	}, controllerName), nil
}
