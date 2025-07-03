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
	"context"
	"fmt"

	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	"k8s.io/kubernetes/pkg/controller/bootstrap"
)

func newBootstrapSignerControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:                names.BootstrapSignerController,
		aliases:             []string{"bootstrapsigner"},
		initFunc:            newBootstrapSignerController,
		isDisabledByDefault: true,
	}
}

func newBootstrapSignerController(ctx context.Context, controllerContext ControllerContext, controllerName string) (Controller, error) {
	client, err := controllerContext.ClientBuilder.Client("bootstrap-signer")
	if err != nil {
		return nil, fmt.Errorf("failed to create a client: %w", err)
	}

	bsc, err := bootstrap.NewSigner(
		client,
		controllerContext.InformerFactory.Core().V1().Secrets(),
		controllerContext.InformerFactory.Core().V1().ConfigMaps(),
		bootstrap.DefaultSignerOptions(),
	)
	if err != nil {
		return nil, fmt.Errorf("error creating BootstrapSigner controller: %w", err)
	}

	return newNamedRunnableFunc(func(ctx context.Context) error {
		bsc.Run(ctx)
		return nil
	}, controllerName), nil
}

func newTokenCleanerControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:                names.TokenCleanerController,
		aliases:             []string{"tokencleaner"},
		initFunc:            newTokenCleanerController,
		isDisabledByDefault: true,
	}
}

func newTokenCleanerController(ctx context.Context, controllerContext ControllerContext, controllerName string) (Controller, error) {
	client, err := controllerContext.ClientBuilder.Client("token-cleaner")
	if err != nil {
		return nil, fmt.Errorf("failed to create a client: %w", err)
	}

	tcc, err := bootstrap.NewTokenCleaner(
		client,
		controllerContext.InformerFactory.Core().V1().Secrets(),
		bootstrap.DefaultTokenCleanerOptions(),
	)
	if err != nil {
		return nil, fmt.Errorf("error creating TokenCleaner controller: %w", err)
	}

	return newNamedRunnableFunc(func(ctx context.Context) error {
		tcc.Run(ctx)
		return nil
	}, controllerName), nil
}
