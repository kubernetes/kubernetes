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

	"k8s.io/kubernetes/pkg/controller/bootstrap"

	"k8s.io/kubernetes/cmd/kube-controller-manager/internal/controller"
	"k8s.io/kubernetes/cmd/kube-controller-manager/internal/controller/run"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
)

func newBootstrapSignerControllerDescriptor() *controller.Descriptor {
	return &controller.Descriptor{
		Name:                names.BootstrapSignerController,
		Aliases:             []string{"bootstrapsigner"},
		Constructor:         newBootstrapSignerController,
		IsDisabledByDefault: true,
	}
}

func newBootstrapSignerController(ctx context.Context, controllerContext controller.Context, controllerName string) (controller.Controller, error) {
	client, err := controllerContext.NewClient("bootstrap-signer")
	if err != nil {
		return nil, err
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

	return run.NewControllerLoop(bsc.Run, controllerName), nil
}

func newTokenCleanerControllerDescriptor() *controller.Descriptor {
	return &controller.Descriptor{
		Name:                names.TokenCleanerController,
		Aliases:             []string{"tokencleaner"},
		Constructor:         newTokenCleanerController,
		IsDisabledByDefault: true,
	}
}

func newTokenCleanerController(ctx context.Context, controllerContext controller.Context, controllerName string) (controller.Controller, error) {
	client, err := controllerContext.NewClient("token-cleaner")
	if err != nil {
		return nil, err
	}

	tcc, err := bootstrap.NewTokenCleaner(
		client,
		controllerContext.InformerFactory.Core().V1().Secrets(),
		bootstrap.DefaultTokenCleanerOptions(),
	)
	if err != nil {
		return nil, fmt.Errorf("error creating TokenCleaner controller: %w", err)
	}

	return run.NewControllerLoop(tcc.Run, controllerName), nil
}
