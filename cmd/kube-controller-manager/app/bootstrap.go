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

	"k8s.io/controller-manager/controller"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	"k8s.io/kubernetes/pkg/controller/bootstrap"
)

func newBootstrapSignerControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:                names.BootstrapSignerController,
		aliases:             []string{"bootstrapsigner"},
		initFunc:            startBootstrapSignerController,
		isDisabledByDefault: true,
	}
}
func startBootstrapSignerController(ctx context.Context, controllerContext ControllerContext, controllerName string) (controller.Interface, bool, error) {
	bsc, err := bootstrap.NewSigner(
		ctx,
		controllerContext.ClientBuilder.ClientOrDie("bootstrap-signer"),
		controllerContext.InformerFactory.Core().V1().Secrets(),
		controllerContext.InformerFactory.Core().V1().ConfigMaps(),
		bootstrap.DefaultSignerOptions(),
	)
	if err != nil {
		return nil, true, fmt.Errorf("error creating BootstrapSigner controller: %v", err)
	}
	go bsc.Run(ctx)
	return nil, true, nil
}

func newTokenCleanerControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:                names.TokenCleanerController,
		aliases:             []string{"tokencleaner"},
		initFunc:            startTokenCleanerController,
		isDisabledByDefault: true,
	}
}
func startTokenCleanerController(ctx context.Context, controllerContext ControllerContext, controllerName string) (controller.Interface, bool, error) {
	tcc, err := bootstrap.NewTokenCleaner(
		ctx,
		controllerContext.ClientBuilder.ClientOrDie("token-cleaner"),
		controllerContext.InformerFactory.Core().V1().Secrets(),
		bootstrap.DefaultTokenCleanerOptions(),
	)
	if err != nil {
		return nil, true, fmt.Errorf("error creating TokenCleaner controller: %v", err)
	}
	go tcc.Run(ctx)
	return nil, true, nil
}
