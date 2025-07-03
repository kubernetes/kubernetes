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

// Package app implements a server that runs a set of active
// components.  This includes replication controllers, service endpoints and
// nodes.
package app

import (
	"context"
	"fmt"

	"golang.org/x/sync/errgroup"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/scale"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	"k8s.io/kubernetes/pkg/controller/podautoscaler"
	"k8s.io/kubernetes/pkg/controller/podautoscaler/metrics"
	resourceclient "k8s.io/metrics/pkg/client/clientset/versioned/typed/metrics/v1beta1"
	"k8s.io/metrics/pkg/client/custom_metrics"
	"k8s.io/metrics/pkg/client/external_metrics"
)

func newHorizontalPodAutoscalerControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:     names.HorizontalPodAutoscalerController,
		aliases:  []string{"horizontalpodautoscaling"},
		initFunc: newHorizontalPodAutoscalerController,
	}
}

func newHorizontalPodAutoscalerController(ctx context.Context, controllerContext ControllerContext, controllerName string) (Controller, error) {
	clientConfig, err := controllerContext.ClientBuilder.Config("horizontal-pod-autoscaler")
	if err != nil {
		return nil, fmt.Errorf("failed to init client config for %s: %w", controllerName, err)
	}

	client, err := controllerContext.ClientBuilder.Client("horizontal-pod-autoscaler")
	if err != nil {
		return nil, fmt.Errorf("failed to create a client: %w", err)
	}

	// we don't use cached discovery because DiscoveryScaleKindResolver does its own caching,
	// so we want to re-fetch every time when we actually ask for it
	scaleKindResolver := scale.NewDiscoveryScaleKindResolver(client.Discovery())
	scaleClient, err := scale.NewForConfig(clientConfig, controllerContext.RESTMapper, dynamic.LegacyAPIPathResolverFunc, scaleKindResolver)
	if err != nil {
		return nil, fmt.Errorf("failed to init HPA scale client: %w", err)
	}

	apiVersionsGetter := custom_metrics.NewAvailableAPIsGetter(client.Discovery())

	resourceClient, err := resourceclient.NewForConfig(clientConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to init the resource client for %s: %w", controllerName, err)
	}

	externalMetricsClient, err := external_metrics.NewForConfig(clientConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to init the external metrics client for %s: %w", controllerName, err)
	}

	metricsClient := metrics.NewRESTMetricsClient(
		resourceClient,
		custom_metrics.NewForConfig(clientConfig, controllerContext.RESTMapper, apiVersionsGetter),
		externalMetricsClient,
	)

	pas := podautoscaler.NewHorizontalController(
		ctx,
		client.CoreV1(),
		scaleClient,
		client.AutoscalingV2(),
		controllerContext.RESTMapper,
		metricsClient,
		controllerContext.InformerFactory.Autoscaling().V2().HorizontalPodAutoscalers(),
		controllerContext.InformerFactory.Core().V1().Pods(),
		controllerContext.ComponentConfig.HPAController.HorizontalPodAutoscalerSyncPeriod.Duration,
		controllerContext.ComponentConfig.HPAController.HorizontalPodAutoscalerDownscaleStabilizationWindow.Duration,
		controllerContext.ComponentConfig.HPAController.HorizontalPodAutoscalerTolerance,
		controllerContext.ComponentConfig.HPAController.HorizontalPodAutoscalerCPUInitializationPeriod.Duration,
		controllerContext.ComponentConfig.HPAController.HorizontalPodAutoscalerInitialReadinessDelay.Duration,
	)
	return newNamedRunnableFunc(func(ctx context.Context) error {
		eg, ctx := errgroup.WithContext(ctx)
		eg.Go(func() error {
			custom_metrics.PeriodicallyInvalidate(
				apiVersionsGetter,
				controllerContext.ComponentConfig.HPAController.HorizontalPodAutoscalerSyncPeriod.Duration,
				ctx.Done())
			return nil
		})

		eg.Go(func() error {
			pas.Run(ctx, int(controllerContext.ComponentConfig.HPAController.ConcurrentHorizontalPodAutoscalerSyncs))
			return nil
		})
		return eg.Wait()
	}, controllerName), nil
}
