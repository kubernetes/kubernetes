/*
Copyright 2024 The Kubernetes Authors.

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

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
	clientgofeaturegate "k8s.io/client-go/features"
	"k8s.io/client-go/metadata"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	svm "k8s.io/kubernetes/pkg/controller/storageversionmigrator"
	"k8s.io/kubernetes/pkg/features"
)

func newStorageVersionMigratorControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:        names.StorageVersionMigratorController,
		aliases:     []string{"svm"},
		constructor: newSVMController,
	}
}

func newSVMController(ctx context.Context, controllerContext ControllerContext, controllerName string) (Controller, error) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.StorageVersionMigrator) ||
		!clientgofeaturegate.FeatureGates().Enabled(clientgofeaturegate.InformerResourceVersion) {
		return nil, nil
	}

	if !controllerContext.ComponentConfig.GarbageCollectorController.EnableGarbageCollector {
		return nil, fmt.Errorf("storage version migrator requires garbage collector")
	}

	if !clientgofeaturegate.FeatureGates().Enabled(clientgofeaturegate.InOrderInformers) {
		err := fmt.Errorf("storage version migrator requires the InOrderInformers feature gate to be enabled")
		return nil, err
	}

	// svm controller can make a lot of requests during migration, keep it fast
	config, err := controllerContext.NewClientConfig(controllerName)
	if err != nil {
		return nil, err
	}

	config.QPS *= 20
	config.Burst *= 100

	client, err := controllerContext.NewClient(controllerName)
	if err != nil {
		return nil, err
	}

	informer := controllerContext.InformerFactory.Storagemigration().V1beta1().StorageVersionMigrations()

	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		return nil, err
	}

	discoveryClient, err := discovery.NewDiscoveryClientForConfig(config)
	if err != nil {
		return nil, err
	}

	metaClient, err := metadata.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create metadata client for %s: %w", controllerName, err)
	}

	return newControllerLoop(concurrentRun(
		svm.NewSVMController(
			ctx,
			client,
			dynamicClient,
			informer,
			controllerName,
			controllerContext.RESTMapper,
			controllerContext.GraphBuilder,
		).Run,
		svm.NewResourceVersionController(
			ctx,
			client,
			discoveryClient,
			metaClient,
			informer,
			controllerContext.RESTMapper,
		).Run,
	), controllerName), nil
}
